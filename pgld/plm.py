"""
plm.py

Protein Language Models for likelihood inference
"""

from __future__ import annotations

import logging
import time

import torch.nn.functional as F
import torch
import os
import numpy as np
from abc import ABC

from constants import AMINO_ACIDS, IGLM_VOCAB
from esme.alphabet import tokenize

log = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True


class PLM(ABC):
    """Abstract base class for protein language models that can predict sequence log-probabilities"""

    def prepare(self, sequences: list):
        """Tokenize string sequences"""
        raise NotImplementedError

    def get_log_probs(self, xs: torch.Tensor):
        """Get log probabilities of tokenized sequences"""
        raise NotImplementedError


class LinearLogLH(PLM, torch.nn.Module):
    """Simple product LH with random categorical distribution at each position"""

    def __init__(
        self,
        device,
        str_len: int,
        alphabet: list,
        unmasked_pos: list = [],
        seed: int = 42,
        default_token_logprob=-100,
    ):
        torch.nn.Module.__init__(self)
        gen = torch.Generator().manual_seed(
            seed
        )  # ensure we always get the same model regardless of global seed
        self.logits = torch.nn.Parameter(
            torch.randn(str_len - len(unmasked_pos), len(alphabet), generator=gen)
        ).to(device)
        self.device = device
        self.alphabet = alphabet
        self.tok2idx = {a: i for i, a in enumerate(alphabet)}
        self.unmasked_pos = unmasked_pos
        self.variable_msk = torch.ones(str_len, dtype=bool)
        self.variable_msk[self.unmasked_pos] = False
        self.str_len = str_len
        self.name = "linear"
        self.default_token_logprob = torch.tensor(default_token_logprob, device=device)

    def prepare(self, sequences: list):
        xs = torch.zeros((len(sequences), self.str_len), dtype=int, device=self.device)
        msk = torch.zeros((len(sequences), self.str_len), dtype=int, device=self.device)
        for i, seq in enumerate(sequences):
            xs[i, : len(seq)] = torch.tensor([self.tok2idx[a] for a in seq])
            msk[i, : len(seq)] = 1
        return {"xs": xs, "mask": msk}

    def get_log_probs(self, batch: dict):
        is_cuda = self.device == "cuda" or (
            hasattr(self.device, "type") and self.device.type == "cuda"
        )
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            log_probs_all = F.log_softmax(self.logits, dim=-1)
        idx = (
            torch.arange(self.logits.shape[0]).unsqueeze(0).expand(len(batch["xs"]), -1)
        )
        log_probs = (
            log_probs_all[idx, batch["xs"][:, self.variable_msk]]
            * batch["mask"][:, self.variable_msk]
        )
        log_probs = torch.clamp(log_probs, min=self.default_token_logprob)
        result = log_probs.sum(dim=-1)

        if is_cuda:
            torch.cuda.synchronize()
        log.info(
            f"[TIMING] LinearLogLH.get_log_probs (n={len(batch['xs'])}): {(time.perf_counter() - t0) * 1000:.1f}ms"
        )
        return result

    def score_tensor_batch(self, peptide_indices: torch.Tensor) -> torch.Tensor:
        """
        Score peptides given as tensor of AA indices (vectorized).

        Args:
            peptide_indices: (U, L) int tensor, values 0-19 for AAs, -1 for masked

        Returns:
            (U,) float tensor of log-likelihood scores
        """
        U, L = peptide_indices.shape

        with torch.no_grad():
            log_probs_all = F.log_softmax(self.logits, dim=-1)  # (L, alphabet_size)

        # Handle variable positions
        L_var = log_probs_all.shape[0]
        if L > L_var:
            peptide_indices = peptide_indices[:, :L_var]
            L = L_var

        # Ensure indices are long type for gather
        peptide_indices = peptide_indices.long()
        valid_mask = peptide_indices >= 0
        indices_safe = peptide_indices.clamp(min=0)

        log_probs_exp = log_probs_all.unsqueeze(0).expand(U, -1, -1)  # (U, L, alphabet)
        gathered = log_probs_exp.gather(2, indices_safe.unsqueeze(-1)).squeeze(
            -1
        )  # (U, L)
        gathered = torch.clamp(gathered, min=self.default_token_logprob)
        gathered = gathered * valid_mask

        return gathered.sum(dim=-1)  # (U,)


class MaskedPLM(PLM, ABC):
    """
    Abstract base class for masked protein language models.
    Provides functionality to compute pseudo log probabilities.
    """

    def get_logits(self, xs: torch.Tensor):
        """
        Get model logits in single precision for a batch of sequences
        """
        raise NotImplementedError

    def get_log_probs(self, xs: torch.Tensor):
        """
        Get pseudo-logLH for a batch of sequences.
        Masks each variable token sequentially
        """
        log_probs = torch.zeros(len(xs), device=self.device)
        # avoid bos and eos
        idx = torch.arange(1, xs.shape[1] - 1, dtype=int)
        # avoid fixed positions
        valid_unmasked_pos = self.unmasked_pos[self.unmasked_pos < xs.shape[1] - 2]
        if len(valid_unmasked_pos) > 0:
            idx[valid_unmasked_pos] = -1
            idx = idx[idx != -1]
        unpadded = xs != self.pad_token
        with torch.inference_mode():
            for i in idx:
                # only calculate probs for seqs that haven't terminated yet
                xs_curr = xs[unpadded[:, i]].clone()
                xs_curr[:, i] = self.mask_token
                # get logits at masked position
                preds = self.get_logits(xs_curr)[:, i]
                # gather probs
                preds = (
                    F.log_softmax(preds, dim=-1)
                    .gather(dim=-1, index=xs[unpadded[:, i]][:, [i]])
                    .squeeze()
                )

                preds = torch.clamp(preds, min=self.default_token_logprob)
                log_probs[unpadded[:, i]] += preds

        return log_probs


class ESM2Wrapper(MaskedPLM):
    """Wrapper for FAIR-ESM models"""

    def __init__(
        self,
        device="cpu",
        model_name: str = "esm2_t6_8M_UR50D",
        unmasked_pos: list = [],
        default_token_logprob: float = -100,
    ):
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.name = model_name
        self.model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}").half()
        self.model.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.unmasked_pos = np.array(unmasked_pos)
        self.device = device
        self.mask_token = self.tokenizer._token_to_id[self.tokenizer.mask_token]
        self.pad_token = self.tokenizer._token_to_id[self.tokenizer.pad_token]
        self.default_token_logprob = torch.tensor(default_token_logprob, device=device)

    def prepare(self, sequences: list):
        return self.tokenizer(sequences, return_tensors="pt")["input_ids"].to(
            self.device
        )

    def get_logits(self, xs: torch.Tensor):
        return self.model(xs).logits


class FastESM2Wrapper(MaskedPLM, torch.nn.Module):
    """
    Wrapper for ESM-Efficient models.

    Multi-GPU Support:
        Call set_n_gpus(n) to wrap the model with DataParallel.
        Batches all 24 masked positions into one forward pass (P*U, L),
        which DataParallel automatically splits across GPUs.
    """

    def __init__(
        self,
        device="cpu",
        model_name: str = "esm2_8m",
        path: str = ".",
        unmasked_pos: list = [],
        default_token_logprob: float = -100,
    ):
        from esme import ESM2
        from esme.alphabet import Alphabet

        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.nn.Module.__init__(self)
        self.model = ESM2.from_pretrained(
            os.path.join(path, f"{model_name}.safetensors"), device=device
        ).eval()
        self.device = device
        self.alphabet = Alphabet()
        self.unmasked_pos = np.array(unmasked_pos)
        self.name = model_name
        self.mask_token = self.alphabet.mask_idx
        self.pad_token = self.alphabet.padding_idx
        self.default_token_logprob = torch.tensor(default_token_logprob, device=device)

        # Build AA index (0-19) to ESM token mapping
        self._aa_to_esm_token = torch.zeros(20, dtype=torch.long, device=device)
        for i, aa in enumerate(AMINO_ACIDS[:20]):
            self._aa_to_esm_token[i] = self.alphabet.token_to_idx[aa]

        # Multi-GPU: set by set_n_gpus()
        self.n_gpus = 1
        self._dp_model = None

    def set_n_gpus(self, n_gpus: int):
        """Enable multi-GPU scoring using DataParallel."""
        if n_gpus <= 1:
            return

        self.n_gpus = n_gpus
        device_ids = list(range(n_gpus))
        self._dp_model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        log.info(f"ESM2: Using DataParallel across {n_gpus} GPUs")

    def forward(self, input_ids: torch.Tensor):
        self.model.forward_representation(
            input_ids, layers=list(range(self.model.layers))
        )

    def prepare(self, sequences):
        return tokenize(sequences).to(self.device)

    def get_logits(self, xs: torch.Tensor):
        return self.model(xs).to(torch.float32)

    def get_log_probs(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Get pseudo-logLH via batched masked forward pass.

        Instead of 24 sequential forward passes, creates all masked versions
        as one batch and runs a single forward pass. DataParallel splits across GPUs.

        Args:
            xs: (U, L) input tokens
        """
        U, L = xs.shape
        device = xs.device

        # Get variable positions to mask (exclude BOS, EOS, and fixed context)
        all_positions = torch.arange(1, L - 1, device=device)
        context_indices = self.unmasked_pos[self.unmasked_pos < L - 2]
        if len(context_indices) > 0:
            is_variable = torch.ones(L - 2, dtype=torch.bool, device=device)
            is_variable[context_indices] = False
            positions = all_positions[is_variable]
        else:
            positions = all_positions

        P = len(positions)  # number of positions to mask (typically 24)

        # Build masked versions: (P, U, L)
        xs_masked = xs.unsqueeze(0).expand(P, -1, -1).clone()
        for i in range(P):
            xs_masked[i, :, positions[i]] = self.mask_token

        # Forward pass: (P*U, L) -> (P*U, L, V) -> (P, U, L, V)
        xs_flat = xs_masked.view(P * U, L)
        with torch.inference_mode():
            model = self._dp_model if self.n_gpus > 1 else self.model
            logits_flat = model(xs_flat).to(torch.float32)
        V = logits_flat.shape[-1]
        logits = logits_flat.view(P, U, L, V)

        # Extract log prob of true token at each masked position
        log_probs = torch.zeros(U, device=device)
        for i in range(P):
            pos = positions[i]
            masked_logits = logits[i, :, pos, :]  # (U, V)
            true_tokens = xs[:, pos]  # (U,)
            pos_log_probs = F.log_softmax(masked_logits, dim=-1)
            pos_log_probs = pos_log_probs.gather(
                dim=-1, index=true_tokens.unsqueeze(-1)
            ).squeeze(-1)
            log_probs += torch.clamp(
                pos_log_probs, min=self.default_token_logprob.item()
            )

        return log_probs

    def score_tensor_batch(self, peptide_indices: torch.Tensor) -> torch.Tensor:
        """
        Score peptides given as AA index tensor via pseudolikelihood.

        Args:
            peptide_indices: (U, L) tensor, values 0-19 for AAs, -1 for padding

        Returns:
            (U,) tensor of pseudolog-likelihood scores
        """
        device = peptide_indices.device
        tokens = self._aa_to_esm_token.to(device)[peptide_indices.clamp(min=0)]
        tokens = torch.where(peptide_indices >= 0, tokens, self.pad_token)

        U = tokens.shape[0]
        cls_col = torch.full(
            (U, 1), self.alphabet.cls_idx, device=device, dtype=tokens.dtype
        )
        eos_col = torch.full(
            (U, 1), self.alphabet.eos_idx, device=device, dtype=tokens.dtype
        )
        xs = torch.cat([cls_col, tokens, eos_col], dim=1)

        return self.get_log_probs(xs)


class AbLang2Wrapper(MaskedPLM):
    """Wrapper for the masked AbLang2 model from Olsen et al."""

    def __init__(
        self, device: str = "cpu", unmasked_pos: list = [], default_token_logprob=-100
    ):
        import ablang2
        from ablang2.models.ablang2.vocab import ablang_vocab

        self.device = device
        self.name = "ablang2-paired"
        self.model = ablang2.pretrained(
            model_to_use="ablang2-paired", random_init=False, ncpu=1, device=device
        )
        self.mask_token = ablang_vocab["*"]
        self.pad_token = ablang_vocab["-"]
        self.unmasked_pos = np.array(
            unmasked_pos
        )  # should include heavy and light chain separator
        self.default_token_logprob = torch.tensor(default_token_logprob, device=device)

    def prepare(self, sequences: list):
        return self.model.tokenizer(
            sequences, pad=True, device=self.device, w_extra_tkns=False
        )

    def get_logits(self, xs):
        with torch.no_grad(), torch.autocast(device_type=self.device):
            logits = self.model.AbLang(xs)
            return logits.to(torch.float32)


def _build_iglm_ascii_lookup() -> torch.Tensor:
    """Build ASCII lookup table for fast char->token conversion."""
    lookup = torch.full((128,), IGLM_VOCAB["[UNK]"], dtype=torch.long)
    for char, tok_id in IGLM_VOCAB.items():
        if (
            len(char) == 1
        ):  # only single-char tokens (AAs), skip special tokens like [PAD]
            lookup[ord(char)] = tok_id
    return lookup


# Module-level constant for IgLM ASCII lookup
_IGLM_ASCII_LOOKUP = _build_iglm_ascii_lookup()


class IgLMWrapper(PLM):
    """Wrapper for the autoregressive IgLM model from Shuai et al."""

    def __init__(
        self,
        device: str = "cpu",
        unmasked_pos: list = [],
        model_name: str = "IgLM",
        use_infill: bool = False,
    ):
        """
        Args:
            device: Device to run the model on.
            unmasked_pos: Positions that are fixed context (left/right flanks).
            model_name: "IgLM" (full, 13M params) or "IgLM-S" (small, 1.5M params).
            use_infill: If True, use infill scoring (conditions on both left and right context).
        """
        import iglm

        self.device = device
        self.name = model_name.lower()
        self.model = iglm.IgLM(model_name=model_name)
        self.model.model.to(device).eval()
        self.context_positions = np.array(unmasked_pos)
        self.pad_token = 0  # [PAD] token ID
        self.use_infill = use_infill

        self._aa_to_iglm_token = torch.zeros(20, dtype=torch.long, device=device)
        for i, aa in enumerate(AMINO_ACIDS[:20]):
            self._aa_to_iglm_token[i] = IGLM_VOCAB[aa]

        self._char_to_token = _IGLM_ASCII_LOOKUP

        # Precompute left context length (contiguous positions from start)
        self._left_ctx_len = 0
        sorted_ctx = sorted(self.context_positions)
        for i, pos in enumerate(sorted_ctx):
            if pos != i:
                break
            self._left_ctx_len = i + 1

        # Right context = total context positions minus left context
        self._right_ctx_len = len(self.context_positions) - self._left_ctx_len

    def prepare(
        self, sequences: list, chain: str = "[HEAVY]", species: str = "[HUMAN]"
    ) -> torch.Tensor:
        """
        Tokenize sequences for IgLM scoring.

        Args:
            sequences: List of AA strings with context already appended (left_ctx + peptide + right_ctx).
            chain: Chain type token ("[HEAVY]" or "[LIGHT]").
            species: Species token (e.g., "[HUMAN]", "[MOUSE]").

        Returns:
            Token tensor of shape (batch_size, seq_len + 3) with [chain, species, ...seq..., SEP].
        """
        chain_tok = IGLM_VOCAB[chain]
        species_tok = IGLM_VOCAB[species]
        sep_tok = IGLM_VOCAB["[SEP]"]

        seq_array = np.array(
            [[ord(c) for c in seq] for seq in sequences], dtype=np.int64
        )
        token_ids = self._char_to_token[seq_array]

        batch_size = token_ids.shape[0]
        prefix = torch.full((batch_size, 2), 0, dtype=torch.long)
        prefix[:, 0] = chain_tok
        prefix[:, 1] = species_tok
        suffix = torch.full((batch_size, 1), sep_tok, dtype=torch.long)

        return torch.cat([prefix, token_ids, suffix], dim=1).to(self.device)

    def _get_left_ctx_mask(self, seq_len: int) -> torch.Tensor | None:
        """Get mask that zeros out left context positions."""
        if self._left_ctx_len == 0:
            return None
        mask = torch.ones(seq_len, device=self.device)
        mask[: min(self._left_ctx_len, seq_len)] = 0
        return mask

    def get_log_probs(self, xs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model.model(xs).logits
        logprobs_all = F.log_softmax(logits, dim=-1)
        logprobs = logprobs_all[..., 1:-2, :].gather(
            dim=-1, index=xs[..., 2:-1][..., None].long()
        )
        logprobs = logprobs.squeeze(-1)

        # The prob of the left context is basically a constant so we can mask it
        mask = self._get_left_ctx_mask(logprobs.shape[-1])
        if mask is not None:
            logprobs = logprobs * mask

        return logprobs.sum(dim=-1)

    def score_tensor_batch(
        self,
        peptide_indices: torch.Tensor,
        chain: str = "[HEAVY]",
        species: str = "[HUMAN]",
    ) -> torch.Tensor:
        """
        Score peptides given as AA index tensor.

        Args:
            peptide_indices: (U, L) tensor, values 0-19 for AAs, -1 for padding
                L = left_ctx_len + variable_len + right_ctx_len
            chain: Chain type token
            species: Species token

        Returns:
            (U,) tensor of log-likelihood scores
        """
        if self.use_infill:
            return self._score_infill(peptide_indices, chain, species)
        else:
            return self._score_autoregressive(peptide_indices, chain, species)

    def _score_autoregressive(
        self,
        peptide_indices: torch.Tensor,
        chain: str = "[HEAVY]",
        species: str = "[HUMAN]",
    ) -> torch.Tensor:
        """Autoregressive scoring: P(variable, right_ctx | left_ctx)."""
        device = peptide_indices.device
        tokens = self._aa_to_iglm_token.to(device)[peptide_indices.clamp(min=0).long()]
        tokens = torch.where(peptide_indices >= 0, tokens, self.pad_token)

        U = tokens.shape[0]
        chain_tok = IGLM_VOCAB[chain]
        spec_tok = IGLM_VOCAB[species]
        sep_tok = IGLM_VOCAB["[SEP]"]

        prefix = torch.tensor(
            [[chain_tok, spec_tok]], device=device, dtype=tokens.dtype
        )
        prefix = prefix.expand(U, -1)
        suffix = torch.full((U, 1), sep_tok, device=device, dtype=tokens.dtype)
        xs = torch.cat([prefix, tokens, suffix], dim=1).long()

        return self.get_log_probs(xs)

    def _score_infill(
        self,
        peptide_indices: torch.Tensor,
        chain: str = "[HEAVY]",
        species: str = "[HUMAN]",
    ) -> torch.Tensor:
        """
        Infill scoring: P(variable | left_ctx, right_ctx).

        Format: [chain, species, left_ctx, MASK, right_ctx, SEP, variable, CLS]
        Only evaluates log-probs for the variable tokens (after SEP).
        """
        device = peptide_indices.device
        U, L = peptide_indices.shape

        # Convert all to IgLM tokens
        tokens = self._aa_to_iglm_token.to(device)[peptide_indices.clamp(min=0).long()]
        tokens = torch.where(peptide_indices >= 0, tokens, self.pad_token)

        # Split into left_ctx, variable, right_ctx
        left_ctx = tokens[:, : self._left_ctx_len]
        right_ctx = (
            tokens[:, -self._right_ctx_len :]
            if self._right_ctx_len > 0
            else tokens[:, :0]
        )
        variable = tokens[:, self._left_ctx_len : L - self._right_ctx_len]
        var_len = variable.shape[1]

        # Build infill format: [chain, species, left_ctx, MASK, right_ctx, SEP, variable, CLS]
        chain_tok = IGLM_VOCAB[chain]
        spec_tok = IGLM_VOCAB[species]
        mask_tok = IGLM_VOCAB["[MASK]"]
        sep_tok = IGLM_VOCAB["[SEP]"]
        cls_tok = IGLM_VOCAB["[CLS]"]

        prefix = torch.tensor(
            [[chain_tok, spec_tok]], device=device, dtype=tokens.dtype
        ).expand(U, -1)
        mask = torch.full((U, 1), mask_tok, device=device, dtype=tokens.dtype)
        sep = torch.full((U, 1), sep_tok, device=device, dtype=tokens.dtype)
        cls = torch.full((U, 1), cls_tok, device=device, dtype=tokens.dtype)

        # [chain, species, left_ctx, MASK, right_ctx, SEP, variable, CLS]
        xs = torch.cat(
            [prefix, left_ctx, mask, right_ctx, sep, variable, cls], dim=1
        ).long()

        # Compute log-probs for entire sequence
        with torch.no_grad():
            logits = self.model.model(xs).logits

        # Only evaluate variable tokens (positions after SEP, before CLS)
        # SEP is at position: 2 + left_ctx_len + 1 + right_ctx_len
        eval_start = 2 + self._left_ctx_len + 1 + self._right_ctx_len  # position of SEP

        # Get logits for variable positions only
        var_logits = logits[:, eval_start : eval_start + var_len, :]

        # Non-AA tokens are not masked, so sequences that would end early (via [CLS])
        # score lower, by design.
        var_logprobs = F.log_softmax(var_logits, dim=-1)
        var_targets = xs[:, eval_start + 1 : eval_start + var_len + 1]
        logprobs = var_logprobs.gather(dim=-1, index=var_targets.unsqueeze(-1)).squeeze(
            -1
        )

        return logprobs.sum(dim=-1)
