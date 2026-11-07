#!/bin/bash

# Smoke test for fast PLM training with average reward + marginal entropy
# Requires: pgld_plm environment, GPU

echo "Launching PLM fast mode smoke test..."

python pgld/train.py \
    landscape=plm_linear \
    policy=iupac \
    policy.num_templates=64 \
    training.optimization.iters=10000 \
    training.optimization.lr=0.02 \
    training.sampling.mc_samples=1 \
    training.sampling.batch_size=256 \
    training.reward_type=average \
    training.reward_aggregation_unit=template \
    training.per_sequence_rewards=false \
    training.entropy.lambda_=1 \
    training.entropy.sequence_type=peptide \
    training.entropy.type=marginal_gain \
    training.entropy.baseline_per_template=true \
    training.pooling.uniqueness=peptide \
    training.pooling.enabled=true \
    training.evaluation.extra_eval=false \
    training.evaluation.checkpointing=false \
    ++training.fast_mode=true

echo "PLM fast mode smoke test complete."
