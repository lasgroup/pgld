#!/bin/bash

# Smoke test for pooled training with IUPAC policy
# Requires: pgld_basic environment
# Tests pooling across mc_samples with aggressive subsampling

ASSAYS=(
    "PAX6_REF_R1"
)

echo "Launching pooled training smoke test..."

for assay in "${ASSAYS[@]}"; do
    echo "Launching job for assay: $assay"
    python pgld/train.py \
        landscape=tf_binding \
	landscape.objectives=null \
	++landscape.objectives.$assay.top_prop=0.05 \
	++landscape.objectives.$assay.invert=false \
	training.entropy.type='gradient_estimate' \
	policy=iupac \
	training.sampling.batch_size=128 \
	training.reward_type=hits \
	training.per_sequence_rewards=false \
	training.sampling.mc_samples=100 \
	training.entropy.lambda_=0.0 \
	training.optimization.iters=500 \
	training.pooling.enabled=true \
	training.pooling.queries_per_batch=10 \
	training.pooling.uniqueness=dna \
	training.evaluation.checkpointing=false
    sleep 1
done

echo "Pooled training smoke test complete."
