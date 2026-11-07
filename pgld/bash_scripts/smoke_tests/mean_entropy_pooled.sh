#!/bin/bash

# Smoke test for template-level credit assignment with IUPAC policy
# Exercises reward_aggregation_unit=template, average rewards, and pooled training
# Requires: pgld_basic environment

ASSAYS=(
    "PAX6_REF_R1"
)

echo "Launching template-level credit assignment smoke test..."

for assay in "${ASSAYS[@]}"; do
    echo "Launching job for assay: $assay"
    python pgld/train.py \
        landscape=tf_binding \
	landscape.objectives=null \
	++landscape.objectives.$assay.top_prop=0.05 \
	++landscape.objectives.$assay.invert=false \
	policy=iupac \
	policy.num_templates=10 \
	training.sampling.batch_size=128 \
	training.reward_type=average \
	training.reward_aggregation_unit=template \
	training.per_sequence_rewards=false \
	training.sampling.mc_samples=10 \
	training.entropy.lambda_=0.01 \
	training.entropy.type=marginal_gain \
	training.optimization.iters=1000 \
	training.pooling.enabled=true \
	training.pooling.gpu_accelerated=false \
	training.pooling.queries_per_batch=10 \
	training.pooling.uniqueness=dna \
	training.evaluation.checkpointing=false

    sleep 1
done

echo "Template-level credit assignment smoke test complete."
