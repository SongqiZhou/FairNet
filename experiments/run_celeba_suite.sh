#!/usr/bin/env bash
# Reproduce every CelebA number the paper reports, in dependency order.
#
#   bash experiments/run_celeba_suite.sh <celeba-root> [out-dir] [seeds...]
#
# The ERM run is first because it writes the shared Stage 1 base model that all
# other variants load, so the Table 1 rows differ only in Stages 2-4.
set -euo pipefail

# A very high core count can make OpenBLAS abort inside scikit-learn; cap it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

DATA_ROOT="${1:?usage: run_celeba_suite.sh <celeba-root> [out-dir] [seeds...]}"
OUT_DIR="${2:-results}"
shift $(( $# > 2 ? 2 : $# ))
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 2)
fi

PYTHON="${PYTHON:-python}"
RUN="$PYTHON experiments/run_experiment.py --data-root $DATA_ROOT --out-dir $OUT_DIR"

for seed in "${SEEDS[@]}"; do
  CKPT="$OUT_DIR/ckpt/celeba_stage1_seed${seed}.pt"
  mkdir -p "$(dirname "$CKPT")"

  # Table 1 rows.
  $RUN --config celeba_erm        --seed "$seed" --stage1-checkpoint "$CKPT"
  $RUN --config celeba_full       --seed "$seed" --stage1-checkpoint "$CKPT"
  $RUN --config celeba_partial    --seed "$seed" --stage1-checkpoint "$CKPT" --sweep-threshold
  $RUN --config celeba_unlabeled  --seed "$seed" --stage1-checkpoint "$CKPT"

  # Table 3 ablations, plus the full-label ablation of Supplementary Table C.
  $RUN --config celeba_ablate_detector          --seed "$seed" --stage1-checkpoint "$CKPT"
  $RUN --config celeba_ablate_contrastive       --seed "$seed" --stage1-checkpoint "$CKPT"
  $RUN --config celeba_ablate_both              --seed "$seed" --stage1-checkpoint "$CKPT"
  $RUN --config celeba_full_ablate_contrastive  --seed "$seed" --stage1-checkpoint "$CKPT"

  # The same backbone trained to convergence, to show how much of FairNet's
  # gain depends on the base model still underfitting the minority group.
  CONVERGED="$OUT_DIR/ckpt/celeba_stage1_converged_seed${seed}.pt"
  $RUN --config celeba_erm_converged  --seed "$seed" --stage1-checkpoint "$CONVERGED"
  $RUN --config celeba_full_converged --seed "$seed" --stage1-checkpoint "$CONVERGED"

  # Supplementary Table 5: FairNet-Partial against the labelled fraction.
  for fraction in 0.001 0.005 0.01 0.05 0.10 0.50; do
    $RUN --config celeba_partial --seed "$seed" \
      --stage1-checkpoint "$CKPT" \
      --set "labeled_fraction=$fraction" \
      --name "celeba_partial_frac${fraction}"
  done
done

$PYTHON experiments/aggregate_results.py --results "$OUT_DIR"
