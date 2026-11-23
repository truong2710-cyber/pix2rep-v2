#!/usr/bin/env bash
# CAMUS baselines: for one strategy (scratch | linear_probing | finetuning), train on
# 1,5,10,25,100% of the training patients, then evaluate each run on the official test set.
#   STRATEGY=scratch        bash scripts/cardiac/sweeps/finetune_camus_strategies.sh
#   STRATEGY=linear_probing bash scripts/cardiac/sweeps/finetune_camus_strategies.sh
# Uses the same nested, quality-stratified patient subsets as finetune_camus_fractions.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export CAMUS_DATASET_FOLDER=${CAMUS_DATASET_FOLDER:-$PWD/data/CAMUS_processed}
export WANDB_MODE=${WANDB_MODE:-online}
STRATEGY=${STRATEGY:?set STRATEGY to scratch, linear_probing or finetuning}
BACKBONE=${BACKBONE:-pretrain_CAMUS_ep12}   # frozen copy of a pretraining checkpoint
TAG=${TAG:-}                               # appended to exp names, e.g. TAG=_ep27
case "$STRATEGY" in
  scratch) SUFFIX=scratch ;;
  linear_probing) SUFFIX=linprobe ;;
  finetuning) SUFFIX=finetune ;;
  *) echo "unknown STRATEGY=$STRATEGY"; exit 1 ;;
esac
mkdir -p output/logs
for pct in ${PCTS:-1 5 10 25 100}; do
  exp="finetune_CAMUS_${pct}pct_${SUFFIX}${TAG}"
  echo "[$(date)] start $exp"
  python -m scripts.cardiac.finetune_pix2repv2 --finetuning_dataset CAMUS \
    --exp_name "$exp" --backbone_name "$BACKBONE" --ckpt_name best --strategy "$STRATEGY" \
    --overrides finetuning.num_patients=$pct > "output/logs/$exp.log" 2>&1
  rc=$?  # capture before $(date) resets $?
  echo "[$(date)] done $exp (exit $rc)"
  python -m scripts.cardiac.evaluate_pix2repv2 --evaluation_dataset CAMUS \
    --finetuning_exp_name "$exp" --split test --ckpt_name best > "output/logs/eval_$exp.log" 2>&1
  rc=$?
  echo "[$(date)] evaluated $exp (exit $rc)"
done
echo "[$(date)] sweep $STRATEGY finished"
