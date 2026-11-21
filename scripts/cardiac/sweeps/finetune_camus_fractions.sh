#!/usr/bin/env bash
# Finetune Pix2Rep-v2 (pretrain_CAMUS/best.ckpt) on CAMUS with 1,5,10,25,100% of training patients.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export CAMUS_DATASET_FOLDER=${CAMUS_DATASET_FOLDER:-$PWD/data/CAMUS_processed}
export WANDB_MODE=online
BACKBONE=${BACKBONE:-pretrain_CAMUS}
mkdir -p output/logs
for pct in ${PCTS:-1 5 10 25 100}; do
  exp="finetune_CAMUS_${pct}pct"
  echo "[$(date)] start $exp"
  python -m scripts.cardiac.finetune_pix2repv2 --finetuning_dataset CAMUS \
    --exp_name "$exp" --backbone_name "$BACKBONE" --ckpt_name best \
    --overrides finetuning.num_patients=$pct > "output/logs/$exp.log" 2>&1
  echo "[$(date)] done $exp (exit $?)"
done
