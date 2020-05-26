#!/usr/bin/env bash

for ckpt_dir in ${MODEL_DIR}/checkpoint-*; do
    MOD_DIR=${ckpt_dir} envsubst < ${CONFIG_FILE} > ${OUTPUT}/new_config.json
    ckpt_num=${ckpt_dir#*-}
    python -m modularqa.inference.configurable_inference \
        --input ${INPUT}/${FILE} \
        --output ${OUTPUT}/ckpt_${ckpt_num}_predictions_${FILE} \
        --config ${OUTPUT}/new_config.json --reader ${DATASET_TYPE}
done