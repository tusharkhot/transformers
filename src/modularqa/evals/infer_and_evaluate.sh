#!/usr/bin/env bash

set -e

L=$L N=$N P=$P K=$K B=$B DS=$DS DLOAD=$DLOAD HLOAD=$HLOAD SLOAD=$SLOAD envsubst < /configs/config.json > ${OUTPUT}/new_config.json

python -m modularqa.inference.configurable_inference \
        --input $INPUT/$FILE \
        --output $OUTPUT/predictions_$FILE \
        --config ${OUTPUT}/new_config.json --reader $DATASET

if [[ "$DATASET" == "drop" ]]; then
  python -m modularqa.evals.drop_eval \
    --gold_path $INPUT/$FILE \
    --prediction_path $OUTPUT/predictions_$FILE \
    --output_path $OUTPUT/metrics.json
else
    python -m modularqa.evals.evaluate_hotpot_squad_format \
    $OUTPUT/predictions_$FILE $INPUT/$FILE > $OUTPUT/metrics.json
fi
