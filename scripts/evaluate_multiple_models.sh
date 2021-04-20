#!/bin/bash

if [ $# -lt 1 ]
then
    echo 'you must provide at least one checkpoint folder path'
    exit 1
fi

NUM_MODELS=$#
MODELS=$@

CONFIG_OUT_DIR=/home/npf290/dev/xl-wsd-code/tmp_configs
mkdir $CONFIG_OUT_DIR
PYTHONPATH=. python src/evaluation/evaluation_config_creation.py --config_template config/config_en_semcor_wngt.test.yaml --out_config_dir $CONFIG_OUT_DIR --model_paths $MODELS

for f in `ls $CONFIG_OUT_DIR`
do
    echo 'sbatch /home/npf290/dev/xl-wsd-code/scripts/evaluate_model.sh ' $CONFIG_OUT_DIR/$f
    sbatch /home/npf290/dev/xl-wsd-code/scripts/evaluate_model.sh $CONFIG_OUT_DIR/$f
done
