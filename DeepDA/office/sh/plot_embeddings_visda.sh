#!/bin/bash

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

echo ""
echo -n "run_id: "
read run_id

echo ""
echo "Input path to weights"
echo -n "OT: "
read OT_dir
echo -n "UOT: "
read UOT_dir
echo -n "POT: "
read POT_dir

echo ""
echo -n "Version (m or TS): "
read version

s_dset_path="./data/visda-2017/train_list.txt"
t_dset_path="./data/visda-2017/validation_list.txt"

output_dir="plot_visda_run${run_id}"

echo "Begin in ${output_dir}"
python plot_embeddings.py \
    --gpu_id ${gpus} \
    --net ResNet50 \
    --dset visda \
    --s_dset_path ${s_dset_path} \
    --t_dset_path ${t_dset_path} \
    --output_dir ${output_dir} \
    --restore_dir ${OT_dir} ${UOT_dir} ${POT_dir} \
    --titles ${version}-OT ${version}-UOT ${version}-POT
echo "Finish in ${output_dir}"