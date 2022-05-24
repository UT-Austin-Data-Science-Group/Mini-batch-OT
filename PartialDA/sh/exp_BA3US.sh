#!/bin/bash
# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

method='BA3US'
for S in {0..3}
do
    for T in {0..3}
    do
        if [ $S != $T ]
        then
        python run_${method}.py --s $S --t $T --dset office_home --net ResNet50 --cot_weight 1. --output $method --gpu_id $gpus
        fi
    done
done