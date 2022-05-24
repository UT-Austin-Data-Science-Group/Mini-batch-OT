#!/bin/bash
# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

# choose the method
METHOD=jdot
SOURCE=svhn
TARGET=mnist
DATA_DIR=./data
K=1
BATCH_SIZE=500
EPOCH=100
TEST_INTERVAL=1
CLASS=10
EPSILON=0.0
TAU=1.
ETA1=0.1
ETA2=0.1
MASS=0.85
LR=4e-4
seed=1980

python train_digits.py --gpu_id ${gpus} --method $METHOD --source_ds $SOURCE --target_ds $TARGET --data_dir $DATA_DIR --k $K --mbsize $BATCH_SIZE --n_epochs $EPOCH --test_interval $TEST_INTERVAL --nclass $CLASS --epsilon $EPSILON --tau $TAU --mass $MASS --lr $LR --eta1 $ETA1 --eta2 $ETA2 --num_workers 8 --seed $seed