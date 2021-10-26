#!/bin/bash
# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
METHOD=jumbot
SOURCE=usps
TARGET=mnist
DATA_DIR=./data
K=1
BATCH_SIZE=500
EPOCH=100
TEST_INTERVAL=1
CLASS=10
if [ $METHOD = 'jumbot' ]
then
    EPSILON=0.1
else
    EPSILON=0
fi
TAU=1.
MASS=0.9
if [ $METHOD = 'jdot' ]
then
    ETA1=0.001
    ETA2=0.0001
else
    ETA1=0.1
    ETA2=0.1
fi
for LR in 2e-4
do
    python train_digits.py --gpu_id ${gpus} --method $METHOD --source_ds $SOURCE --target_ds $TARGET --data_dir $DATA_DIR --k $K --mbsize $BATCH_SIZE --n_epochs $EPOCH --test_interval $TEST_INTERVAL --nclass $CLASS --epsilon $EPSILON --tau $TAU --mass $MASS --lr $LR --eta1 $ETA1 --eta2 $ETA2
done