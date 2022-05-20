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
M=25
if [ $SOURCE = 'usps' ]
then
    SCALE=1
else
    SCALE=2
fi
M=$(echo "scale=0; $M * $SCALE" | bc -l)
TEST_INTERVAL=1
CLASS=10
EPSILON=0.1
TAU=1.
MASS=0.9
ETA1=0.1
ETA2=0.1
SEED=1980

for K in 1 2 4 8
do
    EPOCH=$(echo "scale=0; $K * 10" | bc -l)
    for LR in 2e-4
    do
        python train_digits.py --gpu_id ${gpus} --method $METHOD --source_ds $SOURCE --target_ds $TARGET --data_dir $DATA_DIR --k $K --mbsize $M --n_epochs $EPOCH --test_interval $TEST_INTERVAL --nclass $CLASS --epsilon $EPSILON --tau $TAU --mass $MASS --lr $LR --eta1 $ETA1 --eta2 $ETA2 --num_workers 4 --seed $SEED
    done
done