#!/bin/bash
# python PATH
# export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

gpus=1
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
ot_type=balanced
use_bomb=no
threshold=0.9
run_id=0
# OT parameters
ETA1=0.01
ETA2=0.5
if [ $ot_type = 'unbalanced' ]
then
    EPSILON=0.01
else
    EPSILON=0
fi
TAU=0.5
ITER=10000
TEST_INTERVAL=500
M=65
K=2
BATCH=$(echo "$K * $M" | bc -l)

if [ $ot_type = 'partial' ]
then
    loss_type=POT
elif [ $ot_type = 'unbalanced' ]
then
    loss_type=UOT
else
    loss_type=OT
fi

if [ $use_bomb = yes ]
then
    loss_type=BoMb_${loss_type}
fi

for num in 01
do
    case ${num} in
        01 )
            s_dset_path="./data/office-home/Art.txt"
            t_dset_path="./data/office-home/Clipart.txt"
            output_dir="A2C"
            ;;
        02 )
            s_dset_path="./data/office-home/Art.txt"
            t_dset_path="./data/office-home/Product.txt"
            output_dir="A2P"
            ;;
        03 )
            s_dset_path="./data/office-home/Art.txt"
            t_dset_path="./data/office-home/Real_World.txt"
            output_dir="A2R"
            ;;
        04 )
            s_dset_path="./data/office-home/Clipart.txt"
            t_dset_path="./data/office-home/Art.txt"
            output_dir="C2A"
            ;;
        05 )
            s_dset_path="./data/office-home/Clipart.txt"
            t_dset_path="./data/office-home/Product.txt"
            output_dir="C2P"
            ;;
        06 )
            s_dset_path="./data/office-home/Clipart.txt"
            t_dset_path="./data/office-home/Real_World.txt"
            output_dir="C2R"
            ;;
        07 )
            s_dset_path="./data/office-home/Product.txt"
            t_dset_path="./data/office-home/Art.txt"
            output_dir="P2A"
            ;;
        08 )
            s_dset_path="./data/office-home/Product.txt"
            t_dset_path="./data/office-home/Clipart.txt"
            output_dir="P2C"
            ;;
        09 )
            s_dset_path="./data/office-home/Product.txt"
            t_dset_path="./data/office-home/Real_World.txt"
            output_dir="P2R"
            ;;
        10 )
            s_dset_path="./data/office-home/Real_World.txt"
            t_dset_path="./data/office-home/Art.txt"
            output_dir="R2A"
            ;;
        11 )
            s_dset_path="./data/office-home/Real_World.txt"
            t_dset_path="./data/office-home/Clipart.txt"
            output_dir="R2C"
            ;;
        12 )
            s_dset_path="./data/office-home/Real_World.txt"
            t_dset_path="./data/office-home/Product.txt"
            output_dir="R2P"
            ;;
    esac

    DES="home_${output_dir}_${loss_type}_run${run_id}"
    final_log="home_${loss_type}_run${run_id}"

    for i in {10..10}
    do
        MASS=$(echo "scale=2; $i / 20" | bc -l)
        echo "-- mass = $MASS"
        output_dir="${DES}_mass0${MASS}_k${K}_m${M}_epsilon${EPSILON}"
        echo "Begin in ${output_dir}"
        echo "log in ${final_log}_log.txt"
        # train the model
        python train.py --gpu_id ${gpus} \
                        --net ResNet50 \
                        --dset office-home \
                        --test_interval $TEST_INTERVAL \
                        --s_dset_path ${s_dset_path} \
                        --stratify_source \
                        --t_dset_path ${t_dset_path} \
                        --batch_size $BATCH \
                        --output_dir ${output_dir} \
                        --final_log "${final_log}_log.txt" \
                        --stop_step $ITER \
                        --ot_type ${ot_type} \
                        --eta1 $ETA1 \
                        --eta2 $ETA2 \
                        --epsilon $EPSILON \
                        --tau $TAU \
                        --mass $MASS \
                        --use_bomb ${use_bomb} \
                        --k $K
        echo "Finish in ${output_dir}"
    done
done

echo "Training Finished!!!"
