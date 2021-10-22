#!/bin/bash
# python PATH
# export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

# choose the method
echo "balanced -- full OT"
echo "unbalanced -- unbalanced OT"
echo "partial -- partial OT"
echo -n "choose the loss_type: "
read ot_type
if [ $ot_type = 'partial' ]
then
    loss_type=POT
elif [ $ot_type = 'unbalanced' ]
then
    loss_type=UOT
else
    loss_type=OT
fi

# whether to use OT
echo "use BoMb version (yes/no) : "
read use_bomb
if [ $use_bomb = yes ]
then
    loss_type=BoMb_${loss_type}
fi

echo ""
echo "0 -- default"
echo -n "run_id: "
read run_id

# OT parameters
ETA1=0.005
ETA2=1
if [ $ot_type = 'unbalanced' ]
then
    EPSILON=0.01
else
    EPSILON=0
fi
TAU=0.3
ITER=10000
TEST_INTERVAL=500
M=72
K=1
BATCH=$(echo "$K * $M" | bc -l)

for num in 01
do
    case ${num} in
        01 )
            s_dset_path="./data/visda-2017/train_list.txt"
            t_dset_path="./data/visda-2017/validation_list.txt"
            output_dir="train_val"
            ;;
    esac

    output_dir="home_${output_dir}_${method}"
    final_log="home_${method}"

    case ${loss_type} in
        0 )
            output_dir="${output_dir}"
            ;;
        * )
            output_dir="${output_dir}=${loss_type}"
            final_log="${final_log}=${loss_type}"
            ;;
    esac

    output_dir="${output_dir}_thresh=${threshold}"
    final_log="${final_log}_thresh=${threshold}"

    case ${run_id} in
        0 )
            DES="${output_dir}"
            ;;
        * )
            DES="${output_dir}_run${run_id}"
            final_log="${final_log}_run${run_id}"
            ;;
    esac

    # train the model
    for i in {10..10}
    do
        MASS=$(echo "scale=2; $i / 20" | bc -l)
        echo "-- mass = $MASS"
        output_dir="${DES}_mass0${MASS}_k${K}_m${M}_epsilon${EPSILON}"
        echo "Begin in ${output_dir}"
        echo "log in ${final_log}_log.txt"
        python train.py --gpu_id ${gpus} \
                        --net ResNet50 \
                        --dset visda \
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