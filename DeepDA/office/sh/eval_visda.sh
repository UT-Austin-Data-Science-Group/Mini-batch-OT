#!/bin/bash
# python PATH
# export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

# choose the method
echo ""
echo "0  --  DANN"
echo "1  --  ALDA"
echo "2  --  Classifier only"
echo -n "choose the method: "
read method_choose

case ${method_choose} in
    0 )
        method="DANN"
        ;;
    1 )
        method="ALDA"
        ;;
    2 )
        method="Classifier"
        ;;
    * )
        echo "The choice of method is illegal!"
        exit 1 
        ;;
esac

# choose the loss_type
if [ $method = 'ALDA' ]
then
    echo ""
    echo "all -- ALDA with full losses"
    echo "nocorrect -- ALDA without the target loss"
    echo -n "choose the loss_type: "
    read loss_type
else
    loss_type=none
fi

# whether to use OT
echo "use OT loss (yes/no) : "
read use_ot

if [ $use_ot = 'yes' ]
then
    echo ""
    echo "balanced -- full OT"
    echo "unbalanced -- unbalanced OT"
    echo "partial -- partial OT"
    echo -n "choose the loss_type: "
    read ot_type
    loss_type=${loss_type}_${ot_type}OT
else
    ot_type=balanced # dummy value
fi

# choose the threshold
echo ""
echo "0.9 -- the optimum for office"
echo -n "choose the threshold: "
read threshold

echo ""
echo "0 -- default"
echo -n "run_id: "
read run_id

echo "home_${method}=loss_type=${loss_type}_thresh=${threshold}_${run_id}"

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
            DES="${output_dir}_${run_id}"
            final_log="${final_log}_${run_id}"
            ;;
    esac

    # OT parameters
    ITER=10000
    TEST_INTERVAL=500
    BATCH=72
    EPSILON=0.2

    # eval checkpoint
    for i in {10..10}
    do
        MASS=$(echo "scale=2; $i / 20" | bc -l)
        echo "-- mass = $MASS"
        output_dir="${DES}_mass0${MASS}_batch${BATCH}_epsilon${EPSILON}"
        echo "Begin in eval_${output_dir}"
        echo "log in ${final_log}_log.txt"
        python evaluate.py --method $method \
                        --gpu_id ${gpus} \
                        --net ResNet50 \
                        --dset visda \
                        --test_interval $TEST_INTERVAL \
                        --s_dset_path ${s_dset_path} \
                        --t_dset_path ${t_dset_path} \
                        --batch_size $BATCH \
                        --output_dir eval_${output_dir} \
                        --final_log "${final_log}_log.txt" \
                        --loss_type ${loss_type} \
                        --threshold ${threshold} \
                        --stop_step $ITER \
                        --restore_dir ${output_dir}
        echo "Finish in ${output_dir}"
    done
done

echo "Training Finished!!!"