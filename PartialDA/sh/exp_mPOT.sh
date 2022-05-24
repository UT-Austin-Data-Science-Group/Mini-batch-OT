#!/bin/bash
gpus=0,1
ot_type=pot
method=mOT

# OT parameters
ETA1=0.03
ETA2=7.5
ETA3=1
if [ $ot_type = 'uot' ]
then
    EPSILON=0.01
else
    EPSILON=0.5
fi
TAU=0.06
K=1
M=65
MASS=0.65

for i in {5..15}
do
    MASS=$(echo "scale=2; $i / 20" | bc -l)
    echo "-- mass = $MASS"
    for S in {0..0}
    do
        for T in {0..1}
        do
            if [ $S != $T ]
            then
            OUTPUT=m${ot_type}_k${K}_m${M}_mass0${MASS}
            python run_${method}.py --s $S \
                                    --t $T \
                                    --batch_size $M \
                                    --dset office_home \
                                    --net ResNet50 \
                                    --output $OUTPUT \
                                    --gpu_id $gpus \
                                    --ot_type $ot_type \
                                    --eta1 $ETA1 \
                                    --eta2 $ETA2 \
                                    --eta3 $ETA3 \
                                    --epsilon $EPSILON \
                                    --tau $TAU \
                                    --mass $MASS \
                                    --k $K
            fi
        done
    done
done
