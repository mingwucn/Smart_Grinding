#!/bin/bash

mkdir -p log

allowed_input_types=('pp' 'ae_spec' 'ae_features'  'ae_features+pp')

epochs=20
lr=1e-5
drop=0.4
folds=10
repeat=10
batch_size=8
dataset_mode='ram'
gpu='cpu'
num_workers=0

# for input_type in "${allowed_input_types[@]}"; do
#     while true; do
#         echo "Training with input_type: $input_type"
#         python ./trainer.py --epochs 10 --batch_size 2 --learning_rate 1e-5 --model_name "$input_type" --input_type "$input_type" --verbose_interval 0 --repeat 1 2>&1 | tee >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >>"log/train_${input_type}.txt"; done)

#         if [ $? -eq 0 ]; then
#             echo "Training completed successfully for input_type: $input_type."
#             break
#         else
#             echo "Training failed for input_type: $input_type. Retrying..."
#         fi
#     done
# done

for input_type in "${allowed_input_types[@]}"; do
    echo "Training with input_type: $input_type"
    python ./trainer.py --epochs $epochs --batch_size $batch_size --learning_rate $lr --model_name "$input_type" --input_type "$input_type" --verbose_interval 10 --repeat $repeat --dataset_mode $dataset_mode --gpu $gpu --num_workers $num_workers 2>&1 | tee >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >>"log/train_${input_type}.txt"; done)
done

pkill -u ming