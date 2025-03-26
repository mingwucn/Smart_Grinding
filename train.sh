#!/bin/bash

mkdir -p log

allowed_input_types=('ae_spec' 'vib_spec' 'ae_spec+ae_features' 'vib_spec+vib_features' 'ae_spec+ae_features+vib_spec+vib_features' 'all')

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
    python ./trainer.py --epochs 10 --batch_size 2 --learning_rate 1e-5 --model_name "$input_type" --input_type "$input_type" --verbose_interval 10 --repeat 1 --num_workers 24 --train_mode "classical" 2>&1 | tee >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >>"log/train_${input_type}.txt"; done)
done
