#!/bin/bash

mkdir -p log

while true; do
    python ./trainer.py --epochs 10 --batch_size 2 --learning_rate 1e-5 --model_name 'SmartGrinding_all' --verbose_interval 10 2>&1 | tee >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >> log/train_all.txt; done)
    
    if [ $? -eq 0 ]; then
        echo "Training completed successfully."
        break
    else
        echo "Training failed. Retrying..."
    fi
done
