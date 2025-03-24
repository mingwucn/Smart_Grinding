#!/bin/bash

# Create the log directory if it doesn't exist
mkdir -p log

# Run the training command, print output to the screen, and append to log/train.txt with timestamps
python ./trainer.py --epochs 50 --batch_size 2 --learning_rate 1e-5 --model_name 'SmartGrinding_all' --verbose_interval 10 2>&1 | tee >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >> log/train_all.txt; done)
