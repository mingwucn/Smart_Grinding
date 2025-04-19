#!/bin/bash

mkdir -p log

intermediate_data_type=('physics' 'spec')

for input_type in "${intermediate_data_type[@]}"; do
    echo "Constructing intermediate data: $input_type"
    python ./GrindingData.py --threads=8 --process_type "$input_type" 2>&1 | tee >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >>"log/preprocessing_${input_type}.txt"; done)
done