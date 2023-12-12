#!/bin/bash
export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node 8 finetune.py \
    --plugin "hybrid_parallel" \ #更换成所需的插件模式
    --dataset "./dataset" \  #跟换成自己的数据集路径
    --model_path "Llama2-Chinese-7b-Chat" \ #跟换成自己的模型路径
    --task_name "finetuning" \
    --batch_size 3 \
    --num_epochs 1 \
    --flash_attention \
    --save_dir "./output" #跟换成自己的输出路径
    #2>&1 | tee bs_4_node_2.log