#!/bin/bash

python main_end2end.py --dataset_split_type direct --model_name /userhome/pretrain_model/EgoVLPv2.pth --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
