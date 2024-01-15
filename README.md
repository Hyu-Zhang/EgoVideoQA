<h1 align="center"> <a href=''>Patch Partition, Selection, and Aggregation for<br/> Egocentric Video Question Answering</a></h2>

> **TL;DR:** We unleash an egocentric video question answering model which exhibits strong performance on multiple public datasets.

## 📝 Preparation
### 1. Install Dependencies 
Installs dependencies needed for the code to run.
```bash
conda create -n egovqa python=3.9 pip
pip install torch-1.12.1+cu113-cp39-cp39-linux_x86_64.whl
pip install torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
If you encounter the following error ```ImportError: libGL.so.1: cannot open shared object file: No such file or directory``` during execution , you can resolve it with this command ```apt-get update && apt-get install ffmpeg libsm6 libxext6 -y```. 

### 2. Data Download
You can get the dataset by following the data processing steps provided in the [EgoTaskQA](https://github.com/Buzz-Beater/EgoTaskQA/blob/main/baselines/README.md) work. Also, you can download our processed data directly by following the commands below.
```
wget https://drive.google.com/file/d/1TMJ3qcMt-psDuevw4JaXd7pOzwmMk6wR/view?usp=sharing
tar -zxvf Data.tar.gz && rm Data.tar.gz
# The following links are provided by EgoVLPv2, see https://github.com/facebookresearch/EgoVLPv2/tree/main/EgoTaskQA
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/qa_videos.tgz
tar -xvzf qa_videos.tgz && rm qa_videos.tgz
```
The two folders ```/Data``` and ```/qa_videos``` are placed under the path ```/data```.

### 3. Pretrained Weights
We use the EgoVLPv2 model weights, which are pre-trained on the [EgoClip](https://drive.google.com/file/d/1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl/view?usp=sharing) version of [Ego4D](https://ego4d-data.org/docs/start-here/#cli-download). And you can follow the commands below.
```
wget -c https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/reasoning_unique_cat.pth
# ViT from timm package
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
# RoBERTa from huggingface
https://huggingface.co/roberta-base/tree/main
```
The above files (```EgoVLPv2.pth```, ```reasoning_unique_cat.pth```, and ```jx_vit_base_p16_224-80ecf9dd.pth```) and folder (```/roberta-base```) should be placed under the ```/pretrain_model``` path.

### 4. File Structure
Before the code is executed, make sure the file structure is as shown below.
```
.
├── code
│   ├── README.md
│   ├── EgoNCE_MLM_ITM_Config.yml
│   ├── EgoTaskQA_dataset.py
│   ├── /base
│   ├── /configs
│   ├── /logger
│   ├── main_end2end.py
│   ├── /model
│   ├── parse_config.py
│   ├── reasoning_type_unique_cat.py
│   ├── requirements.txt
│   ├── run.sh
│   ├── transforms.py
│   └── /utils
├── data
│   ├── /Data
│   │   ├── README.md
│   │   ├── /metadata
│   │   ├── /qa
│   │   │   ├── /direct
│   │   │   └── /indirect
│   │   ├── /qa_ori
│   │   │   ├── /direct
│   │   │   └── /indirect
│   │   ├── /raw
│   │   └── /templates
│   └── /qa_videos
└── pretrain_model
    ├── EgoVLPv2.pth
    ├── jx_vit_base_p16_224-80ecf9dd.pth
    ├── reasoning_unique_cat.pth
    └── /roberta-base
```

## 🔧 Fine-tuning
Modify the target path parameters, including ```writer```, ```--basedir```, ```--model_name```, ```data_dir```, ```meta_dir```, and ```unique_dict``` in the file ```main_end2end.py```, ```metadata_dir```, ```unique_dict```, and ```tokenizer``` in the file ```EgoTaskQA_dataset.py```, and ```self.text_model``` and ```vit_model``` in the file ```/model/video_qa_model_linear_end2end.py```.

After that, you can fine-tune model on EgoTaskQA dataset with the following commands. The split type is controlled by the ```--dataset_split_type``` argument.
```
# direct setting
# bash run.sh or
python main_end2end.py --dataset_split_type direct --model_name /userhome/pretrain_model/EgoVLPv2.pth --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
# indirect setting
python main_end2end.py --dataset_split_type indirect --model_name /userhome/pretrain_model/EgoVLPv2.pth --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```
If the fine-tuning stops before completion, the process can be resumed from last saved checkpoint via ```--resume_finetune_model_path``` argument.
```
python main_end2end.py --dataset_split_type direct --model_name /userhome/pretrain_model/EgoVLPv2.pth --resume_finetune_model_path <last_saved_ckpt> --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```
Note: We train 40 epochs for ~20 hours on 4 V100-32G cards, or ~13 hours at 8 V100-32G cards.

## 🎯 Evaluation
To evaluate the fine-tuned checkpoints, you are able to add ```--test_only_model_path``` argument. In addition, we perform evaluation for each epoch in the generated file ```log.txt``` as well.
```
python main_end2end.py --dataset_split_type direct --test_only_model_path <model_best_ckpt> --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```

## 🏆 Fine-tuned Checkpoints
We have provided our fine-tuned model checkpoints, as well as the log file generated during training.
### 1. Direct Setting
| Model | Open | Binary | All | Checkpoint | Log |
| :------: | :------: | :------: | :------: | :------: | :------: |
| [EgoVLP](https://github.com/showlab/EgoVLP) | 31.69 | 71.26 | 42.51 | [Link](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing) | - |
| [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2/tree/main/EgoTaskQA) | 35.56 | 75.60 | 46.26 | [Link](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EgoTaskQA_Finetuned/EgoTaskQA_finetune_direct.tar) | - |
| Ours | 38.95 | 75.86 | 48.69 | [Link]() | [Link]() |

### 2. Indirect Setting
| Model | Open | Binary | All | Checkpoint | Log |
| :------: | :------: | :------: | :------: | :------: | :------: |
| [EgoVLP](https://github.com/showlab/EgoVLP) | 27.04 | 55.28 | 38.69 | [Link](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing) | - |
| [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2/tree/main/EgoTaskQA) | 29.14 | 59.68 | 42.28 | [Link](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EgoTaskQA_Finetuned/EgoTaskQA_finetune_indirect.tar) | - |
| Ours | 32.44 | 63.02 | 45.40 | [Link]() | [Link]() |

## 🎓 Citation
If our work is helpful to you, please cite our paper.

```

```

## ✉️ Contact
Questions and discussions are welcome via `XXX`.

## 🙏 Acknowledgements
We thank the authors from [EgoTaskQA](https://github.com/Buzz-Beater/EgoTaskQA/tree/main) for releasing the dataset and baselines. Also, we thank the authors from [EgoVLP](https://github.com/showlab/EgoVLP?tab=readme-ov-file) and [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2/tree/main) for the exploratory research, which is the beginning of our study.

## 🔖 License
[MIT License]()
