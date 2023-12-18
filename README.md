# Patch Partition, Selection, and Aggregation for Egocentric Video Question Answering

[arXiv]() | [Code]()

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
If you encounter the following error ```ImportError: libGL.so.1: cannot open shared object file: No such file or directory``` during execution , you can resolve it with this command ```apt-get update && apt-get install ffmpeg libsm6 libxext6  -y```. 

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
├── EgoNCE_MLM_ITM_Config.yml
├── EgoTaskQA_dataset.py
├── base
│   ├── __init__.py
│   ├── base_dataset.py
│   └── base_model.py
├── configs
│   ├── egotaskqa.json
│   └── egotaskqa_f32.json
├── logger
│   ├── __init__.py
│   ├── logger.py
│   ├── logger_config.json
│   └── visualization.py
├── main_end2end.py
├── model
│   ├── hcrn.py
│   ├── heads.py
│   ├── model.py
│   ├── patch_selection.py
│   ├── roberta.py
│   ├── uniformer.py
│   ├── video_qa_model_linear_end2end.py
│   └── video_transformer.py
├── parse_config.py
├── reasoning_type_unique_cat.py
├── run.sh
├── transforms.py
└── utils
    ├── __init__.py
    ├── custom_transforms.py
    ├── distributed.py
    ├── html.py
    ├── logging.py
    ├── loss.py
    ├── mAP.py
    ├── nDCG.py
    ├── util.py
    ├── utils.py
    ├── video.py
    ├── video_chunk.py
    ├── video_resize.py
    ├── visualisation.py
    └── visualizer.py
```

## 🏋️‍️ Fine-tuning

## Evaluation
## 🔧 Fine-tuned Checkpoints
 
| Model | Open | Binary | All | Checkpoint | Log |
| ------ | ------ | ------ | ------ | ------ | ------ |
| Direct Settings |
| [EgoVLP](https://github.com/showlab/EgoVLP) | 31.69 | 71.26 | 42.51 | [Link](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing) | - |
| [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2/tree/main/EgoTaskQA) | 35.56 | 75.60 | 46.26 | [Link](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EgoTaskQA_Finetuned/EgoTaskQA_finetune_direct.tar) | - |
| Ours | 38.95 | 75.86 | 48.69 | [Link]() | [Link]() |
| Indirect Settings |
| [EgoVLP](https://github.com/showlab/EgoVLP) | 27.04 | 55.28 | 38.69 | [Link](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing) | - |
| [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2/tree/main/EgoTaskQA) | 29.14 | 59.68 | 42.28 | [Link](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EgoTaskQA_Finetuned/EgoTaskQA_finetune_indirect.tar) | - |
| Ours | 32.44 | 63.02 | 45.40 | [Link]() | [Link]() |



## 🎓 Citation

If you find our work helps, please cite our paper.

```bibtex
@article{kevin2022egovlp,
  title={Egocentric Video-Language Pretraining},
  author={Lin, Kevin Qinghong and Wang, Alex Jinpeng and Soldan, Mattia and Wray, Michael and Yan, Rui and Xu, Eric Zhongcong and Gao, Difei and Tu, Rongcheng and Zhao, Wenzhe and Kong, Weijie and others},
  journal={arXiv preprint arXiv:2206.01670},
  year={2022}
}
```

## ✉️ Contact

This repo is maintained by [Kevin](https://github.com/QinghongLin). Questions and discussions are welcome via `kevin.qh.lin@gmail.com`.

## 🙏 Acknowledgements

This codebase is based on [Frozen](https://github.com/m-bain/frozen-in-time). 

Thanks to [Alex](https://github.com/fingerrec) for the help with DDP and [Mattia](https://github.com/Soldelli) for the help with NLQ and MQ benchmarks.

We thank the EgoTaskQA authors for releasing the dataset and baselines.
EgoVLPv2

## LICENSE

MIT
