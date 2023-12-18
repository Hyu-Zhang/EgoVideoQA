# Patch Partition, Selection, and Aggregation for Egocentric Video Question Answering

[arXiv]()

> **TL;DR:** We pioneer Egocentric Video-Language Pretraining from pretraining dataset, model and development benchmark; the resulted pretrained model exhibits strong performance on five downstream tasks across three egocentric datasets.

![EgoVLP](figures/egovlp_framework.jpg)

## 📝 Preparation
### Install Dependencies 
```bash
conda env create -f environment.yml
source activate egovlp
```
### Data Download

### Pretrained Weights
EgoVLPv2 https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth

### File Structure

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
