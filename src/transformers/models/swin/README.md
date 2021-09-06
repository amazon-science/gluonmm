# Swin transformer

This folder contains the implementation of Swin-transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)). The following table lists all the models we have and its accuracy on ImageNet validation set.


## Model zoo

| Model | Pretrain | Top-1 | Top-5 | Params | Config |
| ----- | -------- | ----- | ----- | ------ | ------ |
| swin_tiny_patch4_window7_224 | ImageNet-1K | 81.4 | 95.5 | 28M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_tiny_patch4_window7_224.yaml) |
| swin_small_patch4_window7_224 | ImageNet-1K | 83.2 | 96.3 | 50M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_small_patch4_window7_224.yaml) |
| swin_base_patch4_window7_224 | ImageNet-1K | 83.6 | 96.5 | 88M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_base_patch4_window7_224.yaml) |
| swin_base_patch4_window12_384 | ImageNet-1K | 84.5 | 96.9 | 88M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_base_patch4_window12_384.yaml) |
| swin_base_patch4_window7_224 | ImageNet-22K | 85.3 | 97.6 | 88M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_base_patch4_window7_224_22kto1k.yaml) |
| swin_base_patch4_window12_384 | ImageNet-22K | 86.4 | 98.1 | 88M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_base_patch4_window12_384_22kto1k.yaml) |
| swin_large_patch4_window7_224 | ImageNet-22K | 86.3 | 97.9 | 197M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_large_patch4_window7_224_22kto1k.yaml) |
| swin_large_patch4_window12_384 | ImageNet-22K | 87.2 | 98.2 | 197M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/swin_large_patch4_window12_384_22kto1k.yaml) |


## Evaluation

To evaluate a pre-trained model on ImageNet validation set with 50K images, run
```
python ./scripts/image_classification/test.py --config-file CONFIG_FILE
```


## Training
To train a model on ImageNet on a single node with 8 GPUS, run
```
python ./scripts/image_classification/train.py --config-file CONFIG_FILE
```


```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
