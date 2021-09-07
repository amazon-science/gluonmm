# CaiT

This folder contains the implementation of CaiT ([Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)). The following table lists all the models we have and its accuracy on ImageNet validation set.


## Model zoo

| Model | Pretrain | Top-1 | Top-5 | Params | Config |
| ----- | -------- | ----- | ----- | ------ | ------ |
| cait_xxs24_patch16_224 | ImageNet-1K | 78.4 | 94.3 | 12M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_xxs24_patch16_224_imagenet1k.yaml) |
| cait_xxs24_patch16_384 | ImageNet-1K | 81.0 | 95.6 | 12M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_xxs24_patch16_384_imagenet1k.yaml) |
| cait_xxs36_patch16_224 | ImageNet-1K | 79.8 | 94.9 | 17M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_xxs36_patch16_224_imagenet1k.yaml) |
| cait_xxs36_patch16_384 | ImageNet-1K | 82.2 | 96.2 | 17M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_xxs36_patch16_384_imagenet1k.yaml) |
| cait_xs24_patch16_384 | ImageNet-1K | 84.1 | 96.9 | 27M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_xs24_patch16_384_imagenet1k.yaml) |
| cait_s24_patch16_224 | ImageNet-1K | 83.5 | 96.6 | 47M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_s24_patch16_224_imagenet1k.yaml) |
| cait_s24_patch16_384 | ImageNet-1K | 85.1 | 97.3 | 47M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_s24_patch16_384_imagenet1k.yaml) |
| cait_s36_patch16_384 | ImageNet-1K | 85.5 | 97.5 | 68M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_s36_patch16_384_imagenet1k.yaml) |
| cait_m36_patch16_384 | ImageNet-1K | 86.1 | 97.7 | 271M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_m36_patch16_384_imagenet1k.yaml) |
| cait_m48_patch16_448 | ImageNet-1K | 86.5 | 97.8 | 356M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/cait_m48_patch16_448_imagenet1k.yaml) |


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
@article{touvron2021cait,
  title={Going deeper with Image Transformers},
  author={Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Herv\'e J\'egou},
  journal={arXiv preprint arXiv:2103.17239},
  year={2021}
}
```
