# DeiT

This folder contains the implementation of DeiT ([Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)). The following table lists all the models we have and its accuracy on ImageNet validation set.


## Model zoo

| Model | Pretrain | Top-1 | Top-5 | Params | Config |
| ----- | -------- | ----- | ----- | ------ | ------ |
| vit_deit_tiny_patch16_224 | ImageNet-1K | 72.2 | 91.1 | 5M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_tiny_patch16_224_imagenet1k.yaml) |
| vit_deit_small_patch16_224 | ImageNet-1K | 79.9 | 95.0 | 22M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_small_patch16_224_imagenet1k.yaml) |
| vit_deit_base_patch16_224 | ImageNet-1K | 82.0 | 95.7 | 86M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_base_patch16_224_imagenet1k.yaml) |
| vit_deit_base_patch16_384 | ImageNet-1K | 83.1 | 96.4 | 87M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_base_patch16_384_imagenet1k.yaml) |
| vit_deit_tiny_distilled_patch16_224 | ImageNet-1K | 74.5 | 91.9 | 6M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_tiny_distilled_patch16_224_imagenet1k.yaml) |
| vit_deit_small_distilled_patch16_224 | ImageNet-1K | 81.2 | 95.4 | 22M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_small__distilled_patch16_224_imagenet1k.yaml) |
| vit_deit_base_distilled_patch16_224 | ImageNet-1K | 83.4 | 96.5 | 87M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_base_distilled_patch16_224_imagenet1k.yaml) |
| vit_deit_base_distilled_patch16_384 | ImageNet-1K | 85.4 | 97.3 | 87M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/deit_base_distilled_patch16_384_imagenet1k.yaml) |


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
@article{touvron2020deit,
  title={Training data-efficient image transformers & distillation through attention},
  author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
  journal={arXiv preprint arXiv:2012.12877},
  year={2020}
}
```
