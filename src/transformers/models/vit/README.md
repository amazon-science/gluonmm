# ViT

This folder contains the implementation of ViT ([An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)). The following table lists all the models we have and its accuracy on ImageNet validation set.


## Model zoo

| Model | Pretrain | Top-1 | Top-5 | Params | Config |
| ----- | -------- | ----- | ----- | ------ | ------ |
| vit_base_patch16_224 | ImageNet-21K | 81.8 | 96.1 | 87M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/vit_base_patch16_224_imagenet1k.yaml) |
| vit_base_patch16_384 | ImageNet-21K | 84.2 | 97.2 | 87M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/vit_base_patch16_384_imagenet1k.yaml) |
| vit_base_patch32_384 | ImageNet-21K | 81.7 | 96.1 | 88M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/vit_base_patch32_384_imagenet1k.yaml) |
| vit_large_patch16_224 | ImageNet-21K | 83.1 | 96.5 | 304M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/vit_large_patch16_224_imagenet1k.yaml) |
| vit_large_patch16_384 | ImageNet-21K | 85.2 | 97.4 | 305M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/vit_large_patch16_384_imagenet1k.yaml) |
| vit_large_patch32_384 | ImageNet-21K | 81.6 | 96.2 | 307M | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/image_classification/config/vit_large_patch32_384_imagenet1k.yaml) |


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
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```
