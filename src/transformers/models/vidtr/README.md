# Vidtr

This folder contains the implementation of Vidtr ([VidTr: Video Transformer Without Convolutions](https://arxiv.org/abs/2104.11746)). The following table lists all the models we have and its accuracy on Kinetics400 validation set.


## Model zoo

| Model | Pretrain | Top-1 | Top-5 | GFLOPs | Config |
| ----- | -------- | ----- | ----- | ------ | ------ |
| VidTr-S | ImageNet-21K | 77.7 | 93.3 | 89 | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/action_recognition/config/vidtr_s_8x8_patch16_224_k400.yaml) |
| VidTr-M | ImageNet-21K | 78.6 | 93.5 | 179 | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/action_recognition/config/vidtr_m_16x4_patch16_224_k400.yaml) |
| VidTr-L | ImageNet-21K | 79.1 | 93.9 | 351 | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/action_recognition/config/vidtr_l_32x2_patch16_224_k400.yaml) |
| Compact-VidTr-S | ImageNet-21K | 75.7 | - | 39 | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/action_recognition/config/cvidtr_s_8x8_patch16_224_k400.yaml) |
| Compact-VidTr-M | ImageNet-21K | 76.7 | - | 59 | [config](https://github.com/amazon-research/gluonmm/blob/main/scripts/action_recognition/config/cvidtr_m_16x4_patch16_224_k400.yaml) |



## Evaluation

To evaluate a pre-trained model on Kinetics400 validation set, run
```
python ./scripts/action_recognition/test.py --config-file CONFIG_FILE
```


## Training
To train a model on Kinetics400 on a single node with 8 GPUS, run
```
python ./scripts/action_recognition/train.py --config-file CONFIG_FILE
```


```
@article{li2021vidtr,
  title={VidTr: Video Transformer Without Convolutions},
  author={Xinyu Li, Yanyi Zhang, Chunhui Liu, Bing Shuai, Yi Zhu, Biagio Brattoli, Hao Chen, Ivan Marsic, Joseph Tighe},
  journal={arXiv preprint arXiv:2104.11746},
  year={2021}
}
```
