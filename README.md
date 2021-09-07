## GluonMM

GluonMM is a library of transformer models for computer vision and multi-modality research. It contains reference implementations of widely adopted baseline models and also research work from Amazon Research.


## Install

First, clone the repository locally,
```
git clone https://github.com/amazon-research/gluonmm.git
```

Then install dependencies,
```
conda create -n gluonmm python=3.7
conda activate gluonmm
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install timm tensorboardX yacs tqdm requests pandas decord scikit-image opencv-python

# Install apex for half-precision training (optional)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

We have extensively tested the usage with PyTorch 1.8.1 and torchvision 0.9.1 with CUDA 10.2.


## Model zoo

### Image classification
- [ViT](https://arxiv.org/abs/2010.11929)
- [DeiT](https://arxiv.org/abs/2012.12877)
- [CaiT](https://arxiv.org/abs/2103.17239)
- [Swin-transformer](https://arxiv.org/abs/2103.14030)

### Video action recognition
- [VidTr](https://arxiv.org/abs/2104.11746)


## Usage

For detailed usage, please refer to the README file in each model family. For example, the training, evaluation and model zoo information of video transformer VidTr can be found at [here](./src/transformers/models/vidtr/README.md).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.


## License

This project is licensed under the Apache-2.0 License.


## Acknowledgement
Parts of the code are heavily derived from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [Swin-transformer](https://github.com/microsoft/Swin-Transformer), [vit-pytorch](https://github.com/lucidrains/vit-pytorch) and [vision_transformer](https://github.com/google-research/vision_transformer/tree/master/vit_jax).
