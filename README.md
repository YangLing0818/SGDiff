# Diffusion-Based Scene Graph to Image Generation with Masked Contrastive Pre-Training
<a href="https://arxiv.org/abs/2211.11138"><img src="https://img.shields.io/badge/arXiv-2211.11138-blue.svg" height=22.5></a>

Official Implementation for [Diffusion-Based Scene Graph to Image Generation with Masked Contrastive Pre-Training](https://arxiv.org/abs/2211.11138). 

## Overview of The Proposed SGDiff

<div align=center><img width="850" alt="image" src="https://user-images.githubusercontent.com/62683396/202852210-d91d6a63-f04d-4a02-ae5f-55f00f8c1ec5.png"></div>

<div align=center><img width="590" alt="image" src="https://github.com/YangLing0818/SGDiff/assets/62683396/bc341a3b-0ff2-4544-b0f6-3fe759b77097"></div>



## Environment
```
git clone https://github.com/YangLing0818/SGDiff.git
cd SGDiff

conda env create -f sgdiff.yaml
conda activate sgdiff
mkdir pretrained
```


## Data and Model Preparation

The instructions of data pre-processing can be [found here](https://github.com/YangLing0818/SGDiff/blob/main/DATA.md).

Our masked contrastive pre-trained models of SG-image pairs for COCO and VG datasets are provided in [here](https://www.dropbox.com/scl/fo/lccvtxuwxxblo3atnxlmg/h?rlkey=duy7dcwmy3a64auqoqiw8dv2e&dl=0), please download them and put them in the 'pretrained' directory.

And the pretrained VQVAE for embedding image to latent can be obtained from https://ommer-lab.com/files/latent-diffusion/vq-f8.zip

## Training
Kindly note that one should not skip the training stage and test directly. For single gpu, one can use
```shell
python trainer.py --base CONFIG_PATH -t --gpus 0,
```

## Sampling

```shell
python testset_ddim_sampler.py
```

## Citation
If you found the codes are useful, please cite our paper
```
@article{yang2022diffusionsg,
  title={Diffusion-based scene graph to image generation with masked contrastive pre-training},
  author={Yang, Ling and Huang, Zhilin and Song, Yang and Hong, Shenda and Li, Guohao and Zhang, Wentao and Cui, Bin and Ghanem, Bernard and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2211.11138},
  year={2022}
}
```
