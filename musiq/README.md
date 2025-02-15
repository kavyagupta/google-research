# MUSIQ: Multi-scale Image Quality Transformer

This directory contains checkpoints and model inference code for the ICCV 2021
paper:
["MUSIQ: Multi-scale Image Quality Transformer"](https://arxiv.org/abs/2108.05997)
by Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang.

*Disclaimer: This is not an official Google product.*

![Model overview](images/overview.png)

## Using the models

The MUSIQ models are available on [TensorFlow Hub](https://tfhub.dev/s?q=musiq)
with documentation and a sample notebook for you to try.

But if you want to go deeper in the code, follow the instructions below.

## Pre-requisite

Install dependencies:

```
pip3 install -r requirements.txt
```

The model checkpoints can be downloaded from:
[download link](https://console.cloud.google.com/storage/browser/gresearch/musiq)

The folder contains the following checkpoints:

-   **ava_ckpt.npz**: Trained on AVA dataset.
-   **koniq_ckpt.npz**: Trained on KonIQ dataset.
-   **paq2piq_ckpt.npz**: Trained on PaQ2PiQ dataset.
-   **spaq_ckpt.npz**: Trained on SPAQ dataset.
-   **imagenet_pretrain.npz**: Pretrained checkpoint on ImageNet.

## Run Inference

```shell
python3 -m musiq.run_predict_image \
  --ckpt_path=/tmp/spaq_ckpt.npz \
  --image_path=/tmp/image.jpeg
  
 TF_CPP_MIN_LOG_LEVEL=0 python3 -m musiq.run_predict_image   --ckpt_path=musiq/checkpoints/spaq_ckpt.npz   --image_path=musiq/Residential_Properties_Annotations/rasters/
```

## Citation

If you find this code is useful for your publication, please cite the original
paper:

```
@inproceedings{jke_musiq_iccv2021,
  title = {MUSIQ: Multi-scale Image Quality Transformer},
  author = {Junjie Ke and Qifei Wang and Yilin Wang and Peyman Milanfar and Feng Yang},
  booktitle = {ICCV},
  year = {2021}
}
```
