# SGDiff
<a href="https://arxiv.org/abs/2206.02779"><img src="https://img.shields.io/badge/arXiv-2206.02779-b31b1b.svg" height=22.5></a>

Official Implementation for Diffusion-Based Scene Graph to Image Generation with Masked Contrastive Pre-Training.

> Abstract: Generating images from graph-structured inputs, such as scene graphs, is uniquely challenging due to the difficulty of aligning nodes and connections in graphs with objects and their relations in images. Most existing methods address this challenge by using scene layouts, which are image-like representations of scene graphs designed to capture the coarse structures of scene images. Because scene layouts are manually crafted, the alignment with images may not be fully optimized, causing suboptimal compliance between the generated images and the original scene graphs. To tackle this issue, we propose to learn scene graph embeddings by directly optimizing their alignment with images. Specifically, we pre-train an encoder to extract both global and local information from scene graphs that are predictive of the corresponding images, relying on two loss functions: masked autoencoding loss and contrastive loss. The former trains embeddings by reconstructing randomly masked image regions, while the latter trains embeddings to discriminate between compliant and non-compliant images according to the scene graph. Given these embeddings, we build a latent diffusion model to generate images from scene graphs. The resulting method, called SGDiff, allows for the semantic manipulation of generated images by modifying scene graph nodes and connections. On the Visual Genome and COCO-Stuff datasets, we demonstrate that SGDiff outperforms state-of-the-art methods, as measured by both the Inception Score and Fréchet Inception Distance (FID) metrics. 

<img width="440" alt="image" src="https://user-images.githubusercontent.com/62683396/202852678-25548437-6a33-4550-9cd8-d2dd955d20dc.png"><img width="480" alt="image" src="https://user-images.githubusercontent.com/62683396/202852055-23000ad2-9f21-41d0-a3e5-0b1b6f5eff36.png">


## Overview of The Proposed SGDiff

<img width="850" alt="image" src="https://user-images.githubusercontent.com/62683396/202852210-d91d6a63-f04d-4a02-ae5f-55f00f8c1ec5.png">

## Qualitative Evaluations
### Image Generation from Scene Graphs

<img width="795" alt="image" src="https://user-images.githubusercontent.com/62683396/202852387-ac3bdee2-ca9f-4ab6-b053-8aa0c47ca798.png">

### Semantic Image Manipulation

<img width="1298" alt="image" src="https://user-images.githubusercontent.com/62683396/202852465-4c41d8b1-1f3a-4eca-9a5b-89e95ca1224c.png">

## The Code and Trained Models

We will release the code and trained models soon.
