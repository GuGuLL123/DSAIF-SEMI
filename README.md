<p align="center">
  <h1 align="center"> Dual Structure-Aware Image Filterings for Semi-supervised Medical Image Segmentation</h1>
</p>


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Semi-supervised image segmentation has attracted great attention recently. 
The key is how to leverage unlabeled images in the training process. 
Most methods maintain consistent predictions of the unlabeled images under variations (\eg, adding noise/perturbations, or creating alternative versions) in the image and/or model level. 
In most image-level variation, medical images often have prior structure information, which has not been well explored.
In this paper, we propose novel dual structure-aware image filterings (DSAIF) as the image-level variations for semi-supervised medical image segmentation. 
Motivated by connected filtering that simplifies image via filtering in structure-aware tree-based image representation, we resort to the dual contrast invariant Max-tree and Min-tree representation.
Specifically, we propose a novel connected filtering that removes topologically equivalent nodes (\ie connected components) having no siblings in the Max/Min-tree. This results in two filtered images preserving topologically critical structure. 
Applying the proposed DSAIF to mutually supervised networks decreases the consensus of their erroneous predictions on unlabeled images. This helps to alleviate the confirmation bias issue of overfitting to noisy pseudo labels of unlabeled images, and thus effectively improves the segmentation performance. 
Extensive experimental results on three benchmark datasets demonstrate that the proposed method significantly/consistently outperforms some state-of-the-art methods. 

<!-- <img width="100%" src="./figs/pipeline.jpg" /> -->

## Requirements
All experiments in our paper were conducted on two NVIDIA GeForce RTX 4090 GPUs.

## Dataset
We follow <a href="https://github.com/ycwu1997/MC-Net">MC-Net</a> for 3D experiments (LA Dataset and Pancreas-NIH Dataset) and <a href="https://github.com/igip-liu/SLC-Net">SLC-Net</a>  for 2D experiments  (PROMISE12 Dataset).


## Usage
The *max_min_tree_cpp* contains the code for Dual Structure-Aware Image Filterings (DSAIF) in C++ format, which requires compilation before it can be used. Additionally, the *Higra* library also provides some functionalities for implementing Dual Structure-Aware Image Filterings. All the experiments in the paper were conducted using C++ code. Since the publication of the paper, we have developed a faster version of the code, utilizing Higra for DSAIF.  When using the Higra version of the code, you need to replace function *tree_process_3D_final* with function *tree_process_3D_higra_final* in the class SemiDataset_Medical_Pancreas_CT  located in the /dataset/semi.py file.

Using the following code to install *Higra* library.
```python
pip install higra
```


### Run the following commands for training and testing. 

```shell
CUDA_VISIBLE_DEVICES=0,1 bash tools/train.sh 2 61423
```


## License
This project is licensed under the MIT License

## Citation
```bibtex
@article{gu2024dual,
  title={Dual structure-aware image filterings for semi-supervised medical image segmentation},
  author={Gu, Yuliang and Sun, Zhichao and Chen, Tian and Xiao, Xin and Liu, Yepeng and Xu, Yongchao and Najman, Laurent},
  journal={Medical Image Analysis},
  pages={103364},
  year={2024},
  publisher={Elsevier}
}

```
