<h1 align="center">LSH-Based Efficient Point Transformer (HEPT)</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2402.12535"><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/HEPT"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://arxiv.org/abs/2402.12535"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2724&color=blue"></a>
</p>

## This HEPT Fork uses PQuant for pruning and quantization

Pruning scripts are not yet uploaded.
Quantization script is uploaded as `src/tracking_quantization.py` (can be modified easily for pruning as well)

There are 2 existing PyTorch model files present under `data/tracking/logs/`:
+ Quantized model: Quantized using Fixed point representation (1 sign bit, 7 integer bits, 8 fractional bits), accuracy on tracking-600 is ~87%
+ Pre trained model: Non quantized version with accuracy on tracking-600 as ~89%, regions = 1

## Datasets
The quantization script has only been tested with tracking-600 dataset. Copy the tracking-600
dataset to the folder `data/tracking/processed/`

## Installation

#### Environment
We are using `torch 2.3.1` and `pyg 2.5.3` with `python 3.10.14` and `cuda 12.1`. Use the following command to install the required packages:
```
conda env create -f pquant_hept_env.yaml
pip install torch_geometric==2.5.3
pip install torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install --no-deps git+https://github.com/calad0i/HGQ2.git
pip install git+https://github.com/ArghyaDas112358/PQuant.git@MDMM
```

#### Running the code

For running the training of the quantized version use
```
python tracking_quantizer.py -m hept
```

Configurations will be loaded from those located in `./configs/` directory.
Quantization script was tested with resuming training from a pretrained (non quantized) model with regions = 1

<br>

## TODO
- [ ] Put more details in the README.
- [ ] Add support for FlashAttn.
- [x] Add support for efficient processing of batched input.
- [x] Add an example of HEPT with minimal code.

## News
- **2024.06:** HEPT has been accepted to ICML 2024 and is selected as an oral presentation (144/9473, 1.5% acceptance rate)!
- **2024.04:** HEPT now supports efficient processing of batched input by this [commit](https://github.com/Graph-COM/HEPT/commit/2e408388a16400050c0eb4c4f7390c3c24078dee). This is implemented via integrating batch indices in the computation of AND hash codes, which is more efficient than naive padding, especially for batches with imbalanced point cloud sizes. **Note:**
  - Only the code in `./example` is updated to support batched input, and the original implementation in `./src` is not updated.
  - The current implementation for batched input is not yet fully tested. Please feel free to open an issue if you encounter any problems.

- **2024.04:** An example of HEPT with minimal code is added in `./example` by this [commit](https://github.com/Graph-COM/HEPT/commit/350a9863d7757e556177c52a44bac2aaf0c6dde8). It's a good starting point for users who want to use HEPT in their own projects. There are minor differences between the example and the original implementation in `./src/models/attention/hept.py`, but they should not affect the performance of the model.


## Introduction
This study introduces a novel transformer model optimized for large-scale point cloud processing in scientific domains such as high-energy physics (HEP) and astrophysics. Addressing the limitations of graph neural networks and standard transformers, our model integrates local inductive bias and achieves near-linear complexity with hardware-friendly regular operations. One contribution of this work is the quantitative analysis of the error-complexity tradeoff of various sparsification techniques for building efficient transformers. Our findings highlight the superiority of using locality-sensitive hashing (LSH), especially OR \& AND-construction LSH, in kernel approximation for large-scale point cloud data with local inductive bias. Based on this finding, we propose LSH-based Efficient Point Transformer (**HEPT**), which combines E2LSH with OR \& AND constructions and is built upon regular computations. HEPT demonstrates remarkable performance in two critical yet time-consuming HEP tasks, significantly outperforming existing GNNs and transformers in accuracy and computational speed, marking a significant advancement in geometric deep learning and large-scale scientific data processing.

<p align="center"><img src="./data/HEPT.png" width=85% height=85%></p>
<p align="center"><em>Figure 1.</em>Pipline of HEPT.</p>
## FAQ

#### How to tune the hyperparameters of HEPT?
There are three key hyperparameters in HEPT:
- `block_size`: block size for attention computation
- `n_hashes`: the number of hash tables, i.e., OR LSH
-  `num_regions`: # of regions HEPT will randomly divide the input space into (Sec. 4.3 in the paper)

We suggest first determine `block_size` and `n_hashes` according to the computational budget, but generally `n_hashes` should be greater than 1. `num_regions` should be tuned according to the local inductive bias of the dataset.


## Reference
```bibtex
@article{miao2024locality,
  title   = {Locality-Sensitive Hashing-Based Efficient Point Transformer with Applications in High-Energy Physics},
  author  = {Miao, Siqi and Lu, Zhiyuan and Liu, Mia and Duarte, Javier and Li, Pan},
  journal = {International Conference on Machine Learning},
  year    = {2024}
}
