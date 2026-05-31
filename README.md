<h1 align="center">LSH-Based Efficient Point Transformer (HEPT)</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2402.12535"><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/HEPT"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://arxiv.org/abs/2402.12535"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2724&color=blue"></a>
</p>

## This HEPT Fork uses PQuant for pruning and quantization

Script for pruning and quantization is uploaded as `src/tracking_quantizer_and_pruner.py`

There are 2 existing PyTorch model files present under `data/tracking/logs/`:

Note: These are old results
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
```

Download PQuant manually, `git clone https://github.com/cern-nextgen/PQuantML`<br><br>
Then, go to  `src/pquant/core/torch/layers.py` and comment out the line: `model(torch.rand(input_shape).to("cuda"))`<br><br>
Now, install PQuant using `pip install <Path to PQuant>`

#### Running the code

Before running the code, change the "PATH" variable in `tracking_quantizer_and_pruner.py` to the folder path that you want the logs to go in.

For running the pruning / quantization script
```
python tracking_quantizer_and_pruner.py
```

Configurations will be loaded from those located in `./configs/` directory.

<br>

Current issue - Unable to replicate results with PQuant's new changes.

## TODOs
- [ ] Use PQuant's dev branch to check if pruning / quantization works with that.
- [ ] When pruning, try using `print(model)` to see if pruning layers are being configured.
- [ ] Input shape (right now it is arbitary) being passed to PQuant might matter, not running the model once on any input might be a problem.
