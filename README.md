
<div class="container" style="text-align: center;">
<h1>Model Immunization from a Condition Number Perspective</h1>
    <div class="authors">
        <div class="author-row">
            <a href="https://amberyzheng.github.io/">Amber Yijia Zheng</a>*,
            <a href="https://best99317.github.io/SiteBai/">Cedar Site Bai</a>*,
            <a href="https://bbullins.github.io/">Brian Bullins</a>,
            <a href="https://raymond-yeh.com/">Raymond A. Yeh</a>
        </div>
        <div class="affiliation" style="color:#1a4e7a;font-weight:600;">Department of Computer Science, Purdue University</div>
        <div class="venue" style="color:#e74c3c;font-weight:700;">ICML 2025 (<span style="color:#e74c3c;font-weight:700;">Oral</span>)</div>
    </div>
    <div class="links">
        <a href="https://arxiv.org/abs/2505.23760" class="btn">Paper</a> |
        <a href="https://www.amberyzheng.com/immu_cond_num/" class="btn">Project Page</a>
    </div>
</div>

## Summary

We provide a theoretical framework for model immunization, showing how the condition number of the feature covariance matrix governs the ability to immunize models against harmful fine-tuning via linear probing, while preserving performance on benign tasks.

## Installation

We recommend using conda to create a new environment with all dependencies:

```bash
conda env create -n model-immunization
conda activate model-immunization
pip install -r requirements.txt
```

## Usage

### Training

To train a model with our proposed regularization method:

```bash
python main.py --config <config_file>.yaml [--seed <seed>] [--digit1 <digit1> --digit2 <digit2>] [--ckpt_path <path>]
```

Arguments:
- `--config`: Path to the configuration file (required)
- `--seed`: Random seed (default: 1)
- `--digit1` and `--digit2`: For MNIST experiments, specify two digits to use (optional)
- `--ckpt_path`: Path to checkpoint for model initialization (optional)

### Configuration

The training configuration is specified in YAML files under the `configs/` directory. Key parameters include:

- Model architecture and feature extractor type
- Training hyperparameters (learning rate, batch size, etc.)
- Regularization parameters (lambda_r1, lambda_r2)
- Dataset settings

### Experiments

#### House Price Regression
```bash
python main.py --config configs/house_price.yaml --seed 1
```

#### MNIST Classification
For each digit pair (e.g., digits 0 and 1):
```bash
python main.py --config configs/mnist.yaml --seed 1 --digit1 0 --digit2 1
```

#### Deep Neural Networks
For ResNet18 with [Cars dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) as harmful task:
```bash
python main.py --config configs/resnet18_cars.yaml --seed 1
```

For ResNet18 with [Country211 dataset](https://github.com/openai/CLIP/blob/main/data/country211.md) as harmful task:
```bash
python main.py --config configs/resnet18_country211.yaml --seed 1
```

For ViT with Cars dataset as harmful task:
```bash
python main.py --config configs/vit_cars.yaml --seed 1
```

For ViT with Country211 dataset as harmful task:
```bash
python main.py --config configs/vit_country211.yaml --seed 1
```

Note: For DeepNet experiments, you'll need to download the ImageNet-1K dataset and specify its path in the config file.

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{zheng2025model,
  title={Model Immunization from a Condition Number Perspective},
  author={Zheng, Amber Yijia* and Bai, Cedar Site* and Bullins, Brian and Yeh, Raymond A.},
  booktitle={Proc. ICML},
  year={2025}
}
```

## License

This work is licensed under the Apache-2.0 license.
