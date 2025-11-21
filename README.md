# Self-Paced Learning for Images of Antinuclear Antibodies

Official PyTorch implementation of **"Self-Paced Learning for Images of Antinuclear Antibodies"** published in **IEEE Transactions on Medical Imaging (TMI)**.

This repository contains the code for multi-label classification of Antinuclear Antibodies (ANA) images using self-paced learning with adaptive sample weighting and pseudo-label training.



## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Place image files in `./data/images/` directory
- Prepare CSV annotation file with the following columns:
  - `path`: Image path
  - `TARGET`: Labels (space-separated numbers, e.g., "0 3 5")
  - `Split`: Dataset split (train/val/test)

### 3. Update Configuration

Edit `config.py` or `config_single.py` to update data paths:

```python
DIR_TRAIN_IMAGES = './data/images/'
DIR_TEST_IMAGES = './data/images/'
PATH_TRAIN_ANNFILE = 'your_annotations.csv'
PATH_TEST_ANNFILE = 'your_annotations.csv'
```

### 4. Train Model

**Full training command (recommended):**

```bash
python main.py --saveModel --lr 1e-3 --weight_lr 1e-3 --initWeight iw-sample --updateLR ulr-adaptive --granularity label --sampling --trainingLabel pseudo
```

**Simple training command:**

```bash
python main.py --saveModel --lr 1e-3
```

**Single-label training:**

```bash
python main_single.py --saveModel --lr 1e-3
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Model learning rate | 5e-3 |
| `--weight_lr` | Sample weight learning rate | 5e-3 |
| `--epoch` | Number of epochs | 200 |
| `--trainBatchSize` | Training batch size | 32 |
| `--saveModel` | Save model checkpoints | False |
| `--initWeight` | Weight initialization (iw-ones/iw-data/iw-sample) | iw-ones |
| `--updateLR` | Learning rate update strategy (ulr-ones/ulr-adaptive) | ulr-ones |
| `--trainingLabel` | Training label type (real/pseudo) | real |
| `--granularity` | Weight granularity (label/sample) | label |
| `--sampling` | Enable weighted sampling | False |

## Features

- Multi-label classification
- Adaptive sample weighting
- Pseudo-label training
- Weighted random sampling
- Early stopping mechanism
- Multiple evaluation metrics (Accuracy, F1-score, mAP)

## Output Files

Training generates the following files:
- `{model_name}/output.txt` - Training log
- `{model_name}/{epoch}_f1mi.pt` - Best micro-F1 model
- `{model_name}/{epoch}_f1ma.pt` - Best macro-F1 model
- `{model_name}/{epoch}_acc.pt` - Best accuracy model
- `{model_name}/{epoch}_mAP.pt` - Best mAP model
- `{model_name}/results.json` - Evaluation results

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA (recommended for GPU acceleration)

See `requirements.txt` for full dependencies.

## Notes

- Supports standard image formats (jpg, png, etc.)
- Ensure paths in config files point to your data
- GPU will be used automatically if available
- CSV file must contain `path`, `TARGET`, `Split` columns


## License

This code is released for academic research use only.

