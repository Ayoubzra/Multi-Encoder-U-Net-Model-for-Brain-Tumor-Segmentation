# Multi-Encoder U-Net for Brain Tumor Segmentation

This repository contains the implementation of a Multi-Encoder U-Net for multimodal 3D brain tumor segmentation using BraTS 2023 dataset.

## Folder Structure

- `requirements.txt` : Python libraries needed
- `get_data_from_synapse.py` : download and extract dataset from Synapse
- `metrics.py` : dice coefficients and evaluation metrics
- `datagenerator.py` : custom Keras data generator
- `unet_model.py` : multi-encoder U-Net model
- `main_notebook.ipynb` : main workflow, visualization, and training
- `README.md` : this file

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download data from Synapse using your token:
```python
from get_data_from_synapse import download_and_extract_data
train_path, val_path = download_and_extract_data(token="YOUR_TOKEN_HERE")
```

3. Launch `main_notebook.ipynb` for training, visualization, and evaluation.
