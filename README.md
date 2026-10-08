# PSLE-YOLO: Infrared Tiny Ship Detection

Official implementation of PSLE-YOLO, a lightweight object detection framework for infrared tiny ship detection in complex maritime environments.

## Overview

Infrared tiny ship detection is challenging due to weak target responses, limited spatial information, and interference from clouds, coastlines, and complex sea clutter.

PSLE-YOLO is developed based on YOLOv8 to improve the detection of small and low-contrast ship targets. The framework combines spatial-preserving downsampling, efficient multi-scale feature fusion, local feature enhancement, and global feature selection to strengthen target representations while maintaining computational efficiency.

## Project Structure

```text
PSLE-YOLO/
├── ultralytics/        # Model architecture and detection framework
├── data/               # Dataset information and download links
├── demo/               # Detection visualizations
├── mytrain.py          # Model training
├── myval.py            # Model validation
├── mytest.py           # Model inference
├── requirements.txt    # Python dependencies
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hnnsk/PSLE-YOLO.git
cd PSLE-YOLO
```

### 2. Create the Environment

We recommend using a separate Conda environment.

```bash
conda create -n psle-yolo python=3.10 -y
conda activate psle-yolo
```

### 3. Install Dependencies

Install a compatible version of PyTorch according to your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Experiments are conducted on two publicly available infrared ship detection datasets.

| Dataset | Description | Download |
|---|---|---|
| NUDT-SIRST-Sea | Infrared tiny ship detection in complex maritime scenes | [Official Repository](https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection) |
| TISD | Three-band thermal infrared ship detection dataset | [Dataset Paper](https://doi.org/10.3390/rs14215297) |

Please download the datasets from their original sources and prepare the corresponding dataset configuration files.

Detailed dataset information and download instructions are available in [data/README.md](data/README.md).

## Usage

The repository provides scripts for model training, validation, and inference.

Before running the scripts, update the dataset paths, model configuration paths, and checkpoint paths according to your local environment.

### Training

Configure the model YAML and dataset YAML paths in `mytrain.py`, then run:

```bash
python mytrain.py
```

### Validation

Specify the trained model checkpoint and dataset configuration in `myval.py`, then run:

```bash
python myval.py
```

### Inference

Specify the trained model checkpoint and input image directory in `mytest.py`, then run:

```bash
python mytest.py
```

Detection results are saved to the configured output directory.

## Detection Demo

Qualitative detection results on NUDT-SIRST-Sea and TISD are provided to illustrate the performance of PSLE-YOLO in different maritime environments.

**[View Detection Results](demo/README.md)**

## Acknowledgments

This project is built upon the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework.

We sincerely thank the authors of NUDT-SIRST-Sea and TISD for providing the datasets used in this research.
