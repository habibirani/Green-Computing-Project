# Green-Computing-Project
This repository contains code for training, evaluating, and applying quantization and pruning techniques to a transformer model designed for analyzing time series data. The project was undertaken as part of the Advanced Green Computing course (CS7333) at the Department of Computer Science, Texas State University.


## Table of Contents
- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Model](#model)
- [Dataset](#dataset)
- [Training Procedure](#Training-Procedure)
- [Inference](#inference)
- [Results](#results)
- [Discussion](#discussion)
- [Contributing](#contributing)
- [Citation](#citation)


## Description
This project aims to address the substantial energy consumption and carbon footprint associated with training and deploying these models. By focusing on optimization techniques such as pruning, quantization, and knowledge distillation, specifically tailored to the domain of time series data, the research endeavors to enhance the computational efficiency and environmental sustainability of transformer models. Employing the UniMiB SHAR dataset for human activity recognition as a testbed, this study will systematically evaluate the impact of various optimization strategies on model performance, energy consumption, and speed.


## Dependencies
- Python 3.10
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn
- PyTorch

## Installation

To install and run this repository, follow these steps:

```bash
git clone https://github.com/habibirani/Green-Computing-project.git
cd Green-Computing-project
conda env create -f environment.yml

```

## Model
  - [Transformer-encoder] The TimeSeriesTransformer model is designed for processing sequential data, with a focus on time-series analysis. It comprises a patch-based embedding layer and a Transformer encoder architecture. The patch embedding layer preprocesses input sequences, while the Transformer encoder captures temporal dependencies using self-attention mechanisms across multiple encoder layers.
  
## Dataset

The UniMib dataset consists of time series data collected from wearable sensors for human activity recognition tasks. It contains accelerometer and gyroscope readings capturing various activities such as walking, running, sitting, and standing. The dataset is provided in CSV format and includes timestamps along with sensor readings. To facilitate easy usage, a data loader script 'data_loader.py' is provided in the 'data' folder, offering functions for loading, preprocessing, and splitting the dataset into training, validation, and test sets.

## Training Procedure

The Transformer model was trained using  Adam optimizer, cross-entropy loss. Hyperparameters such as learning rate, batch size, number of epochs, etc. were tuned in this phase. The training process included validation to monitor model performance.

## Inference

Model accuracy was evaluated using standard metrics such as classification accuracy. The average inference time per sample was measured to assess the computational efficiency of the models. Then post-training static quantization was employed using the quantize_dynamic function with a specified quantization configuration and Pruning based on the L1 norm of the weight tensors was applied to eliminate less significant connections. Again model accuracy was evaluated using classification accuracy and inference time was measured.

## Results

The scripts folder contains various Jupyter notebokes for training, evaluating, and demonstrating the performance of the transformer model on the UniMib dataset. These scripts include training the model with different configurations and hyperparameters, evaluating the trained models on the test set, and performing inference with the trained models. Additionally, there are scripts for applying quantization and pruning techniques to the trained models, such as showcasing the impact of quantization and pruning on model size, accuracy, and inference time. The results obtained from running these scripts provide insights into the effectiveness of different training strategies and optimization techniques for enhancing the performance of the transformer model on activity recognition tasks.

## Discussion

These experimental results demonstrate the effectiveness of quantization and pruning techniques in reducing model size and computational complexity while preserving model accuracy to a considerable extent.

<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

<!-- CITATION -->
## Citation
If you use this repository in your research please use the following BibTeX entry:

```bibtex
@Misc{TransformerInferenceEnergy,
  title = {Optimizing Transformer Models for Energy Efficiency
           in Time Series Classification},
  author = {Irani, Habib},
  howpublished = {Github},
  year = {2024}
  url = {https://github.com/habibirani/Green-Computing-Project}
}
```

