# FarSeg-Network-for-Segmentation-on-Seasonet-Dataset

## Introduction

This repository contains all the necessary files for training, testing, and evaluating a FarSeg model for segmentation. Below is a guide on how to explore the repository and understand the key components.

## Repository Structure

- **`main.py`**: The entry point of the project. This script is responsible for launching the entire process, including training and testing with the various options to pass via command line.
- **`solver.py`**: Contains the core logic for training and testing the model. This file manages data loading, model training, evaluation, and saving checkpoints.
- **`model.py`**: Defines the neural network architecture used in the project.
- **`utils.py`**: Contains helper functions used throughout the project, such as data processing and other utilities.
- **Directories:**
  - **`checkpoints/`**: Stores model checkpoints saved during training.
  - **`visualizations/`**: Contains generated plots and visualized results.
  - **`runs/`**: Holds logs and metadata for tracking training experiments (tensorboard data).

## How to Clone and Launch the Repository

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/refedico/FarSeg-Network-for-Segmentation-on-Seasonet-Dataset.git
   cd FarSeg-Network-for-Segmentation-on-Seasonet-Dataset
   ```

2. **Set up the environment**:

   - Ensure you have the required dependencies installed. You can use a virtual environment or Conda to manage packages.

3. **Run the main script**:

   ```bash
   python main.py
   ```

   This will trigger the model training and testing process.

4. **Check training and testing logic**:

   - If you want to understand or modify how training and evaluation are handled, refer to `solver.py`.

5. **Modify the model architecture**:

   - You can adjust the neural network by editing `model.py`.


6. **Run inference and visualize results**:

   - Open `test.ipynb` in Jupyter Notebook to run inference on images and visualize the model's performance.

## Conclusion

This repository is structured to facilitate seamless training, testing, and evaluation. Feel free to explore and modify the scripts as needed!

