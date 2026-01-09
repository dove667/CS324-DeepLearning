# Assignment 2 Code Reproduction Guide

This guide provides instructions on how to run the code for Assignment 2, covering MLP, CNN, and RNN.

## Environment Setup

Ensure you have the following dependencies installed:
*   Python 3.x
*   PyTorch
*   Torchvision
*   NumPy
*   Matplotlib
*   Scikit-learn
*   Jupyter Notebook

## Part 1: MLP on CIFAR-10

### PyTorch Implementation
1.  Navigate to the `Part 1` directory:
    ```bash
    cd "Part 1"
    ```
2.  run the training script:
    ```bash
    python pytorch_train_mlp.py
    ```
3.  Open and run the `CIFAR10.ipynb` notebook to train and test the PyTorch-based MLP model on CIFAR-10:
    ```bash
    jupyter notebook CIFAR10.ipynb
    ```

## Part 2: CNN on CIFAR-10

1.  Navigate to the `Part 2` directory:
    ```bash
    cd "Part 2"
    ```
2.  Run the training script:
    ```bash
    python cnn_train.py
    ```

## Part 3: RNN on Palindrome

1.  Navigate to the `Part 3` directory:
    ```bash
    cd "Part 3"
    ```

2.  Open and run the `task2.ipynb` notebook (or `task3.ipynb` if renamed) to see the RNN implementation and training for the Palindrome task.
    ```bash
    jupyter notebook task2.ipynb
    ```
3.  Alternatively, if there is a standalone training script (e.g., `train.py`), you can run it directly:
    ```bash
    python train.py
    ```

## Notes
*   Data will be automatically downloaded to the `12410106_assignment2/data` directory if not present.
*   Ensure you are in the correct directory before running scripts to avoid path issues.
