# Cloud Classification using PyTorch

This project demonstrates how to build and train a deep learning model to classify different types of clouds using PyTorch. It explores both building a simple Convolutional Neural Network (CNN) from scratch and utilizing transfer learning with a pre-trained ResNet50 model.

## Project Overview

The goal of this project is to accurately classify cloud images into different categories. The dataset used contains images of various cloud types.

The notebook covers the following steps:

1.  **Data Loading and Preprocessing:** Loading the dataset, applying transformations (including data augmentation for training), and creating data loaders.
2.  **Custom CNN Model:** Defining and training a simple CNN model for cloud classification.
3.  **Transfer Learning with ResNet50:** Loading a pre-trained ResNet50 model, modifying it for the specific classification task, and fine-tuning the model.
4.  **Evaluation:** Evaluating the performance of both models using metrics like precision and recall, and visualizing the results with a confusion matrix.
5.  **Sample Prediction:** Demonstrating how to make predictions on new images.

## Getting Started

To run this project, you will need to have Python and the necessary libraries installed. The project is designed to be run in a Google Colab environment, which provides the required dependencies and GPU access.

1.  **Open the Notebook:** Open the provided `.ipynb` notebook file in Google Colab.
2.  **Run Cells:** Execute each code cell sequentially. The notebook includes comments and markdown cells explaining each step.
3.  **Data:** The code assumes the cloud images dataset is available as a `.zip` file and is unzipped in the Colab environment. You will need to provide the path to your dataset.

## Dependencies

The following Python libraries are required to run this project:

-   `torch`
-   `torchvision`
-   `torchmetrics`
-   `matplotlib`
-   `numpy`
-   `seaborn`
-   `sklearn`

These dependencies will be installed automatically when you run the `!pip install` commands in the Colab notebook.

## Dataset

The project uses a dataset of cloud images. The code assumes the data is organized into subdirectories for training and testing, with each subdirectory containing folders for each cloud class.

## Model Architecture

The project explores two model architectures:

1.  **Custom CNN:** A simple CNN with convolutional layers, activation functions, pooling layers, and a fully connected layer.
2.  **ResNet50:** A pre-trained ResNet50 model from `torchvision.models`, fine-tuned for cloud classification.

## Training

Both models are trained using the following components:

-   **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
-   **Optimizer:** Adam (`optim.Adam`)
-   **Learning Rate Scheduler:** StepLR (`lr_scheduler.StepLR`) is used for the ResNet50 model.

## Evaluation

The models are evaluated using:

-   **Precision**
-   **Recall**
-   **Confusion Matrix**
