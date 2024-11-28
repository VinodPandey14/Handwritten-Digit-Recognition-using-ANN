# Handwritten Digit Recognizer

This project uses a deep learning model to recognize handwritten digits from the MNIST dataset. The model is built using **TensorFlow** and **Keras** and trained to classify digits ranging from 0 to 9.

The model is designed to accept handwritten digit input, make predictions, and output the recognized digit.

## Project Overview

- **Dataset**: MNIST (containing 60,000 training images and 10,000 test images of handwritten digits)
- **Model**: Convolutional Neural Network (ANN) trained using Keras.
- **Tech Stack**:
  - **Backend**: Python, TensorFlow, Keras
  - **Notebook**: Google Colab for interactive execution of code
- **Features**:
  - Data preprocessing (scaling, reshaping).
  - Model training with Keras.
  - Evaluation and accuracy metrics.
  - Prediction using the trained model.

## How to Use

1. **Open the Google Colab Notebook**:
   - [Click here to access the notebook](https://colab.research.google.com/drive/12FB91--UB7ke-iClP8lyipGfFm4zki1Z?usp=sharing).

2. **Run the Notebook**:
   - Go through each cell in the notebook to train the model and evaluate it.
   - You can modify the hyperparameters or retrain the model to improve its accuracy.

3. **Making Predictions**:
   - After training the model, you can use it to predict handwritten digits.
   - The notebook will show an example of how to use the model to predict new images.

## Requirements

To run this project locally, you'll need the following Python libraries:
- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`

You can install the necessary libraries using `pip`:
```bash
pip install tensorflow keras numpy matplotlib
