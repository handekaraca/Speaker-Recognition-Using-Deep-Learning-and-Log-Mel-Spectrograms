# Speaker Recognition Using Deep Learning and Log-Mel Spectrograms

This project presents a high-accuracy speaker recognition system developed using deep learning techniques. Real-world voice recordings from five well-known political figures were processed and classified using Convolutional Neural Networks (CNNs) based on their log-Mel spectrogram representations.

## Project Overview

The goal of this project is to build a robust speaker recognition model that can accurately identify individuals based on their voice characteristics. The system leverages deep learning to capture subtle variations in speech such as pitch, formants, and speech rate, using CNNs trained on log-Mel spectrogram features.

## Technologies & Tools

* Python
* TensorFlow / Keras
* Librosa
* NumPy / Pandas / Matplotlib
* Kaggle (Dataset Source)

## Dataset

* **Source:** [Speaker Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/vaibhavkumar/speaker-recognition-dataset)
* **Speakers:** Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret Thatcher, Nelson Mandela
* **Format:** 1-second PCM audio clips, 16kHz sampling rate
* **Balance:** Equal number of samples for each speaker
* **Noise samples** included for augmentation and robustness testing

##  Methodology

1. **Preprocessing**

   * Pre-emphasis filtering
   * Short-Time Fourier Transform (STFT)
   * Conversion to log-Mel spectrograms
   * Normalization to \[0, 1] scale

2. **Model Architecture**

   * Convolutional Neural Network (CNN)
   * Layers: Conv2D + ReLU + MaxPooling + Dropout
   * Final Softmax layer for multiclass classification
   * Optimizer: Adam (lr=0.001)
   * Loss: Categorical Crossentropy
   * Techniques: Early stopping, learning rate scheduler, data augmentation

3. **Data Augmentation**

   * Time shifting
   * Background noise addition

## Results

* **Test Accuracy:** 100%
* **Precision / Recall / F1-Score (per class):** 1.00
* **No misclassifications** (as shown in the confusion matrix)
* Robust to background noise and speaker gender/accents

## Visualizations

* Training/validation accuracy and loss plots
* Confusion matrix
* Epoch-wise performance table

## Key Insights

* CNNs combined with log-Mel spectrograms provide powerful speaker embeddings.
* Even with a limited dataset, proper preprocessing and augmentation can yield high generalization.
* The model is lightweight and suitable for real-time applications and potentially mobile deployment.
