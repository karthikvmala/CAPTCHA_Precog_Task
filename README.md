# CAPTCHA Text Generation Using CNN

This repository contains a Convolutional Recurrent Neural Network (CRNN) implementation using PyTorch for variable-length CAPTCHA recognition. The model uses a combination of CNN for feature extraction, followed by Bidirectional LSTM for sequence modeling, and CTC Loss for training the network with variable-length sequences.

## Requirements
- Python 3.x
- PyTorch 1.7+
- Matplotlib

## Dataset Creation
To evaluate CAPTCHA recognition, a custom dataset was created with three difficulty levels:

### 1. Easy Set:
- Fixed font, uniform capitalization.
- Plain white background.

### 2. Hard Set:
- Multiple fonts, fluctuating capitalization.
- Noisy or textured backgrounds to simulate real-world CAPTCHAs.

### 3. Bonus Set (Optional):
- Background color-dependent text rendering:
  - **Green Background** → Text appears normally.
  - **Red Background** → Text is reversed in the image, but the model must predict the correct word.

## Task-1: CNN Classification
A subset of 100 words from both the Easy and Hard datasets was selected to train a CNN-based classifier. The goal was to classify each CAPTCHA image into one of 100 labels corresponding to different words.

### Model Architecture:
1. **Initial Model:**
   - 3 convolutional layers
   - Fully connected layers for classification.
2. **Experimentation:**
   - Increased the number of convolutional layers to capture deeper image features.
   - Added Batch Normalization to prevent overfitting.
   - Fine-tuned learning rates and batch sizes.
3. **Applied augmentation** to increase dataset size by color intensity and elastic transformations.

## Task-2: CRNN-LSTM for Sequence Recognition
Unlike CNN classification (which assigns an entire image to a single label), Task-2 focuses on recognizing variable-length CAPTCHA text sequences. We implemented a CRNN (CNN + RNN) with LSTM layers, combined with Connectionist Temporal Classification (CTC) Loss for training.

### Model Architecture:
1. **Convolutional Layers (Feature Extraction)**
   - Extracts spatial features from images.
   - Uses 2 convolutional layers with max pooling.
2. **Recurrent Layers (Temporal Modeling)**
   - Uses Bidirectional LSTM to learn sequence dependencies.
3. **Fully Connected Layer (Final Prediction)**
   - Outputs a sequence of character probabilities.
4. **CTC Loss**
   - Handles alignment-free training, enabling the model to recognize words of varying lengths.
