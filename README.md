# MNIST Handwritten Digit Recognition with SFML Drawing Interface

This project demonstrates the integration of training a neural network for MNIST digit recognition with a custom SFML-based canvas for drawing and predicting digits in real time. It allows users to hand-draw digits using a mouse on a 28x28 grid, simulating MNIST images, and predicts the drawn digit using a pre-trained neural network.

## How It Works
1. **Drawing**: Draw a digit on the SFML canvas window.
2. **Prediction**: Once finished, press the `Enter` key to predict the drawn digit using the trained neural network model.
3. **Result**: The drawn digit is displayed in grayscale, and the predicted digit is printed on the console.

## Key Components
- **Neural Network**: 
  - The network consists of an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation.
  - Weights and biases are saved and loaded from binary files to allow for efficient training and prediction.
  
- **Input Handling**:
  - The canvas allows freehand drawing using the mouse, and the image is processed into 28x28 grayscale to match the MNIST dataset's resolution.
  
- **Training and Prediction**:
  - The model can be trained on the MNIST dataset and saved for later use. It can also load an existing model and perform real-time predictions on drawn images.