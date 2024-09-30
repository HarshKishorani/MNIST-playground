#pragma once
#include <math.h>
#include <cstdlib>
#include <vector>
#include <iostream>

#define INPUT_SIZE 784

class Layer
{
public:
    std::vector<float> weights;  // A flattened array representing the weight matrix.
    std::vector<float> biases;   // An array for the biases of each neuron.
    int input_size, output_size; // Input and output size of a layer

    /// @brief Initialize the layer and its weights and biases
    /// @param in_size Input size of the layer
    /// @param out_size Output size of the layer
    Layer(int in_size, int out_size);

    /// @brief Forward pass: The process of computing the output of a layer in a neural network given the input.
    /// It computes the weighted sum of inputs for each neuron and adds the bias to get the output.
    ///
    /// @param input Pointer to the input data (from the previous layer or input layer in the network).
    /// @param output Pointer to the array that will hold the computed output of the current layer.
    void forward(std::vector<float> &input, std::vector<float> &output);

    /// @brief Backward pass, the reversed flow of the forward pass - propagating the error from the output layer back through hidden layers to the input layer.
    /// The function updates the weights and biases of the layer based on the gradients from the output.
    ///
    /// @param input Pointer to the input data (the same input used during the forward pass).
    /// @param output_grad Pointer to the gradient of the loss with respect to the output (provided by the layer ahead in the network).
    /// @param input_grad Pointer to store the gradient of the loss with respect to the input (used to propagate gradients backward to the previous layer).
    ///        If null, it means we do not need to compute gradients for the input (i.e., for the first layer).
    /// @param lr Learning rate, a scalar value that controls how much we adjust the weights and biases based on the gradients.
    void backward(std::vector<float> &input, std::vector<float> &output_grad, std::vector<float> &input_grad, float lr);
};