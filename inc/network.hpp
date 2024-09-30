#pragma once
#include <layer.hpp>
#include "input_data.hpp"

#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10

class Network
{
private:
    Layer *hidden;
    Layer *output; // MNIST Neural Network

    void softmax(std::vector<float> &input, int size);

    /// @brief Train the network on a single Aexample, performing a forward pass followed by a backward pass.
    /// This function updates the network's weights and biases based on the computed gradients.
    /// @param input Pointer to the input data for this training example.
    /// @param label The correct label (class) for this training example (used to calculate the loss).
    /// @param lr Learning rate, which controls how much to adjust the weights and biases based on the gradients.
    void trainSingle(std::vector<float> &input, int label, float lr);

public:
    Network();
    ~Network();

    /// @brief Perform prediction using the neural network by performing a forward pass and returning the class with the highest probability.
    /// @param input Pointer to the input data for which we want to predict the class.
    /// @return The index of the class (label) with the highest probability.
    int predict(std::vector<float> &input);

    /// @brief Saves the trained network (weights and biases) to a file.
    /// @param filename The file path where the network will be saved.
    void save_network(std::string filename);

    /// @brief Loads the network (weights and biases) from a file.
    /// @param filename The file path from where the network will be loaded.
    void load_network(std::string filename);

    /// @brief Trains the neural network on the provided dataset over multiple epochs using stochastic gradient descent.
    ///
    /// This function handles the main training loop of the neural network. It divides the dataset into training and test sets
    /// and performs the forward and backward passes to optimize the network’s weights and biases. After each epoch, it evaluates
    /// the network's accuracy on the test set.
    ///
    /// @param data The dataset object containing the training images and labels (as `InputData`).
    /// @param learning_rate The learning rate used to update the weights and biases during training.
    /// @param trainSplit A float value representing the fraction of data to be used for training (e.g., 0.8 for 80% training, 20% testing).
    /// @param epochs The number of times the training process iterates over the entire training dataset.
    /// @param batchSize The number of samples to process before updating the network’s weights (batch size).
    void trainNetwork(InputData &data,
                      float learning_rate,
                      float trainSplit,
                      int epochs,
                      int batchSize);
};