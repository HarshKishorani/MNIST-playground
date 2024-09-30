#pragma once
#include <layer.hpp>

#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10

class Network
{
public:
    Layer *hidden;
    Layer *output; // MNIST Neural Network

    void softmax(std::vector<float> &input, int size);

    Network();
    ~Network();

    /// @brief Saves the trained network (weights and biases) to a file.
    /// @param net The network to save.
    /// @param filename The file path where the network will be saved.
    void save_network(std::string &filename);

    /// @brief Loads the network (weights and biases) from a file.
    /// @param net The network to initialize and load.
    /// @param filename The file path from where the network will be loaded.
    void load_network(std::string &filename);

    /// @brief Perform prediction using the neural network by performing a forward pass and returning the class with the highest probability.
    /// @param net Pointer to the network, which contains the hidden and output layers.
    /// @param input Pointer to the input data for which we want to predict the class.
    /// @return The index of the class (label) with the highest probability.
    int predict(std::vector<float> &input);

    /// @brief Train the network on a single Aexample, performing a forward pass followed by a backward pass.
    /// This function updates the network's weights and biases based on the computed gradients.
    /// @param net Pointer to the network, which contains the hidden and output layers.
    /// @param input Pointer to the input data for this training example.
    /// @param label The correct label (class) for this training example (used to calculate the loss).
    /// @param lr Learning rate, which controls how much to adjust the weights and biases based on the gradients.
    void train(std::vector<float> &input, int label, float lr);
};