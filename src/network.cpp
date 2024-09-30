#include "network.hpp"

void Network::softmax(std::vector<float> &input, int size)
{
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max)
            max = input[i];
    for (int i = 0; i < size; i++)
    {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

Network::Network()
{
    this->hidden = new Layer(INPUT_SIZE, HIDDEN_SIZE);
    this->output = new Layer(HIDDEN_SIZE, OUTPUT_SIZE);
}

Network::~Network()
{
    free(this->hidden);
    free(this->output);
}

int Network::predict(std::vector<float> &input)
{
    // Arrays to store the intermediate hidden layer output and the final output (probabilities).
    std::vector<float> hidden_output(HIDDEN_SIZE), final_output(OUTPUT_SIZE);

    // Forward pass through the hidden layer.
    this->hidden->forward(input, hidden_output);

    // Apply the ReLU activation function to the hidden layer's output.
    // ReLU sets all negative values to 0, keeping positive values unchanged.
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;
    }

    // Forward pass through the output layer.
    this->output->forward(hidden_output, final_output);

    // Apply the softmax function to convert the raw output scores (logits) into probabilities.
    softmax(final_output, OUTPUT_SIZE);

    // Find the index of the maximum probability in the output layer.
    // This corresponds to the predicted class label.
    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
    {
        if (final_output[i] > final_output[max_index])
        {
            max_index = i;
        }
    }

    // Return the index of the class with the highest probability.
    return max_index;
}

void Network::train(std::vector<float> &input, int label, float lr)
{
    // Arrays to store intermediate values and gradients
    std::vector<float> hidden_output(HIDDEN_SIZE);
    std::vector<float> final_output(OUTPUT_SIZE);
    std::vector<float> output_grad(OUTPUT_SIZE, 0);
    std::vector<float> hidden_grad(HIDDEN_SIZE, 0);

    // Forward Pass: Input to Hidden Layer
    this->hidden->forward(input, hidden_output);

    // Apply ReLU activation function on the hidden layer output
    // ReLU (Rectified Linear Unit) sets any negative value to 0, keeping positive values unchanged.
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0; // ReLU Activation
    }

    // Forward Pass: Hidden Layer to Output Layer
    this->output->forward(hidden_output, final_output);

    // Apply the softmax function to the output layer, converting logits into probabilities
    softmax(final_output, OUTPUT_SIZE);

    // Compute the gradient of the loss with respect to the output.
    // This is based on the difference between the predicted output (final_output[i]) and the true label (one-hot encoded).
    // The loss gradient for the correct class is negative and positive for incorrect classes.
    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_grad[i] = final_output[i] - (i == label); // Softmax-CrossEntropy gradient

    // Backward Pass: Propagate the gradient from the output layer to the hidden layer.
    // This updates the weights and biases of the output layer.
    this->output->backward(hidden_output, output_grad, hidden_grad, lr);

    // Backpropagate Through ReLU Activation:
    // Only propagate the gradient for neurons where ReLU was active (output > 0).
    // The gradient is 0 for inputs where ReLU "deactivated" the neuron (output <= 0).
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0; // Derivative of ReLU
    }

    // Backward Pass: Propagate the gradient from the hidden layer to the input layer.
    // This updates the weights and biases of the hidden layer.
    // Since this is the input layer, we do not need to compute further gradients (hence, input_grad is NULL).
    std::vector<float> nullOutputGrad;
    this->hidden->backward(input, hidden_grad, nullOutputGrad, lr);
}
