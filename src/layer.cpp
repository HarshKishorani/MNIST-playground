#include "layer.hpp"

Layer::Layer(int in_size, int out_size)
{
    int n = in_size * out_size; // Total number of weights required in the layer.
    float scale = sqrtf(2.0f / in_size);

    this->input_size = in_size;
    this->output_size = out_size;
    this->weights = std::vector<float>(n);
    this->biases = std::vector<float>(out_size, 0.f);

    // We use 'He Initialization' to set the weights.
    for (int i = 0; i < n; i++)
    {
        this->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
}

void Layer::forward(std::vector<float> &input, std::vector<float> &output)
{
    // Loop over each output node (neuron) in the layer (i.e., for each neuron in this layer).
    for (int i = 0; i < this->output_size; i++)
    {
        // Start by setting the output to the bias of the current neuron.
        // Each neuron has its own bias term that is independent of the input.
        output[i] = this->biases[i];

        // Loop over each input coming from the previous layer.
        for (int j = 0; j < this->input_size; j++)
        {
            // Calculate the contribution of the current input to the output of the neuron.
            // The weight corresponding to the connection between input j and output i is located at index (j * output_size + i).
            // Multiply the input value by the corresponding weight and accumulate it in the output.
            output[i] += (input[j] * this->weights[j * this->output_size + i]);
        }
    }
}

void Layer::backward(std::vector<float> &input, std::vector<float> &output_grad, std::vector<float> &input_grad, float lr)
{
    // Loop over each output node (neuron) of the layer
    for (int i = 0; i < this->output_size; i++)
    {
        // Loop over each input node of the layer
        for (int j = 0; j < this->input_size; j++)
        {
            // Calculate the index for accessing the weight between input j and output i
            int idx = j * this->output_size + i;

            // Compute the gradient of the loss with respect to the weight:
            // This is the gradient of the loss with respect to the output times the input value.
            // grad = ∂L/∂w_ij = output_grad[i] * input[j]
            float grad = output_grad[i] * input[j];

            // Update the weight by subtracting the product of the learning rate and the gradient.
            // w_ij = w_ij - lr * grad
            this->weights[idx] -= lr * grad;

            // If input_grad is not empty, compute the gradient of the loss with respect to the input j.
            // This is done by summing up the gradient of the loss with respect to each output i
            // multiplied by the weight connecting input j to output i.
            // input_grad[j] += ∂L/∂o_i * w_ij
            if (input_grad.size() != 0)
            {
                input_grad[j] += output_grad[i] * this->weights[idx];
            }
        }

        // Update the bias for output i.
        // The gradient of the loss with respect to the bias is simply the output gradient.
        // b_i = b_i - lr * output_grad[i]
        this->biases[i] -= lr * output_grad[i];
    }
}
