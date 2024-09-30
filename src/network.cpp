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

void Network::save_network(std::string filename)
{
    std::cout << "=> Saving Network...." << std::endl;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file to save network: " << filename << std::endl;
        exit(1);
    }

    // Save hidden layer weights and biases
    file.write(reinterpret_cast<char *>(this->hidden->weights.data()), this->hidden->weights.size() * sizeof(float));
    file.write(reinterpret_cast<char *>(this->hidden->biases.data()), this->hidden->biases.size() * sizeof(float));

    // Save output layer weights and biases
    file.write(reinterpret_cast<char *>(this->output->weights.data()), this->output->weights.size() * sizeof(float));
    file.write(reinterpret_cast<char *>(this->output->biases.data()), this->output->biases.size() * sizeof(float));

    file.close();
    std::cout << "=> Network saved at : " << filename << std::endl;
}

void Network::load_network(std::string filename)
{
    std::cout << "=> Loading Network...." << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file to load network: " << filename << std::endl;
        exit(1);
    }

    // Initialize the hidden and output layers (ensure memory allocation)
    this->hidden = new Layer(INPUT_SIZE, HIDDEN_SIZE);
    this->output = new Layer(HIDDEN_SIZE, OUTPUT_SIZE);

    // Load hidden layer weights and biases
    file.read(reinterpret_cast<char *>(this->hidden->weights.data()), this->hidden->weights.size() * sizeof(float));
    file.read(reinterpret_cast<char *>(this->hidden->biases.data()), this->hidden->biases.size() * sizeof(float));

    // Load output layer weights and biases
    file.read(reinterpret_cast<char *>(this->output->weights.data()), this->output->weights.size() * sizeof(float));
    file.read(reinterpret_cast<char *>(this->output->biases.data()), this->output->biases.size() * sizeof(float));

    file.close();
    std::cout << "=> Network loaded from : " << filename << std::endl;
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

void Network::trainSingle(std::vector<float> &input, int label, float lr)
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

void Network::trainNetwork(InputData &data,
                           float learning_rate,
                           float trainSplit,
                           int epochs,
                           int batchSize)
{
    // Loading Data to feed the network
    int nImages = data.nImages;
    int nLabels = data.nLabels;
    std::vector<unsigned char> images = data.images, labels = data.labels;

    printf("=> Starting training with %d epoch(s).\n", epochs);

    std::vector<float> img(INPUT_SIZE); // buffer for normalized image

    // Calculate the number of training and test examples.
    int train_size = (nImages * trainSplit);
    int test_size = nImages - train_size;

    // Training loop that iterates through multiple epochs.
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0; // Variable to track total loss over the epoch.

        // Iterate over the training data in batches.
        for (int i = 0; i < train_size; i += batchSize)
        {
            for (int j = 0; j < batchSize && i + j < train_size; j++)
            {
                int idx = i + j; // Current data index.

                // Normalize the input image (convert pixel values from 0-255 to 0-1).
                for (int k = 0; k < INPUT_SIZE; k++)
                {
                    img[k] = images[idx * INPUT_SIZE + k] / 255.0f;
                }

                // Train the network with the current image and its label.
                this->trainSingle(img, labels[idx], learning_rate);

                // Compute the loss for this batch (optional for tracking).
                std::vector<float> hidden_output(HIDDEN_SIZE), final_output(OUTPUT_SIZE);
                this->hidden->forward(img, hidden_output);

                // Apply ReLU activation to the hidden layer's output.
                for (int k = 0; k < HIDDEN_SIZE; k++)
                {
                    hidden_output[k] = hidden_output[k] > 0 ? hidden_output[k] : 0; // ReLU Activation
                }

                // Forward pass from hidden layer to output layer.
                this->output->forward(hidden_output, final_output);
                this->softmax(final_output, OUTPUT_SIZE);

                // Calculate the loss using the negative log-likelihood of the true label.
                total_loss += -logf(final_output[labels[idx]] + 1e-10f); // Avoid log(0) by adding a small epsilon.
            }
        }

        // Testing phase: Evaluate accuracy on the test set.
        int correct = 0; // Track the number of correct predictions.
        for (int i = train_size; i < nImages; i++)
        {
            // Normalize the input image.
            for (int k = 0; k < INPUT_SIZE; k++)
            {
                img[k] = images[i * INPUT_SIZE + k] / 255.0f;
            }

            // Predict the label for the current test image.
            if (this->predict(img) == labels[i])
            {
                correct++; // Increment correct count if the prediction matches the label.
            }
        }

        // Print the epoch results: accuracy and average loss.
        printf("   - Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f\n", epoch + 1, (float)correct / test_size * 100, total_loss / train_size);
    }
}
