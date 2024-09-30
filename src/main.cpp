#include "network.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#define IMAGE_SIZE 28
#define TRAIN_IMG_PATH "../../data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../../data/train-labels.idx1-ubyte"

#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8

int swap_endian(int value)
{
    unsigned char *bytes = reinterpret_cast<unsigned char *>(&value);
    std::reverse(bytes, bytes + sizeof(int));
    return value;
}

void read_mnist_images(std::vector<unsigned char> &images, int &nImages)
{
    std::ifstream file(TRAIN_IMG_PATH, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Error opening image file: " + std::string(TRAIN_IMG_PATH));
    }

    int temp, rows, cols;
    file.read(reinterpret_cast<char *>(&temp), sizeof(int));
    file.read(reinterpret_cast<char *>(&nImages), sizeof(int));
    nImages = swap_endian(nImages);

    file.read(reinterpret_cast<char *>(&rows), sizeof(int));
    file.read(reinterpret_cast<char *>(&cols), sizeof(int));
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    images.resize(nImages * rows * cols);
    file.read(reinterpret_cast<char *>(images.data()), images.size());
    file.close();
}

void read_mnist_labels(std::vector<unsigned char> &labels, int &nLabels)
{
    std::ifstream file(TRAIN_LBL_PATH, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Error opening label file: " + std::string(TRAIN_LBL_PATH));
    }

    int temp;
    file.read(reinterpret_cast<char *>(&temp), sizeof(int));
    file.read(reinterpret_cast<char *>(&nLabels), sizeof(int));
    nLabels = swap_endian(nLabels);

    labels.resize(nLabels);
    file.read(reinterpret_cast<char *>(labels.data()), labels.size());
    file.close();
}

void display_image(sf::RenderWindow &window, const std::vector<unsigned char> &images, const std::vector<unsigned char> &labels, int imageIndex)
{
    int offset = imageIndex * IMAGE_SIZE * IMAGE_SIZE;
    sf::Font font;
    if (!font.loadFromFile("../../fonts/PixelifySans-VariableFont_wght.ttf"))
    {
        std::cout << "Failed to load font file" << std::endl;
        exit(EXIT_FAILURE);
    }

    sf::Text text;
    text.setFont(font);
    text.setCharacterSize(20);
    text.setFillColor(sf::Color::White);
    text.setPosition(30, 30);

    unsigned char label = labels[imageIndex];
    std::ostringstream stringStream;
    stringStream << "Label: " << static_cast<int>(label);
    text.setString(stringStream.str());

    sf::Image image;
    image.create(IMAGE_SIZE, IMAGE_SIZE);
    for (int y = 0; y < IMAGE_SIZE; y++)
    {
        for (int x = 0; x < IMAGE_SIZE; x++)
        {
            unsigned char pixelValue = images[offset + y * IMAGE_SIZE + x];
            sf::Color color(pixelValue, pixelValue, pixelValue);
            image.setPixel(x, y, color);
        }
    }

    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
    sprite.setScale(20.f, 20.f);

    window.clear();
    window.draw(sprite);
    window.draw(text);
    window.display();
}

int main()
{
    Network net; // Neural network structure containing hidden and output layers.
    std::vector<unsigned char> images, labels;
    int nImages = 0, nLabels = 0;
    float learning_rate = LEARNING_RATE;
    std::vector<float> img(INPUT_SIZE); // Learning rate and buffer for normalized image

    try
    {
        read_mnist_images(images, nImages);
        read_mnist_labels(labels, nLabels);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Number of images : " << nImages << std::endl;
    std::cout << "Number of labels : " << nLabels << std::endl;

    // Calculate the number of training and test examples.
    int train_size = (nImages * TRAIN_SPLIT);
    int test_size = nImages - train_size;

    // Training loop that iterates through multiple epochs.
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_loss = 0; // Variable to track total loss over the epoch.

        // Iterate over the training data in batches.
        for (int i = 0; i < train_size; i += BATCH_SIZE)
        {
            for (int j = 0; j < BATCH_SIZE && i + j < train_size; j++)
            {
                int idx = i + j; // Current data index.

                // Normalize the input image (convert pixel values from 0-255 to 0-1).
                for (int k = 0; k < INPUT_SIZE; k++)
                {
                    img[k] = images[idx * INPUT_SIZE + k] / 255.0f;
                }

                // Train the network with the current image and its label.
                net.train(img, labels[idx], learning_rate);

                // Compute the loss for this batch (optional for tracking).
                std::vector<float> hidden_output(HIDDEN_SIZE), final_output(OUTPUT_SIZE);
                net.hidden->forward(img, hidden_output);

                // Apply ReLU activation to the hidden layer's output.
                for (int k = 0; k < HIDDEN_SIZE; k++)
                {
                    hidden_output[k] = hidden_output[k] > 0 ? hidden_output[k] : 0; // ReLU Activation
                }

                // Forward pass from hidden layer to output layer.
                net.output->forward(hidden_output,final_output);
                net.softmax(final_output, OUTPUT_SIZE);

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
            if (net.predict(img) == labels[i])
            {
                correct++; // Increment correct count if the prediction matches the label.
            }
        }

        // Print the epoch results: accuracy and average loss.
        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f\n", epoch + 1, (float)correct / test_size * 100, total_loss / train_size);
    }

    return 0;
}