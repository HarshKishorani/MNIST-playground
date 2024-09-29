#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#define IMAGE_SIZE 28

namespace FileUtils
{
    /// @brief Helper function to swap the byte order (big-endian to little-endian).
    int swap_endian(int value)
    {
        unsigned char *bytes = reinterpret_cast<unsigned char *>(&value);
        std::reverse(bytes, bytes + sizeof(int));
        return value;
    }

    /// @brief Function to read MNIST image data from a file
    /// @param filename is the path to the image file
    /// @param images is a vector to store the loaded image data
    /// @param nImages stores the total number of images in the dataset
    void read_mnist_images(const std::string &filename, std::vector<unsigned char> &images, int &nImages)
    {
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Error opening image file: " + filename);
        }

        int temp, rows, cols;

        // Read the first 4 bytes (magic number), not needed for processing
        file.read(reinterpret_cast<char *>(&temp), sizeof(int));

        // Read the number of images (32-bit integer)
        file.read(reinterpret_cast<char *>(&nImages), sizeof(int));
        nImages = swap_endian(nImages); // Swap byte order if needed

        // Read the number of rows and columns (each 32-bit integers)
        file.read(reinterpret_cast<char *>(&rows), sizeof(int));
        file.read(reinterpret_cast<char *>(&cols), sizeof(int));
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        // Allocate memory to hold all image data (each image is 'rows x cols' bytes)
        images.resize(nImages * rows * cols);

        // Read the image data into the allocated memory
        file.read(reinterpret_cast<char *>(images.data()), images.size());

        file.close();
    }

    /// @brief Function to read MNIST label data from a file
    /// @param filename is the path to the label file
    /// @param labels is a vector to store the loaded label data
    /// @param nLabels stores the total number of labels
    void read_mnist_labels(const std::string &filename, std::vector<unsigned char> &labels, int &nLabels)
    {
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Error opening label file: " + filename);
        }

        int temp;

        // Read the first 4 bytes (magic number), not needed for processing
        file.read(reinterpret_cast<char *>(&temp), sizeof(int));

        // Read the number of labels (32-bit integer)
        file.read(reinterpret_cast<char *>(&nLabels), sizeof(int));
        nLabels = swap_endian(nLabels); // Swap byte order if needed

        // Allocate memory to hold the label data
        labels.resize(nLabels);

        // Read the label data into the allocated memory
        file.read(reinterpret_cast<char *>(labels.data()), labels.size());

        file.close();
    }
} // namespace FileUtils

namespace ImageUtils
{
    void display_image(sf::RenderWindow &window, const std::vector<unsigned char> &images, const std::vector<unsigned char> &labels, int imageIndex = 0)
    {
        // Calculate the offset for the image based on its index
        int offset = imageIndex * IMAGE_SIZE * IMAGE_SIZE;

        // Load font
        sf::Font font;
        if (!font.loadFromFile("../../fonts/PixelifySans-VariableFont_wght.ttf"))
        {
            std::cout << "Failed to load font file" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Create text object and set properties
        sf::Text text;
        text.setFont(font);                  // Assign the font that will be alive during the window lifetime
        text.setCharacterSize(20);           // Set character size
        text.setFillColor(sf::Color::White); // Set text color
        text.setPosition(30, 30);

        // Display the label of the current image
        unsigned char label = labels[imageIndex]; // Access the label corresponding to the image

        // Create a string stream for dynamic text
        std::ostringstream stringStream;
        stringStream << "Label: " << static_cast<int>(label); // Cast label to int since it's an unsigned char
        text.setString(stringStream.str());

        sf::Image image;
        image.create(IMAGE_SIZE, IMAGE_SIZE);

        // Set the pixels from the MNIST image to the SFML image
        for (int y = 0; y < IMAGE_SIZE; y++)
        {
            for (int x = 0; x < IMAGE_SIZE; x++)
            {
                // Access the pixel from the MNIST image data
                unsigned char pixelValue = images[offset + y * IMAGE_SIZE + x];

                // Convert grayscale value (0-255) to a white-to-black color
                sf::Color color(pixelValue, pixelValue, pixelValue); // grayscale (r, g, b all the same)

                image.setPixel(x, y, color);
            }
        }

        // Create a texture and sprite to display the image
        sf::Texture texture;
        texture.loadFromImage(image);
        // texture.setSmooth(true); // Enable smooth scaling

        sf::Sprite sprite(texture);

        // Scale the 28x28 image up to 280x280 for better visibility
        sprite.setScale(20.f, 20.f); // Adjust scaling

        // Clear and draw the sprite
        window.clear();
        window.draw(sprite);
        window.draw(text);
        window.display();
    }
} // namespace ImageUtils
