#pragma once
#include <vector>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>

#define IMAGE_SIZE 28

class InputData
{
private:
    int swap_endian(int value);
    void read_mnist_labels(const std::string trainLabelsPath);
    void read_mnist_images(const std::string trainImagesPath);

public:
    std::vector<unsigned char> images;
    std::vector<unsigned char> labels;
    int nImages, nLabels;

    InputData();

    void readData(const std::string trainImagesPath, const std::string trainLabelsPath);

    void display_image(sf::RenderWindow &window, int imageIndex, int predictedIndex = -1);
    void display_image(sf::RenderWindow &window, std::vector<unsigned char> image);
};
