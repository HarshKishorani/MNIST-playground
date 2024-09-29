#include <SFML/Graphics.hpp>
#include <iostream>
#include "utils.hpp"

int main()
{
    std::vector<unsigned char> images, labels;
    int nImages = 0, nLabels = 0;

    try
    {
        // Replace with the correct file paths
        FileUtils::read_mnist_images("../../data/train-images.idx3-ubyte", images, nImages);
        FileUtils::read_mnist_labels("../../data/train-labels.idx1-ubyte", labels, nLabels);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    sf::RenderWindow window(sf::VideoMode(700, 700), "MNIST Image Display");
    sf::Time t3 = sf::seconds(1.5f);

    // Main loop to display the first image
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Display the first image (index 0) from the dataset
        for (int i = 0; i < nImages; i++)
        {
            ImageUtils::display_image(window, images, labels, i);
            sf::sleep(t3);
        }
    } 

    return 0;
}