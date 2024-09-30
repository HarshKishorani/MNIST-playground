#include "network.hpp"

#define TRAIN_IMG_PATH "../../data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../../data/train-labels.idx1-ubyte"
#define MODEL_SAVE_PATH "../../trained_network.bin"

#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define TRAIN_SPLIT 0.8

void saveAndLoadNetworkExample(sf::RenderWindow &window)
{
    Network net;

    InputData inputData;
    inputData.readData(TRAIN_IMG_PATH, TRAIN_LBL_PATH);

    // net.trainNetwork(inputData, LEARNING_RATE, TRAIN_SPLIT, 2, BATCH_SIZE);
    // net.save_network(MODEL_SAVE_PATH);

    net.load_network(MODEL_SAVE_PATH);
    // Normalize the input image.
    std::vector<float> img(INPUT_SIZE);
    for (int k = 0; k < INPUT_SIZE; k++)
    {
        img[k] = inputData.images[9 * INPUT_SIZE + k] / 255.0f;
    }
    inputData.display_image(window, 9, net.predict(img));
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(700, 700), "MNIST Image Display");
    saveAndLoadNetworkExample(window);
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
    }
    return 0;
}