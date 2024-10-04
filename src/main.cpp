#include "network.hpp"

#define TRAIN_IMG_PATH "../../data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "../../data/train-labels.idx1-ubyte"
#define MODEL_PATH "../../trained_network.bin"

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
    // net.save_network(MODEL_PATH);

    net.load_network(MODEL_PATH);
    // Normalize the input image.
    std::vector<float> img(INPUT_SIZE);
    for (int k = 0; k < INPUT_SIZE; k++)
    {
        img[k] = inputData.images[9 * INPUT_SIZE + k] / 255.0f;
    }
    inputData.display_image_from_data(window, 9, net.predict(img));
}

std::vector<float> normalizeImage(const std::vector<unsigned char> &images, int imageIndex)
{
    std::vector<float> float_image(IMAGE_SIZE * IMAGE_SIZE);

    // Calculate the offset for the image in the dataset
    int offset = imageIndex * IMAGE_SIZE * IMAGE_SIZE;

    for (int y = 0; y < IMAGE_SIZE; y++)
    {
        for (int x = 0; x < IMAGE_SIZE; x++)
        {
            // Normalize the pixel value from [0, 255] to [0.0, 1.0]
            float_image[y * IMAGE_SIZE + x] = images[offset + y * IMAGE_SIZE + x] / 255.0f;
        }
    }

    return float_image;
}

int brushRadius = 1;
const int canvasSize = 560; // 560x560 pixels canvas
const int pixelSize = 20;   // Each "pixel" is 20x20 pixels, representing a single MNIST pixel

void clearImage(std::vector<sf::RectangleShape> &pixels)
{
    for (auto &pixel : pixels)
    {
        pixel.setFillColor(sf::Color::Black); // Reset pixels to black
    }
}

// This function simulates drawing with a brush
void applyBrush(int xIndex, int yIndex, std::vector<sf::RectangleShape> &pixels)
{
    for (int dy = -brushRadius; dy <= brushRadius; dy++)
    {
        for (int dx = -brushRadius; dx <= brushRadius; dx++)
        {
            int newX = xIndex + dx;
            int newY = yIndex + dy;

            // Make sure we're within bounds
            if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28)
            {
                int index = newY * 28 + newX;
                pixels[index].setFillColor(sf::Color::White);
            }
        }
    }
}

int main()
{
    InputData inputData;
    inputData.readData(TRAIN_IMG_PATH, TRAIN_LBL_PATH);

    Network net;
    net.load_network(MODEL_PATH);

    sf::RenderWindow window(sf::VideoMode(canvasSize, canvasSize), "Canvas Window");
    sf::RenderWindow renderWindow(sf::VideoMode(canvasSize, canvasSize), "Image Window");

    // Display the first dataset image using your display_image function
    inputData.display_image(renderWindow, normalizeImage(inputData.images, 0));

    // 28x28 grid to simulate the MNIST image
    std::vector<sf::RectangleShape> pixels(28 * 28);

    // Initialize the pixel grid
    for (int y = 0; y < 28; y++)
    {
        for (int x = 0; x < 28; x++)
        {
            sf::RectangleShape pixel(sf::Vector2f(pixelSize - 1, pixelSize - 1)); // Create each pixel
            pixel.setPosition(x * pixelSize, y * pixelSize);                      // Position it in the grid
            pixel.setFillColor(sf::Color::Black);                                 // Set the initial color to black
            pixels[y * 28 + x] = pixel;
        }
    }

    bool isDrawing = false; // Track if we are drawing

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }

            // Start drawing when the left mouse button is pressed
            if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left)
            {
                isDrawing = true;
            }

            // Stop drawing when the left mouse button is released
            if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left)
            {
                isDrawing = false;
            }

            if (event.type == sf::Event::KeyPressed && event.key.scancode == sf::Keyboard::Scan::Space)
            {
                clearImage(pixels);
            }

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Enter)
            {
                std::vector<float> img(INPUT_SIZE);

                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        sf::RectangleShape pixel = pixels[y * 28 + x];

                        // Get the pixel's fill color
                        sf::Color color = pixel.getFillColor();

                        // Convert the color to grayscale
                        float grayscaleValue = color.r / 255.0f; // Normalize to [0, 1]

                        // Store it in the input vector
                        img[y * 28 + x] = grayscaleValue;
                    }
                }

                // Display the drawn image
                inputData.display_image(renderWindow, img);

                // Give the network the image to predict
                std::cout << "=> Prediction: " << net.predict(img) << std::endl;
                clearImage(pixels);
            }
        }

        if (isDrawing)
        {
            // Get mouse position within the window
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            int xIndex = mousePos.x / pixelSize;
            int yIndex = mousePos.y / pixelSize;

            if (xIndex >= 0 && xIndex < 28 && yIndex >= 0 && yIndex < 28)
            {
                applyBrush(xIndex, yIndex, pixels); // Apply brush to simulate smoother strokes
            }
        }

        // Render the pixels
        window.clear();
        for (auto &pixel : pixels)
        {
            window.draw(pixel);
        }
        window.display();
    }
    return 0;
}