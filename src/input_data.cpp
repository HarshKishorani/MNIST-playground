#include "input_data.hpp"

int InputData::swap_endian(int value)
{
    unsigned char *bytes = reinterpret_cast<unsigned char *>(&value);
    std::reverse(bytes, bytes + sizeof(int));
    return value;
}

void InputData::read_mnist_labels(const std::string trainLabelsPath)
{
    std::ifstream file(trainLabelsPath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Error opening label file: " + std::string(trainLabelsPath));
    }

    int temp;
    file.read(reinterpret_cast<char *>(&temp), sizeof(int));
    file.read(reinterpret_cast<char *>(&nLabels), sizeof(int));
    nLabels = swap_endian(nLabels);

    labels.resize(nLabels);
    file.read(reinterpret_cast<char *>(labels.data()), labels.size());
    file.close();
}

void InputData::read_mnist_images(const std::string trainImagesPath)
{
    std::ifstream file(trainImagesPath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Error opening image file: " + std::string(trainImagesPath));
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

InputData::InputData()
{
}

void InputData::readData(const std::string trainImagesPath, const std::string trainLabelsPath)
{
    try
    {
        this->read_mnist_images(trainImagesPath);
        this->read_mnist_labels(trainLabelsPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "=> Read number of images : " << nImages << std::endl;
    std::cout << "=> Read number of labels : " << nLabels << std::endl;
}

void InputData::display_image_from_data(sf::RenderWindow &window, int imageIndex, int predictedIndex)
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
    stringStream << "Label: " << static_cast<int>(label) << "\n";
    if (predictedIndex != -1)
        stringStream << "Prediction: " << predictedIndex << "\n";

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

void InputData::display_image(sf::RenderWindow &window, std::vector<float> img)
{
    sf::Image image;
    image.create(IMAGE_SIZE, IMAGE_SIZE);
    for (int y = 0; y < IMAGE_SIZE; y++)
    {
        for (int x = 0; x < IMAGE_SIZE; x++)
        {
            // Convert the float value (0.0 - 1.0) back to grayscale (0 - 255)
            unsigned char pixelValue = static_cast<unsigned char>(img[y * IMAGE_SIZE + x] * 255);
            sf::Color color(pixelValue, pixelValue, pixelValue); // Grayscale color
            image.setPixel(x, y, color);
        }
    }

    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
    sprite.setScale(20.f, 20.f); // Scale the image for better visibility

    window.clear();
    window.draw(sprite);
    window.display();
}
