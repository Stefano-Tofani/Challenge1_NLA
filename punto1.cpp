#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main(int argc, char* argv[]) {

  //PUNTO 1 'Load the image as an Eigen matrix with size m × n. Each entry in the matrix corresponds
  //to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
  //Report the size of the matrix.'

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  const char* input_image_path = argv[1];

  // Load the image using stb_image
  int width, height, channels;
  // for greyscale images force to load only one channel
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path << std::endl;
    return 1;
  }

  std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
  

  // PUNTO 2 'Introduce a noise signal into the loaded image by adding random fluctuations of color
  //ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.'

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> dark_image(height, width);
  // matrice originale reshape come vettore
  for (int i=0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {

        //double noise = 50.0 * Eigen::internal::random<double>(-1.0, 1.0);
        double val = static_cast<double>(image_data[i * width + j]); //+ noise;

        // Limita il valore nell'intervallo valido [0, 255]
        //val = std::clamp(val, 0.0, 255.0);

        dark_image(i, j) = static_cast<unsigned char>(val);
       

    }
  }

  // Mappa i dati della matrice come un vettore colonna di dimensione (height * width)
Eigen::Map<const Eigen::Matrix<unsigned char, Dynamic, 1>> w_original(dark_image.data(), height * width);

// Calcola la norma convertendo in double
std::cout << "Norma del vettore senza rumore v: \n" << w_original.cast<double>().norm() << std::endl;

// Preapariamo la vera matrice dark_image con rumore
 for (int i=0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {

        double noise = 50.0 * Eigen::internal::random<double>(-1.0, 1.0);
        
        // 1. Estrapola il valore originale del pixel in double e aggiungi il rumore
        double val = static_cast<double>(dark_image(i, j)) + noise;

        // 2. Applica il clamp lavorando sui double (evita errori di tipi misti)
        val = std::clamp(val, 0.0, 255.0);

        // 3. Salva nella matrice solo dopo aver sistemato il range
        dark_image(i, j) = static_cast<unsigned char>(val);
       

    }
  }

    // Free memory!!!
  stbi_image_free(image_data);

  // Save the image using stb_image_write
  const std::string output_image_path1 = "dark_image.png";
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     dark_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
  }
  
// Mappa i dati della matrice come un vettore colonna di dimensione (height * width)
Eigen::Map<const Eigen::Matrix<unsigned char, Dynamic, 1>> w_vec(dark_image.data(), height * width);

// Calcola la norma convertendo in double
std::cout << "Norma del vettore w: \n" << w_vec.cast<double>().norm() << std::endl;

return 0;

}
