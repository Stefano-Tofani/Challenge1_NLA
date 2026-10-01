#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <vector>

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

//Punto4
// 1. Definisci il numero totale di pixel
int N = height * width;

// 2. Inizializza la matrice sparsa di tipo double
Eigen::SparseMatrix<double> A1(N, N);

// 3. Usa un vettore di Triplet per riempirla velocemente
std::vector<Eigen::Triplet<double>> triplets;
// Pre-allochiamo la memoria: al massimo 9 elementi non nulli per ogni riga
triplets.reserve(N * 9);

// 4. Doppio ciclo sulle coordinate 2D dell'immagine
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        
        // Indice 1D della riga della matrice A1
        int row_idx = y * width + x;
        
        // Cicliamo sul kernel 3x3 (offset da -1 a +1 per y e x)
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                
                int neighbor_y = y + ky;
                int neighbor_x = x + kx;
                
                // CONTROLLO DEI BORDI: verifichiamo che il vicino esista
                if (neighbor_y >= 0 && neighbor_y < height && 
                    neighbor_x >= 0 && neighbor_x < width) {
                    
                    // Indice 1D della colonna (il pixel da cui leggiamo il colore)
                    int col_idx = neighbor_y * width + neighbor_x;
                    
                    // Assegniamo il peso corretto
                    double weight = 0.0;
                    if (ky == 0 && kx == 0) {
                        weight = 4.0 / 12.0; // Centro
                    } else {
                        weight = 1.0 / 12.0; // Vicini
                    }
                    
                    // Salviamo il valore
                    triplets.push_back(Eigen::Triplet<double>(row_idx, col_idx, weight));
                }
            }
        }
    }
}

// 5. Costruisci la matrice sparsa dai Triplet
A1.setFromTriplets(triplets.begin(), triplets.end());


// Calcola la norma convertendo in double
std::cout << "Norma del vettore senza rumore v: \n" << w_original.cast<double>().norm() << std::endl;

// Adesso puoi stampare il numero di elementi non nulli richiesto dalla consegna
std::cout << "Elementi non nulli in A1: " << A1.nonZeros() << std::endl;



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

//PUNTO5 Apply the previous smoothing filter to the noisy image by performing the matrix vector
//multiplication A1w. Export and upload the resulting image.

Eigen::VectorXd w_double = w_vec.cast<double>();

// 3. Esegui la moltiplicazione matrice-vettore: A1 * w
Eigen::VectorXd g_double = A1 * w_double;

// 4. Converti il risultato in unsigned char per salvarlo come immagine
Eigen::Matrix<unsigned char, Dynamic, 1> g_char = g_double.cast<unsigned char>();

// 
//Eigen::Map<const Eigen::Matrix<unsigned char, Dynamic, 1>> g_vec(g_char.data(), height * width);
// 5. Salva il risultato come immagine
const std::string output_image_path2 = "dark_image_first_filter.png";
  if (stbi_write_png(output_image_path2.c_str(), width, height, 1,
                     g_char.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
  }

// Calcola la norma convertendo in double
std::cout << "Norma del vettore w: \n" << w_vec.cast<double>().norm() << std::endl;

return 0;

}
