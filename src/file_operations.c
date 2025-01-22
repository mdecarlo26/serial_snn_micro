#include <stdio.h>
#include <stdlib.h>
#include "file_operations.h"

// Function to load weights from a file
void load_weights(const char *filename, float **weights, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open weight file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%f", &weights[i][j]);
        }
    }
    fclose(file);
}

// Function to load data from a file
void load_data(const char *filename, float *data, int num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open data file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_samples; i++) {
        fscanf(file, "%f", &data[i]);
    }
    fclose(file);
}

// Function to save the output to a file
void save_output(const char *filename, unsigned char *output, int num_neurons) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_neurons; i++) {
        fprintf(file, "%d ", output[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}
