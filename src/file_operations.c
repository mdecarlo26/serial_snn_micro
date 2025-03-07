#include <stdio.h>
#include <stdlib.h>
#include "file_operations.h"


int read_spike_data(const char* filename, char ***spikes) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open spike_data.bin");
        return 1;
    }

    // Read spike data into pre-allocated 3D array
    size_t items_read = fread(spikes[0][0], sizeof(char), NUM_SAMPLES * TIME_WINDOW * INPUT_SIZE, file);
    if (items_read != NUM_SAMPLES * TIME_WINDOW * INPUT_SIZE) {
        fprintf(stderr, "Warning: Expected %d spike values, but read %zu\n", NUM_SAMPLES * TIME_WINDOW * INPUT_SIZE, items_read);
    }

    fclose(file);
    return 0;
}

// Function to read label data from binary file
int read_labels(const char* filename, char *labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open labels.bin");
        return 1;
    }

    // Read label data into pre-allocated array
    size_t items_read = fread(labels, sizeof(char), NUM_SAMPLES, file);
    if (items_read != NUM_SAMPLES) {
        fprintf(stderr, "Warning: Expected %d labels, but read %zu\n", NUM_SAMPLES, items_read);
    }

    fclose(file);
    return 0;
}

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

// Function to load biases from a file
void load_bias(const char *filename, float *bias, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open bias file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &bias[i]);
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

// Function to read a CSV file and dump it into a 2D array
void load_csv(const char *filename, float **array, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open CSV file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%f,", &array[i][j]) != 1) {
                perror("Failed to read value from CSV file");
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}
