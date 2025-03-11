#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "file_operations.h"


int read_spike_data(const char* filename, char ***spikes) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening CSV file for reading");
        return 1;
    }
    
    char line[2048];
    int row = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strlen(line) <= 1)
            continue;  // skip empty lines

        if (row >= NUM_SAMPLES * TIME_WINDOW) {
            fprintf(stderr, "Error: More rows in CSV than expected.\n");
            fclose(fp);
            return 1;
        }
        
        // Determine which sample and time step this row belongs to.
        int sample = row / TIME_WINDOW;
        int time_idx = row % TIME_WINDOW;
        
        int col = 0;
        char *token = strtok(line, ",");
        while (token && col < INPUT_SIZE) {
            int value = atoi(token);
            spikes[sample][time_idx][col] = (char)value;
            token = strtok(NULL, ",");
            col++;
        }
        
        if (col != INPUT_SIZE) {
            fprintf(stderr, "Error: Row %d expected %d values, got %d.\n", row, INPUT_SIZE, col);
            fclose(fp);
            return 1;
        }
        row++;
    }
    
    if (row != NUM_SAMPLES * TIME_WINDOW) {
        fprintf(stderr, "Error: Total rows read (%d) does not match expected (%d).\n",
                row, NUM_SAMPLES * TIME_WINDOW);
        fclose(fp);
        return 1;
    }
    
    fclose(fp);
    return 0;
}

// Function to read label data from binary file
int read_labels(const char* filename, char *labels, int num_samples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening labels CSV file");
        return 1;
    }
    
    char line[1024];
    if (!fgets(line, sizeof(line), fp)) {
        perror("Error reading from labels CSV file");
        fclose(fp);
        return 1;
    }
    fclose(fp);
    
    // Remove newline if present.
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') {
        line[len - 1] = '\0';
    }
    
    int count = 0;
    char *token = strtok(line, ",");
    while (token != NULL && count < num_samples) {
        int value = atoi(token);
        labels[count] = (char)value;
        count++;
        token = strtok(NULL, ",");
    }
    
    if (count != num_samples) {
        fprintf(stderr, "Error: Expected %d labels, but found %d\n", num_samples, count);
        return 1;
    }
    
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
