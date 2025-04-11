#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
// #include <sys/time.h>

#include "define.h"
#include "file_operations.h"
#include "rate_encoding.h"
#include "snn_network.h"
// #include "debug.h"
#include "dummy.h"

Snn_Network snn_network;

static uint8_t ping_pong_buffer_1[MAX_NEURONS][BITMASK_BYTES] = {0};
static uint8_t ping_pong_buffer_2[MAX_NEURONS][BITMASK_BYTES] = {0};

// 3. Fully static memory for ping-pong buffers

// 6. Fully static memory for labels
char labels[NUM_SAMPLES] = {0};

// Function prototypes
void initialize_input_spikes(char **input, int num_neurons);
void dump_classification(FILE *output_file, int sample_index, int classification, char* labels);

// Memory allocation functions
char*** allocate_spike_array();
void free_spike_array(char*** spikes);
int validate_spike_data(char ***spikes);
char *allocate_labels(int num_samples);
void free_labels(char *labels);

int main() {
    srand((unsigned int)time(NULL));

    // Example initialization
    snn_network.num_layers = NUM_LAYERS;
    int neurons_per_layer[] = {INPUT_SIZE, HIDDEN_LAYER_1, NUM_CLASSES};
    // for (int i = 0; i < MAX_NEURONS; i++) {
    //     ping_pong_buffer_1_data[i] = ping_pong_buffer_1_blocks[i];
    //     ping_pong_buffer_2_data[i] = ping_pong_buffer_2_blocks[i];
    // }

    // float weights_fc1_data[HIDDEN_LAYER_1][INPUT_SIZE] = {0};
    // float weights_fc2_data[NUM_CLASSES][HIDDEN_LAYER_1] = {0};
    float* weights_fc1[HIDDEN_LAYER_1];
    float* weights_fc2[NUM_CLASSES];

    for (int i = 0; i < HIDDEN_LAYER_1; i++) {
        weights_fc1[i] = weights_fc1_data[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        weights_fc2[i] = weights_fc2_data[i];
    }

    initialize_network(neurons_per_layer, weights_fc1, weights_fc2, bias_fc1, bias_fc2);
    zero_network();
    printf("Network initialized\n");

    // Print model overview
    // print_model_overview();


    // 4. Fully static memory for spike array (3D)
    static uint8_t initial_spikes[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES] = {0};

    labels[0] = label;

    
    printf("Making Spikes\n");
    rate_encoding_3d(input_data, NUM_SAMPLES, TIME_WINDOW, INPUT_BYTES, initial_spikes);
    printf("\033[1;32mSpikes Made\033[0m\n");

    // Read data into allocated arrays
    // if (read_spike_data("../mnist_input_spikes.csv", initial_spikes) || read_labels("../mnist_labels.csv", labels, NUM_SAMPLES)) {
    // // if (read_spike_data("../input_spikes.csv", initial_spikes) || read_labels("../labels.csv", labels, NUM_SAMPLES)) {
    //     free_spike_array(initial_spikes);
    //     free(labels);
    //     return 1;
    // }

    // Load initial spikes from CSV file
    // float **initial_spikes = (float **)malloc(NUM_SAMPLES * sizeof(float *));
    // for (int i = 0; i < NUM_SAMPLES; i++) {
    //     initial_spikes[i] = (float *)malloc(TIME_WINDOW * sizeof(float));
    // }
    // load_csv("spikes.csv", initial_spikes, NUM_SAMPLES, TIME_WINDOW);
    // if (!validate_spike_data(initial_spikes)) {
    //     printf("Invalid spike data\n");
    //     free_spike_array(initial_spikes);
    //     free(labels);
    //     return 1;
    // }
    // printf("%d\n", initial_spikes[0][3][0]);

    // Process each data point
    FILE *output_file = fopen("model_output.txt", "w");
    if (output_file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }

    // struct timeval start, stop;
    // gettimeofday(&start, NULL);

    printf("\033[1;32mStarting Sim\033[0m\n");
    for (int d = 0; d < NUM_SAMPLES; d++) {
        printf("\r\033[KSample: \033[1;37m%d\033[0m/%d", d+1, NUM_SAMPLES);
        fflush(stdout);

        int classification = inference(initial_spikes, ping_pong_buffer_1, ping_pong_buffer_2, d);
        dump_classification(output_file, d, classification, labels);
    }
    printf("\n");
    printf("\033[1;32mSim Finished\033[0m\n");
    fclose(output_file);

    // gettimeofday(&stop, NULL);

    // printf("Simulation took %lu ms\n", (stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec) / 1000);
    // printf("Average time per sample: %ld ms\n", ((stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec) / 1000) / NUM_SAMPLES);

    return 0;
}


// Function to free the allocated memory
void free_network() {
    for (int l = 0; l < snn_network.num_layers; l++) {
        for (int i = 0; i < snn_network.layers[l].num_neurons; i++) {
            if (snn_network.layers[l].weights[i] != NULL) {
                free(snn_network.layers[l].weights[i]);
            }
        }
        free(snn_network.layers[l].weights);
        free(snn_network.layers[l].bias);
        free(snn_network.layers[l].neurons);
    }
    free(snn_network.layers);
}




void dump_classification(FILE *output_file, int sample_index, int classification, char* labels) {
    fprintf(output_file, "Sample %d: Classification = %d, Label = %d\n", sample_index, classification, labels[sample_index]);
}

// Function to print the weight matrix and biases

char*** allocate_spike_array() {
    char *data_block = malloc(NUM_SAMPLES * TIME_WINDOW * INPUT_SIZE * sizeof(char));
    if (!data_block) {
        perror("Memory allocation error for data block");
        return NULL;
    }
    
    // Allocate the array of sample pointers.
    char ***spikes = malloc(NUM_SAMPLES * sizeof(char **));
    if (!spikes) {
        perror("Memory allocation error for spike array");
        free(data_block);
        return NULL;
    }
    
    // For each sample, allocate an array of pointers (one per time step).
    for (int i = 0; i < NUM_SAMPLES; i++) {
        spikes[i] = malloc(TIME_WINDOW * sizeof(char *));
        if (!spikes[i]) {
            perror("Memory allocation error for spike row pointers");
            for (int j = 0; j < i; j++) {
                free(spikes[j]);
            }
            free(spikes);
            free(data_block);
            return NULL;
        }
        // Set each time pointer to the correct offset in the contiguous block.
        for (int j = 0; j < TIME_WINDOW; j++) {
            spikes[i][j] = data_block + (i * TIME_WINDOW * INPUT_SIZE) + (j * INPUT_SIZE);
        }
    }
    return spikes;
}

void free_spike_array(char*** spikes) {
    if (spikes) {
        free(spikes[0][0]);  // Free the contiguous data block.
        for (int i = 0; i < NUM_SAMPLES; i++) {
            free(spikes[i]);  // Free each sample's row pointer array.
        }
        free(spikes);  // Free the array of sample pointers.
    }
}

int validate_spike_data(char ***spikes) {
    if (!spikes || !spikes[0] || !spikes[0][0]) {
        fprintf(stderr, "Invalid spike pointer structure\n");
        return 0;
    }

    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < TIME_WINDOW; j++) {
            for (int k = 0; k < INPUT_SIZE; k++) {
                char value = spikes[i][j][k];
                if (value != 0 && value != 1) {
                    fprintf(stderr, "Invalid value at [%d][%d][%d]: %d\n", i, j, k, value);
                    return 0;
                }
            }
        }
    }
    printf("All spike data loaded correctly.\n");
    return 1;
}

char *allocate_labels(int num_samples) {
    char *labels = malloc(num_samples * sizeof(char));
    if (!labels) {
        perror("Failed to allocate memory for labels");
        return NULL;
    }
    return labels;
}

void free_labels(char *labels) {
    free(labels);
}
