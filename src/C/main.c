#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "define.h"
#include "file_operations.h"
#include "rate_encoding.h"
#include "snn_network.h"
// #include "debug.h"
#include "dummy.h"

Snn_Network snn_network;

char labels[NUM_SAMPLES] = {0};

// Function prototypes
void dump_classification(FILE *output_file, int sample_index, int classification, char* labels);

int validate_spike_data(char ***spikes);

int main() {
    srand((unsigned int)time(NULL));

    snn_network.num_layers = NUM_LAYERS;
    int neurons_per_layer[] = {INPUT_SIZE, HIDDEN_LAYER_1, NUM_CLASSES};

    initialize_network(neurons_per_layer, weights_fc1_data, weights_fc2_data, bias_fc1, bias_fc2);
    zero_network();
    printf("Network initialized\n");

    static uint8_t initial_spikes[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES] = {0};

    labels[0] = label;

    printf("Making Spikes\n");
    rate_encoding_3d(input_data, NUM_SAMPLES, TIME_WINDOW, INPUT_SIZE, initial_spikes);
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

    time_t start, end;

    printf("\033[1;32mStarting Sim\033[0m\n");
    for (int d = 0; d < NUM_SAMPLES; d++) {
        printf("\r\033[KSample: \033[1;37m%d\033[0m/%d", d+1, NUM_SAMPLES);
        fflush(stdout);

        start = time(NULL);

        int classification = inference(initial_spikes, d);

        end = time(NULL);

        dump_classification(output_file, d, classification, labels);
    }
    printf("\n");
    printf("\033[1;32mSim Finished\033[0m\n");
    printf("Time taken: %ld seconds\n", (double)(end - start));
    fclose(output_file);

    return 0;
}



void dump_classification(FILE *output_file, int sample_index, int classification, char* labels) {
    fprintf(output_file, "Sample %d: Classification = %d, Label = %d\n", sample_index, classification, labels[sample_index]);
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
