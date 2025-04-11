#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "define.h"
#include "rate_encoding.h"
#include "snn_network.h"
#include <time.h>

void rate_encoding(float *data, int data_size, int time_window, int max_rate, unsigned char **spike_trains) {
    for (int i = 0; i < data_size; i++) {
        int num_spikes = (int)(data[i] * max_rate);
        for (int j = 0; j < num_spikes; j++) {
            int spike_time = rand() % time_window;
            spike_trains[i][spike_time] = 1;
        }
    }
}

void print_spike_trains(unsigned char **spike_trains, int data_size, int time_window) {
    for (int i = 0; i < data_size; i++) {
        printf("Spike train for data[%d]: ", i);
        for (int j = 0; j < time_window; j++) {
            printf("%d", spike_trains[i][j]);
        }
        printf("\n");
    }
}

int bernoulli_trial(float p){
    return (rand() / (float)RAND_MAX) < p ? 1 : 0; // bernoulli trial with probability p
}

void rate_encoding_3d(float *data, int dim1, int dim2, int dim3, uint8_t spikes[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES]) {
    float prob =0;
    int spike = 0;
    int out = 0;
    for (int c = 0; c < dim1; c++) {
        for (int h = 0; h < dim2; h++) {
            for (int w = 0; w < dim3; w++) {
                prob = data[w] / 255.0; // Normalize the data to [0, 1]
                // spikes[c][h][w] =  bernoulli_trial(prob);
                spike = bernoulli_trial(prob); // Perform Bernoulli trial
                printf("Spike at [%d][%d][%d]: %d\n", c, h, w, spike);
                set_input_spike(spikes, c, h, w, spike); // Set the spike in the 3D array
                printf("Spike at [%d][%d][%d]: %d\n", c, h, w, get_input_spike(spikes, c, h, w)); // Print the spike value
            }
        }
    }
}