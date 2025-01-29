#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rate_encoding.h"

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