#ifndef DEBUG_H
#define DEBUG_H

#include "snn_network.h"

// Debug Print Functions
void print_weights(float **weights, float *bias, int rows, int cols);
void print_model_overview();
void print_neuron_states(Layer *layer);
void print_spike_buffer(const char **buffer, int size);
void print_ping_pong_buffers(const char **buffer1, const char **buffer2, int size);
void print_firing_counts(int **firing_counts, int num_neurons, int num_chunks);
#endif // DEBUG_H
