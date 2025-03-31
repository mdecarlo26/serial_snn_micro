#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

#include "define.h"

void load_weights(const char *filename, float **weights, int rows, int cols);
void load_bias(const char *filename, float *bias, int size);
void load_data(const char *filename, float *data, int num_samples);
void save_output(const char *filename, unsigned char *output, int num_neurons);
void load_csv(const char *filename, float **array, int rows, int cols);
int read_spike_data(const char* filename, char ***spikes);
int read_labels(const char* filename, char *labels, int num_samples);

#endif // FILE_OPERATIONS_H
