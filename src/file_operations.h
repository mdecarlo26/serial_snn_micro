#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

void load_weights(const char *filename, float **weights, int rows, int cols);
void load_bias(const char *filename, float *bias, int size);
void load_data(const char *filename, float *data, int num_samples);
void save_output(const char *filename, unsigned char *output, int num_neurons);

#endif // FILE_OPERATIONS_H
