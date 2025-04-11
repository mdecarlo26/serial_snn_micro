#ifndef RATE_ENCODING_H
#define RATE_ENCODING_H

void rate_encoding(float *data, int data_size, int time_window, int max_rate, unsigned char **spike_trains);
void print_spike_trains(unsigned char **spike_trains, int data_size, int time_window);
int bernoulli_trial(float p);
void rate_encoding_3d(float *data, int dim1, int dim2, int dim3, uint8_t spikes[NUM_SAMPLES][TIME_WINDOW][INPUT_BYTES]);
#endif // RATE_ENCODING_H