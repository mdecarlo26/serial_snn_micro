#ifndef RATE_ENCODING_H
#define RATE_ENCODING_H

void rate_encoding(float *data, int data_size, int time_window, int max_rate, unsigned char **spike_trains);
void print_spike_trains(unsigned char **spike_trains, int data_size, int time_window);

#endif // RATE_ENCODING_H