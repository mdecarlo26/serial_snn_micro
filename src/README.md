# Spiking Neural Network Simulation

This project simulates a spiking neural network (SNN) using a simple model with multiple layers of neurons. Each neuron has a membrane potential, a voltage threshold, and a decay rate. The simulation processes input spikes through the network layers and updates the neurons' states based on the input and their weights.

## Files

- `main.c`: The main source file containing the implementation of the SNN simulation.
- `Makefile`: The makefile for building and running the project.

## Compilation and Execution

To compile and run the project, use the provided Makefile. The following targets are available:

- `all`: Compile the project.
- `clean`: Remove the build directory and all compiled files.
- `redo`: Clean, compile, and run the project.
- `run`: Run the compiled executable.

### Example Commands

```sh
# Compile the project
make all

# Run the compiled executable
make run

# Clean the build directory
make clean

# Clean, compile, and run the project
make redo
```

## High-Level Approach

The high-level approach of the simulation in `main.c` involves the following steps:

1. **Network Initialization**: The network is initialized with a specified number of neurons per layer. Each neuron is assigned initial values for membrane potential, voltage threshold, and decay rate.

2. **Input Initialization**: Input spikes for the first layer are initialized with random values.

3. **Layer Processing**: Each layer of the network is processed for a specified number of time steps (`TAU`). During each time step, the neurons' states are updated based on the input spikes and their weights. The membrane potential of each neuron is decayed, and neurons fire if their membrane potential exceeds the voltage threshold.

4. **Output**: The output of the last layer is printed, showing which neurons fired.

5. **Memory Cleanup**: The dynamically allocated memory for the network is freed.

## Algorithm Approach

The algorithm approach of taking an event-driven SNN and bringing it to a serial CPU involves the following steps:

1. **Initialization**: The network is initialized with the specified number of neurons per layer. Each neuron is assigned initial values for membrane potential, voltage threshold, and decay rate. The weights connecting neurons are also initialized.

2. **Input Spikes Initialization**: Input spikes for the first layer are initialized with random values. This simulates the initial input to the network.

3. **Processing Layers**: Each layer of the network is processed sequentially. For each layer:
   - The input spikes are processed for a specified number of time steps (`TAU`).
   - During each time step, the neurons' states are updated based on the input spikes and their weights.
   - The membrane potential of each neuron is decayed, and neurons fire if their membrane potential exceeds the voltage threshold.
   - The output spikes of the current layer become the input spikes for the next layer.

4. **Output**: After processing all layers, the output of the last layer is printed, showing which neurons fired.

5. **Memory Cleanup**: The dynamically allocated memory for the network is freed to avoid memory leaks.

## Code Overview

### Data Structures

- `Neuron`: Represents a single neuron with the following properties:
  - `membrane_potential`: The current membrane potential of the neuron.
  - `voltage_thresh`: The voltage threshold for the neuron to fire.
  - `decay_rate`: The rate at which the membrane potential decays over time.

- `Layer`: Represents a layer of neurons with the following properties:
  - `neurons`: An array of `Neuron` structs.
  - `weights`: A 2D array of weights connecting neurons in this layer to neurons in the previous layer.
  - `num_neurons`: The number of neurons in this layer.

- `Network`: Represents the entire neural network with the following properties:
  - `layers`: An array of `Layer` structs.
  - `num_layers`: The number of layers in the network.

### Functions

- `initialize_network(int neurons_per_layer[])`: Initializes the network with the specified number of neurons per layer, setting initial values for membrane potentials, voltage thresholds, and decay rates.

- `free_network()`: Frees the dynamically allocated memory for the network.

- `set_bit(unsigned char buffer[], int index, int value)`: Sets a bit in the buffer at the specified index to the given value.

- `get_bit(const unsigned char buffer[], int index)`: Gets the value of a bit from the buffer at the specified index.

- `simulate_layer(const unsigned char input[], unsigned char output[], float weights[][MAX_NEURONS], int num_neurons, int input_size)`: Simulates neuron firing in a layer based on the input and weights.

- `update_layer(const unsigned char input[], unsigned char output[], Layer *layer, int input_size)`: Updates the entire layer based on the input buffer, applying decay and checking for neuron firing.

- `initialize_input_spikes(unsigned char input[], int num_neurons)`: Initializes input spikes for the first layer with random values.

### Main Function

The `main` function initializes the network, sets up the input spikes, processes each layer for a specified number of time steps (`TAU`), and prints the output of the last layer.

## License

This project is licensed under the MIT License.