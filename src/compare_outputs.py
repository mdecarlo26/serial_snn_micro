import numpy as np

# Load the outputs from the SNN Torch model
snn_torch_output = np.loadtxt("final_output.txt")

# Load the outputs from your model
your_model_output = np.loadtxt("model_output.txt", dtype=int)

# Compare the outputs
def compare_outputs(snn_torch_output, your_model_output):
    num_samples, num_neurons = snn_torch_output.shape
    mismatches = 0

    for sample in range(num_samples):
        for neuron in range(num_neurons):
            snn_torch_spike = snn_torch_output[sample, neuron]
            your_model_spike = your_model_output[sample, neuron]
            if snn_torch_spike != your_model_spike:
                print(f"Mismatch at sample {sample}, neuron {neuron}: SNN Torch = {snn_torch_spike}, Your Model = {your_model_spike}")
                mismatches += 1

    total_comparisons = num_samples * num_neurons
    match_percentage = ((total_comparisons - mismatches) / total_comparisons) * 100

    print(f"\nTotal Comparisons: {total_comparisons}")
    print(f"Total Mismatches: {mismatches}")
    print(f"Match Percentage: {match_percentage:.2f}%")

# Run the comparison
compare_outputs(snn_torch_output, your_model_output)
