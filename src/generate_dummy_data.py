import numpy as np

# Parameters
num_samples = 100
mean_class_1 = 1
mean_class_2 = -1
std_dev = 1

# Generate data for class 1
data_class_1 = np.random.normal(mean_class_1, std_dev, num_samples)
labels_class_1 = np.ones(num_samples)

# Generate data for class 2
data_class_2 = np.random.normal(mean_class_2, std_dev, num_samples)
labels_class_2 = np.zeros(num_samples)

# Combine the data and labels
data = np.concatenate((data_class_1, data_class_2)).astype(float)
labels = np.concatenate((labels_class_1, labels_class_2)).astype(int)

# Save the data and labels to text files
np.savetxt("data.txt", data)
np.savetxt("labels.txt", labels)
