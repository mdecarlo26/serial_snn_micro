import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikegen as spikegen
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Define the network architecture
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2)
        return spk2, mem2

# Load the dummy data
data = np.loadtxt("dummy_data.txt")
labels = np.loadtxt("dummy_labels.txt")

# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the network
num_inputs = 1
num_hidden = 10
num_outputs = 1
net = SNN(num_inputs, num_hidden, num_outputs)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        spk_out, mem_out = net(batch_data)
        loss = criterion(spk_out, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the final weights and output to text files
np.savetxt("weights_fc1.txt", net.fc1.weight.detach().numpy())
np.savetxt("weights_fc2.txt", net.fc2.weight.detach().numpy())

# Generate final output for the entire dataset
with torch.no_grad():
    spk_out, _ = net(data)
np.savetxt("final_output.txt", spk_out.detach().numpy())
