import numpy as np

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from snntorch import functional as SF
from torch.utils.data import DataLoader, TensorDataset

# Load Data
base = 'c:\\Drexel\\Research\\Kand\\snn_micro\\src'
data = torch.tensor([[float(line.strip())] for line in open(base + "\\data.txt")])
labels = torch.tensor([int(float(line.strip())) for line in open(base + "\\labels.txt")])  # Make sure labels are 0 or 1
b_size = 1
time_steps = 10

# Prepare Dataset and DataLoader
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)

# Define SNN with 2 LIF layers, each with 10 neurons
class TwoLayerSNN(nn.Module):
    def __init__(self):
        super(TwoLayerSNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # First layer: input -> 10 neurons
        self.lif1 = snn.Leaky(beta=0.8)
        self.fc2 = nn.Linear(10, 2)  # Second layer: 10 neurons -> 2 output neurons (for 2 classes)
        self.lif2 = snn.Leaky(beta=0.8, output=True)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        for step in range(time_steps):  # Simulating 20 time steps
            cur = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        # Return summed spikes over time as logits (raw values for classification)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Instantiate the model
model = TwoLayerSNN()

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2,betas=(0.9, 0.999))

# Training Loop
initial_spikes = []
num_epochs = 20
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Convert to spike trains
        batch_data = spikegen.rate(batch_data, num_steps=time_steps)  # Convert data to spike train
        # print(batch_data.shape)
        # print(batch_data.detach().numpy().reshape(-1))
        if epoch == 0:
            temp = batch_data.detach().clone()
            initial_spikes.append(temp.detach().numpy().reshape(-1))
        
        optimizer.zero_grad()
        spk_rec, _ = model(batch_data)
        
        # Use summed spikes over time as the input to the loss function
        logits = spk_rec  # This is the output of the model
        
        # Compute the loss (CrossEntropyLoss expects logits as raw output)
        loss = loss_fn(logits.sum(0), batch_labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Test the Model
with torch.no_grad():
    test_data = spikegen.rate(data, num_steps=time_steps)  # Convert test data to spike train
    spk_rec, _ = model(test_data)
    
    # Use summed spikes over time to determine predictions
    logits = spk_rec
    predictions = logits.sum(0).argmax(1)  # Get class predictions based on maximum spikes
    accuracy = (predictions == labels).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

print(len(initial_spikes))
print(initial_spikes[0].shape)
stacked_spikes = np.stack(initial_spikes, axis=-1)
np.savetxt("spikes.csv", stacked_spikes, delimiter=",")
np.savetxt("weights_fc1.txt", model.fc1.weight.detach().numpy())
np.savetxt("weights_fc2.txt", model.fc2.weight.detach().numpy())
np.savetxt("bias_fc1.txt", model.fc1.bias.detach().numpy())
np.savetxt("bias_fc2.txt", model.fc2.bias.detach().numpy())
