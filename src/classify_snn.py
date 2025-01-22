import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from torch.utils.data import DataLoader, TensorDataset

# Load Data
data = torch.tensor([[float(line.strip())] for line in open("data.txt")])
labels = torch.tensor([int(float(line.strip())) for line in open("labels.txt")])  # Ensure labels are 0 or 1

# Prepare Dataset and DataLoader
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define SNN with 2 LIF layers, each with 10 neurons
class TwoLayerSNN(nn.Module):
    def __init__(self):
        super(TwoLayerSNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # First layer: input -> 10 neurons
        self.lif1 = snn.Leaky(beta=0.8)
        self.fc2 = nn.Linear(10, 2)  # Second layer: 10 neurons -> 2 output neurons (for 2 classes)
        self.lif2 = snn.Leaky(beta=0.8)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        
        for step in range(20):  # Simulating 20 time steps
            cur = self.fc1(x)
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        # Return summed spikes over time (this is the final classification logits)
        summed_spikes = torch.stack(spk2_rec, dim=0).sum(0)  # Shape: [batch_size, num_classes]
        
        return summed_spikes

# Instantiate the model
model = TwoLayerSNN()

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Convert to spike trains
        batch_data = spikegen.rate(batch_data, num_steps=20)  # Convert data to spike train
        
        optimizer.zero_grad()
        logits = model(batch_data)  # Get the final summed spikes over time
        
        # Compute the loss (CrossEntropyLoss expects logits with shape [batch_size, num_classes])
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Test the Model
with torch.no_grad():
    test_data = spikegen.rate(data, num_steps=20)
    logits = model(test_data)  # Get the final summed spikes over time
    
    # Get predictions from the logits
    predictions = logits.argmax(1)  # Get class predictions based on maximum spikes
    accuracy = (predictions == labels).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
