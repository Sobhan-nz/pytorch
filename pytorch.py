import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# داده‌های فرضی
inputs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
targets = torch.tensor([[1.0], [0.0]])

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished. Final loss:", loss.item())
