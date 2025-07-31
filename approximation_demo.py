# approximation_demo.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.special import erf        # <-- import scipy's erf
import matplotlib.pyplot as plt


# 1. Prepare the dataset: dense grid in [-5,5]
x = np.linspace(-5, 5, 20000, dtype=np.float32).reshape(-1, 1)
# Use scipy.special.erf here
y = 0.5 * x * (1 + erf(x / np.sqrt(2)))  # GELU ground truth

# Convert to tensors and create DataLoader
tensor_x = torch.from_numpy(x)
tensor_y = torch.from_numpy(y)
dataset = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# 2. Define the modified rational activation
class RationalActivation(nn.Module):
    def __init__(self, m=5, n=4):
        super().__init__()
        # Numerator coefficients a0...am
        self.a = nn.Parameter(torch.randn(m+1) * 0.1)
        # Denominator coefficients b1...bn (b0 is implicitly 1)
        self.b = nn.Parameter(torch.randn(n) * 0.1)

    def forward(self, x):
        # x: tensor of shape [B, 1]
        # Compute P(x) = sum_{i=0}^m a_i * x^i
        P = sum(self.a[i] * x.pow(i) for i in range(self.a.shape[0]))
        # Compute H(x) = sum_{j=1}^n b_j * x^j
        H = sum(self.b[j-1] * x.pow(j) for j in range(1, self.b.shape[0] + 1))
        # Denominator Q(x) = 1 + |H(x)|
        Q = 1.0 + torch.abs(H)
        return P / Q

# 3. Define the MLP using the rational activation
class GeluApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.act = RationalActivation(m=5, n=4)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# 4. Instantiate model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GeluApproximator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch}/{num_epochs} - MSE Loss: {avg_loss:.6f}")

# 6. Evaluate the maximum approximation error on a fine grid
model.eval()
with torch.no_grad():
    xs = torch.linspace(-5, 5, 10001, device=device).unsqueeze(1)
    # Use torch.erf here
    ys_true = 0.5 * xs * (1 + torch.erf(xs / torch.sqrt(torch.tensor(2.0, device=device))))
    ys_pred = model(xs)
    max_error = (ys_true - ys_pred).abs().max().item()
    print(f"Max approximation error over [-5,5]: {max_error:.6e}")
# Plot the results
plt.figure(figsize=(6,4))
plt.plot(xs.cpu().numpy(), ys_true.cpu().numpy(), label="GELU")
plt.plot(xs.cpu().numpy(), ys_pred.cpu().numpy(), label="Rational Approx")
plt.title("GELU vs. Rational Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Save to a file instead of (or before) plt.show()
out_path = "/user1/res/cvpr/soumyo.b_r/Understanding Informer/gelu_vs_rational.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot saved to {out_path}")

# If you *do* have a display, you can still call show():
# plt.show()

# 8. Save learned coefficients
a_coeffs = model.act.a.detach().cpu().numpy()
b_coeffs = model.act.b.detach().cpu().numpy()
np.save("/user1/res/cvpr/soumyo.b_r/Understanding Informer/rational_num_coeffs.npy", a_coeffs)
np.save("/user1/res/cvpr/soumyo.b_r/Understanding Informer/rational_den_coeffs.npy", b_coeffs)
print("Numerator coefficients saved to rational_num_coeffs.npy")
print("Denominator coefficients saved to rational_den_coeffs.npy")