import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# Load datasets
train_df = pd.read_csv("labelled_train.csv")
val_df = pd.read_csv("labelled_validation.csv")
test_df = pd.read_csv("labelled_test.csv")

print("Training set:", train_df.shape)
print("Validation set:", val_df.shape)
print("Test set:", test_df.shape)

# Separate features and target
X_train = train_df.drop("sus_label", axis=1)
y_train = train_df["sus_label"]

X_val = val_df.drop("sus_label", axis=1)
y_val = val_df["sus_label"]

X_test = test_df.drop("sus_label", axis=1)
y_test = test_df["sus_label"]

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(
    y_train.values.reshape(-1, 1),
    dtype=torch.float32
)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(
    y_val.values.reshape(-1, 1),
    dtype=torch.float32
)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(
    y_test.values.reshape(-1, 1),
    dtype=torch.float32
)

# Neural network
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# Train for exactly 10 epochs
print("\nTraining model...\n")

for epoch in range(10):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch + 1}/10] "
        f"Loss: {loss.item():.6f}"
    )

# Validation accuracy
model.eval()

with torch.no_grad():

    val_preds = model(X_val)
    val_preds = (val_preds >= 0.5).float()

    val_accuracy = (
        (val_preds == y_val)
        .float()
        .mean()
        .item()
    )

# Test accuracy
with torch.no_grad():

    test_preds = model(X_test)
    test_preds = (test_preds >= 0.5).float()

    test_accuracy = (
        (test_preds == y_test)
        .float()
        .mean()
        .item()
    )

print("\nResults")
print("-" * 40)
print(f"Validation Accuracy: {val_accuracy:.6f}")
print(f"Test Accuracy: {test_accuracy:.6f}")

# Save model
torch.save(
    model.state_dict(),
    "cyber_attack_model.pth"
)

print("\nModel saved as cyber_attack_model.pth")