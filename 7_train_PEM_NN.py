import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import joblib

from NN_PEM import BoundingBoxErrorNet, multivariate_gaussian_nll

# === Set Seed for Reproducibility ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# === Load and Clean Dataset ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/matched_boxes_with_error_v8s_5class_cleaned.csv')

# Remove rows where 'occluded' == 3
df = df[df['occluded'] != 3].reset_index(drop=True)

# Select input and output features
input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X = df[input_cols].copy()
y = df[output_cols].copy()

# === Feature Scaling ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers for later use (optional)
# import joblib
# joblib.dump(scaler_X, 'input_scaler.pkl')
# joblib.dump(scaler_y, 'output_scaler.pkl')

# === Split Dataset ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === Custom Dataset ===
class BBoxErrorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Create DataLoaders ===
train_dataset = BBoxErrorDataset(X_train, y_train)
val_dataset = BBoxErrorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("\nDataLoader complete\n")

# === Optional Save Processed Data ===
# processed_df = pd.concat([pd.DataFrame(X_scaled, columns=input_cols), pd.DataFrame(y_scaled, columns=output_cols)], axis=1)
# processed_df.to_csv('processed_dataset.csv', index=False)

# === Model, Optimizer, and Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoundingBoxErrorNet(in_features=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Train")

num_epochs = 200
counter = 0
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        mu_pred, L_pred = model(X_batch)
        loss = multivariate_gaussian_nll(y_batch, mu_pred, L_pred)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mu_pred, L_pred = model(X_batch)
            loss = multivariate_gaussian_nll(y_batch, mu_pred, L_pred)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model and do early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "NN_PEM_v8s_5class_v21_2.pth")
    else:
        counter += 1
        if counter >= 5:
            print("Early stopping triggered!")
            break

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

print("NN_PEM_v8s_5class_v21_2")
# Optional: save model
# torch.save(model.state_dict(), 'PEM_NN_sampled500.pth')

# === Plot training and validation loss ===
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('NN_PEM_v8s_5class_v21_2.png')
plt.show()