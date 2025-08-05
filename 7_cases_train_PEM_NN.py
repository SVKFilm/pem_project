import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib

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

# === Encode pred_label ===
label_encoder = LabelEncoder()
df['pred_label_encoded'] = label_encoder.fit_transform(df['pred_label'])
# joblib.dump(label_encoder, 'label_encoder.pkl')

# Select input and output features
# input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

# X = df[input_cols].copy()
y = df[output_cols].copy()

# === Feature Combinations ===
feature_sets = {
    # "case1": ['pred_label_encoded', 'truncated', 'occluded', 'right', 'down', 'forward'],
    # "case2": ['pred_label_encoded', 'truncated', 'occluded'], 
    # "case3": ['pred_label_encoded', 'right', 'down', 'forward'],
    # "case4": ['truncated', 'occluded', 'right', 'down', 'forward'],
    # "case5": ['truncated', 'occluded'],
    "case6": ['right', 'down', 'forward']
}

# === Custom Dataset ===
class BBoxErrorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Training Loop for Each Case ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for case_name, input_cols in feature_sets.items():
    print(f"\n=== Training {case_name} ===")
    
    X = df[input_cols].copy()

    # Feature Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # DataLoaders
    train_loader = DataLoader(BBoxErrorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(BBoxErrorDataset(X_val, y_val), batch_size=32, shuffle=False)

    # Model Setup
    model = BoundingBoxErrorNet(in_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"{case_name}_best_model.pth")
        else:
            counter += 1
            if counter >= 5:
                print("Early stopping triggered!")
                break

        print(f"[{case_name}] Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{case_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{case_name}_loss_curve.png")
    plt.close()