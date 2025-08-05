import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import joblib
import numpy as np
from PIL import Image
import os

from NN_PEM import BoundingBoxErrorNet

# === Load scalers ===
# scaler_X = joblib.load('input_scaler.pkl')
# scaler_y = joblib.load('output_scaler.pkl')

# === Load sample data ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/matched_boxes_with_error_v8s_5class_cleaned.csv')
# df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/sampled_error/matched_boxes_with_error_conf7560_500_cleaned_toTrain.csv')
df = df[df['occluded'] != 3].reset_index(drop=True)

input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X_raw = df[input_cols].copy()
y_raw = df[output_cols].copy()

# === Fit scalers ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = BoundingBoxErrorNet(in_features=5).to(device)
model.load_state_dict(torch.load("case4_best_model.pth", map_location=device))
model.eval()

# === Inference ===
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    mu_pred, L_pred = model(X_tensor)
    Sigma_pred = torch.bmm(L_pred, L_pred.transpose(1, 2))  # Covariance
    # print(mu_pred.shape)
    # print(Sigma_pred.shape)

# === Visualization ===
def plot_bounding_boxes(image_path):
    # Extract image_id from filename
    image_filename = os.path.basename(image_path)
    image_id = int(os.path.splitext(image_filename)[0])

    # Get all matching indices
    indices = df[df['image_name'] == image_id].index.tolist()
    if not indices:
        print(f"No matching entries for image {image_filename}")
        return

    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    for idx in indices:
        X_input = X_raw.iloc[idx]
        W_prime = df.loc[idx, ['pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax']].values
        # print(df.loc[idx, ['image_name']].values)
        W = df.loc[idx, ['gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax']].values

        mu = mu_pred[idx].cpu().numpy()
        cov = Sigma_pred[idx].cpu().numpy()
        std = np.sqrt(np.diag(cov))

        pred_mean = scaler_y.inverse_transform(mu.reshape(1, -1))[0]
        pred_upper = scaler_y.inverse_transform((mu + std).reshape(1, -1))[0]
        pred_lower = scaler_y.inverse_transform((mu - std).reshape(1, -1))[0]

        # Ground truth box
        ax.add_patch(patches.Rectangle((W[0], W[1]), W[2]-W[0], W[3]-W[1], linewidth=2, edgecolor='g', facecolor='none', label='W (Ground Truth)' if idx == indices[0] else None))

        # Predicted box
        ax.add_patch(patches.Rectangle((W_prime[0], W_prime[1]), W_prime[2]-W_prime[0], W_prime[3]-W_prime[1], linewidth=2, edgecolor='r', facecolor='none', label="W'" if idx == indices[0] else None))

        # Predicted mean
        pred_box = W_prime + pred_mean
        pred_box = W + pred_mean
        ax.add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1], linewidth=2, edgecolor='b', facecolor='none', label="W+μ" if idx == indices[0] else None))

        # Confidence bounds
        # upper_box = W_prime + pred_upper
        # lower_box = W_prime + pred_lower

        # ax.add_patch(patches.Rectangle((upper_box[0], upper_box[1]), upper_box[2]-upper_box[0], upper_box[3]-upper_box[1], linewidth=1, edgecolor='cyan', facecolor='none', label='+1σ' if idx == indices[0] else None))
        # ax.add_patch(patches.Rectangle((lower_box[0], lower_box[1]), lower_box[2]-lower_box[0], lower_box[3]-lower_box[1], linewidth=1, edgecolor='magenta', facecolor='none', label='-1σ' if idx == indices[0] else None))

        # Sampled boxes
        # samples = np.random.multivariate_normal(mean=mu, cov=cov, size=5)
        # for s in samples:
        #     # sample_box = W_prime + scaler_y.inverse_transform(s.reshape(1, -1))[0]
        #     sample_box = W + scaler_y.inverse_transform(s.reshape(1, -1))[0]
        #     ax.add_patch(patches.Rectangle((sample_box[0], sample_box[1]), sample_box[2]-sample_box[0], sample_box[3]-sample_box[1], linewidth=1, edgecolor='orange', linestyle='--', facecolor='none', alpha=0.8))

    ax.legend()
    ax.set_title(f"Image {image_filename}: Bounding Box Error Prediction")
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.set_aspect('equal')
    plt.show()

# === Example usage ===
im_path = "C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/train/000173.png"
# im_path = "C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/val/000076.png"
plot_bounding_boxes(im_path)