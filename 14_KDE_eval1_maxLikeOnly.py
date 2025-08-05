import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from scipy.stats import multivariate_normal
from itertools import product
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Progress bar
import joblib

from NN_PEM import BoundingBoxErrorNet

# === Load Evaluation Dataset ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/test/matched_boxes_with_error_v8s_5class_cleaned.csv')
df = df[df['occluded'] != 3].reset_index(drop=True)

# === Encode pred_label using saved LabelEncoder ===
label_encoder = joblib.load('label_encoder.pkl')
df['pred_label_encoded'] = label_encoder.transform(df['pred_label'])

# === Columns for Input and Output ===
input_cols = ['right', 'down', 'forward']   # Adjust input columns here
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X_raw = df[input_cols].copy()
z_raw = df[output_cols].copy()
W_gt = df[['gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax']].values
W_pred = df[['pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax']].values
gt_labels = df['gt_label'].values

# === Normalize input features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values)

# === Load KDE Hyperparameters and Training Data ===
# with open("KDE_model_PEM_best.pkl", "rb") as f:
#     config = pickle.load(f)

# h_rel_vec = np.array([config["h_rel"][key] for key in input_cols])
# H_con = np.cov(z_raw.T) * config["H_con_scale"]
# x_train = X_raw.values
# z_train = z_raw.values

# === z_range Grid for KDE with high resolution ===
# print("Generating KDE z_range grid...")
# z_vals = np.linspace(-1000, 1000, 20) 
# z_range = np.array(list(product(z_vals, z_vals, z_vals, z_vals)))
# print(f"z_range shape: {z_range.shape}")

# # === KDE Prediction Function ===
# def kde_predict(x_query, x_train, z_train, h_rel_vec, H_con, z_range):
#     x_diff = (x_train - x_query) / h_rel_vec
#     weights = np.exp(-0.5 * np.sum(x_diff ** 2, axis=1))
#     weights /= np.sum(weights)

#     pdf = np.zeros(z_range.shape[0])
#     for i in range(len(x_train)):
#         pdf += weights[i] * multivariate_normal(mean=z_train[i], cov=H_con).pdf(z_range)

#     pdf /= np.trapezoid(pdf, x=z_range[:, 0])  # Normalize over first dim
#     mu_map = z_range[np.argmax(pdf)]
#     return mu_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoundingBoxErrorNet(in_features=len(input_cols)).to(device)
model.load_state_dict(torch.load("case6_best_model.pth", map_location=device))
model.eval()

# === NN Inference ===
print("Running NN inference...")
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    mu_pred, L_pred = model(X_tensor)
    Sigma_pred = torch.bmm(L_pred, L_pred.transpose(1, 2))

# === IoU Computation ===
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# === Evaluate Mean-based IoU from NN and KDE ===
ori_ious = []
per_class_ious = {}
nn_ious = []
# kde_ious = []
mu_map_list = []

print("Evaluating samples with tqdm...")

for i in tqdm(range(len(df)), desc="Evaluating"):
    label = gt_labels[i]
    x_input = X_raw.iloc[i].values
    W = W_gt[i]
    Wp = W_pred[i]

    # === Original IoUs ===
    iou_ori = compute_iou(W, Wp)
    ori_ious.append(iou_ori)

    # === NN prediction (mean error) ===
    E_nn = mu_pred[i].cpu().numpy()
    W_nn = W + E_nn
    iou_nn = compute_iou(W_nn, Wp)
    nn_ious.append(iou_nn)

    if label not in per_class_ious:
        per_class_ious[label] = {"ori": [], "nn": []}
    per_class_ious[label]["ori"].append(iou_ori)
    per_class_ious[label]["nn"].append(iou_nn)

    # === KDE MAP prediction ===
    # mu_map = kde_predict(x_input, x_train, z_train, h_rel_vec, H_con, z_range)
    # mu_map_list.append(mu_map)
    # W_kde = W + mu_map
    # kde_ious.append(compute_iou(W_kde, Wp))

# === Final Output ===
# === Overall IoUs ===
print(f"\n[Original ] Overall Mean IoU: {np.mean(ori_ious):.4f}")
print(f"[NN       ] Overall Mean IoU: {np.mean(nn_ious):.4f}")
# print(f"[KDE] Mean IoU (Mean-based): {np.mean(kde_ious):.4f}")

# === Class-wise IoUs ===
print("\nClass-wise IoU Results:")
for label in sorted(per_class_ious.keys()):
    class_ori = np.mean(per_class_ious[label]["ori"])
    class_nn = np.mean(per_class_ious[label]["nn"])
    print(f"  [{label:<10}] Original: {class_ori:.4f} | NN: {class_nn:.4f}")
