import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from NN_PEM import BoundingBoxErrorNet
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# === Load Dataset ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/test/matched_boxes_with_error_v8s_5class_cleaned.csv')
df = df[df['occluded'] != 3].reset_index(drop=True)

# === Encode pred_label using saved LabelEncoder ===
label_encoder = joblib.load('label_encoder.pkl')
df['pred_label_encoded'] = label_encoder.transform(df['pred_label'])

# === Columns ===
input_cols = ['right', 'down', 'forward']  # Adjust input columns here
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X_raw = df[input_cols].copy()
z_raw = df[output_cols].copy()
W_gt = df[['gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax']].values
W_pred = df[['pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax']].values
gt_labels = df['gt_label'].values

# === Normalize input ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values)

# === Load KDE Hyperparameters (stubbed for now) ===
# with open("KDE_model_PEM_best.pkl", "rb") as f:
#     config = pickle.load(f)
# h_rel_vec = np.array([config["h_rel"][key] for key in input_cols])
# H_con = np.cov(z_raw.T) * config["H_con_scale"]
# x_train = X_raw.values
# z_train = z_raw.values

# === Load Neural Network ===
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

# === IoU Function ===
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

# === Sampling-based IoU Evaluation ===
SAMPLES = 50
ori_ious = []
nn_sampled_ious = []
# kde_sampled_ious = []
per_class_sampled = {}

print(f"Running sampling-based evaluation with {SAMPLES} samples each...")

for i in tqdm(range(len(df)), desc="Sampling Eval"):
    label = gt_labels[i]
    W = W_gt[i]
    Wp = W_pred[i]

    # === Original IoU ===
    ori_iou = compute_iou(W, Wp)
    ori_ious.append(ori_iou)

    # === NN Sampling ===
    mu_i = mu_pred[i].cpu().numpy()
    Sigma_i = Sigma_pred[i].cpu().numpy()
    sampled_errors = np.random.multivariate_normal(mu_i, Sigma_i, SAMPLES)
    ious = [compute_iou(W + e, Wp) for e in sampled_errors]
    iou_nn = np.mean(ious)
    nn_sampled_ious.append(iou_nn)

    # === Store by class ===
    if label not in per_class_sampled:
        per_class_sampled[label] = {"ori": [], "nn": []}
    per_class_sampled[label]["ori"].append(ori_iou)
    per_class_sampled[label]["nn"].append(iou_nn)

    # === KDE Sampling (stub) ===
    # x_query = X_raw.iloc[i].values
    # x_diff = (x_train - x_query) / h_rel_vec
    # weights = np.exp(-0.5 * np.sum(x_diff ** 2, axis=1))
    # weights /= np.sum(weights)
    # idx_sampled = np.random.choice(len(z_train), size=SAMPLES, p=weights)
    # sampled_kde = np.random.multivariate_normal(z_train[idx_sampled[0]], H_con, SAMPLES)
    # ious_kde = [compute_iou(W + e, Wp) for e in sampled_kde]
    # kde_sampled_ious.append(np.mean(ious_kde))

# === Final Results ===
print(f"\n[Original ] Overall Mean IoU: {np.mean(ori_ious):.4f}")
print(f"[NN       ] Overall Mean IoU (Sampling-based): {np.mean(nn_sampled_ious):.4f}")
# print(f"[KDE      ] Overall Mean IoU (Sampling-based): {np.mean(kde_sampled_ious):.4f}")

# === Class-wise Results ===
print("\nClass-wise IoU Results (Sampling-based):")
for label in sorted(per_class_sampled.keys()):
    mean_ori = np.mean(per_class_sampled[label]["ori"])
    mean_nn = np.mean(per_class_sampled[label]["nn"])
    print(f"  [{label:<10}] Original: {mean_ori:.4f} | NN: {mean_nn:.4f}")
    # Optionally print KDE later
