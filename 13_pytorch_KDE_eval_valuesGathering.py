import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from itertools import product
import joblib
import argparse

# === Parse Command-Line Argument for Case Name ===
parser = argparse.ArgumentParser()
parser.add_argument("--case", type=str, required=True)
args = parser.parse_args()
case_name = args.case

# === Define Input Columns Based on Case Name ===
if case_name == "case1":
    input_cols = ['pred_label_encoded', 'truncated', 'occluded', 'right', 'down', 'forward']
elif case_name == "case2":
    input_cols = ['pred_label_encoded', 'truncated', 'occluded']
elif case_name == "case3":
    input_cols = ['pred_label_encoded', 'right', 'down', 'forward']
elif case_name == "case4":
    input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
elif case_name == "case5":
    input_cols = ['truncated', 'occluded']
elif case_name == "case6":
    input_cols = ['right', 'down', 'forward']
else:
    raise ValueError(f"Unknown case name: {case_name}")

# === Paths and Output Naming ===
csv_out_path = f"kde_predictions_{case_name}.csv"

# === Load Evaluation Dataset ===
df = pd.read_csv("error/test/matched_boxes_with_error_v8s_5class_cleaned.csv")
df = df[df["occluded"] != 3].reset_index(drop=True)

# === Encode pred_label using saved LabelEncoder ===
import joblib
label_encoder = joblib.load("label_encoder.pkl")
df["pred_label_encoded"] = label_encoder.transform(df["pred_label"])

# === Config ===
# case_name = "case4"
# csv_out_path = f"kde_predictions_{case_name}.csv"
n_samples = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Input Columns ===
# input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X_raw = df[input_cols].values
z_raw = df[output_cols].values

# === Load KDE Hyperparameters ===
with open(f"KDE_{case_name}_model_PEM_best.pkl", "rb") as f:
    config = pickle.load(f)

h_rel_vec = np.array([config["h_rel"][key] for key in input_cols])
H_con_np = np.cov(z_raw.T) * config["H_con_scale"]

# === Convert to Torch ===
x_train = torch.tensor(X_raw, dtype=torch.float32, device=device)
z_train = torch.tensor(z_raw, dtype=torch.float32, device=device)
h_rel_vec = torch.tensor(h_rel_vec, dtype=torch.float32, device=device)
H_con = torch.tensor(H_con_np, dtype=torch.float32, device=device)

# === z_range grid ===
z = np.linspace(-1, 1, 20)
z_vals = 50 * z**3
z_range_np = np.array(list(product(z_vals, z_vals, z_vals, z_vals)))
z_range = torch.tensor(z_range_np, dtype=torch.float32, device=device)

# === KDE PDF calculation ===
def multivariate_normal_pdf(x, mean, cov_inv, cov_det):
    diff = x - mean
    exponent = -0.5 * torch.sum(diff @ cov_inv * diff, dim=1)
    norm_const = torch.sqrt((2 * np.pi) ** x.shape[1] * cov_det)
    return torch.exp(exponent) / norm_const

# === Precompute covariance inverse and det ===
H_con_inv = torch.inverse(H_con)
H_con_det = torch.det(H_con)

# === Main KDE Loop ===
results = []

for idx in tqdm(range(len(df))):
    x_query = x_train[idx]
    x_diff = (x_train - x_query) / h_rel_vec
    weights = torch.exp(-0.5 * torch.sum(x_diff**2, dim=1))
    weights /= weights.sum()

    # Evaluate KDE PDF at all z_range
    pdf_vals = torch.zeros(z_range.shape[0], device=device)
    for i in range(len(x_train)):
        mean = z_train[i]
        pdf_vals += weights[i] * multivariate_normal_pdf(z_range, mean, H_con_inv, H_con_det)

    # Normalize PDF
    pdf_vals /= torch.trapezoid(pdf_vals, z_range[:, 0])

    # MAP (max PDF)
    mu_map_idx = torch.argmax(pdf_vals)
    mu_map = z_range[mu_map_idx]

    # CDF + Sampling
    cdf_vals = torch.cumsum(pdf_vals, dim=0)
    cdf_vals /= cdf_vals[-1]

    rand_vals = torch.rand(n_samples, device=device)
    sample_indices = torch.searchsorted(cdf_vals, rand_vals)
    sampled = z_range[sample_indices]

    row = {"index": idx}
    for j in range(4):
        row[f"mu_map_{j}"] = mu_map[j].item()
    for i in range(n_samples):
        for j in range(4):
            row[f"sample_{i}_{j}"] = sampled[i][j].item()

    results.append(row)

# === Save Results ===
df_kde = pd.DataFrame(results).set_index("index")
df_kde.to_csv(csv_out_path)
print(f"\n✅ Saved CSV: {csv_out_path}")
