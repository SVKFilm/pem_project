import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import multivariate_normal
from itertools import product
import pickle
import joblib

# === Set Random Seed for Reproducibility ===
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# === Load Evaluation Dataset ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/test/matched_boxes_with_error_v8s_5class_cleaned.csv')
df = df[df['occluded'] != 3].reset_index(drop=True)

# === Encode pred_label using saved LabelEncoder ===
label_encoder = joblib.load('label_encoder.pkl')
df['pred_label_encoded'] = label_encoder.transform(df['pred_label'])

case_name = "case4"
csv_out_path = f"kde_predictions_{case_name}.csv"

# input_cols = ['pred_label_encoded', 'truncated', 'occluded', 'right', 'down', 'forward'] #case1
# input_cols = ['pred_label_encoded', 'truncated', 'occluded'] #case2
# input_cols = ['pred_label_encoded', 'right', 'down', 'forward'] #case3
input_cols = ['truncated', 'occluded', 'right', 'down', 'forward'] #case4
# input_cols = ['truncated', 'occluded'] #case5
# input_cols = ['right', 'down', 'forward'] #case6
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X_raw = df[input_cols].copy()
z_raw = df[output_cols].copy()
# W_gt = df[['gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax']].values
# W_pred = df[['pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax']].values
# image_ids = df['image_name'].values

# === Load KDE Hyperparameters ===
with open(f"KDE_{case_name}_model_PEM_best.pkl", "rb") as f:
    config = pickle.load(f)

h_rel_vec = np.array([config["h_rel"][key] for key in input_cols])
H_con = np.cov(z_raw.T) * config["H_con_scale"]
x_train = X_raw.values
z_train = z_raw.values

# === Create z_range grid ===
z = np.linspace(-1, 1, 20)
z_vals = 50 * z**3  # Center bias
z_range = np.array(list(product(z_vals, z_vals, z_vals, z_vals)))

# === Output directory ===
output_dir = f"kde_predictions_{case_name}"
os.makedirs(output_dir, exist_ok=True)

# === KDE Prediction Function ===
def kde_predict(x_query, x_train, z_train, h_rel_vec, H_con, z_range, n_samples=50):
    x_diff = (x_train - x_query) / h_rel_vec
    weights = np.exp(-0.5 * np.sum(x_diff**2, axis=1))
    weights /= np.sum(weights)

    pdf = np.zeros(z_range.shape[0])
    for i in range(len(x_train)):
        pdf += weights[i] * multivariate_normal(mean=z_train[i], cov=H_con).pdf(z_range)

    pdf /= np.trapezoid(pdf, x=z_range[:, 0])
    mu_map = z_range[np.argmax(pdf)]

    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    rand_vals = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, rand_vals)
    samples = z_range[sample_indices]

    return mu_map, samples

# === Run Inference and Collect Results ===
print(f"🔍 Running KDE inference for {len(df)} rows...")

records = []
for idx in tqdm(range(len(df)), desc="KDE Inference"):
    x_input = X_raw.iloc[idx].values
    mu_map, samples = kde_predict(x_input, x_train, z_train, h_rel_vec, H_con, z_range, n_samples=50)

    row = {"index": idx}
    row.update({f"mu_map_{i}": mu_map[i] for i in range(4)})
    for i in range(samples.shape[0]):
        for j in range(4):
            row[f"sample_{i}_{j}"] = samples[i, j]

    records.append(row)

# === Save to CSV ===
df_kde = pd.DataFrame(records).set_index("index")
df_kde.to_csv(csv_out_path)
print(f"\n✅ KDE prediction results saved to: {csv_out_path}")