import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold
from itertools import product

# === Load and Clean Dataset ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/matched_boxes_with_error_v8s_5class_cleaned.csv')
df = df[df['occluded'] != 3].reset_index(drop=True)

# === Select input and output features ===
input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']
x_data = df[input_cols].values  # shape (N, 5)
z_data = df[output_cols].values  # shape (N, 4)

# === Output noise covariance (can also be estimated) ===
H_con = np.cov(z_data.T)  # Full covariance across 4 outputs

#debugs
# print(H_con.shape)
# print(H_con)

# === LOO-CV Bandwidth Tuning (Diagonal Input Kernel) ===
def loo_log_likelihood_diag(x_data, z_data, h_rel_vec, h_con_mat):
    N = x_data.shape[0]
    total_loglik = 0.0

    for i in tqdm(range(N)):
        x_loo = np.delete(x_data, i, axis=0)
        z_loo = np.delete(z_data, i, axis=0)

        x_diff = (x_data[i] - x_loo) / h_rel_vec
        weights = np.exp(-0.5 * np.sum(x_diff**2, axis=1))
        weights /= np.sum(weights)

        pdf_vals = np.array([
            multivariate_normal(mean=z_loo[j], cov=h_con_mat).pdf(z_data[i])
            for j in range(N - 1)
        ])
        p_i = np.sum(weights * pdf_vals)
        total_loglik += np.log(p_i + 1e-12)

    return total_loglik / N

# === k-Fold CV Bandwidth Tuning (Diagonal Input Kernel) ===
def kfold_log_likelihood(x_data, z_data, h_rel_vec, h_con_mat, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42) # seed=42
    total_loglik = 0.0
    total_points = 0

    for train_idx, val_idx in kf.split(x_data):
        # x_train, z_train = x_data[train_idx], z_data[train_idx]
        # x_val, z_val = x_data[val_idx], z_data[val_idx]

        # for i in tqdm(range(len(x_val))):
        #     x_diff = (x_val[i] - x_train) / h_rel_vec
        #     weights = np.exp(-0.5 * np.sum(x_diff**2, axis=1))
        #     weights /= np.sum(weights)

        #     pdf_vals = np.array([
        #         multivariate_normal(mean=z_train[j], cov=h_con_mat).pdf(z_val[i])
        #         for j in range(len(z_train))
        #     ])
        #     p_i = np.sum(weights * pdf_vals)
        #     total_loglik += np.log(p_i + 1e-12)
        #     total_points += 1

        x_train, z_train = x_data[train_idx], z_data[train_idx]
        x_val, z_val = x_data[val_idx], z_data[val_idx]

        # === Compute Relevance Weights (K_rel) ===
        x_val_exp = x_val[:, np.newaxis, :]
        x_train_exp = x_train[np.newaxis, :, :]
        x_diff = (x_val_exp - x_train_exp) / h_rel_vec
        dists_sq = np.sum(x_diff**2, axis=-1)
        weights = np.exp(-0.5 * dists_sq)
        weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize

        # === Compute Conditional Kernel (K_con) ===
        mvn = multivariate_normal(mean=np.zeros(z_train.shape[1]), cov=h_con_mat)
        z_val_exp = z_val[:, np.newaxis, :]
        z_train_exp = z_train[np.newaxis, :, :]
        z_diff = z_val_exp - z_train_exp
        pdf_vals = np.exp(mvn.logpdf(z_diff))

        # === Combine Weights ===
        p_approx = np.sum(weights * pdf_vals, axis=1)
        total_loglik += np.sum(np.log(p_approx + 1e-12))
        total_points += len(x_val)

    return total_loglik / total_points

# === Full Hyperparameter tuning ===
input_grid = [
    [0.3, 0.5, 0.8],
    [0.3, 0.5, 0.8],
    [0.3, 0.5, 0.8],
    [0.3, 0.5, 0.8],
    [0.3, 0.5, 0.8]
]
scale_factors = [0.5, 1.0, 2.0]
H_base = np.cov(z_data.T)

best_score = -np.inf
best_config = None
scores = []

# All combinations of h_rel_vec and H_con scale
for h_rel_vals in tqdm(list(product(*input_grid))):
    h_rel_vec = np.array(h_rel_vals)

    for scale in scale_factors:
        H_con_scaled = H_base * scale

        score = kfold_log_likelihood(x_data, z_data, h_rel_vec, H_con_scaled, k=5)

        # Store best config
        if score > best_score:
            best_score = score
            best_config = {
                "h_rel": dict(zip(input_cols, h_rel_vec.tolist())),
                "H_con_scale": scale,
                "score": score
            }

        # Append to log
        scores.append((
            dict(zip(input_cols, h_rel_vec.tolist())),  # named h_rel
            scale,
            score
        ))

        print(f"h = {h_rel_vec}, scale = {scale:.2f} → log-likelihood = {score:.4f}")


# Convert scores list into DataFrame
df_scores = pd.DataFrame([
    {
        **{f"h_rel_{k}": v for k, v in h_dict.items()},
        'H_con_scale': scale,
        'log_likelihood': ll
    }
    for (h_dict, scale, ll) in scores
])
df_scores.to_csv("kde_hyperparam_search_log.csv", index=False)
print("📄 Saved all tuning results to 'kde_hyperparam_search_log.csv'")

# === Save best kernels ===
with open("KDE_model_PEM_best.pkl", "wb") as f:
    pickle.dump(best_config, f)
print("💾 Saved best hyperparameters to 'best_kde_hyperparams.pkl'")

# # === Final KDE Function ===
# def conditional_kde_vector_full(x_sim, z_range, x_data, z_data, h_rel_vec, h_con_mat):
#     N = x_data.shape[0]
#     weights = np.array([
#         np.exp(-0.5 * np.sum(((x_sim - x_data[i]) / h_rel_vec)**2))
#         for i in range(N)
#     ])
#     weights /= np.sum(weights)

#     pdf = np.zeros(z_range.shape[0])
#     for i in range(N):
#         pdf += weights[i] * multivariate_normal(mean=z_data[i], cov=h_con_mat).pdf(z_range)

#     pdf /= np.trapezoid(pdf, x=z_range[:, 0])
#     return pdf

# # === Example Query and Visualization ===
# x_sim = np.array([0, 0, 100, 50, 20])  # Example input (modify as needed)
# z_range = np.linspace(-100, 100, 300).reshape(-1, 1)  # z₁ range

# h_rel_vec = np.ones(x_data.shape[1]) * best_h
# pdf = conditional_kde_vector_full(x_sim, z_range, x_data, z_data, h_rel_vec, H_con)

# # === Plot conditional PDF for z₁ ===
# plt.figure(figsize=(8, 4))
# plt.plot(z_range, pdf, label=f"Estimated $p(z_1 \\mid x={x_sim.tolist()})$")
# plt.xlabel("z₁ (e.g., error_xmin)")
# plt.ylabel("Density")
# plt.grid(True)
# plt.title("Conditional KDE for First Output Dimension")
# plt.legend()
# plt.tight_layout()
# plt.show()