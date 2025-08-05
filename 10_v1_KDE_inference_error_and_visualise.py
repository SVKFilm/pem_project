import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal
import pickle
from itertools import product
from tqdm import tqdm

# === Load data ===
df = pd.read_csv('C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/matched_boxes_with_error_v8s_5class_cleaned.csv')
df = df[df['occluded'] != 3].reset_index(drop=True)

input_cols = ['truncated', 'occluded', 'right', 'down', 'forward']
output_cols = ['error_xmin', 'error_ymin', 'error_xmax', 'error_ymax']

X_raw = df[input_cols].copy()
z_raw = df[output_cols].copy()

# === Load best hyperparameters ===
with open("KDE_case4_model_PEM_best.pkl", "rb") as f:
    config = pickle.load(f)

h_rel_vec = np.array([config["h_rel"][key] for key in input_cols])
H_con = np.cov(z_raw.T) * config["H_con_scale"]

# Store training data arrays for KDE
x_train = X_raw.values
z_train = z_raw.values

# === Final KDE Function - not used ===
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

def kde_predict(x_query, x_train, z_train, h_rel_vec, H_con, z_range, n_samples=1):
    # Step 1: KDE weights in input space
    #debug
    # print(x_train.shape)
    # print(x_query.shape)
    # print(h_rel_vec.shape)
    # fsf
    x_diff = (x_train - x_query) / h_rel_vec
    # print(x_diff.shape)
    # eff
    weights = np.exp(-0.5 * np.sum(x_diff**2, axis=1))
    # print(np.max(weights))
    # print(weights.shape)
    # print(weights)
    # print(np.sum(weights))
    # eff
    weights /= np.sum(weights)

    # Step 2: Evaluate KDE over grid of z values
    pdf = np.zeros(z_range.shape[0])
    # print(z_range.shape[0])
    # rfjs
    for i in range(len(x_train)):
        pdf += weights[i] * multivariate_normal(mean=z_train[i], cov=H_con).pdf(z_range)
        # print(np.all(weights[i] * multivariate_normal(mean=z_train[i], cov=H_con).pdf(z_range) == np.zeros(z_range.shape[0])))
    
    # Step 3: Normalize PDF - not needed
    pdf /= np.trapezoid(pdf, x=z_range[:, 0])
    # print(np.max(pdf))
    top_indices = np.argsort(pdf)[-5:][::-1]  # Indices of top 5 in descending order
    top_zs = z_range[top_indices]             # Corresponding z values
    top_probs = pdf[top_indices]              # Their densities

    # Print results
    for rank, (z_val, prob) in enumerate(zip(top_zs, top_probs), 1):
        print(f"#{rank}: z = {z_val}, pdf = {prob:.6f}")
        # print(np.argmax(pdf))
    # e
    # Step 4: MAP estimate
    mu_map = z_range[np.argmax(pdf)]

    # Step 5: Weighted random sampling from z_range
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    rand_vals = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, rand_vals)
    samples = z_range[sample_indices]
    print(mu_map)
    print(samples)

    # Plot PDF - for debug
    # plt.figure(figsize=(10, 4))
    # plt.plot(pdf, label='KDE PDF over z_range')
    # plt.axvline(np.argmax(pdf), color='r', linestyle='--', label='MAP Index')
    # plt.title("KDE PDF across z_range samples")
    # plt.xlabel("z_range Index")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    return mu_map, samples

# z_vals = np.linspace(-50, 50, 20)  
z = np.linspace(-1, 1, 20)
z_vals = 50 * z**3  # new - Cubic bias toward center
# z_vals
z_range = np.array(list(product(z_vals, z_vals, z_vals, z_vals)))

# === Visualisation ===
def plot_bounding_boxes(image_path):
    image_filename = os.path.basename(image_path)
    image_id = int(os.path.splitext(image_filename)[0])
    indices = df[df['image_name'] == image_id].index.tolist()

    if not indices:
        print(f"No matching entries for image {image_filename}")
        return

    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    for idx in tqdm(indices):
        X_input = X_raw.iloc[idx].values
        W_pred = df.loc[idx, ['pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax']].values
        W_gt = df.loc[idx, ['gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax']].values

        # === KDE full density prediction ===
        mu_map, samples = kde_predict(X_input, x_train, z_train, h_rel_vec, H_con, z_range, n_samples=1)

        # === Plot boxes ===
        ax.add_patch(patches.Rectangle((W_gt[0], W_gt[1]), W_gt[2]-W_gt[0], W_gt[3]-W_gt[1],
                                       linewidth=2, edgecolor='g', facecolor='none', label='W (GT)' if idx == indices[0] else None))
        ax.add_patch(patches.Rectangle((W_pred[0], W_pred[1]), W_pred[2]-W_pred[0], W_pred[3]-W_pred[1],
                                       linewidth=2, edgecolor='r', facecolor='none', label="W'" if idx == indices[0] else None))
        pred_box = W_gt + mu_map
        ax.add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                       linewidth=2, edgecolor='b', facecolor='none', label="W+MAP" if idx == indices[0] else None))

        # for s in samples:
        #     sample_box = W_gt + s
        #     ax.add_patch(patches.Rectangle((sample_box[0], sample_box[1]), sample_box[2]-sample_box[0], sample_box[3]-sample_box[1],
        #                                    linewidth=1, edgecolor='orange', linestyle='--', facecolor='none', alpha=0.8))

    ax.legend()
    ax.set_title(f"Image {image_filename}: KDE Bounding Box Error Prediction")
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.set_aspect('equal')
    plt.show()

im_path = "C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/val/000021.png"
plot_bounding_boxes(im_path)
