import pandas as pd
import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List

# Directories
prediction_dir = "C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/sampled_prediction/conf075_500"  # directory containing CSV prediction files
ground_truth_dir = "C:/Users/User/Desktop/UoE/DISS/Datasets/sampled_dataset/labels/training/label_2"  # directory containing ground truth label .txt files

# YOLO predicted boxes (x_min, y_min, x_max, y_max) - sampled data
# pred_boxes = np.array([
#     [363.6302, 189.7239, 545.5192, 297.1926],
#     [821.1246, 180.4119, 924.5131, 253.2139],
#     [559.2828, 182.8428, 638.9226, 229.3388],
#     [1005.2284, 195.1554, 1242.0000, 374.0997],
#     [745.8887, 172.6562, 786.5232, 195.7948],
#     [667.2655, 179.4496, 704.2049, 202.8741],
#     [804.5766, 181.8971, 876.5080, 234.6684],
#     [785.9456, 175.2713, 841.6179, 217.9536],
#     [606.9636, 180.8615, 649.5353, 216.1779],
#     [631.8679, 178.9830, 665.3527, 206.7648],
#     [614.9254, 181.6044, 660.2128, 211.8418],
# ])

# # Ground truth boxes
# gt_boxes = np.array([
#     [1013.39, 182.46, 1241.00, 374.00],
#     [354.43, 185.52, 549.52, 294.49],
#     [859.54, 159.80, 879.68, 221.40],
#     [819.63, 178.12, 926.85, 251.56],
#     [800.54, 178.06, 878.75, 230.56],
#     [558.55, 179.04, 635.05, 230.61],
#     [598.30, 178.68, 652.25, 218.17],
#     [784.59, 178.04, 839.98, 220.10],
#     [663.74, 175.36, 707.21, 204.15],
# ])

# IoU function
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Build IoU matrix
# iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
# for i, pred in enumerate(pred_boxes):
#     for j, gt in enumerate(gt_boxes):
#         iou_matrix[i, j] = compute_iou(pred, gt)

# # Convert to cost matrix for Hungarian algorithm
# cost_matrix = 1 - iou_matrix
# row_ind, col_ind = linear_sum_assignment(cost_matrix)

# # Output matched pairs and IoU values
# matches = list(zip(row_ind, col_ind))
# ious = [iou_matrix[r, c] for r, c in matches]

# for (pred_idx, gt_idx), iou in zip(matches, ious):
#     print(f"Pred {pred_idx} matched with GT {gt_idx} — IoU: {iou:.4f}")

# Store matching results
results = {}

# Matching and data collection
combined_data = []

# Iterate over prediction CSV files
for filename in os.listdir(prediction_dir):
    if filename.endswith(".csv"):
        image_name = os.path.splitext(filename)[0]
        pred_path = os.path.join(prediction_dir, filename)
        gt_path = os.path.join(ground_truth_dir, f"{image_name}.txt")

        # Read prediction boxes
        pred_df = pd.read_csv(pred_path)
        pred_boxes = pred_df[["xmin", "ymin", "xmax", "ymax"]].values
        pred_labels = pred_df["class_label"].values if "class_label" in pred_df else None

        # Read ground truth boxes
        gt_boxes = []
        gt_labels = []
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 8:
                        label = parts[0]
                        try:
                            xmin = float(parts[4])
                            ymin = float(parts[5])
                            xmax = float(parts[6])
                            ymax = float(parts[7])
                            gt_boxes.append([xmin, ymin, xmax, ymax])
                            gt_labels.append(label)
                        except ValueError:
                            continue

        # Match predictions to ground truth using Hungarian algorithm based on IoU
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou_matrix[i, j] = 1 - compute_iou(gt_box, pred_box)  # Hungarian minimizes cost

            row_ind, col_ind = linear_sum_assignment(iou_matrix)

            matched_pairs = []

            # Example output
            # '000097': [{'gt_box': [536.07, 176.04, 569.72, 194.78], 'pred_box': [536.96045, 178.99469, 569.20416, 194.50488], 'iou': np.float64(0.7930625996547742), 'gt_label': 'Car', 'pred_label': 'Car'}, 
            # {'gt_box': [717.03, 174.45, 738.76, 193.71], 'pred_box': [715.4706, 176.17818, 740.95123, 193.40366], 'iou': np.float64(0.7747658454661955), 'gt_label': 'Car', 'pred_label': 'Car'}]
            # TODO: Find error between gt and pred and append to matched_pairs
            # then create the error dataset, by image, and combined entries for PEM training
            
            for r, c in zip(row_ind, col_ind):
                gt_box = gt_boxes[r]
                pred_box = pred_boxes[c]
                iou_score = 1 - iou_matrix[r, c]
                error = (np.array(gt_box) - pred_box).tolist()

                combined_data.append({
                    "image_name": image_name,
                    "gt_xmin": gt_box[0],
                    "gt_ymin": gt_box[1],
                    "gt_xmax": gt_box[2],
                    "gt_ymax": gt_box[3],
                    "pred_xmin": pred_box[0],
                    "pred_ymin": pred_box[1],
                    "pred_xmax": pred_box[2],
                    "pred_ymax": pred_box[3],
                    "iou": iou_score,
                    "gt_label": gt_labels[r],
                    "pred_label": pred_labels[c] if pred_labels is not None else None,
                    "error_xmin": error[0],
                    "error_ymin": error[1],
                    "error_xmax": error[2],
                    "error_ymax": error[3],
                })

            results[image_name] = matched_pairs

results_summary = {img: len(pairs) for img, pairs in results.items()}
print(len(results))
# print(results_summary)
# print(results)

# Save combined results to CSV
combined_df = pd.DataFrame(combined_data)
output_path = "C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/sampled_error/matched_boxes_with_error_conf075.csv"
combined_df.to_csv(output_path, index=False)