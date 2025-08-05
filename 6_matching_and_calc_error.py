import pandas as pd
import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List
from tqdm import tqdm

# Directories
prediction_dir = "C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/prediction/v8s_5class_test"  # directory containing CSV prediction files
ground_truth_dir = "C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/labels/training/label_2"  # directory containing ground truth label .txt files

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

# Store matching results
results = {}

# Matching and data collection
combined_data = []

# Iterate over prediction CSV files
for filename in tqdm(os.listdir(prediction_dir)):
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
        gt_extra_info = []  # Store extra features
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 8:
                        label = parts[0]
                        try:
                            label = parts[0]
                            xmin = float(parts[4])
                            ymin = float(parts[5])
                            xmax = float(parts[6])
                            ymax = float(parts[7])
                            truncated = float(parts[1])
                            occluded = int(parts[2])
                            right = float(parts[11])
                            down = float(parts[12])
                            forward = float(parts[13])

                            gt_boxes.append([xmin, ymin, xmax, ymax])
                            gt_labels.append(label)
                            gt_extra_info.append({
                                "truncated": truncated,
                                "occluded": occluded,
                                "right": right,
                                "down": down,
                                "forward": forward
                            })
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
            
            for r, c in zip(row_ind, col_ind):
                gt_box = gt_boxes[r]
                pred_box = pred_boxes[c]
                iou_score = 1 - iou_matrix[r, c]
                error = (pred_box - np.array(gt_box)).tolist() # new

                if iou_score > 0.4 and gt_labels[r] in ["Car", "Pedestrian", "Van", "Cyclist", "Truck"]:
                    extra = gt_extra_info[r]
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
                        "truncated": extra["truncated"],
                        "occluded": extra["occluded"],
                        "right": extra["right"],
                        "down": extra["down"],
                        "forward": extra["forward"]
                    })

            results[image_name] = matched_pairs

results_summary = {img: len(pairs) for img, pairs in results.items()}
print(len(results))
# print(results_summary)
# print(results)

# Save combined results to CSV
combined_df = pd.DataFrame(combined_data)
output_path = "C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/error/test/matched_boxes_with_error_v8s_5class_cleaned.csv"
combined_df.to_csv(output_path, index=False)