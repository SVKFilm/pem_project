import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train2/weights/best.pt')

# Image path
image_path = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/train/000031.png'

# Label path
label_path = "C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/labels/training/label_2/000031.txt"

# Run prediction
results = model.predict(source=image_path, save=False, conf=0.5)

# Load original image (for display)
original_img = cv2.imread(image_path)
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Copy images for drawing
gt_img = original_img_rgb.copy()
pred_img = original_img_rgb.copy()

# --------------------
# Draw Ground Truth Boxes (Car or Pedestrian)
# --------------------
with open(label_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        # print(parts)
        cls_name = parts[0]
        # if cls_name not in ['Car', 'Pedestrian']:
        #     continue  # Skip other classes

        # Extract bounding box from 5th to 8th entries (index 4 to 7)
        xmin, ymin, xmax, ymax = map(float, parts[4:8])
        print(cls_name, xmin, ymin, xmax, ymax)
        # Draw rectangle
        cv2.rectangle(gt_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
        # Add label
        cv2.putText(gt_img, cls_name, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)

# --------------------
# Filter and Draw Predictions
# --------------------
class_thresholds = {
    'Car': 0.75,
    'Pedestrian': 0.60,
    'Van': 0.60,
    'Cyclist': 0.60,
    'Truck': 0.60
}

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[c] for c in class_ids]

    for box, score, cls_id, cls_name in zip(boxes, scores, class_ids, class_names):
        if cls_name in class_thresholds and score >= class_thresholds[cls_name]:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(pred_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            label = f"{cls_name} {score:.2f}"
            cv2.putText(pred_img, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

# --------------------
# Display Side-by-Side
# --------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.axis('off')
plt.imshow(gt_img)

plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.axis('off')
plt.imshow(pred_img)

plt.tight_layout()
plt.show()