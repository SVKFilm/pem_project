from ultralytics import YOLO
import pandas as pd
import os
from tqdm import tqdm

# Load the model
model = YOLO('runs/detect/train2/weights/best.pt')

# Define per-class confidence thresholds
class_thresholds = {
    'Car': 0.75,
    'Pedestrian': 0.60,
    'Van': 0.60,
    'Cyclist': 0.60,
    'Truck': 0.60
}

# Output directory for CSV results
output_dir = 'C:/Users/User/Desktop/UoE/DISS/car_ped_prediction/prediction/v8s_5class_test'
os.makedirs(output_dir, exist_ok=True)

def predict_and_save(source):
    # Run inference
    results = model.predict(source=source, conf=0.5)

    # Process each result
    for result in results:
        image_path = result.path  # full path to image
        image_name = os.path.basename(image_path)
        csv_name = os.path.splitext(image_name)[0] + '.csv'
        # print(csv_name)   #debug
        csv_path = os.path.join(output_dir, csv_name)

        boxes = result.boxes.xyxy.cpu().numpy()        # [N, 4] - xmin, ymin, xmax, ymax
        scores = result.boxes.conf.cpu().numpy()       # [N,]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[c] for c in class_ids]

        # Filter by per-class thresholds
        filtered_data = []
        for box, score, cls_id, cls_name in zip(boxes, scores, class_ids, class_names):
            if cls_name in class_thresholds and score >= class_thresholds[cls_name]:
                filtered_data.append({
                    'xmin': box[0], 'ymin': box[1],
                    'xmax': box[2], 'ymax': box[3],
                    'confidence': score,
                    'class_id': cls_id,
                    'class_label': cls_name,
                    'image_name': image_name
                })

        # Save only if we have valid predictions
        if filtered_data:
            df = pd.DataFrame(filtered_data)
            df.to_csv(csv_path, index=False)
            print(f"Saved predictions to {csv_path}")
        else:
            print(f"No predictions passed thresholds for {image_name}")

# Predict bounding boxes
# Define the image folder
image_folder_train = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/train'
image_folder_val = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/val'
image_folder_test = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/test'

# List all image files (filtering for .jpg, .png, etc.)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Gather image paths from both folders
image_paths = []
# for folder in [image_folder_train, image_folder_val]:
for folder in [image_folder_test]:
    image_paths.extend(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    )

# Run predictions with progress bar
for image_path in tqdm(image_paths, desc="Predicting"):
    predict_and_save(image_path)