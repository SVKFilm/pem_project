from ultralytics import YOLO

# Train model
# Load a pre-trained model (you can choose yolov8n.pt, yolov8s.pt, etc.)
# model = YOLO('yolov8s.pt')
model = YOLO('runs/detect/train/weights/best.pt')

# Train on custom dataset - version1
# model.train(
#     data='dataset.yaml',  # path to your dataset.yaml
#     epochs=50,
#     imgsz=640,
#     patience=5,
#     verbose=True
# )

# Train on custom dataset - version2
model.train(
    data='dataset2.yaml',  # path to your dataset.yaml
    epochs=15,
    imgsz=640,
    patience=3,
    verbose=True
)

# Validation
# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run validation
# metrics = model.val(data='dataset.yaml')  # Returns dictionary of metrics - version1
metrics = model.val(data='dataset2.yaml')  # Returns dictionary of metrics - version2
print("Validation metrics:")
print(metrics)  # Optional: See mAP, precision, recall, etc.

