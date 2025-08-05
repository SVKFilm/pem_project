import os

# Convert labels to YOLO format

# Settings
input_labels_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/labels/training/label_2'
output_labels_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/labels/training/label_2_toTrain'
image_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/image_2'
# Version1
# classes_to_keep = ['Car', 'Pedestrian']
# class_map = {'Car': 0, 'Pedestrian': 1}  # YOLO class IDs
# Version2
classes_to_keep = ['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck']
class_map = {'Car': 0, 'Pedestrian': 1, 'Van': 2, 'Cyclist': 3, 'Truck': 4}  # YOLO class IDs

os.makedirs(output_labels_dir, exist_ok=True)

for label_file in os.listdir(input_labels_dir):
    if not label_file.endswith('.txt'):
        print("Skipped file:", label_file)
        continue

    input_path = os.path.join(input_labels_dir, label_file)
    output_path = os.path.join(output_labels_dir, label_file)

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] not in class_map:
                continue

            class_id = class_map[parts[0]]
            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])

            # Convert to YOLO format (normalized x_center, y_center, width, height)
            img_filename = label_file.replace('.txt', '.png')
            img_path = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_path):
                continue

            from PIL import Image
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")