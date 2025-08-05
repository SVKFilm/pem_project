import shutil
import os
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

img_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training/image_2'
label_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/labels/training/label_2_toTrain'
out_img_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/images/training'
out_label_dir = 'C:/Users/User/Desktop/UoE/DISS/Datasets/full_dataset/labels/training'

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(f"{out_img_dir}/{split}", exist_ok=True)
    os.makedirs(f"{out_label_dir}/{split}", exist_ok=True)

# Collect and shuffle image filenames
image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
random.shuffle(image_files)

# Split into train/val/test
n = len(image_files)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

splits = {
    'train': train_files,
    'val': val_files,
    'test': test_files
}

# Copy images and labels with progress bar
for split_name, file_list in splits.items():
    print(f"Processing {split_name} set...")
    for fname in tqdm(file_list, desc=f"Copying {split_name}"):
        # Copy image
        shutil.copy(os.path.join(img_dir, fname), f"{out_img_dir}/{split_name}/{fname}")

        # Copy label
        label_name = fname.replace('.png', '.txt')
        shutil.copy(os.path.join(label_dir, label_name), f"{out_label_dir}/{split_name}/{label_name}")
