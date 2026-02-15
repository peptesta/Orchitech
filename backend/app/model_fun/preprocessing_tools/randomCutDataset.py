import os
import random

PATH_TO_DATASET = '../dataset/raw/mini/'
MAX_IMAGES = 30

def cutDataset(directory, maxImages=100):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            images = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            if len(images) > maxImages:
                images_to_delete = random.sample(images, len(images) - maxImages)
                for image in images_to_delete:
                    os.remove(os.path.join(dir_path, image))
                print(f"Balanced {dir_path}: Removed {len(images_to_delete)} images")

if __name__ == "__main__":
    cutDataset(PATH_TO_DATASET, MAX_IMAGES)