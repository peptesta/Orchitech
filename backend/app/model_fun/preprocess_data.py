import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from app.model_fun.preprocessing_tools.AugmentedDataset import AugmentedDataset
from dotenv import dotenv_values
import sys
import os
from sklearn.model_selection import train_test_split
import numpy as np

import app.model_fun.preprocessing_tools.rotation as rotation
import app.model_fun.preprocessing_tools.normalization as normalization
import app.model_fun.preprocessing_tools.resizing as resizing
from app.model_fun.preprocessing_tools.dataset_tool import TensorDatasetWithName, ImageFolderWithName, DynamicAugmentation, CalculateAugmentationSize
from app.model_fun.preprocessing_tools.reproducibility import set_seed, seed_worker


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = dotenv_values(".env")

# Servono solo per mostrare qualche immagine alla fine della fase di preprocessing, giusto per essere sicuri che sia tutto ok
def show_n_images(dataset, n=5):
    for i, (image, label, name) in enumerate(dataset):
        print(f"Image {i+1} - Label: {label} - Name: {name}")
        if i >= n:
            break
        show_image(image)

def show_image(image):
    image = image.permute(1, 2, 0)
    img = Image.fromarray((image.numpy() * 255).astype('uint8'))
    img.show()

# Questa funzione permette di applicare ulteriori trasformazioni alle immagini
# In questo caso le immagini vengono convertite in tensori e normalizzate
def getTransforms(width, height, shouldNormalize, mean, std):
    transforms_list = [T.Resize((height, width)), T.ToTensor()]
    if shouldNormalize:
        mean = normalization.get_mean()
        std = normalization.get_std()
        transforms_list.append(normalization.NormalizeImageTransform(mean, std))
    return T.Compose(transforms_list)

def getDataset(path, width, height, shouldNormalize, mean=None, std=None):
    # ImageFolderWithName è una classe che estende ImageFolder, le trasformazioni vengono applicate nell'ordine in cui vengono definite
    dataset = ImageFolderWithName(path, transform=getTransforms(width, height, shouldNormalize, mean, std), allow_empty=True)
    return dataset


def preprocess_data_to_tensor(loader):
    processed_image = []
    processed_label = []
    processed_filename = []

    total_images = 0
    for image, label, filename in loader:
        total_images += len(image)
        logging.info(f"Processed {total_images} images")
        processed_image.append(image)
        processed_label.append(label)
        processed_filename.extend(filename)

    processed_image = torch.cat(processed_image)
    processed_label = torch.cat(processed_label)
    processed_filename = np.array(processed_filename, dtype=str)

    logging.info(f"Processed {total_images} images")
    logging.info(f"Processed {len(processed_image)} images")
    for i in range(len(processed_image)):
        if processed_image[i].shape[0] != 3:
            logging.error(f"Image {processed_filename[i]} has shape {processed_image[i].shape}")

    return TensorDatasetWithName((processed_image, processed_label), processed_filename, classes=loader.dataset.classes)



def preprocessData(rawDataPath, outDataPath, width, height, batchSize, numWorkers, shouldNormalize, seed=42, mean=None, std=None):
    set_seed(seed)
    dataset = getDataset(rawDataPath, width, height, shouldNormalize, mean, std)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True, worker_init_fn=seed_worker)
    processed_dataset = preprocess_data_to_tensor(loader)
    os.makedirs(os.path.dirname(outDataPath), exist_ok=True)
    torch.save({'data': processed_dataset}, outDataPath)
    logging.info(f"Dataset saved at {outDataPath}")
    show_n_images(processed_dataset, 2) # Vengono visualizzate 2 immagini per dataset
    return processed_dataset
    
# Questa funzione calcola la media e la deviazione standard delle immagini del dataset
def fresh_normalization(rawDataPath, width, height, batchSize, numWorkers, seed):
    set_seed(seed)
    dataset = ImageFolderWithName(rawDataPath, transform=T.ToTensor())
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True, worker_init_fn=seed_worker)
    return normalization.calculate_fresh_mean_std(loader)

def DynamicAugmentationWrapper(RAW_DATA_PATH, WORK_PROCESSED_DATA_PATH_BALANCED, OUT_PROCESSED_DATA_PATH, CLASS_NAMES, TRAIN_SIZE, VALIDATION_SIZE, WIDTH, HEIGHT, DYNAMIC_AUGMENTATION, augmentation_functions):
    minSize, maxSize = CalculateAugmentationSize(RAW_DATA_PATH, CLASS_NAMES, augmentation_functions)
    size = minSize if DYNAMIC_AUGMENTATION else maxSize # se DYNAMIC_AUGMENTATION = True allora il dataset verrà bilanciato, la funzione DynamicAugmentation necessita di sapere il numero minimo di immagini. Se gli viene passato un numero alto, allora il dataset verrà augmentato totalmente
    print(f'Augmentation size: {size}')
    DynamicAugmentation(size, RAW_DATA_PATH, WORK_PROCESSED_DATA_PATH_BALANCED, OUT_PROCESSED_DATA_PATH, CLASS_NAMES, TRAIN_SIZE, VALIDATION_SIZE, WIDTH, HEIGHT, augmentation_functions)

