import torch
import torchvision.transforms as T
import os
from PIL import Image
import app.model_fun.preprocessing_tools.rotation as rotation, app.model_fun.preprocessing_tools.resizing as resizing
from dotenv import dotenv_values
import shutil

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import numpy as np
import random

config = dotenv_values(".env")

# Viene utilizzato in fase di preprocessing per salvare i dati in un file salvando anche i nomi dei file
class TensorDatasetWithName(Dataset):
    def __init__(self, tensors, file_paths, classes=None):
        assert len(tensors[0]) == len(file_paths), "Mismatch between data and file paths length"
        self.tensors = tensors
        self.file_names = [os.path.basename(path) for path in file_paths]
        self.classes = classes if classes is not None else []

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (self.file_names[index],)

    def __len__(self):
        return len(self.tensors[0])

# Serve a caricare le immagini da un percorso specifico
# Estende ImageFolder per poter avere il nome del file
# Le immagini devono essere in cartelle con la seguente struttura:
# path
# ├── class1
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── class2
# │   ├── img1.jpg
# │   └── ...
# └── ...
class ImageFolderWithName(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithName, self).__getitem__(index)
        path, _ = self.samples[index]
        file_name = os.path.basename(path)
        return original_tuple[0], original_tuple[1], file_name

# Il dataset viene caricato da una singola cartella
# Le immagini devono essere direttamente nella cartella
# path
# ├── img1.jpg
# ├── img2.jpg
# └── ...
class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Carica immagine come RGB
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)


def getDatasetFromFile(file_path):
    processed_data_dict = torch.load(file_path, weights_only=False)
    processed_dataset = processed_data_dict['data']
    return processed_dataset


def preprocessImage(image, width, height):
    if rotation.is_landscape(image):
        image = rotation.rotate_image(image)
    if not resizing.is_correct_image_size(image, width, height):
        image = resizing.resize_image(image, width, height)
    return image

# Augmenta le immagini in data_path e le salva in output_path
# max_images_per_class: numero massimo di immagini per classe, se raggiunto il ciclo si interrompe. In questo modo si bilanciano le classi
# Le immagini non vengono normalizzate in questa fase
def augmentDataPath(data_path, output_path, max_images_per_class, width, height, augmentation_functions):
    class_count = 0
    for function in augmentation_functions:
        print(f'Processing {function.__name__}...')
        for idx, foto in enumerate(os.listdir(data_path)):
            if class_count >= max_images_per_class and function != rotation.identity: # In questo modo l'identità viene sempre applicata
                print(f'Processed {class_count} images')
                return 
            img = Image.open(os.path.join(data_path, foto))
            img = preprocessImage(img, width, height) # Ritaglia l'immagine
            original_name = os.path.splitext(foto)[0]
            output_file_path = os.path.join(output_path, f'{function.__name__}_{original_name}_{idx}.jpg') # Al file viene aggiunto l'indice e il nome della funzione
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            img = function(img) # Applica la funzione di trasformazione
            img.save(output_file_path)
            class_count += 1
            if class_count % 10 == 0:
                print(f'Processed {class_count} images')
    print(f'Processed {class_count} images')


# Divide le immagini in data_path in train, validation e test
# train_size: percentuale di immagini da mettere nel training set
# validation_size: percentuale di immagini da mettere nel validation set
# Le immagini vengono copiate nelle cartelle corrispondenti
def SplitData(data_path, output_path, class_name, train_size, validation_size):
    train_path = os.path.join(output_path, 'train', class_name)
    validation_path = os.path.join(output_path, 'validation', class_name)
    test_path = os.path.join(output_path, 'test', class_name)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    images = os.listdir(os.path.join(data_path, class_name))
    random.shuffle(images)

    for idx, foto in enumerate(images):
        if idx < train_size*len(os.listdir(os.path.join(data_path, class_name))):
            shutil.copy(os.path.join(data_path, class_name, foto), train_path)
        elif idx < (train_size+validation_size)*len(os.listdir(os.path.join(data_path, class_name))):
            shutil.copy(os.path.join(data_path, class_name, foto), validation_path)
        else:
            shutil.copy(os.path.join(data_path, class_name, foto), test_path)

    return train_path, validation_path, test_path


# size: numero di immagini per classe
# raw_data_path: percorso delle immagini originali
# work_processed_data_path_balanced: percorso di lavoro per le immagini augmentate
# out_processed_data_path: percorso di output per le immagini augmentate
# class_names: nomi delle classi
# train_size: percentuale di immagini da mettere nel training set
# validation_size: percentuale di immagini da mettere nel validation set
# width: larghezza delle immagini
# height: altezza delle immagini
# augmentation_functions: funzioni di trasformazione da applicare alle immagini
def DynamicAugmentation(size, raw_data_path, work_processed_data_path_balanced, out_processed_data_path, class_names, train_size, validation_size, width, height, augmentation_functions):
    for class_name in class_names:
        print(f'Processing {class_name}...')
        train_path, validation_path, test_path = SplitData(raw_data_path, work_processed_data_path_balanced, class_name, train_size, validation_size)

        os.makedirs(os.path.join(out_processed_data_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(out_processed_data_path, 'validation', class_name), exist_ok=True)
        os.makedirs(os.path.join(out_processed_data_path, 'test', class_name), exist_ok=True)

        augmentDataPath(train_path, os.path.join(out_processed_data_path, 'train', class_name), size*train_size, width, height, augmentation_functions)
        augmentDataPath(validation_path, os.path.join(out_processed_data_path, 'validation', class_name), size*0, width, height, augmentation_functions)
        augmentDataPath(test_path, os.path.join(out_processed_data_path, 'test', class_name), size*0, width, height, augmentation_functions)

    return True

# Calcola il numero di immagini minimo e massimo per classe
# raw_data_path: percorso delle immagini originali
# class_names: nomi delle classi
# augmentation_functions: funzioni di trasformazione da applicare alle immagini
# Restituisce il numero minimo e massimo di immagini 
def CalculateAugmentationSize(raw_data_path, class_names, augmentation_functions):
    min_size = 1000000
    max_size = 0
    for class_name in class_names:
        class_path = os.path.join(raw_data_path, class_name)
        num_files = len(os.listdir(class_path))
        if num_files < min_size:
            min_size = num_files
        if num_files > max_size:
            max_size = num_files
    return min_size * len(augmentation_functions), max_size * len(augmentation_functions)

