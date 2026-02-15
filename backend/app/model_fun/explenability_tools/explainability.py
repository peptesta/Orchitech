import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import Normalize, LinearSegmentedColormap
from dotenv import dotenv_values
from torch.nn import functional as F
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from captum.attr._utils.visualization import _normalize_attr

import app.model_fun.preprocessing_tools.normalization as normalization
from app.model_fun.explenability_tools.occlusion import display_occlusion_for_save, display_occlusion

from collections import defaultdict

config = dotenv_values(".env")

# CLASS_NAMES = eval(config['CLASS_NAMES'])
# CLASS_SIZE = len(CLASS_NAMES)


def inference(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        values = model(image)
        _, predicted = torch.max(values, 1)
    return values, predicted

def getPercentages(values):
    return F.softmax(values, dim=1) * 100

# Permette di visualizzare l'interfaccia interattiva (vedi file test_model.py)
def showAndTestImages(dataset, model, device, classNames, slidingWindowSize, stride):
    plt.ion()
    fig, ax = plt.subplots(1, len(classNames)+1, figsize=(18, 3))
    plt.subplots_adjust(bottom=0.2)  # Spazio per il pulsante e il testo di confidenza
    model.to(device)
    ig = IntegratedGradients(model)

    class InteractiveViewer:
        def __init__(self):
            self.dataset_iter = iter(dataset)
            self.current_image = None
            self.current_label = None
            self.current_filename = None
            self.confidence_text = fig.text(0.01, 0.1, '', ha='left', va='center', fontsize=8)
            self.occlusion_text_x, self.occlusion_text_y = ax[1].get_position().x0, ax[1].get_position().y0 - 0.1
            self.occlusion_text = fig.text(self.occlusion_text_x, self.occlusion_text_y, '', ha='left', va='center', fontsize=9)

        def display_next(self, event=None):
            try:
                data = next(self.dataset_iter)
                if isinstance(data, tuple) and len(data) == 3:  # Dataset con etichette (immagine, etichetta, nomefile)
                    image, label, filename = data
                elif isinstance(data, tuple) and len(data) == 2:  # Dataset con filename ma senza etichette (immagine, nomefile)
                    image, filename = data
                    label = None
                else:  # Dataset senza etichette e senza filename
                    image, label, filename = data, None, None

                self.current_image = image.to(device)
                self.current_label = label.to(device) if label is not None else None
                self.current_filename = filename

                prediction_values, predicted = inference(model, self.current_image.unsqueeze(0), device)

                display_image(ax[0], self.current_image, predicted, self.current_label, classNames)
                for idx, _ in enumerate(classNames):
                    display_occlusion(fig, ax[idx + 1], self.current_image, model, idx, self.occlusion_text, slidingWindowSize, stride, classNames)
                displayConfidence(self.confidence_text, prediction_values, classNames)
                displayFilename(fig.text(0.95, 0.02, '', ha='right', va='center', fontsize=9),  self.current_filename)

                bg_color = 'white' if self.current_label is not None and predicted == self.current_label else 'lightcoral'
                fig.patch.set_facecolor(bg_color)

                fig.canvas.draw_idle()
            except StopIteration:
                print("Fine del dataset.")
                plt.close(fig)

    viewer = InteractiveViewer()
    viewer.display_next()

    ax_button = plt.axes([0.85, 0.05, 0.1, 0.075])  # Posizione del pulsante
    next_button = Button(ax_button, 'Next')
    next_button.on_clicked(viewer.display_next)

    plt.show(block=True)


def generateOutputImages(dataset, model, device, classNames, output_dir, slidingWindowSize, stride):
    model.to(device)
    ig = IntegratedGradients(model)
    os.makedirs(output_dir, exist_ok=True)
    class_counters = defaultdict(int)

    for idx, data in enumerate(dataset):
        # I dataset di test non sempre hanno le etichette, vanno gestiti i due casi
        # nel file dataset_tool.py vengono definiti i diversi tipi di dataset
        if isinstance(data, tuple):
            if len(data) == 3:  # Dataset con etichette
                image, label, filename = data
            elif len(data) == 2:  # Dataset senza etichette
                image, filename = data
                label = None
            else:
                image, label, filename = data, None, None
        else:
            image, label, filename = data, None, None

        if isinstance(label, torch.Tensor):
            label = label.item()

        
        image = image.to(device)
        label_dir = os.path.join(output_dir, f"{dataset.classes[label]}" if label is not None else "unlabeled")
        os.makedirs(label_dir, exist_ok=True)

        # Per numerare le immagini in modo da non sovrascriverle
        if label is not None:
            class_counters[label] += 1
            class_index = class_counters[label]
        else:
            class_counters["unlabeled"] += 1
            class_index = class_counters["unlabeled"]

        prediction_values, predicted = inference(model, image.unsqueeze(0), device)

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)
        display_image(ax[0], image, predicted, label, classNames)
        display_occlusion_for_save(ax[1], image, model, predicted if label is None else label, slidingWindowSize, stride, classNames)
        display_integrated_gradients(ax[2], image, ig, predicted if label is None else label, classNames)
        displayConfidence(fig.text(0.01, 0.1, '', ha='left', va='center', fontsize=9), prediction_values, classNames,)
        displayFilename(fig.text(0.95, 0.02, '', ha='right', va='center', fontsize=9), filename)

        bg_color = 'white' if label is not None and predicted == label else 'lightcoral'
        fig.patch.set_facecolor(bg_color)

        label_name = classNames[predicted.item()] if label is None else classNames[label] 
        output_path = os.path.join(label_dir, f"{class_index}_{filename[:-4]}.png") # Rimuove l'estensione dal nome del file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)

        print(f"Immagine salvata: {output_path}")

    print("Tutte le immagini sono state elaborate e salvate.")



def displayFilename(text, filename):
    text.set_text(f'Filename: {filename}')

def displayConfidence(text, values, classNames):
    confidence_text = 'Attributions:\n'
    percentages = getPercentages(values)
    for i, percentage in enumerate(percentages.squeeze()):
        confidence_text += f'{classNames[i]}: {percentage:.2f}%\n'
    text.set_text(confidence_text)

def display_image(ax, image, predicted, label, classNames):
    image = normalization.denormalize_image(image, normalization.get_mean(), normalization.get_std()).cpu()
    ax.clear()
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.axis('off')
    predicted_text = f'Predicted: {classNames[predicted.item()]}'
    label_text = f'Label: {classNames[label]}' if label is not None else 'Label: Undefined'
    ax.set_title(f'{predicted_text}\n{label_text}', fontsize=12)

def display_integrated_gradients(ax, image, ig, label, classNames):
    attributions = ig.attribute(image.unsqueeze(0), target=label) 
    attribution_map = np.sum(np.abs(attributions.squeeze(0).cpu().numpy()), axis=0)
    ax.clear()
    im = ax.imshow(attribution_map, cmap='hot')
    ax.set_title(f'IntegratedGradients attributions\nfor class {classNames[label]}')
    ax.axis('off')
    if not hasattr(ax, 'colorbar') or ax.colorbar is None:
        cbar = plt.colorbar(im, ax=ax)
        ax.colorbar = cbar

