import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from captum.attr import Occlusion
from captum.attr._utils.visualization import _normalize_attr
import app.model_fun.preprocessing_tools.normalization as normalization


config = dotenv_values(".env")

# SLIDING_WINDOW_SIZE = int(config['SLIDING_WINDOW_SIZE'])
# STRIDE = int(config['SLIDING_WINDOW_STRIDE'])

# CLASS_NAMES = eval(config['CLASS_NAMES'])
# CLASS_SIZE = len(CLASS_NAMES)

def display_occlusion(fig, ax, image, model, label, occlusion_text, slidingWindowSize, stride, classNames):
    occlusion = Occlusion(model)
    
    global_attribution = occlusion.attribute(
        image.unsqueeze(0),
        sliding_window_shapes=(3, slidingWindowSize, slidingWindowSize),
        strides=(3, stride, stride)
    )

    class_map = get_attribution_class_map(global_attribution)
    label_attributions = global_attribution[label, :, :, :]
    
    label_attributions = label_attributions.squeeze(0).cpu().detach().numpy()
    label_attributions = np.transpose(label_attributions, (1, 2, 0))

    image_np = normalization.denormalize_image(image, normalization.get_mean(), normalization.get_std()).cpu().numpy().transpose(1, 2, 0)

    norm_attr = _normalize_attr(label_attributions, sign="all", outlier_perc=2, reduction_axis=2)
    cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    vmin, vmax = -1, 1

    ax.clear()
    ax.imshow(np.mean(image_np, axis=2), cmap="gray")
    ax.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5 )
    ax.set_title('Occlusion attributions\nfor class ' + classNames[label], fontsize=8)
    ax.axis('off')

    add_occlusion_hover_text(fig, ax, class_map, global_attribution, occlusion_text, classNames)


def get_attribution_class_map(global_attribution):
    global_attribution = global_attribution.squeeze(0).cpu().detach().numpy()
    class_map = np.argmax(global_attribution, axis=0)
    # print(f'Class map shape: {class_map.shape}')
    # for i in range(class_map.shape[0]):
    #     print(f'Row {i}', end='')
    #     for j in range(0, class_map.shape[1], STRIDE):
    #         print(f'\n', end='')
    #         for k in range(0, class_map.shape[2], STRIDE):
    #             print(f'{class_map[i, j, k]} ', end='')
    #     print('\n')
    # print('\n')
    return class_map

def add_occlusion_hover_text(fig, ax, class_map, global_attribution, occlusion_text, classNames):
    def update_occlusion_text(x, y):
        if 0 <= x < class_map.shape[2] and 0 <= y < class_map.shape[1]:
            class_id = int(class_map[0, y, x])
            # occlusion_text.set_text(f'Class: {CLASS_NAMES[class_id]}')
            actual_class_vector = global_attribution[:, 0, y, x].cpu().detach().numpy()
            ordered_classes = np.argsort(actual_class_vector)[::-1]
            occlusion_text_str = ''
            for i in range(len(classNames)):
                occlusion_text_str += f'{classNames[ordered_classes[i]]}: {actual_class_vector[ordered_classes[i]]:.2f}\n'
            occlusion_text.set_text(occlusion_text_str)

        else:
            occlusion_text.set_text('')

    def on_hover(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            update_occlusion_text(x, y)

    fig.canvas.mpl_connect('motion_notify_event', on_hover)


def get_occlusion_attribution_list(image, occlusion, slidingWindowSize, stride, classNames):
    attribution_list = []
    for target_class in range(len(classNames)):
        attributions = occlusion.attribute(
            image.unsqueeze(0),
            target=target_class,
            sliding_window_shapes=(3, slidingWindowSize, slidingWindowSize),
            strides=(3, stride, stride)
        )
        attribution_list.append(attributions.squeeze(0).cpu().detach().numpy())
    return attribution_list


def display_occlusion_for_save(ax, image, model, label, slidingWindowSize, stride, classNames):
    occlusion = Occlusion(model)
    
    attributions = occlusion.attribute(
        image.unsqueeze(0),
        target=label,
        sliding_window_shapes=(3, slidingWindowSize, slidingWindowSize),
        strides=(3, stride, stride)
    )
    print(label)
    
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))

    image_np = normalization.denormalize_image(image, normalization.get_mean(), normalization.get_std()).cpu().numpy().transpose(1, 2, 0)

    norm_attr = _normalize_attr(attributions, sign="all", outlier_perc=2, reduction_axis=2)
    cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    vmin, vmax = -1, 1

    ax.clear()
    im = ax.imshow(np.mean(image_np, axis=2), cmap="gray")
    im_attr = ax.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)
    ax.set_title(f'Occlusion attributions\nfor class {classNames[label]}')
    ax.axis('off')

    # Aggiungi una barra laterale
    if not hasattr(ax, 'colorbar') or ax.colorbar is None:
        cbar = plt.colorbar(im_attr, ax=ax)
        ax.colorbar = cbar


