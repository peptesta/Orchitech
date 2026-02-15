from PIL import Image
from typing import List, Tuple, Optional, Any
import numpy as np
import collections
import torch
from app.model_fun.inference import getValues6ClassModel, getValues1vsAllModel
CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']

# --- TTA AUGMENTATION FUNCTION DEFINITION ---

def createAugmentedImages(image: Image.Image) -> List[Image.Image]:
    """
    Generates a list of augmented views of a single image for Test Time Augmentation (TTA).
    Includes rotations (0, 90, 180, 270 degrees) and horizontal flips.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    rotations = [
        image,                                # 0 degrees (Original)
        image.rotate(90, expand=True),       # 90 degrees
        image.rotate(180, expand=True),      # 180 degrees
        image.rotate(270, expand=True)       # 270 degrees
    ]
    
    augmented_views = []
    
    for img_view in rotations:
        # 1. Add the view without flip
        augmented_views.append(img_view)
        
        # 2. Add the horizontal flip
        img_flipped = img_view.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_views.append(img_flipped)

    return augmented_views

# --- TTA AGGREGATION FUNCTION (Modified for Percentage) ---

def strategy_mean(probs_array: np.ndarray) -> Tuple[int, float]:
    """Simple arithmetic mean across all views."""
    all_means = np.mean(probs_array, axis=0)
    final_idx = np.argmax(all_means)
    return int(final_idx), float(all_means[final_idx])

def strategy_trimmed_mean(probs_array: np.ndarray) -> Tuple[int, float]:
    if probs_array.shape[0] <= 2: return strategy_mean(probs_array)
    sorted_probs = np.sort(probs_array, axis=0)
    trimmed = sorted_probs[1:-1, :]
    final_means = np.mean(trimmed, axis=0)
    final_idx = np.argmax(final_means)
    return int(final_idx), float(final_means[final_idx])

def strategy_max_confidence(probs_array: np.ndarray) -> Tuple[int, float]:
    """Pick the single highest confidence value found in any view."""
    # Find the coordinates (view, class) of the maximum value
    view_idx, class_idx = np.unravel_index(np.argmax(probs_array), probs_array.shape)
    return int(class_idx), float(probs_array[view_idx, class_idx])

def strategy_hybrid_vote(probs_array: np.ndarray) -> Tuple[int, float]:
    """Your custom hybrid logic with frequency voting and confidence tie-breaker."""
    CONFIDENCE_TOLERANCE = 5.0
    vote_counts = collections.defaultdict(int)
    conf_sums = collections.defaultdict(float)
    
    for view_probs in probs_array:
        max_p = np.max(view_probs)
        candidates = np.where(view_probs >= (max_p - CONFIDENCE_TOLERANCE))[0]
        for idx in candidates:
            vote_counts[idx] += 1
            conf_sums[idx] += view_probs[idx]
            
    max_votes = max(vote_counts.values())
    top_candidates = [i for i, v in vote_counts.items() if v == max_votes]
    
    if len(top_candidates) == 1:
        best_idx = top_candidates[0]
    else:
        # Tie-breaker: mean confidence of tied classes (only where they were candidates)
        best_idx = max(top_candidates, key=lambda i: conf_sums[i] / vote_counts[i])
        
    return int(best_idx), float(conf_sums[best_idx] / vote_counts[best_idx])

def strategy_borda_count(probs_array: np.ndarray) -> Tuple[int, float]:
    """Aggregation based on ranking rather than raw values."""
    num_classes = probs_array.shape[1]
    borda_scores = np.zeros(num_classes)
    for view_probs in probs_array:
        ranks = np.argsort(view_probs) # Sorts low to high
        for score, class_idx in enumerate(ranks):
            borda_scores[class_idx] += score
    
    best_idx = np.argmax(borda_scores)
    # Return index and the mean prob for that class as confidence
    return int(best_idx), float(np.mean(probs_array[:, best_idx]))

# --- MAIN TTA AGGREGATION FUNCTION ---

def aggregate_tta_results(
    model, 
    onevall_models, 
    tta_images: List[Image.Image], 
    model_strategy: str, 
    transform_pipeline, 
    device: str,
    aggregation_func=strategy_hybrid_vote  # <-- Default strategy
) -> Tuple[int, float, List[float], Optional[str]]:
    
    all_probs_list = []
    
    # 1. Collect predictions
    for i, tta_img in enumerate(tta_images):
        try:
            tensor_tta = transform_pipeline(tta_img).unsqueeze(0).to(device)
            _, _, probs, err = perform_inference(model, onevall_models, tensor_tta, model_strategy, device)
            
            if err: continue
            all_probs_list.append(probs)
        except Exception as e:
            print(f"DEBUG: Error view {i+1}: {e}", flush=True)

    if not all_probs_list:
        return -1, 0.0, None, "TTA failed: No successful inference."

    # 2. Convert to Numpy
    probs_array = np.array(all_probs_list)
    
    # 3. Apply the selected strategy
    final_idx, final_conf = aggregation_func(probs_array)
    
    # Calculate overall means for the UI/Response
    all_means = np.mean(probs_array, axis=0).tolist()

    print(f"DEBUG: TTA Finished using {aggregation_func.__name__}")
    print(f"DEBUG: Predicted: {final_idx} ({CLASS_NAMES[final_idx]}) | Conf: {final_conf:.2f}%", flush=True)

    return final_idx, final_conf, all_means, None

# --- STANDARD INFERENCE FUNCTION DEFINITION ---

def perform_inference(model, onevall_models, tensor: torch.Tensor, strategy: str, device) -> Tuple[int, float, Any, Optional[str]]:
    """
    Executes the prediction based on the selected strategy.
    
    Returns:
        (predicted_index, confidence [0-100], probability_dict [0-100], error_message)
        NOTE: Assuming getValues...Model already returns values in the desired scale.
    """
    if tensor is None:
        return -1, 0.0, None, "No image tensor provided."
    
    try:
        if strategy == "standard":
            idx, conf, probs = getValues6ClassModel(model, tensor, device)
            # Assuming conf/probs are already scaled by getValues6ClassModel
            print(f"DEBUG: Standard 5-Class Result -> Class: {idx}, Conf: {conf:.4f}, Probs: {probs}", flush=True)
            return idx, conf, probs, None
        
        elif strategy == "1vsall":
            idx, conf, probs = getValues1vsAllModel(onevall_models, tensor, device)
            # Assuming conf/probs are already scaled by getValues1vsAllModel
            print(f"DEBUG: 1vsAll Model Result -> Class: {idx}, Conf: {conf:.4f}, Probs: {probs}", flush=True)
            if idx == -1:
                return -1, 0.0, probs, "No class predicted with sufficient confidence."
            return idx, conf, probs, None
            
    except Exception as e:
        print(f"DEBUG: Inference Error encountered: {str(e)}", flush=True)
        return -1, 0.0, None, f"Inference Error: {str(e)}"

    return -1, 0.0, None, "Unknown Strategy"