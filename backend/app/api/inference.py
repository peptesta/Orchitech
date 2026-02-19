# --- IMPORTS ---
import io
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from PIL import Image
from dotenv import dotenv_values
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64

# --- LOCAL IMPORTS ---
from app import model_state                                 
from app.model_fun.preprocess_data import getTransforms     
from app.fun.tta_logic import perform_inference
from app.fun.explainability_fun import (
    generate_explanation,                                              
    image_to_base64                                                               
)

# Initialize Blueprint
inference_bp = Blueprint('inference', __name__)

config = dotenv_values(".env")

WIDTH = int(config.get("WIDTH", 256))
HEIGHT = int(config.get("HEIGHT", 512))
MEAN = [float(x) for x in config.get("MEAN", '0.5414286851882935 0.5396731495857239 0.3529253602027893').split()]
STD = [float(x) for x in config.get("STD", '0.2102500945329666 0.23136012256145477 0.19928686320781708').split()]

try:
    from app.cropping_fun.fasterrcnn_crop import crop
    HAS_EXTERNAL_CROP = True
except ImportError:
    HAS_EXTERNAL_CROP = False

def process_single_image(image_data, model, onevall_models, device, CLASS_NAMES, 
                        transform_pipeline, model_strategy, crop_mode, explain_method):
    """
    Process a single image through the inference pipeline.
    Returns a dictionary with results.
    """
    try:
        # Handle both file objects and base64/base64 strings
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            image = Image.open(io.BytesIO(image_data.read())).convert('RGB')
        
        tensor_original = transform_pipeline(image).unsqueeze(0).to(device)
        
        # Cropping logic
        image_cropped = None
        tensor_cropped = None
        crop_boxes = []
        crop_scores = []
        crop_error = None
        
        if crop_mode in ['external', 'compare'] and HAS_EXTERNAL_CROP:
            try:
                image_cropped, crop_boxes, crop_scores = crop(image.copy())
                if image_cropped is not None:
                    tensor_cropped = transform_pipeline(image_cropped).unsqueeze(0).to(device)
            except Exception as e:
                crop_error = f"Cropping failed: {str(e)}"
        
        # Determine tensors to use
        primary_tensor = tensor_original
        secondary_tensor = None
        
        if crop_mode == "external" and tensor_cropped is not None:
            primary_tensor = tensor_cropped
        elif crop_mode == "compare" and tensor_cropped is not None:
            secondary_tensor = tensor_cropped
        
        # Inference
        prim_idx, prim_conf, prim_probs, prim_err = perform_inference(
            model, onevall_models, primary_tensor, model_strategy, device
        )
        
        sec_idx, sec_conf, sec_probs, sec_err = -1, 0.0, None, None
        if secondary_tensor is not None:
            sec_idx, sec_conf, sec_probs, sec_err = perform_inference(
                model, onevall_models, secondary_tensor, model_strategy, device
            )
        
        # Build result
        result = {
            'success': True,
            'model_strategy': model_strategy,
            'crop_mode': crop_mode,
            'predicted_class': CLASS_NAMES[prim_idx] if prim_idx != -1 else "Unknown",
            'confidence': prim_conf,
            'all_classes_probs': prim_probs,
            'error': prim_err or crop_error,
        }
        
        # Add comparison results if applicable
        if crop_mode == "compare":
            result.update({
                'predicted_class_cropped': CLASS_NAMES[sec_idx] if sec_idx != -1 else "Unknown",
                'confidence_cropped': sec_conf,
                'all_classes_probs_cropped': sec_probs,
            })
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@inference_bp.route('/inference', methods=['POST'])
def run_inference_endpoint():
    """Single image inference endpoint (backwards compatible)"""
    model, onevall_models, device, CLASS_NAMES = model_state.get_models()
    
    if model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        image_file = request.files['image']
        model_strategy = request.form.get("model_strategy", "standard")
        crop_mode = request.form.get("crop_mode", "integrated")
        explain_method = request.form.get("explain_method", "none")
        
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        
        result = process_single_image(
            image_file, model, onevall_models, device, CLASS_NAMES,
            transform_pipeline, model_strategy, crop_mode, explain_method
        )
        
        if not result['success']:
            return jsonify({'error': result['error']}), 500
            
        # Add image data for single request
        image_file.seek(0)
        original_image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        result['image'] = image_to_base64(original_image)
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@inference_bp.route('/inference/batch', methods=['POST'])
def run_batch_inference_endpoint():
    """
    Batch inference endpoint with threading support.
    
    Expected form data:
    - images: multiple files
    - model_strategy: str (default: "standard")
    - crop_mode: str (default: "integrated") 
    - use_smart_crop: str "true"/"false" (default: "false")
    - max_workers: int (default: 4)
    """
    model, onevall_models, device, CLASS_NAMES = model_state.get_models()
    
    if model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        # Get parameters
        model_strategy = request.form.get("model_strategy", "standard")
        crop_mode = request.form.get("crop_mode", "integrated")
        use_smart_crop = request.form.get("use_smart_crop", "false").lower() == "true"
        max_workers = int(request.form.get("max_workers", 4))
        
        # Override crop_mode if use_smart_crop is specified
        if use_smart_crop:
            crop_mode = "external"  # Use smart cropping
        else:
            crop_mode = "integrated"  # No smart cropping
        
        # Get all images
        images = request.files.getlist('images')
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        
        results = []
        errors = []
        
        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_image,
                    img, model, onevall_models, device, CLASS_NAMES,
                    transform_pipeline, model_strategy, crop_mode, "none"
                ): idx 
                for idx, img in enumerate(images)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result['success']:
                        results.append({
                            'index': idx,
                            'filename': images[idx].filename,
                            **result
                        })
                    else:
                        errors.append({
                            'index': idx,
                            'filename': images[idx].filename,
                            'error': result['error']
                        })
                except Exception as e:
                    errors.append({
                        'index': idx,
                        'filename': images[idx].filename,
                        'error': str(e)
                    })
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])
        errors.sort(key=lambda x: x['index'])
        
        return jsonify({
            'total_processed': len(results),
            'total_errors': len(errors),
            'crop_mode_used': crop_mode,
            'results': results,
            'errors': errors
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@inference_bp.route('/inference/benchmark', methods=['POST'])
def run_benchmark_inference():
    """
    Specialized endpoint for benchmark testing.
    Returns predictions in a format easy to parse for metrics calculation.
    """
    model, onevall_models, device, CLASS_NAMES = model_state.get_models()
    
    if model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        model_strategy = request.form.get("model_strategy", "standard")
        use_smart_crop = request.form.get("use_smart_crop", "false").lower() == "true"
        max_workers = int(request.form.get("max_workers", 4))
        
        crop_mode = "external" if use_smart_crop else "integrated"
        
        images = request.files.getlist('images')
        labels = request.form.getlist('labels')  # Expected class labels
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        
        predictions = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {
                executor.submit(
                    process_single_image,
                    img, model, onevall_models, device, CLASS_NAMES,
                    transform_pipeline, model_strategy, crop_mode, "none"
                ): {'idx': i, 'label': labels[i] if i < len(labels) else None, 'filename': img.filename}
                for i, img in enumerate(images)
            }
            
            for future in as_completed(future_to_info):
                info = future_to_info[future]
                try:
                    result = future.result()
                    predictions.append({
                        'true_label': info['label'],
                        'predicted_label': result.get('predicted_class') if result['success'] else None,
                        'confidence': result.get('confidence', 0),
                        'success': result['success'],
                        'error': result.get('error'),
                        'filename': info['filename']
                    })
                except Exception as e:
                    predictions.append({
                        'true_label': info['label'],
                        'predicted_label': None,
                        'success': False,
                        'error': str(e),
                        'filename': info['filename']
                    })
        
        return jsonify({
            'predictions': predictions,
            'model_strategy': model_strategy,
            'use_smart_crop': use_smart_crop
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500