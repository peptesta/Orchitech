# --- IMPORTS ---
import io
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from PIL import Image
from dotenv import dotenv_values
import traceback

# --- LOCAL IMPORTS ---
from app import model_state                                 
from app.model_fun.preprocess_data import getTransforms     
from app.fun.tta_logic import (
    perform_inference,                                      
    createAugmentedImages,                                  
    aggregate_tta_results                                             
)
from app.fun.explainability_fun import (
    generate_explanation,                                              
    image_to_base64                                                               
)

# Initialize Blueprint
inference_bp = Blueprint('inference', __name__)

tta_mode = False  
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

@inference_bp.route('/inference', methods=['POST'])
def run_inference_endpoint():
    model, onevall_models, device, CLASS_NAMES = model_state.get_models()

    if model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    # --- 1. INIZIALIZZAZIONE VARIABILI (Evita UnboundLocalError) ---
    tensor_original = None
    image_cropped = None
    tensor_cropped = None 
    crop_error = None
    crop_boxes = []
    crop_scores = []

    try:
        image_file = request.files['image']
        original_image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        model_strategy = request.form.get("model_strategy", "standard")
        crop_mode = request.form.get("crop_mode", "integrated")     
        explain_method = request.form.get("explain_method", "none")

        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor_original = transform_pipeline(original_image).unsqueeze(0).to(device)

        # --- 2. GESTIONE CROP (Unpacking Corretto) ---
        if crop_mode in ['external', 'compare'] and HAS_EXTERNAL_CROP:
            try:
                # Spacchettiamo la tupla restituita dal nuovo modulo
                image_cropped, crop_boxes, crop_scores = crop(original_image.copy())
                
                if image_cropped is not None:
                    tensor_cropped = transform_pipeline(image_cropped).unsqueeze(0).to(device)
            except Exception as e:
                crop_error = f"Cropping failed: {str(e)}"
                print(f"DEBUG Error: {crop_error}")

        # --- 3. LOGICA DI ASSEGNAZIONE SORGENTE ---
        primary_tensor = tensor_original
        secondary_tensor = None
        
        if crop_mode == "external" and image_cropped is not None:
            if tensor_cropped is not None:
                 primary_tensor = tensor_cropped
        
        elif crop_mode == "compare" and image_cropped is not None:
            if tensor_cropped is not None:
                secondary_tensor = tensor_cropped

        # --- 4. INFERENZA ---
        # Eseguiamo l'inferenza usando primary_tensor (che è o l'originale o il crop)
        prim_idx, prim_conf, prim_probs, prim_err = perform_inference(
            model, onevall_models, primary_tensor, model_strategy, device
        )
        
        sec_idx, sec_conf, sec_probs, sec_err = -1, 0.0, None, None
        if secondary_tensor is not None:
            sec_idx, sec_conf, sec_probs, sec_err = perform_inference(
                model, onevall_models, secondary_tensor, model_strategy, device
            )

        # --- 5. EXPLAINABILITY (XAI) ---
        # Usiamo sempre i tensori definiti sopra per coerenza
        primary_xai_b64 = generate_explanation(model, primary_tensor, prim_idx, explain_method)
        secondary_xai_b64 = None
        if secondary_tensor is not None:
            secondary_xai_b64 = generate_explanation(model, secondary_tensor, sec_idx, explain_method)

        # --- 6. COSTRUZIONE RISPOSTA ---
        return jsonify({
            'model_strategy': model_strategy,
            'crop_mode': crop_mode,
            'image': image_to_base64(original_image),
            'image_cropped': image_to_base64(image_cropped) if image_cropped is not None else None, 
            'predicted_class': CLASS_NAMES[prim_idx] if prim_idx != -1 else "Unknown",
            'confidence': prim_conf,
            'all_classes_probs': prim_probs,
            'error': prim_err or crop_error,
            'predicted_external_class': CLASS_NAMES[sec_idx] if sec_idx != -1 else "Unknown",
            'confidence_external': sec_conf,
            'all_classes_probs_external': sec_probs,
            'crop_details': {'boxes': crop_boxes, 'scores': crop_scores},
            'integrated_gradients': primary_xai_b64 if explain_method == 'integrated_gradients' else None,
            'integrated_gradients_external': secondary_xai_b64 if explain_method == 'integrated_gradients' else None
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500