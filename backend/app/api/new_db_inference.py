import os
import io
import torch
import time
import threading
import gc
from flask import Blueprint, request, jsonify
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Configuration
MAX_WORKERS = 2
new_db_inference_bp = Blueprint('new_db_inference', __name__)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

try:
    from app.cropping_fun.fasterrcnn_crop import crop
except ImportError:
    print("CRITICAL ERROR: Check app.cropping_fun.fasterrcnn_crop path")

def process_image_logic(file_storage):
    """
    Core logic for single image inference with aggressive memory cleanup.
    Base64 encoding removed to prevent memory crashes.
    """
    start_time = time.time()
    thread_id = threading.get_ident()
    filename = file_storage.filename
    
    img = None
    
    try:
        # Load image
        img = Image.open(file_storage.stream).convert("RGB")
        
        # 1. INFERENCE ONLY
        with torch.no_grad():
            _, all_boxes, all_scores = crop(img)
        
        total_time = time.time() - start_time
        print(f"+++ [THREAD {thread_id}] DONE: {filename} in {total_time:.3f}s", flush=True)
        
        return {
            "boxes": all_boxes,
            "scores": all_scores,
            "count": len(all_scores),
            "error": False
        }

    except Exception as e:
        print(f"!!! [THREAD {thread_id}] ERROR on {filename}: {str(e)}")
        return {"error": True, "message": str(e)}

    finally:
        # MEMORY CLEANUP
        if img:
            img.close()
            del img
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@new_db_inference_bp.route('/dbinference', methods=['POST'])
def run_inference():
    if 'images' not in request.files:
        return jsonify({"error": "No 'images' key found"}), 400

    files = request.files.getlist('images')
    num_files = len(files)
    
    print(f"\n[SERVER] Received batch of {num_files} images.", flush=True)
    global_start = time.time()

    # Parallel execution
    results = list(executor.map(process_image_logic, files))

    # Constructing light response (No Base64 images)
    final_response = {
        "bounding_box": [],
        "scores": [],
        "bb_count": []
    }

    for res in results:
        if res.get("error"):
            final_response["bounding_box"].append([])
            final_response["scores"].append([])
            final_response["bb_count"].append(0)
        else:
            final_response["bounding_box"].append(res["boxes"])
            final_response["scores"].append(res["scores"])
            final_response["bb_count"].append(res["count"])

    global_duration = time.time() - global_start
    print(f"[SERVER] Batch completed in {global_duration:.3f}s.\n")

    return jsonify(final_response)