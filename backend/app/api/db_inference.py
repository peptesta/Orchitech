import os
import io
import base64
import torch
import time
import threading
import gc
from flask import Blueprint, request, jsonify
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Configurazione
MAX_WORKERS = 4
db_inference_bp = Blueprint('db_inference', __name__)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Import del modello con gestione errore migliorata
try:
    from app.cropping_fun.fasterrcnn_crop import crop
except ImportError:
    print("ERRORE CRITICO: Verificare il percorso di app.cropping_fun.fasterrcnn_crop")

def process_image_logic(file_storage):
    """
    Logica per singola immagine con pulizia aggressiva della memoria.
    """
    start_time = time.time()
    thread_id = threading.get_ident()
    filename = file_storage.filename
    
    # Inizializzazione variabili per cleanup sicuro nel finally
    img = None
    preview_img = None
    buffered = None
    
    print(f"--- [THREAD {thread_id}] Inizio: {filename} ---", flush=True)
    
    try:
        # Caricamento immagine
        img = Image.open(file_storage.stream).convert("RGB")
        
        # 1. ESECUZIONE INFERENZA
        # Assicuriamoci che i tensori non restino appesi
        with torch.no_grad():
            _, all_boxes, all_scores = crop(img)
        
        # 2. OTTIMIZZAZIONE ANTEPRIMA (Resize)
        preview_img = img.copy()
        preview_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # 3. CODIFICA BASE64
        buffered = io.BytesIO()
        preview_img.save(buffered, format="JPEG", quality=75, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        total_time = time.time() - start_time
        print(f"+++ [THREAD {thread_id}] FINITO: {filename} in {total_time:.3f}s", flush=True)
        
        return {
            "image_b64": f"data:image/jpeg;base64,{img_str}",
            "boxes": all_boxes,
            "scores": all_scores,
            "count": len(all_scores),
            "error": False
        }

    except Exception as e:
        print(f"!!! [THREAD {thread_id}] ERRORE su {filename}: {str(e)}")
        return {"error": True, "message": str(e)}

    finally:
        # --- LOGICA DI LIBERAZIONE MEMORIA ---
        # Rimuoviamo i riferimenti agli oggetti Pillow
        if img:
            img.close()
            del img
        if preview_img:
            preview_img.close()
            del preview_img
        if buffered:
            buffered.close()
            del buffered
        
        # Forza il Garbage Collector di Python
        gc.collect()
        
        # Svuota la cache CUDA se presente
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@db_inference_bp.route('/dbinference', methods=['POST'])
def run_inference():
    if 'images' not in request.files:
        return jsonify({"error": "Nessuna chiave 'images' nella richiesta"}), 400

    files = request.files.getlist('images')
    num_files = len(files)
    
    print(f"\n[SERVER] Ricevuta batch di {num_files} immagini.", flush=True)
    global_start = time.time()

    # Esecuzione parallela
    results = list(executor.map(process_image_logic, files))

    # Costruzione risposta finale
    final_response = {
        "images": [],
        "bounding_box": [],
        "scores": [],
        "bb_count": []
    }

    for res in results:
        if res.get("error"):
            final_response["images"].append("")
            final_response["bounding_box"].append([])
            final_response["scores"].append([])
            final_response["bb_count"].append(0)
        else:
            final_response["images"].append(res["image_b64"])
            final_response["bounding_box"].append(res["boxes"])
            final_response["scores"].append(res["scores"])
            final_response["bb_count"].append(res["count"])

    global_duration = time.time() - global_start
    print(f"[SERVER] Batch completato in {global_duration:.3f}s. Media: {global_duration/num_files:.3f}s/img\n")

    return jsonify(final_response)