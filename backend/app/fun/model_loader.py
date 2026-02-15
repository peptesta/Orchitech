# app/model_loader.py

import os
import torch
from dotenv import dotenv_values
from typing import List, Any, Dict

try:
    from app.model_fun.inference import loadModel, loadDevice
except ImportError as e:
    print(f"Error importing model_fun dependencies: {e}")
    raise

config = dotenv_values(".env")

GPU_AVAILABLE = torch.cuda.is_available() and config.get("GPU", "False").lower() in ('true', '1', 't')
SIXCLASS_MODEL_PATH = config.get("SIXCLASS_MODEL_PATH", "app/models/detection_models/5Class/model.pt")
ONEVSALL_MODEL_DIR = config.get("1VSALL_MODEL_DIR", "app/models/detection_models/1vall")
CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']


# --- FUNZIONE DI CARICAMENTO MODELLI ---

def load_resources() -> Dict[str, Any]:
    """
    Carica il device, il modello 5-Class e i modelli 1-vs-All.

    Ritorna un dizionario contenente:
    - 'device': il device di PyTorch
    - 'model': il modello 5-Class
    - 'onevall_models': una lista dei modelli 1-vs-All
    """
    
    # Inizializza il device (dipende dalla configurazione)
    device = loadDevice(forceCpu=not GPU_AVAILABLE)
    
    model = None
    onevall_models = []
    
    print(f"--- SERVER STARTUP: Loading models... ---", flush=True)

    # 1. Load 5-Class Model
    try:
        current_model_path = SIXCLASS_MODEL_PATH
        if not os.path.exists(current_model_path):
            fallback = "app/models/model.pt"
            if os.path.exists(fallback):
                print(f"Warning: Configured path not found. Using fallback: {fallback}", flush=True)
                current_model_path = fallback
            else:
                raise FileNotFoundError(f"Main model not found at {SIXCLASS_MODEL_PATH}")
        
        model = loadModel(current_model_path, len(CLASS_NAMES), device)
        print("Success: six Class Model loaded.", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load 5-Class Model. {e}", flush=True)
        # In un modulo di utilità, è meglio sollevare l'errore per fermare l'avvio del server
        raise RuntimeError(f"Failed to load 5-Class Model: {e}")

    # 2. Load 1-vs-All Models
    try:
        if not os.path.exists(ONEVSALL_MODEL_DIR):
            print(f"Warning: 1-vs-All directory not found at {ONEVSALL_MODEL_DIR}", flush=True)
        else:
            loaded_ovr = []
            for class_name in CLASS_NAMES:
                model_file = os.path.join(ONEVSALL_MODEL_DIR, class_name, 'model.pt')
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Missing 1-vs-All model for: {class_name}")
                
                # Load binary model (output size 2)
                ovr_model = loadModel(model_file, 2, device)
                loaded_ovr.append(ovr_model)
            
            onevall_models = loaded_ovr
            print("Success: 1-vs-All Models loaded.", flush=True)
    except Exception as e:
        print(f"Error: Failed to load 1-vs-All models. {e}", flush=True)
        # Anche qui, solleviamo l'errore per un avvio pulito
        raise RuntimeError(f"Failed to load 1-vs-All models: {e}")

    # 3. Ritorna i modelli caricati e il device
    return {
        "device": device,
        "model": model,
        "onevall_models": onevall_models
    }