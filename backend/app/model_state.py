model = None
onevall_models = None
device = None
CLASS_NAMES = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]

def load_and_set_models(resources):
    """Assegna le risorse (modelli e device) caricate dal loader."""
    global model, onevall_models, device
    model = resources.get('model')
    onevall_models = resources.get('onevall_models', [])
    device = resources.get('device')

def get_models():
    """Ritorna i modelli e il device per l'uso negli endpoint."""
    global model, onevall_models, device, CLASS_NAMES
    return model, onevall_models, device, CLASS_NAMES