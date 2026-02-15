# Usa un'immagine Python leggera ma completa
FROM python:3.10-slim

# Imposta la cartella di lavoro
WORKDIR /app

# 1. Installa le dipendenze di sistema
# NOTA: Abbiamo cambiato 'libgl1-mesa-glx' con 'libgl1' per compatibilità con Debian 12
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia il file dei requisiti
COPY requirements.txt .

# 2. Aggiorna pip
RUN pip install --upgrade pip

# 3. Installa PyTorch versione CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Installa le librerie base
RUN pip install flask python-dotenv requests

# 5. Installa le altre librerie dal file requirements.txt
# (Ignorerà torch se già installato)
RUN pip install -r requirements.txt

# Copia tutto il resto del codice
COPY . .

# Flask di default ascolta su 5000
EXPOSE 5000

# Comando di default
CMD ["python", "-m", "app.main"]
