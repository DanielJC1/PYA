"""
Configuracion general del proyecto de cuantificacion de odio.
"""
import os
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Crear directorios si no existen
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Configuración TF-IDF
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE = (1, 3)

# Configuración del predictor híbrido
MODEL_WEIGHT = 0.7   # peso del clasificador TF-IDF
RULES_WEIGHT = 0.3   # peso de las reglas de contexto
HATE_THRESHOLD = 0.5 # umbral para clasificar como odio

# Clase labels
CLASS_LABELS = {
    0: "no_odio",
    1: "odio"
}

# Semilla para reproducibilidad
RANDOM_SEED = 42