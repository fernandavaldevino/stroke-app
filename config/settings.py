import os
from pathlib import Path

# Caminhos
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'healthcare-dataset-stroke-data.csv'
DATA_PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' 

# Modelos
MODEL_NAME = 'training_stroke_model.pkl'
SCALER_NAME = 'scaler_stroke.pkl'
ENCODERS_NAME = 'encoders_stroke.pkl'

# API
API_HOST = os.getenv('API_HOST', 'localhost')
API_PORT = int(os.getenv('API_PORT', 5000))

# Streamlit
STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', 8501))

# ETL
THRESHOLD_STROKE = 0.6
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 42