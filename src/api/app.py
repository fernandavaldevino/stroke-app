from flask import Flask, request, jsonify
from src.preprocessing.preprocessing import StrokePreprocessor
from config.settings import TEST_SIZE, RANDOM_STATE, DATA_PROCESSED_PATH, MODEL_NAME, SCALER_NAME, ENCODERS_NAME
import pickle
import pandas as pd
import traceback

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

try:
    model_path = DATA_PROCESSED_PATH / MODEL_NAME
    scaler_path = DATA_PROCESSED_PATH / SCALER_NAME
    encoders_path = DATA_PROCESSED_PATH / ENCODERS_NAME

    print("Carregando modelo...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Modelo carregado")
    
    print("Carregando preprocessador...")
    preprocessor = StrokePreprocessor(scaler_path, encoders_path)
    print("✓ Preprocessador carregado")
    
except Exception as e:
    print(f"✗ Erro: {e}")
    traceback.print_exc()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'ok', 'mensagem': 'API funcionando'})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        print(f"Dados recebidos: {data}")
        
        # Converter para DataFrame
        df = pd.DataFrame([data])
        print(f"DataFrame original:\n{df}\n")
        
        # Preprocessar
        df_processed = preprocessor.preprocess(df)
        print(f"DataFrame processado:\n{df_processed}\n")
        print(f"Valores min/max:\n{df_processed.describe()}\n")
        
        # Predição
        y_pred_proba = model.predict_proba(df_processed)[0, 1]
        print(f"Probabilidade bruta: {y_pred_proba}")
        
        y_pred = (y_pred_proba >= 0.2).astype(int)
        
        resultado = {
            'probabilidade': float(y_pred_proba),
            'predicao': int(y_pred),
            'risco': 'Alto' if y_pred == 1 else 'Baixo'
        }
        
        print(f"Resultado: {resultado}\n")
        return jsonify(resultado)
    
    except Exception as e:
        print(f"Erro: {e}")
        traceback.print_exc()
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')