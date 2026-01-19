import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class StrokePreprocessor:
    def __init__(self, scaler_path, encoders_path):
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        self.categorical_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status', 'ever_married']
        
        # Ordem do Scaler
        self.feature_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                               'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    
    def preprocess(self, data):
        """
        Preprocessa os dados antes de fazer predição
        data: DataFrame com as features
        """
        try:
            df = data.copy()
            
            # Reordenando colunas na ordem esperada pelo scaler
            df = df[self.feature_order]
            
            # 1. Encode categorias
            for col in self.categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    
                    try:
                        df[col] = self.encoders[col].transform(df[col])
                    except ValueError as e:
                        print(f"Valor desconhecido em {col}: {df[col].values}")
                        df[col] = self.encoders[col].transform([self.encoders[col].classes_[0]] * len(df))
            
            # 2. Converter para float
            df = df.astype(float)
            
            # 3. Scale
            df_scaled = self.scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
            
            return df_scaled
        
        except Exception as e:
            print(f"Erro no preprocessamento: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        preprocessor = StrokePreprocessor('scaler_stroke.pkl', 'encoders_stroke.pkl')
        print("✓ Preprocessador carregado com sucesso")
    except Exception as e:
        print(f"✗ Erro: {e}")
        import traceback
        traceback.print_exc()