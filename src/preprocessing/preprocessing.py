import pickle
import pandas as pd
import numpy as np
import re
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

            # Normalizar/renomear colunas recebidas (caso clientes enviem variações)
            # Ex: 'Residence type', 'residence_type', 'Residence_type' -> 'Residence_type'
            orig_cols = df.columns.tolist()
            norm_map = {re.sub(r'[^0-9a-zA-Z]', '_', col).lower(): col for col in orig_cols}
            expected_norm = {re.sub(r'[^0-9a-zA-Z]', '_', col).lower(): col for col in self.feature_order}

            rename_dict = {}
            for norm, exp_col in expected_norm.items():
                if norm in norm_map:
                    rename_dict[norm_map[norm]] = exp_col

            if rename_dict:
                df = df.rename(columns=rename_dict)

            # Reordenando colunas na ordem esperada pelo scaler
            missing = [c for c in self.feature_order if c not in df.columns]
            if missing:
                raise ValueError(f"Colunas faltando para preprocessamento: {missing}. Colunas recebidas: {orig_cols}")
            df = df[self.feature_order]
            
            # 1. Encode categorias
            for col in self.categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

                    if col in self.encoders:
                        try:
                            df[col] = self.encoders[col].transform(df[col])
                        except ValueError:
                            print(f"Valor desconhecido em {col}: {df[col].values}")
                            df[col] = self.encoders[col].transform([self.encoders[col].classes_[0]] * len(df))
                    else:
                        # Encoder não disponível: tentar mapeamentos simples conhecidos
                        print(f"Aviso: encoder para '{col}' não encontrado. Tentando mapeamento simples.")
                        if col == 'ever_married':
                            map_em = {'yes': 1, 'no': 0, 'true': 1, 'false': 0}
                            df[col] = df[col].str.lower().map(map_em)
                            if df[col].isnull().any():
                                df[col] = df[col].fillna(0)
                        elif col == 'Residence_type':
                            map_res = {'urban': 1, 'rural': 0}
                            df[col] = df[col].str.lower().map(map_res)
                            if df[col].isnull().any():
                                df[col] = df[col].fillna(0)
                        else:
                            # Fallback: factorizar categorias localmente
                            codes, uniques = pd.factorize(df[col])
                            df[col] = codes
            
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
