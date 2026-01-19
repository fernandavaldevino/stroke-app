# Standard Library
import logging
import pickle

# Data Manipulation
import pandas as pd

# Scikit-Learn - Model Selection
from sklearn.model_selection import train_test_split

# Scikit-Learn - Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix

# Scikit-Learn - Preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Scikit-Learn - Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# XGBoost
from xgboost import XGBClassifier

# CatBoost
from catboost import CatBoostClassifier

# Imbalanced-Learn
from imblearn.over_sampling import SMOTE

from config.settings import TEST_SIZE, RANDOM_STATE, DATA_PROCESSED_PATH, MODEL_NAME, SCALER_NAME, ENCODERS_NAME


class ModelTraining:
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def categorical_encoding(self, df):
        pass


    def impute_knn_smoking_status(self, df):
        pass


    def scaling(self, df):
        pass


    def split_data(self, df, target_column: str):
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return X, y
        

    def train(self, df):
        X = df.drop("stroke", axis=1)
        y = df["stroke"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    stratify=y,         # mantém proporção de classes
                                                    test_size=TEST_SIZE, 
                                                    random_state=RANDOM_STATE)
        
        self.log.info(f"\nDistribuição do target:\n{y.value_counts(normalize=True)}")

        # 01. Codificação de Variáveis Categóricas
        self.log.info(f"- Codificação de Variáveis Categóricas")

        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        encoders = {}
        categorical_cols = ['gender', 'work_type', 'smoking_status']

        for col in categorical_cols:
            le = LabelEncoder()
            X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
            X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
            encoders[col] = le
            self.log.info(f"  {col}: {len(le.classes_)} classes → {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Convertendo todos os dados para int
        X_train_processed = X_train_processed.astype(int)
        X_test_processed = X_test_processed.astype(int)

        # 02. Imputação KNN
        self.log.info(f"- Imputação KNN")

        self.log.info(f"\nTrain set: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)\
                        Test set:  {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)\
                        \nUnknown no train: {(X_train['smoking_status'] == 'Unknown').sum()} ({(X_train['smoking_status'] == 'Unknown').sum() / len(X_train) * 100:.1f}%)\
                        Unknown no test:  {(X_test['smoking_status'] == 'Unknown').sum()} ({(X_test['smoking_status'] == 'Unknown').sum() / len(X_test) * 100:.1f}%)"
                        )
        
        knn_imputer = KNNImputer(n_neighbors=5)

        # Ajustar com dados de TREINO >> Isto "aprende" o padrão dos dados conhecidos
        self.log.info(f"\n  1. Ajustando KNN com {X_train_processed.shape[0]} registros de treino...")
        X_train_imputed = knn_imputer.fit_transform(X_train_processed)
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train_processed.columns)

        # Converter colunas categóricas de volta para inteiro - Treino
        for col in categorical_cols:
            X_train_imputed[col] = X_train_imputed[col].round().astype(int)

        # Aplicar o mesmo imputer nos dados de TESTE >> Não reajusta, apenas usa o padrão aprendido do treino
        print(f"  2. Aplicando KNN no test set (usando padrão do treino)...")
        X_test_imputed = knn_imputer.transform(X_test_processed)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test_processed.columns)


        # Converter colunas categóricas de volta para inteiro - Teste
        for col in categorical_cols:
            X_test_imputed[col] = X_test_imputed[col].round().astype(int)

        # Converter smoking_status de volta para nomes legíveis (apenas para visualização)
        le_smoking = encoders['smoking_status']  # Use o encoder específico
        X_train_smoking_original = le_smoking.inverse_transform(X_train_imputed['smoking_status'].astype(int))
        X_test_smoking_original = le_smoking.inverse_transform(X_test_imputed['smoking_status'].astype(int))

        self.log.info(f"\n  Distribuição após imputação (Train):")
        print(pd.Series(X_train_smoking_original).value_counts())


        # 03. Normalização (Scaling)
        self.log.info(f"- Normalização (Scaling)")

        # Criar StandardScaler
        scaler = StandardScaler()

        # Ajustar com treino
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        # Aplicar ao teste
        X_test_scaled = scaler.transform(X_test_imputed)

        # Converter para DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_imputed.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_imputed.columns)


        # 04. Balanceamento apenas nos dados de treino
        self.log.info(f"- Balanceamento apenas nos dados de treino")
        self.log.info(f"\nAntes do SMOTE:")
        self.log.info(f"  Classe 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        self.log.info(f"  Classe 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")

        # Aplicar SMOTE apenas no treino
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        self.log.info(f"\nDepois do SMOTE:")
        self.log.info(f"  Classe 0: {(y_train_balanced == 0).sum()} ({(y_train_balanced == 0).sum() / len(y_train_balanced) * 100:.1f}%)")
        self.log.info(f"  Classe 1: {(y_train_balanced == 1).sum()} ({(y_train_balanced == 1).sum() / len(y_train_balanced) * 100:.1f}%)")
        

        # 05. Treinamento dos modelos
        self.log.info(f"- Treinamento dos modelos")

        # 5.1. Instanciar modelos com parâmetros relevantes
        arvore = DecisionTreeClassifier(
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )

        log_reg = LogisticRegression(
            max_iter=200,
            random_state=42,
            class_weight='balanced'
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=3,     # ajuda em classes minoritárias
            n_jobs=-1
        )

        catboost = CatBoostClassifier(
            iterations=100,
            depth=10,
            learning_rate=0.1,
            verbose=0,
            random_state=42,
            auto_class_weights='Balanced'
        )

        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            learning_rate_init=0.01,    # controla velocidade de aprendizado
            random_state=42
        )

        gaus = GaussianNB()


        models = {
            'Decision Tree': arvore,
            'Logistic Regression': log_reg,
            'Random Forest': rf,
            'XGBoost': xgb,
            'CatBoost': catboost,
            'MLP': mlp,
            'Gaussian NB': gaus
        }

        # 5.2. Treinamento dos modelos
        arvore.fit(X_train_balanced, y_train_balanced) 
        log_reg.fit(X_train_balanced, y_train_balanced) 
        rf.fit(X_train_balanced, y_train_balanced) 
        xgb.fit(X_train_balanced, y_train_balanced) 
        catboost.fit(X_train_balanced, y_train_balanced)
        mlp.fit(X_train_balanced, y_train_balanced) 
        gaus.fit(X_train_balanced, y_train_balanced)


        # 5.3. Avaliação dos modelos
        # arvore
        y_pred_arvore = arvore.predict(X_test_scaled)
        y_pred_arvore_proba = arvore.predict_proba(X_test_scaled)[:, 1]

        # log_reg
        y_pred_log_reg = log_reg.predict(X_test_scaled)
        y_pred_log_reg_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

        # rf
        y_pred_rf = rf.predict(X_test_scaled)
        y_pred_rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

        # xgb
        y_pred_xgb = xgb.predict(X_test_scaled)
        y_pred_xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

        # catboost
        y_pred_catboost = catboost.predict(X_test_scaled)
        y_pred_catboost_proba = catboost.predict_proba(X_test_scaled)[:, 1]

        # mlp
        y_pred_mlp = mlp.predict(X_test_scaled)
        y_pred_mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]

        # gaus
        y_pred_gaus = gaus.predict(X_test_scaled)
        y_pred_gaus_proba = gaus.predict_proba(X_test_scaled)[:, 1]


        # 5.4. Comparação dos modelos
        print("=" * 80)
        print("COMPARAÇÃO DE TODOS OS MODELOS")
        print("=" * 80)
        print(f"{'model':<20} {'Accuracy':>10} {'Precision':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
        print("=" * 80)

        for name, model in models.items():
            y_pred_model = model.predict(X_test_scaled)
            y_pred_model_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            acc = accuracy_score(y_test, y_pred_model)
            prec = precision_score(y_test, y_pred_model, zero_division=0)
            f1 = f1_score(y_test, y_pred_model, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_model_proba)
            
            print(f"{name:<20} {acc:>10.3f} {prec:>10.3f} {f1:>10.3f} {auc:>10.3f}")

        print("=" * 80)


        # 5.5. Comparação de thresholds

        print("\n" + "="*80)
        print("COMPARAÇÃO DE THRESHOLDS - Logistic Regression")
        print("="*80)

        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

        for threshold in thresholds:
            y_pred_adj = (y_pred_log_reg_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_adj)
            precision = precision_score(y_test, y_pred_adj)
            recall = recall_score(y_test, y_pred_adj)
            cm = confusion_matrix(y_test, y_pred_adj)
            
            print(f"\nThreshold {threshold}:")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  TP: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TN: {cm[1,1]}")
        
        print('\n')
        
        y_pred_final = (y_pred_log_reg_proba >= 0.6).astype(int)

        self.log.info(f"Modelo Final com Threshold 0.6:")
        self.log.info(f"  F1-Score: {f1_score(y_test, y_pred_final):.4f}")
        self.log.info(f"  Recall: {recall_score(y_test, y_pred_final):.4f}")
        self.log.info(f"  Precision: {precision_score(y_test, y_pred_final):.4f}")
        self.log.info(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_final)}")


        # 5.6. Exportar modelo final treinado, o scaler e os encoders
        # Salvar o modelo
        model_path = DATA_PROCESSED_PATH / MODEL_NAME
        with open(model_path, 'wb') as f:
            pickle.dump(gaus, f)

        # Salvar o scaler
        scaler_path = DATA_PROCESSED_PATH / SCALER_NAME
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Salvar os encoders
        encoders_path = DATA_PROCESSED_PATH / ENCODERS_NAME
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)