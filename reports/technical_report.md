# Relatório Técnico: Análise de Predição de AVC (Stroke Prediction)

## Resumo

Este relatório apresenta uma análise completa do dataset de predição de AVC, incluindo análise exploratória, estratégias de pré-processamento, implementação de 7 modelos de machine learning e interpretação dos resultados. O objetivo principal foi desenvolver um modelo capaz de predizer a ocorrência de AVC com base em características demográficas e de saúde dos pacientes.

## 1. Introdução

O dataset utilizado contém informações sobre 5.110 pacientes com 12 características diferentes, incluindo dados demográficos (idade, gênero), condições de saúde (hipertensão, doenças cardíacas), estilo de vida (status de fumante, tipo de trabalho) e medidas fisiológicas (nível de glicose, IMC). A variável target é binária, indicando se o paciente teve ou não um AVC.

## 2. Análise Exploratória dos Dados

### 2.1 Características do Dataset
- **Tamanho**: 5.110 registros com 12 colunas
- **Variável Target**: Stroke (0 = Não teve AVC, 1 = Teve AVC)
- **Desbalanceamento**: Aproximadamente 95,1% dos casos são negativos (sem AVC) e 4,9% são positivos (com AVC)

### 2.2 Principais Descobertas
- **Idade**: Forte correlação com AVC - pacientes mais velhos têm maior risco

![Stroke Age](/assets/stroke_age.png)

- **Nível de Glicose**: Pacientes com AVC tendem a ter níveis mais elevados
- **Hipertensão e Doenças Cardíacas**: Fatores de risco significativos
- **Status de Fumante**: 30% dos dados marcados como "Unknown"
- **IMC**: 201 valores ausentes (3,93% do dataset)

### 2.3 Visualizações Geradas
- Distribuição das variáveis categóricas
- Boxplots para variáveis numéricas
- Matriz de correlação
- Análise da distribuição da variável target

## 3. Estratégias de Pré-processamento

### 3.1 Tratamento de Dados Ausentes
**IMC (Body Mass Index)**:
- **Problema**: 201 valores ausentes (3,93%)
- **Solução Adotada**: Imputação pela mediana (28,1)
- **Justificativa**: A mediana é robusta a outliers e mantém todos os registros

**Alternativa Considerada**: Exclusão dos registros com valores ausentes
- **Vantagens**: Dados 100% confiáveis, sem viés introduzido
- **Desvantagens**: Perda de 201 registros, redução do poder estatístico

### 3.2 Tratamento de Variáveis Categóricas
**Codificação Binária**:
- `ever_married`: Yes=1, No=0
- `Residence_type`: Urban=1, Rural=0

**Label Encoding**:
- `gender`: Female=0, Male=1
- `work_type`: 5 categorias codificadas (0-4)
- `smoking_status`: 4 categorias codificadas (0-3)

### 3.3 Remoção de Outliers e Dados Inconsistentes
- **Gênero "Other"**: 1 registro removido (representava apenas 0,02% dos dados)
- **Coluna ID**: Removida por não ter valor preditivo
- **Outliers do IMC**: Identificados usando IQR, mas mantidos no dataset

### 3.4 Tratamento do Desbalanceamento
**Técnica Utilizada**: SMOTE (Synthetic Minority Oversampling Technique)
- **Antes**: 95,1% classe 0, 4,9% classe 1
- **Depois**: 50% classe 0, 50% classe 1
- **Aplicação**: Apenas no conjunto de treino para evitar data leakage

![Smote](https://github.com/fernandavaldevino/stroke-app/blob/main/assets/smote.png)


### 3.5 Normalização
**StandardScaler**: Aplicado a todas as variáveis para padronizar as escalas
- **Justificativa**: Melhora a performance de algoritmos sensíveis à escala (SVM, Redes Neurais, Regressão Logística)

### 3.6 Imputação KNN para Smoking Status
**Problema**: 30% dos dados marcados como "Unknown"
**Solução**: KNNImputer com k=5 vizinhos
- **Processo**: Treino no conjunto de treino, aplicação no conjunto de teste
- **Resultado**: Mantém a distribuição natural dos dados

## 4. Modelos Implementados

### 4.1 Modelos Testados
1. **Decision Tree** - Modelo baseado em árvore de decisão
2. **Logistic Regression** - Modelo linear probabilístico
3. **Random Forest** - Ensemble de árvores de decisão
4. **XGBoost** - Gradient boosting otimizado
5. **CatBoost** - Gradient boosting para dados categóricos
6. **MLP (Multi-Layer Perceptron)** - Rede neural artificial
7. **Gaussian Naive Bayes** - Modelo probabilístico bayesiano

### 4.2 Configuração dos Modelos
**Logistic Regression**:
- `max_iter=200`
- `class_weight='balanced'`
- `random_state=42`

**Random Forest**:
- `n_estimators=100`
- `max_depth=10`
- `class_weight='balanced'`

**XGBoost**:
- `n_estimators=100`
- `max_depth=10`
- `scale_pos_weight=3` (para classes desbalanceadas)

**CatBoost**:
- `iterations=100`
- `auto_class_weights='Balanced'`

## 5. Resultados e Interpretação

### 5.1 Performance dos Modelos (Threshold Padrão 0.5)

| Modelo | Accuracy | Precision | F1-Score | AUC-ROC |
|--------|----------|-----------|----------|---------|
| **Logistic Regression** | 0.746 | 0.138 | 0.235 | **0.837** |
| Random Forest | 0.865 | 0.172 | 0.250 | 0.791 |
| CatBoost | 0.902 | 0.179 | 0.219 | 0.795 |
| XGBoost | 0.917 | 0.170 | 0.175 | 0.782 |
| Gaussian NB | 0.728 | 0.120 | 0.206 | 0.799 |
| MLP | 0.908 | 0.145 | 0.161 | 0.735 |
| Decision Tree | 0.833 | 0.115 | 0.174 | 0.668 |

### 5.1.1 Perfomance de Todos os Modelos

![Comparação de todos os modelos](/assets/comparacao_modelos.png)

### 5.2 Otimização de Threshold

**Logistic Regression com Threshold Otimizado (0.6)**:
- **F1-Score**: 0.2879 (melhor resultado)
- **Precision**: 0.1776
- **Recall**: 0.7600
- **Justificativa**: Maximiza o F1-Score, balanceando precisão e recall

### 5.3 Feature Importance (Logistic Regression)

| Feature | Importância |
|---------|-------------|
| **Age** | 2.032 |
| Avg Glucose Level | 0.188 |
| Hypertension | 0.147 |
| Residence Type | 0.115 |
| Heart Disease | 0.053 |
| Work Type | 0.035 |
| Gender | 0.029 |
| Ever Married | 0.027 |
| BMI | 0.003 |
| Smoking Status | 0.001 |

![Feature Importance](/assets/featureImportance.png)

## 6. Por que a Regressão Logística foi o Melhor Modelo?

### 6.1 Vantagens Identificadas

1. **Melhor AUC-ROC (0.837)**: Indica excelente capacidade de discriminação entre classes
2. **Interpretabilidade**: Coeficientes facilmente interpretáveis como log-odds
3. **Robustez**: Menos propenso ao overfitting comparado a modelos mais complexos
4. **Eficiência Computacional**: Treinamento rápido e predições eficientes
5. **Probabilidades Calibradas**: Fornece probabilidades bem calibradas para tomada de decisão médica

### 6.2 Características do Dataset que Favorecem a Regressão Logística

1. **Relações Lineares**: As variáveis mostram relações aproximadamente lineares com o log-odds do target
2. **Dataset Balanceado Pós-SMOTE**: A regressão logística se beneficia de classes balanceadas
3. **Variáveis Normalizadas**: O StandardScaler otimiza a performance do modelo linear
4. **Tamanho Moderado do Dataset**: Evita a necessidade de modelos mais complexos

### 6.3 Comparação com Outros Modelos

**Random Forest e XGBoost**:
- Maior accuracy mas menor AUC-ROC
- Possível overfitting nos dados de treino
- Menor capacidade de generalização

**Redes Neurais (MLP)**:
- Requer mais dados para performance ótima
- Maior complexidade sem benefício proporcional

**Naive Bayes**:
- Assume independência entre features (violada neste dataset)
- Performance inferior em métricas chave

## 7. Conclusões e Recomendações

### 7.1 Principais Achados

1. **Idade é o fator mais importante** para predição de AVC
2. **Nível de glicose e hipertensão** são fatores de risco significativos
3. **Regressão Logística com threshold 0.6** oferece o melhor equilíbrio entre métricas
4. **SMOTE foi essencial** para lidar com o desbalanceamento das classes

### 7.2 Limitações

1. **Dataset Relativamente Pequeno**: 5.110 registros podem não capturar toda a variabilidade populacional
2. **Dados Ausentes**: 30% de "Unknown" no status de fumante pode impactar a performance
3. **Validação Externa**: Necessária validação em outros datasets/populações

---

**Data do Relatório**: 10 de Janeiro de 2026  
**Autores**: Fernanda Valdevino / Marcos Câmara
