# 🏥 Aplicação de Predição para Risco de AVC (Acidente Vascular Cerebral)

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-orange)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53.0-red)

**Sistema inteligente de predição de risco de AVC (Acidente Vascular Cerebral) utilizando Machine Learning**

</div>

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Características](#características)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pipeline de Dados](#pipeline-de-dados)
- [API REST](#api-rest)
- [Modelo de Machine Learning](#modelo-de-machine-learning)
- [Feature Importance](#feature-importance)
- [Notebooks](#notebooks)

---

## 🎯 Sobre o Projeto

Este projeto implementa um sistema completo de predição de risco de AVC utilizando técnicas avançadas de Machine Learning. A aplicação foi desenvolvida como parte de um projeto acadêmico e oferece uma interface interativa para avaliação de risco baseada em dados clínicos e demográficos de pacientes.

### Objetivos

Fornecer uma ferramenta de apoio à decisão médica que permita avaliar o risco de AVC em pacientes com base em características como idade, IMC (Índice de Massa Corporal), hipertensão, doenças cardíacas, histórico de tabagismo, entre outros fatores. Além disso, consiste num projeto acadêmico de conclusão da Fase 1 da Pós-Tech FIAP (8IADT).

### Dataset

O projeto utiliza o dataset **Healthcare Dataset - Stroke Data** disponível publicamente no [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), contendo informações de pacientes e a ocorrência ou não de AVC.

---

## ✨ Características

- 🤖 **Modelo de ML de Alta Performance**: Utiliza CatBoost com tratamento de classes desbalanceadas
- 🌐 **API REST**: Endpoint Flask para integração com outros sistemas
- 🎨 **Interface Interativa**: Dashboard Streamlit intuitivo e responsivo
- 📊 **Análise Exploratória**: Notebooks completos com visualizações e insights
- ⚙️ **Pipeline ETL Automatizado**: Extração, transformação e carga de dados
- 🔄 **Pré-processamento Robusto**: Tratamento de missing values, encoding e normalização
- 📈 **Métricas de Avaliação**: Análise detalhada de performance do modelo
- 🎯 **Feature Importance**: Identificação dos fatores mais relevantes para predição

---

## 🛠 Tecnologias Utilizadas

### Core
- **Python 3.12+** - Linguagem principal
- **CatBoost 1.2.8** - Algoritmo de gradient boosting
- **Scikit-learn 1.5.2** - Ferramentas de ML e pré-processamento
- **Pandas 2.3.3** - Manipulação de dados
- **NumPy 2.3.4** - Computação numérica

### Visualização
- **Matplotlib 3.10.7** - Gráficos estáticos
- **Seaborn 0.13.2** - Visualizações estatísticas
- **Plotly 6.3.1** - Gráficos interativos

### Web & API
- **Flask 3.1.2** - Framework para API REST
- **Streamlit 1.53.0** - Framework para interface web

### Ferramentas Adicionais
- **imbalanced-learn 0.12.3** - Técnicas para dados desbalanceados
- **XGBoost 3.1.3** - Algoritmo alternativo de boosting
- **Jupyter Notebook 7.4.7** - Ambiente de desenvolvimento interativo

---

## 🏗 Arquitetura do Sistema

```
┌─────────────────┐
│   Raw Data      │
│   (CSV)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ETL Pipeline  │
│  - Extract      │
│  - Transform    │
│  - Load         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Encoding     │
│  - Scaling      │
│  - Feature Eng. │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│  (CatBoost)     │
└────────┬────────┘
         │
         ├─────────────────┐
         ▼                 ▼
┌─────────────────┐  ┌──────────────┐
│   Flask API     │  │  Streamlit   │
│   (Port 5000)   │  │  (Port 8501) │
└─────────────────┘  └──────────────┘
```

---

## 📋 Pré-requisitos

Antes de iniciar, certifique-se de ter instalado:

- **Sistema Operacional**: macOS, Linux ou Windows
- **Python**: Versão 3.9 ou superior (recomendado 3.12+)
- **pip**: Gerenciador de pacotes Python
- **Git**: Para controle de versão
- **Make** (Opcional): Para utilizar os comandos do Makefile

### Verificando as versões instaladas

```bash
python3 --version
pip3 --version
git --version
make --version  # Opcional
```

---

## 🚀 Instalação

### 1. Clone o Repositório

```bash
git clone <repo-url>
cd stroke-app
```

### 2. Crie e Ative um Ambiente Virtual

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)
```powershell
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows (CMD)
```cmd
python3 -m venv .venv
.\.venv\Scripts\activate.bat
```

> **💡 Nota**: Se você utiliza um alias para `python3` ou `pip3`, substitua conforme necessário.

### 3. Atualize o pip

```bash
pip3 install --upgrade pip
```

### 4. Instale as Dependências

```bash
pip3 install -r requirements.txt
```

### 5. Verifique a Instalação

```bash
python3 -c "import catboost, flask, streamlit; print('✓ Instalação bem-sucedida!')"
```

---

## 💻 Como Usar

Antes de executar o app, é necessário observar os seguintes pontos:

**Obs 1:** Certifique-se de não ter nenhum prompt aberto além do que dará o comando abaixo.

**Obs 2:** Antes de dar o comando abaixo, certifique-se de substituir o ```<path>``` do comando: ```sys.path.insert(0, '<path>stroke-app')``` pelo seu _path_ absoluto, nos arquivos abaixo:

- ```src/api/app.py```: 
- ```main.py```: 


### Método 1: Usando Makefile (Recomendado para macOS)

#### Executar o aplicativo completo (ETL + API + Streamlit)
Na pasta-raiz do projeto, executar o comando:

```bash
make app
```

Este comando irá:
1. Executar o pipeline ETL e treinar o modelo
2. Iniciar a API Flask na porta 5000 (_default_)
3. Iniciar o Streamlit na porta 8501 (_default_)

#### Comandos individuais

```bash
# Apenas treinar o modelo
make train

# Apenas executar o ETL
make etl

# Apenas iniciar a API
make api

# Apenas iniciar o Streamlit
make streamlit
```

### Método 2: Execução Manual

#### Passo 1: Treinar o Modelo

Antes de utilizar a aplicação, é necessário treinar o modelo:

```bash
python3 main.py
```

Este comando irá:
- Extrair dados do CSV
- Realizar pré-processamento
- Treinar o modelo CatBoost
- Salvar o modelo e artefatos em `data/processed/`

#### Passo 2: Iniciar a API Flask

Em um terminal, execute:

```bash
python3 -m src.api.app
```

A API estará disponível em: `http://localhost:5000` (porta _default_ do projeto)

#### Passo 3: Iniciar o Streamlit

Em **outro terminal**, após ativar novamente o .venv, execute:

```bash
streamlit run streamlit/streamlit_app.py
```

O Streamlit abrirá automaticamente no navegador em: `http://localhost:8501` (porta _default_)

### Acessando a Aplicação

1. Abra o navegador em `http://localhost:8501`
2. Preencha os dados do paciente no formulário
3. Clique em "Prever Risco de AVC"
4. Visualize a predição e a probabilidade de risco

---

## 📁 Estrutura do Projeto

```
stroke-app/
│
├── main.py                          # Script principal para executar o ETL
├── Makefile                         # Comandos automatizados
├── requirements.txt                 # Dependências do projeto
├── Readme.md                        # Documentação
│
├── assets/                          # Recursos visuais
│   ├── featureImportance.png       # Gráfico de importância de features
│   └── scatterplot-idadeBmi.png    # Análise de correlação
│
├── catboost_info/                   # Logs de treinamento do CatBoost
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   └── time_left.tsv
│
├── config/                          # Configurações do projeto
│   └── settings.py                 # Parâmetros e constantes
│
├── data/                            # Dados do projeto
│   ├── raw/                        # Dados originais
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/                  # Dados processados e artefatos
│       ├── encoders_stroke.pkl     # Encoders salvos
│       ├── scaler_stroke.pkl       # Scaler salvo
│       └── training_stroke_model.pkl # Modelo treinado
│
├── notebooks/                       # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb      # Análise exploratória
│   ├── 02_data_preprocessing.ipynb        # Pré-processamento
│   └── 03_model_training.ipynb            # Treinamento do modelo
│
├── src/                             # Código fonte
│   ├── api/                        # API REST
│   │   └── app.py                  # Aplicação Flask
│   │
│   ├── etl/                        # Pipeline ETL
│   │   ├── __init__.py
│   │   ├── etl.py                  # Orquestrador ETL
│   │   ├── extract.py              # Extração de dados
│   │   ├── transform.py            # Transformação
│   │   └── test.py                 # Testes ETL
│   │
│   ├── models/                     # Modelos de ML
│   │   └── model_training.py       # Treinamento
│   │
│   └── preprocessing/              # Pré-processamento
│       └── preprocessing.py        # Classes de preprocessamento
│
└── streamlit/                       # Interface web
    └── streamlit_app.py            # Aplicação Streamlit
```

### Descrição das Pastas Principais

| Pasta | Descrição |
|-------|-----------|
| `config/` | Arquivos de configuração e parâmetros (TEST_SIZE, RANDOM_STATE, paths, etc.) |
| `data/raw/` | Dados originais (CSV) - não versionar dados sensíveis |
| `data/processed/` | Modelos treinados, encoders e scalers salvos |
| `notebooks/` | Análises exploratórias e experimentação |
| `src/api/` | API REST Flask para servir predições |
| `src/etl/` | Pipeline de extração, transformação e carga de dados |
| `src/models/` | Scripts de treinamento de modelos |
| `src/preprocessing/` | Funções de pré-processamento e feature engineering |
| `streamlit/` | Interface web interativa |
| `catboost_info/` | Logs e métricas de treinamento do CatBoost |

---

## 🔄 Pipeline de Dados

### 1. Extração (Extract)
- Leitura do dataset CSV
- Validação de integridade dos dados
- Identificação de missing values

### 2. Transformação (Transform)
- **Tratamento de Missing Values**: Imputação baseada em mediana/moda
- **Encoding Categórico**: 
  - Label Encoding para variáveis ordinais
  - One-Hot Encoding para variáveis nominais
- **Feature Engineering**: Criação de novas features relevantes
- **Normalização**: StandardScaler para features numéricas
- **Balanceamento**: Técnicas para lidar com classes desbalanceadas

### 3. Carga (Load)
- Salvamento de modelos treinados
- Persistência de encoders e scalers
- Geração de métricas e relatórios

---

## 🌐 API REST

### Endpoints Disponíveis

#### `GET /`
**Descrição**: Verifica o status da API

**Resposta**:
```json
{
  "status": "ok",
  "mensagem": "API funcionando"
}
```

#### `POST /predict`
**Descrição**: Realiza predição de risco de AVC

**Request Body**:
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 1,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

**Resposta**:
```json
{
  "probabilidade": 0.7542,
  "predicao": 1,
  "risco": "Alto"
}
```

### Exemplo de Uso com cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

### Exemplo de Uso com Python

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "gender": "Female",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 95.5,
    "bmi": 25.3,
    "smoking_status": "never smoked"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 🤖 Modelo de Machine Learning

### Algoritmo: Logistic Regression Classifier

O **Logistic Regression** foi escolhido por suas vantagens:

• ✅ Simples e interpretável
• ✅ Rápido tempo de treinamento e predição
• ✅ Baixo custo computacional
• ✅ Fornece probabilidades de classificação
• ✅ Funciona bem com problemas linearmente separáveis
• ✅ Regularização integrada (L1/L2) previne overfitting
• ✅ Excelente baseline para comparação com outros modelos
• ✅ Funciona bem com alta dimensionalidade
• ✅ Requer pouca memória
• ✅ Probabilidades calibradas por padrão


### Métricas de Avaliação

| Métrica | Valor |
|---------|-------|
| Acurácia | 74.6% | >> Não aconselhável para modelos 
| Precisão | 17.8% |
| Recall | 76.00% |
| F1-Score | 28.8% |
| AUC-ROC | 0.837 |


Matriz de Confusão:
|  | Negativo | Positivo |
|---|---|---|
| **Negativo** | 796 | 176 |
| **Positivo** | 12 | 38 |


### Tratamento de Classes Desbalanceadas

O dataset apresenta desbalanceamento significativo (stroke vs. não-stroke). Técnicas aplicadas:

- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class Weights**: Ajuste de pesos nas classes
- **Threshold Tuning**: Ajuste do limiar de decisão (0.6 ao invés de 0.2)

---

## 📊 Feature Importance

As features mais importantes identificadas pelo modelo são apresentadas em ordem decrescente de relevância:

![Feature Importance](assets/featureImportance.png)

### Top 10 Features

1. **Age (Idade)** - Fator de risco mais relevante
2. **Avg Glucose Level** - Nível médio de glicose
3. **BMI** - Índice de Massa Corporal
4. **Heart Disease** - Presença de doença cardíaca
5. **Hypertension** - Hipertensão
6. **Smoking Status** - Histórico de tabagismo
7. **Work Type** - Tipo de trabalho
8. **Ever Married** - Estado civil
9. **Residence Type** - Tipo de residência
10. **Gender** - Gênero

### Análise de Correlação: Idade vs BMI

![Scatterplot - Idade vs BMI](assets/scatterplot-idadeBmi.png)

Este gráfico demonstra a relação entre idade e BMI, onde:
- **Vermelho**: Pacientes que tiveram AVC
- **Azul**: Pacientes que não tiveram AVC

**Insight**: Observa-se que o avanço da idade está fortemente correlacionado com a ocorrência de AVC, enquanto a variação do BMI não apresenta um padrão claro, corroborando com o menor score de importância dessa feature.

---

## 📓 Notebooks

### 01_exploratory_analysis.ipynb
**Análise Exploratória de Dados (EDA)**

- Estatísticas descritivas
- Distribuição de variáveis
- Análise de correlações
- Identificação de outliers
- Visualizações interativas

### 02_data_preprocessing.ipynb
**Pré-processamento e Feature Engineering**

- Tratamento de missing values
- Encoding de variáveis categóricas
- Normalização e padronização
- Criação de novas features
- Divisão em treino/teste

### 03_model_training.ipynb
**Treinamento e Avaliação de Modelos**

- Comparação de algoritmos
- Tuning de hiperparâmetros
- Cross-validation
- Métricas de performance
- Análise de erros

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"

**Solução**: Certifique-se de que o ambiente virtual está ativado e as dependências instaladas:
```bash
source .venv/bin/activate  # macOS/Linux
pip3 install -r requirements.txt
```

### Erro: "Port already in use"

**Solução**: Mate o processo que está usando a porta:
```bash
# macOS/Linux
lsof -ti:5000 | xargs kill -9  # Para API Flask
lsof -ti:8501 | xargs kill -9  # Para Streamlit

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Erro: "Model file not found"

**Solução**: Execute o treinamento do modelo primeiro:
```bash
python3 main.py
```

### Erro ao carregar o modelo

**Solução**: Verifique se os arquivos estão em `data/processed/`:
```bash
ls -la data/processed/
```

---

## 👨‍💻 Autores

Desenvolvido como projeto acadêmico da **Pós-Tech FIAP - Turma 8IADT - Pós IA para Devs**.

🐙 [@fernandavaldevino](!http://github.com/fernandavaldevino)

🐙 [@marcosvrc](!http://github.com/marcosvrc)

---

<div align="center">

Made with ❤️ and ☕

</div>
