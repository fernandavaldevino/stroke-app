import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Previsão de AVC", layout="wide")

st.title("🏥 Sistema de Previsão de AVC (Acidente Vascular Cerebral)")
st.subheader("Análise de Risco com Machine Learning")

st.markdown("---")

# Input dos dados
st.subheader("📋 Dados do Paciente")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Idade", min_value=0, max_value=100, value=30)
    hypertension = st.selectbox(
        "Hipertensão", 
        [0, 1], 
        format_func=lambda x: "Não" if x == 0 else "Sim"
    )
    heart_disease = st.selectbox(
        "Doença Cardíaca", 
        [0, 1], 
        format_func=lambda x: "Não" if x == 0 else "Sim"
    )

with col2:
    bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
    avg_glucose_level = st.number_input("Nível de Glicose", min_value=50.0, max_value=300.0, value=150.0)
    gender_map = {"Masculino": "Male", "Feminino": "Female"}
    gender_pt = st.selectbox("Gênero", list(gender_map.keys()))
    gender = gender_map[gender_pt]

with col3:
    work_type_map = {
        "Setor Privado": "Private",
        "Autônomo": "Self-employed",
        "Servidor Público": "Govt_job",
        # "Criança": "children",        Esta opção não é relevante para o dataset
        "Nunca Trabalhou": "Never_worked"
    }
    work_type_pt = st.selectbox("Tipo de Trabalho", list(work_type_map.keys()))
    work_type = work_type_map[work_type_pt]
    
    residence_map = {"Urbana": "Urban", "Rural": "Rural"}
    residence_pt = st.selectbox("Tipo de Residência", list(residence_map.keys()))
    Residence_type = residence_map[residence_pt]
    
    smoking_map = {
        "Ex-Fumante": "formerly smoked",
        "Nunca fumou": "never smoked",
        "Fuma": "smokes",
        "Desconhecido": "Unknown"
    }
    smoking_pt = st.selectbox("Status de Fumo", list(smoking_map.keys()))
    smoking_status = smoking_map[smoking_pt]
    
    married_map = {"Sim": "Yes", "Não": "No"}
    married_pt = st.selectbox("É ou já foi casado?", list(married_map.keys()))
    ever_married = married_map[married_pt]

st.markdown("---")

# Botão de predição
if st.button("🔍 Fazer Predição", use_container_width=True):
    
    # Preparar dados (com valores em inglês para a API)
    dados = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'avg_glucose_level': avg_glucose_level,
        'gender': gender,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'smoking_status': smoking_status,
        'ever_married': ever_married
    }
    
    print(f"Enviando dados: {dados}")
    
    try:
        # Chamar API
        response = requests.post('http://localhost:5000/predict', json=dados, timeout=5)
        
        # ⭐ DEBUG:
        print(f"Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code != 200:
            st.error(f"❌ Erro da API (Status {response.status_code}): {response.text}")
        else:
            resultado = response.json()
            
            # Exibir resultado
            st.markdown("---")
            st.subheader("📊 Resultado da Predição:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probabilidade de AVC", f"{resultado['probabilidade']:.2%}")
            
            with col2:
                risco = resultado['risco']
                risco_pt = "Alto" if risco == "Alto" else "Baixo"
                cor = "🔴" if risco == "Alto" else "🟢"
                st.metric("Nível de Risco", f"{cor} {risco_pt}")
            
            with col3:
                predicao_pt = "AVC" if resultado['predicao'] == 1 else "Sem AVC"
                st.metric("Predição", predicao_pt)
            
            st.markdown("---")
            
            # Interpretação
            if resultado['predicao'] == 1:
                st.warning("⚠️ **RISCO ALTO DE AVC** - Recomenda-se avaliação médica imediata!")
            else:
                st.success("✅ **RISCO BAIXO DE AVC** - Mantenha hábitos saudáveis")
    
    except requests.exceptions.ConnectionError as e:
        st.error(f"❌ Não consegue conectar na API: {str(e)}")
        print(f"ConnectionError: {str(e)}")
    
    except requests.exceptions.Timeout:
        st.error(f"❌ Timeout: API não respondeu a tempo")
        print("Timeout!")
    
    except Exception as e:
        st.error(f"❌ Erro na predição: {str(e)}")
        print(f"Exceção: {str(e)}")

st.markdown("---")
st.info("ℹ️ Este é um sistema de previsão baseado em Machine Learning. Sempre consulte um médico para diagnóstico final.")


st.markdown("---")

# Teste de conexão
st.subheader("🔧 Teste de Conexão")

if st.button("Testar Conexão com API"):
    try:
        response = requests.get('http://localhost:5000/')
        st.success(f"✅ API está online! Resposta: {response.json()}")
    except Exception as e:
        st.error(f"❌ Não consegue conectar na API: {str(e)}")

st.markdown("---")