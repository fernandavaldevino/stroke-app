.PHONY: train api streamlit run-all
PROJECT_PATH := $(shell pwd)	# pega o path do projeto atual

train:
	@echo "📊 Treinando modelo..."
	python3 main.py

api:
	@echo "🔧 Iniciando API Flask..."
	python3 -m src.api.app

streamlit:
	@echo "🎨 Iniciando Streamlit..."
	streamlit run streamlit/streamlit_app.py --server.port=8501

app:
	@echo "🚀 Iniciando o App"
	osascript -e "tell application \"Terminal\" to do script \"cd '$(PROJECT_PATH)' && make etl\""
	sleep 2
	osascript -e "tell application \"Terminal\" to do script \"cd '$(PROJECT_PATH)' && make api\""
	sleep 2
	osascript -e "tell application \"Terminal\" to do script \"cd '$(PROJECT_PATH)' && make streamlit\""
