.PHONY: train api streamlit run-all
PROJECT_PATH := $(shell pwd)	# pega o path do projeto atual

train:
	@echo "📊 Treinando modelo..."
	python main.py

api:
	@echo "🔧 Iniciando API Flask na porta 5000..."
	python src/api/main.py

streamlit:
	@echo "🎨 Iniciando Streamlit na porta 8501..."
	streamlit run streamlit/streamlit_app.py --server.port=8501

etl:
	@echo "⚙️ Executando ETL completo..."
	python main.py


app:
	@echo "🚀 Iniciando o App"
	osascript -e "tell application \"Terminal\" to do script \"cd '$(PROJECT_PATH)' && make etl\""
	sleep 2
	osascript -e "tell application \"Terminal\" to do script \"cd '$(PROJECT_PATH)' && make api\""
	sleep 2
	osascript -e "tell application \"Terminal\" to do script \"cd '$(PROJECT_PATH)' && make streamlit\""
