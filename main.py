import logging as log
import sys
sys.path.insert(0, '/Users/fmbv/Documents/Fernanda/git/postech/stroke-app')

from src.etl.etl import ETL


# Configurar o logging
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    
    try:
        print("----- Iniciando ETL -----")
        etl = ETL()
        etl.run()
        print("ETL executada com sucesso!")

    except Exception as e:
        print(f"Erro ao executar o ETL: {e}")
        import traceback
        traceback.print_exc()
