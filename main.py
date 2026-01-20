import logging as log
import sys
from pathlib import Path
from src.etl.etl import ETL


# Adicionar o diretório raiz do projeto ao sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar o logging
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":

    try:
        log.info("----- Iniciando ETL -----")
        etl = ETL()
        etl.run()
        log.info("ETL executada com sucesso!")

    except Exception as e:
        log.error(f"Erro ao executar o ETL: {e}")
        import traceback
        traceback.print_exc()
