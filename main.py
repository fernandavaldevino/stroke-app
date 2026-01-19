import sys
sys.path.insert(0, '/Users/fmbv/Documents/Fernanda/git/postech/stroke-app')

from src.etl.etl import ETL

if __name__ == "__main__":
    print("----- Iniciando ETL -----")
    
    try:
        etl = ETL()
        etl.run()
        print("ETL executada com sucesso!")

    except Exception as e:
        print(f"Erro ao executar o ETL: {e}")
        import traceback
        traceback.print_exc()
