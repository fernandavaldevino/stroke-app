import sys
sys.path.insert(0, '/Users/fmbv/Documents/Fernanda/git/postech/stroke-app')

from src.etl.extract import Extract

class Pipeline():
    def __init__(self):
        self.extract = Extract()

    def run(self):
        print('----- Running ETL Pipeline -----')
        print('Extracting data...')
        df = self.extract.run()
        print(f"df.shape: {df.shape}")

# Executa
pipeline = Pipeline()
pipeline.run()