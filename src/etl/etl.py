import sys
sys.path.insert(0, '/Users/fmbv/Documents/Fernanda/git/postech/stroke-app')

import logging

from src.etl.extract import Extract
from src.etl.transform import Transform
from src.models.model_training import ModelTraining

class ETL():
    def __init__(self):
        self.extract = Extract()
        self.transform = Transform()
        self.model_training = ModelTraining()
        self.logging = logging.getLogger(__name__)

    def run(self):
        df = self.extract.run()
        df = self.transform.run(df)
        self.model_training.train(df)  

        self.logging.info("\n\nPipeline executada com sucesso.")

