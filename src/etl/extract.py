import pandas as pd

from config.settings import DATA_RAW_PATH

class Extract():
    def __init__(self):
        self.path = DATA_RAW_PATH
    
    def run(self):
        df = pd.read_csv(self.path)
        return df
