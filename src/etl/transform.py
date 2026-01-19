import logging


class Transform():
    def __init__(self):
        self.mapping_binary = {'Yes': 1, 'No': 0, 'Urban': 1, 'Rural': 0}
        self.log = logging.getLogger(__name__)

    
    def drop_duplicates(self, df):
        lines_before = df.shape[0]
        df_transformed = df.drop_duplicates()
        lines_after = df_transformed.shape[0]
        if lines_before != lines_after:
            self.log.info(f"- Linhas duplicadas removidas: {lines_before - lines_after}")
        else:
            self.log.info("- Nenhuma linha duplicada")
        return df_transformed


    def exclude_other(self, df):
        df_transformed = df = df.loc[df['gender'] != 'Other']
        return df_transformed


    def exclude_columns(self, df, columns: list):
        df_transformed = df.drop(columns=columns, axis=1)
        return df_transformed
    

    def replace_binary_columns(self, df, column: str, mapping: dict):
        cols_to_replace = ['ever_married', 'Residence_type']
        df[cols_to_replace] = df[cols_to_replace].replace(mapping)
        return df


    def fill_NaN(self, df, column: str, value):
        df[column] = df[column].fillna(value)
        return df
    

    def run(self, df):
        df_cleaned = self.drop_duplicates(df)
        df_cleaned = self.exclude_other(df_cleaned)
        df_cleaned = self.exclude_columns(df_cleaned, columns=['id'])
        df_cleaned = self.replace_binary_columns(df_cleaned, column=['ever_married', 'Residence_type'], mapping=self.mapping_binary)
        df_cleaned = self.fill_NaN(df_cleaned, column='bmi', value=df_cleaned['bmi'].median())
        
        return df_cleaned