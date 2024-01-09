import pandas as pd

def load_data():
    raw_data = pd.read_csv("C:/Users/eveli/ai_core/EDA/loan_payments.csv")
    return raw_data

df_raw = load_data()


class DataTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def float_to_int(self, column):
        self.df_raw[column] = self.df_raw[column].astype("int64")
    
    def object_to_datetime(self, column):
        self.df_raw[column] = pd.to_datetime(self.df_raw[column])
        
    def round_values(self, column):
        self.df_raw[column] = df_raw[column].apply(lambda x:round(x,2))      
    
#transform = DataTransform(df_raw)

#transform.float_to_int()

#transform.object_to_datetime()

#transform.round_values()
