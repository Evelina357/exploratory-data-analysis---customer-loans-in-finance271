import pandas as pd

def load_data():
    raw_data = pd.read_csv("C:/Users/eveli/ai_core/EDA/loan_payments.csv")
    return raw_data

df_raw = load_data()


class DataTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def change_to_int(self, column):
        self.df_raw[column] = self.df_raw[column].astype("int64")
    
    def change_to_datetime(self, column):
        self.df_raw[column] = pd.to_datetime(self.df_raw[column])
        
    def round_values(self, column):
        self.df_raw[column] = df_raw[column].apply(lambda x:round(x,2))
        
    def organise_values(self):
        self.df_raw["term"].replace({"36 months": "36"}, inplace=True)
        self.df_raw["term"].replace({"60 months": "60"}, inplace=True)
        self.df_raw["loan_status"].replace({"Does not meet the credit policy. Status:Charged Off": "Charged Off"}, inplace=True)
        self.df_raw["loan_status"].replace({"Does not meet the credit policy. Status:Fully Paid": "Fully Paid"}, inplace=True)
        self.df_raw["verification_status"].replace({"Source Verified": "Verified"}, inplace=True)
    
#transform = DataTransform(df_raw)

#transform.change_to_int()

#transform.change_to_datetime()

#transform.round_values()

#transform.organise_values()
