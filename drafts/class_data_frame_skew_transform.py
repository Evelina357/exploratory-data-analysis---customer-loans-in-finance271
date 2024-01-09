
import pandas as pd 
from scipy.stats import yeojohnson 
from scipy import stats
import numpy as np 
import seaborn as sns

class DataFrameSkewTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def transform_skew_square_root(self, column_name):
        self.df_raw[column_name] = self.df_raw[column_name]**(1/2)
            
    def transform_skew_log(self, column_name):
        self.df_raw[column_name] = self.df_raw[column_name].map(lambda i: np.log(i) if i>0 else 0)
            
    def transform_skew_yeojohnson(self, column_name):
        yeojohnson_colmn = self.df_raw[column_name]
        yeojohnson_colmn = stats.yeojohnson(yeojohnson_colmn)
        yeojohnson_colmn = pd.Series(yeojohnson_colmn[0])
        self.df_raw[column_name] = yeojohnson_colmn
            
# skew_transform = DataFrameSkewTransform(df_raw)

# skew_transform.transform_skew_square_root()

# skew_transform.transform_skew_log()

# skew_transform.transform_skew_yeojohnson()