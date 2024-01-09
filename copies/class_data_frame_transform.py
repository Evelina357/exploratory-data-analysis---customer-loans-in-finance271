
import pandas as pd 
import numpy as np 

from scipy.stats import yeojohnson 
from scipy import stats
import seaborn as sns

class DataFrameTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def impute_with_median(self, column_name):
        self.df_raw[column_name] = self.df_raw[column_name].fillna(self.df_raw[column_name].median())
        print(self.df_raw[column_name].info(), "\n")
        
    def impute_with_mode(self, column_name):
        self.df_raw[column_name] = self.df_raw[column_name].fillna(self.df_raw[column_name].mode()[0])
        print(self.df_raw[column_name].info(), "\n")
    
    def drop_rows(self, column_name):
        self.df_raw.dropna(how="any", subset=[column_name], inplace=True)
        print(self.df_raw[column_name].info(), "\n")
    
    def drop_columns(self, column_name):
        self.df_raw.drop(column_name, axis=1, inplace=True)
        
    def transform_skew_square_root(self, column_name):
        self.df_raw[column_name] = self.df_raw[column_name]**(1/2)
            
    def transform_skew_log(self, column_name):
        self.df_raw[column_name] = self.df_raw[column_name].map(lambda i: np.log(i) if i>0 else 0)
            
    def transform_skew_yeojohnson(self, column_name):
        yeojohnson_colmn = self.df_raw[column_name]
        yeojohnson_colmn = stats.yeojohnson(yeojohnson_colmn)
        yeojohnson_colmn = pd.Series(yeojohnson_colmn[0])
        self.df_raw[column_name] = yeojohnson_colmn
        
    def categ_data_outliers_transform(self):
        self.df_raw["home_ownership"].replace({"NONE": "OTHER"}, inplace=True)
        self.df_raw["loan_status"].replace({"Does not meet the credit policy. Status:Charged Off": "Charged Off"}, inplace=True)
        self.df_raw["loan_status"].replace({"Does not meet the credit policy. Status:Fully Paid": "Fully Paid"}, inplace=True)
        self.df_raw["verification_status"].replace({"Source Verified": "Verified"}, inplace=True)
        self.df_raw.drop(self.df_raw[self.df_raw['payment_plan'] == 'y'].index, inplace=True)
        
    def iqr_outliers_removal(self, column_name):
        print(f'The length of data frame BEFORE removal of {column_name} outlier: {len(self.df_raw)}')
        Q1 = np.quantile(self.df_raw[column_name], 0.25)
        Q3 = np.quantile(self.df_raw[column_name], 0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        outliers = self.df_raw[(self.df_raw[column_name] < (Q1 - (threshold * IQR))) | (self.df_raw[column_name] > (Q3 + (threshold * IQR)))]
        self.df_raw.drop(index=outliers.index, inplace=True)
        print(f'The length of data frame AFTER removal of {column_name} outlier: {len(self.df_raw)}\n')
        
    def flooring_capping_outliers_transform(self, column_name):
        print(f"The min value of {column_name} BEFORE outlier transformation is {self.df_raw[column_name].min()}")
        print(f"The max value of {column_name} BEFORE outlier transformation is {self.df_raw[column_name].max()}\n")
        flooring = np.quantile(self.df_raw[column_name], 0.10)
        capping = np.quantile(self.df_raw[column_name], 0.90)
        print(f'The flooring for the lower values of {column_name} is: {flooring}')
        print(f'The capping for the higher values of {column_name} is: {capping}\n')
        self.df_raw[column_name] = np.where(self.df_raw[column_name] < flooring, flooring, self.df_raw[column_name])
        self.df_raw[column_name] = np.where(self.df_raw[column_name] > capping, capping, self.df_raw[column_name])
        print(f"The min value of {column_name} AFTER outlier transformation is {self.df_raw[column_name].min()}")
        print(f"The max value of {column_name} AFTER outlier transformation is {self.df_raw[column_name].max()}\n")
        
# df_transform = DataFrameTransform(df_raw)

# df_transform.impute_with_median()

# df_transform.impute_with_mode()

# df_transform.drop_rows()

# df_transform.drop_columns()


# df_transform.transform_skew_square_root()

# df_transform.transform_skew_log()

# df_transform.transform_skew_yeojohnson()


# df_transform.categ_data_outliers_transform()

# df_transform.iqr_outliers_removal()

# df_transform.flooring_capping_outliers_transform()

