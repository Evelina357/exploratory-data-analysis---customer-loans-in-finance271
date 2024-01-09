
import pandas as pd 
import numpy as np 

class DataFrameOutliersTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw

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
        
    def flooring_capping_outliers_removal(self, column_name):
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
            
# outl_transform = DataFrameOutliersTransform(df_raw)

# outl_transform.categ_data_outliers_transform()

# outl_transform.iqr_outliers_removal()

# outl_transform.flooring_capping_outliers_removal()
