# %%
import pandas as pd
import numpy as np 

class DataFrameInfo:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def show_colm_dtypes(self):
        self.df_raw.info()
        print("\n")
        
    def show_colm_head(self):
        all_columns = ["loan_amount", "funded_amount", "funded_amount_inv", "int_rate", "instalment", "annual_inc", "issue_date", "dti", "delinq_2yrs", "earliest_credit_line", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_accounts", "total_accounts", "out_prncp", "out_prncp_inv", "total_payment", "total_payment_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_date", "last_payment_amount", "next_payment_date", "last_credit_pull_date", "collections_12_mths_ex_med", "mths_since_last_major_derog"]
        for column in all_columns:
            print(self.df_raw[column].head(25))
            
    def show_colm_tail(self):
        all_columns = ["loan_amount", "funded_amount", "funded_amount_inv", "int_rate", "instalment", "annual_inc", "issue_date", "dti", "delinq_2yrs", "earliest_credit_line", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_accounts", "total_accounts", "out_prncp", "out_prncp_inv", "total_payment", "total_payment_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_date", "last_payment_amount", "next_payment_date", "last_credit_pull_date", "collections_12_mths_ex_med", "mths_since_last_major_derog"]
        for column in all_columns:
            print(self.df_raw[column].tail(25))
        
    def show_colm_stat_values(self):
        print(self.df_raw.describe())
        
    def show_categ_colm_values_count(self):
        cat_colm = [self.df_raw["term"], self.df_raw["grade"], self.df_raw["sub_grade"], self.df_raw["employment_length"], self.df_raw["home_ownership"], self.df_raw["verification_status"], self.df_raw["loan_status"], self.df_raw["payment_plan"], self.df_raw["purpose"], self.df_raw["policy_code"], self.df_raw["application_type"] ]
        for colm in cat_colm:
            print(colm.value_counts(), "\n")
        
    def show_df_shape(self):
        print(f"The shape of the data frame is: {self.df_raw.shape}", "\n")
        
    def show_null_percentage(self):
        print(f"The percentage of Nan values is each column: \n{self.df_raw.isnull().sum()/len(self.df_raw)}")
        
#info = DataFrameInfo(df_raw)

#info.show_colm_dtypes()

#info.show_colm_head()

#info.show_colm_tail()

#info.show_colm_stat_values()  

#info.show_categ_colm_values_count() 

#info.show_df_shape()

#info.show_null_percentage()
# %%
