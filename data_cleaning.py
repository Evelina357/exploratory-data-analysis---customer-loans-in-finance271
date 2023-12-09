# %%
import pandas as pd

def load_data():
    raw_data = pd.read_csv("C:/Users/eveli/ai_core/EDA/loan_payments.csv")
    return raw_data

df_raw = load_data()
'''
The function loads the "loan_payments.csv" database from the local computer.

df_raw calls the function which returns the locally stored database in a table format. 
'''

# %%
class DataTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def float_to_int(self):
        self.df_raw.funded_amount_inv = self.df_raw["funded_amount_inv"].astype("int64")
        self.df_raw.annual_inc = self.df_raw["annual_inc"].astype("int64")
        
    def object_to_datetime(self):
        self.df_raw.issue_date = pd.to_datetime(self.df_raw.issue_date)
        self.df_raw.earliest_credit_line = pd.to_datetime(self.df_raw.earliest_credit_line)
        self.df_raw.last_payment_date = pd.to_datetime(self.df_raw.last_payment_date)
        self.df_raw.next_payment_date = pd.to_datetime(self.df_raw.next_payment_date)
        self.df_raw.last_credit_pull_date = pd.to_datetime(self.df_raw.last_credit_pull_date)
        
    def organise_values(self):
        self.df_raw["verification_status"].replace({"Source Verified": "Verified"}, inplace=True)
        self.df_raw["total_rec_late_fee"] = df_raw["total_rec_late_fee"].apply(lambda x:round(x,2))
        self.df_raw["collection_recovery_fee"] = df_raw["collection_recovery_fee"].apply(lambda x:round(x,2))      
    
transform = DataTransform(df_raw)

int_dtype = transform.float_to_int()

datetime_dtype = transform.object_to_datetime()

org_dtype = transform.organise_values()

df_raw.info()

# %%

class DataFrameInfo:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def show_colm_dtypes(self):
        self.df_raw.info()
        
    def show_colm_stat_values(self):
        print(self.df_raw.describe())
        
    def show_categ_colm_values_count(self):
        print(self.df_raw["term"].value_counts(), "\n")
        print(self.df_raw["grade"].value_counts(), "\n")
        print(self.df_raw["sub_grade"].value_counts(), "\n")
        print(self.df_raw["employment_length"].value_counts(), "\n")
        print(self.df_raw["home_ownership"].value_counts(), "\n")
        print(self.df_raw["verification_status"].value_counts(), "\n")
        print(self.df_raw["loan_status"].value_counts(), "\n")
        print(self.df_raw["payment_plan"].value_counts(), "\n")
        print(self.df_raw["purpose"].value_counts(), "\n")
        print(self.df_raw["policy_code"].value_counts(), "\n")
        print(self.df_raw["application_type"].value_counts(), "\n")
        
    def show_df_shape(self):
        print(self.df_raw.shape, "\n")
        
    def show_null_percentage(self):
        print(self.df_raw.isnull().sum()/len(self.df_raw))
        
info = DataFrameInfo(df_raw)

colm_dtype = info.show_colm_dtypes()

colm_stats = info.show_colm_stat_values()  

value_count = info.show_categ_colm_values_count() 

df_shape = info.show_df_shape()

null_percent = info.show_null_percentage()

# %%
import seaborn as sns
import matplotlib.pyplot as plot
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
import missingno as msno

class Plotter:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def show_null_values(self):
        print(self.df_raw.isnull().sum()/len(self.df_raw), "\n")
        msno.matrix(self.df_raw) 
        
    def test_norm_distr_funded_amount(self):
        data = self.df_raw["funded_amount"]
        stat, p = normaltest(data, nan_policy='omit')
        print(" Statistics=%.3f, p=%.3f" % (stat, p), "\n")
        qq_plot = qqplot(self.df_raw["funded_amount"], scale=1, line='q')
        plot.title("The distribution of funded_amount values")
        plot.show()
        print(f'\nThe median of funded_amount is {df_raw["funded_amount"].median()}')
        print(f'The mean of funded_amount is {df_raw["funded_amount"].mean()}')
        
    def test_norm_distr_int_rate(self):
        data = self.df_raw["int_rate"]
        stat, p = normaltest(data, nan_policy='omit')
        print(" Statistics=%.3f, p=%.3f" % (stat, p), "\n")
        qq_plot = qqplot(self.df_raw["int_rate"], scale=1, line='q')
        plot.title("The distribution of int_rate values")
        plot.show()
        print(f'\nThe median of int_rate is {df_raw["int_rate"].median()}')
        print(f'The mean of int_rate is {df_raw["int_rate"].mean()}')
        
    def test_norm_distr_collections_12_mths_ex_med(self):
        data = self.df_raw["collections_12_mths_ex_med"]
        stat, p = normaltest(data, nan_policy='omit')
        print(" Statistics=%.3f, p=%.3f" % (stat, p), "\n")
        qq_plot = qqplot(self.df_raw["collections_12_mths_ex_med"], scale=1, line='q')
        plot.title("The distribution of collections_12_mths_ex_med values")
        plot.show()
        print(f'\nThe median of collections_12_mths_ex_med is {df_raw["collections_12_mths_ex_med"].median()}')
        print(f'The mean of collections_12_mths_ex_med is {df_raw["collections_12_mths_ex_med"].mean()}')
        
    def show_disc_prob_distr_term(self):
        plot.rc("axes.spines", top=False, right=False)
        probs = self.df_raw["term"].value_counts(normalize=True)
        dpd=sns.barplot(y=probs.index, x=probs.values, color='b')
        plot.xlabel("Values")
        plot.ylabel("Probability")
        plot.title("Discrete Probability Distribution")
        plot.show()
        print("\nThe value counts in  ")
        print(self.df_raw["term"].value_counts())
        print(f'\nThe mode of term column is {self.df_raw["term"].mode()[0]}')
        
    def show_disc_prob_distr_employment_length(self):
        plot.rc("axes.spines", top=False, right=False)
        probs = self.df_raw["employment_length"].value_counts(normalize=True)
        dpd=sns.barplot(y=probs.index, x=probs.values, color='b')
        plot.xlabel("Values")
        plot.ylabel("Probability")
        plot.title("Discrete Probability Distribution")
        plot.show()
        print("\nThe value counts in  ")
        print(self.df_raw["employment_length"].value_counts())
        print(f'\nThe mode of employment_length column is {self.df_raw["employment_length"].mode()[0]}', "\n")
        
    def show_skewness_loan_amount(self):
        self.df_raw["loan_amount"].hist(bins=50)
        print(f"Skew of loan_amount column is {self.df_raw['loan_amount'].skew()}", "\n") #This indicates a moderate positive skew. The data requires a transformation.
        
    def show_skewness_funded_amount(self):
        self.df_raw["funded_amount"].hist(bins=50)
        print(f"Skew of funded_amount column is {self.df_raw['funded_amount'].skew()}", "\n") #This indicates a moderate positive skew. The data requires a transformation.
        
    def show_skewness_funded_amount_inv(self):
        self.df_raw["funded_amount_inv"].hist(bins=50)
        print(f"Skew of funded_amount_inv column is {self.df_raw['funded_amount_inv'].skew()}", "\n") #This indicates a moderate positive skew. The data requires a transformation.
        
    def show_skewness_int_rate(self):
        self.df_raw["int_rate"].hist(bins=50)
        print(f"Skew of int_rate column is {self.df_raw['int_rate'].skew()}", "\n") #This indicates that a distribution of data is approximately symmetric. Data does not require a transformation.
        
    def show_skewness_instalment(self):
        self.df_raw["instalment"].hist(bins=50)
        print(f"Skew of instalment column is {self.df_raw['instalment'].skew()}", "\n") #This indicates a moderate positive skew. The data requires a transformation.
        
    def show_skewness_annual_inc(self):
        self.df_raw["annual_inc"].hist(bins=50)
        print(f"Skew of annual_inc column is {self.df_raw['annual_inc'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_dti(self):
        self.df_raw["dti"].hist(bins=50)
        print(f"Skew of dti column is {self.df_raw['dti'].skew()}", "\n") #This indicates that a distribution of data is approximately symmetric. Data does not require a transformation.
        
    def show_skewness_delinq_2yrs(self):
        self.df_raw["delinq_2yrs"].hist(bins=50)
        print(f"Skew of delinq_2yrs column is {self.df_raw['delinq_2yrs'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_inq_last_6mths(self):
        self.df_raw["inq_last_6mths"].hist(bins=50)
        print(f"Skew of inq_last_6mths column is {self.df_raw['inq_last_6mths'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_open_accounts(self):
        self.df_raw["open_accounts"].hist(bins=50)
        print(f"Skew of open_accounts column is {self.df_raw['open_accounts'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_total_accounts(self):
        self.df_raw["total_accounts"].hist(bins=50)
        print(f"Skew of total_accounts column is {self.df_raw['total_accounts'].skew()}", "\n") #This indicates a moderate positive skew. The data requires a transformation.
        
    def show_skewness_out_prncp(self):
        self.df_raw["out_prncp"].hist(bins=50)
        print(f"Skew of out_prncp column is {self.df_raw['out_prncp'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_out_prncp_inv(self):
        self.df_raw["out_prncp_inv"].hist(bins=50)
        print(f"Skew of out_prncp_inv column is {self.df_raw['out_prncp_inv'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_total_payment(self):
        self.df_raw["total_payment"].hist(bins=50)
        print(f"Skew of total_payment column is {self.df_raw['total_payment'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
    def show_skewness_total_rec_prncp(self):
        self.df_raw["total_rec_prncp"].hist(bins=50)
        print(f"Skew of total_rec_prncp column is {self.df_raw['total_rec_prncp'].skew()}", "\n") #This indicates a strong positive skew. The data requires a transformation.
        
visuals = Plotter(df_raw)

null_values = visuals.show_null_values()

nd_funded_amount = visuals.test_norm_distr_funded_amount()

nd_int_rate = visuals.test_norm_distr_int_rate()

nd_collections_12_mths_ex_med = visuals.test_norm_distr_collections_12_mths_ex_med()

dpd_term = visuals.show_disc_prob_distr_term()

dpd_employment_length = visuals.show_disc_prob_distr_employment_length()

skew_loan_amount = visuals.show_skewness_loan_amount()

skew_funded_amount = visuals.show_skewness_funded_amount()

skew_funded_amount_inv = visuals.show_skewness_funded_amount_inv()

skew_int_rate = visuals.show_skewness_int_rate()

skew_instalment = visuals.show_skewness_instalment()

skew_annual_inc = visuals.show_skewness_annual_inc()

skew_dti = visuals.show_skewness_dti()

skew_delinq_2yrs = visuals.show_skewness_delinq_2yrs()

skew_inq_last_6mths = visuals.show_skewness_inq_last_6mths()

skew_open_accounts = visuals.show_skewness_open_accounts()

skew_total_accounts = visuals.show_skewness_total_accounts()

skew_out_prncp = visuals.show_skewness_out_prncp()

skew_out_prncp_inv = visuals.show_skewness_out_prncp_inv()

skew_total_payment = visuals.show_skewness_total_payment()

skew_total_rec_prncp = visuals.show_skewness_total_rec_prncp()
# %%
class DataFrameTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def impute_funded_amount(self):
        self.df_raw["funded_amount"] = self.df_raw["funded_amount"].fillna(self.df_raw["funded_amount"].median())
        print(self.df_raw["funded_amount"].info(), "\n")
        
    def impute_int_rate(self):
        self.df_raw["int_rate"] = self.df_raw["int_rate"].fillna(self.df_raw["int_rate"].median())
        print(self.df_raw["int_rate"].info(), "\n")
        
    def impute_collections_12_mths_ex_med(self):
        self.df_raw["collections_12_mths_ex_med"] = self.df_raw["collections_12_mths_ex_med"].fillna(self.df_raw["collections_12_mths_ex_med"].median())
        print(self.df_raw["collections_12_mths_ex_med"].info(), "\n")
        
    def impute_term(self):
        self.df_raw["term"] = self.df_raw["term"].fillna(self.df_raw["term"].mode()[0])
        print(self.df_raw["term"].info(), "\n")
        
    def impute_employment_length(self):
        self.df_raw["employment_length"] = self.df_raw["employment_length"].fillna(self.df_raw["employment_length"].mode()[0])
        print(self.df_raw["employment_length"].info(), "\n")
    
    def drop_rows_last_payment_date(self):
        self.df_raw.dropna(how="any", subset=["last_payment_date"], inplace=True)
        print(self.df_raw["last_payment_date"].info(), "\n")
        
    def drop_rows_last_credit_pull_date(self):
        self.df_raw.dropna(how="any", subset=["last_credit_pull_date"], inplace=True)
        print(self.df_raw["last_credit_pull_date"].info(), "\n")
    
    def drop_columns(self):
        self.df_raw = self.df_raw.drop(columns=["mths_since_last_delinq", "mths_since_last_record", "next_payment_date", "mths_since_last_major_derog"], inplace=True)
        
        
df_transform = DataFrameTransform(df_raw)

no_nan_funded_amount = df_transform.impute_funded_amount()

no_nan_int_rate = df_transform.impute_int_rate()

no_nan_coll_12_mths_ex_med = df_transform.impute_collections_12_mths_ex_med()

no_nan_term = df_transform.impute_term()

no_nan_employment_length = df_transform.impute_employment_length()

df_transform.drop_rows_last_payment_date()

df_transform.drop_rows_last_credit_pull_date()

no_columns = df_transform.drop_columns()

visuals.show_null_values()

# %%
m4_df = df_raw.copy()
# %%
m4_df.info()
# %%
from scipy.stats import yeojohnson 
from scipy import stats
import numpy as np 
import seaborn as sns

class DataFrameSkewTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def transform_skew_loan_amount(self):
        self.df_raw["loan_amount"] = self.df_raw.loan_amount**(1/2)
        visuals.show_skewness_loan_amount() #Result of square Root Transformation: 0.165
        
    def transform_skew_funded_amount(self):
        self.df_raw["funded_amount"] = self.df_raw.funded_amount**(1/2) 
        visuals.show_skewness_funded_amount() # Result of Square Root Transformation is working: 0.180
        
    def transform_skew_funded_amount_inv(self):
        self.df_raw["funded_amount_inv"] = self.df_raw.funded_amount_inv**(1/2)
        visuals.show_skewness_funded_amount_inv() # Result of square Root Transformation: 0.072
        
    def transform_skew_instalment(self):
        self.df_raw["instalment"] = self.df_raw.instalment**(1/2) # Result of square Root Transformation: 0.246
        visuals.show_skewness_instalment()
        
    def transform_skew_annual_inc(self):
        self.df_raw["annual_inc"] = self.df_raw["annual_inc"].map(lambda i: np.log(i) if i>0 else 0)
        t=sns.histplot(self.df_raw["annual_inc"],label="Skewness: %.2f"%(self.df_raw["annual_inc"].skew()))
        t.legend() # Result of log transformation: 0.14
        visuals.show_skewness_annual_inc()
        
    def transform_skew_delinq_2yrs(self):
        yeojohnson_delinq_2yrs = self.df_raw["delinq_2yrs"]
        yeojohnson_delinq_2yrs = stats.yeojohnson(yeojohnson_delinq_2yrs)
        yeojohnson_delinq_2yrs = pd.Series(yeojohnson_delinq_2yrs[0])
        self.df_raw["delinq_2yrs"] = yeojohnson_delinq_2yrs
        t=sns.histplot(yeojohnson_delinq_2yrs, label="Skewness: %.2f"%(yeojohnson_delinq_2yrs.skew()))
        t.legend() #Result of Yeo-Johnson transformation: 1.87(lowest result in comparison to other methods).
        visuals.show_skewness_delinq_2yrs()
        
    def transform_skew_inq_last_6mths(self):
        yeojohnson_inq_last_6mths = self.df_raw["inq_last_6mths"]
        yeojohnson_inq_last_6mths = stats.yeojohnson(yeojohnson_inq_last_6mths)
        yeojohnson_inq_last_6mths = pd.Series(yeojohnson_inq_last_6mths[0])
        self.df_raw["inq_last_6mths"] = yeojohnson_inq_last_6mths
        t=sns.histplot(yeojohnson_inq_last_6mths, label="Skewness: %.2f"%(yeojohnson_inq_last_6mths.skew()))
        t.legend() # Result of Yeo-Johnson transformation: 0.25
        visuals.show_skewness_inq_last_6mths()
        
    def transform_skew_open_accounts(self):
        self.df_raw["open_accounts"] = self.df_raw["open_accounts"].map(lambda i: np.log(i) if i>0 else 0)
        t=sns.histplot(self.df_raw["open_accounts"],label="Skewness: %.2f"%(self.df_raw["open_accounts"].skew()))
        t.legend() # Result of a log transformation: -0.47
        visuals.show_skewness_open_accounts()
        
    def transform_skew_total_accounts(self):
        self.df_raw["total_accounts"] = self.df_raw.total_accounts**(1/2) # Result of a Square Root Transformation: 0.109
        visuals.show_skewness_total_accounts()
        
    def transform_skew_out_prncp(self):
        yeojohnson_out_prncp = self.df_raw["out_prncp"]
        yeojohnson_out_prncp = stats.yeojohnson(yeojohnson_out_prncp)
        yeojohnson_out_prncp = pd.Series(yeojohnson_out_prncp[0])
        self.df_raw["out_prncp"] = yeojohnson_out_prncp
        t=sns.histplot(yeojohnson_out_prncp, label="Skewness: %.2f"%(yeojohnson_out_prncp.skew()))
        t.legend() # Result of Yeo-Johnson method: 0.53 (lowest result)
        visuals.show_skewness_out_prncp()
        
    def transform_skew_out_prncp_inv(self):
        yeojohnson_out_prncp_inv = self.df_raw["out_prncp_inv"]
        yeojohnson_out_prncp_inv = stats.yeojohnson(yeojohnson_out_prncp_inv)
        yeojohnson_out_prncp_inv = pd.Series(yeojohnson_out_prncp_inv[0])
        self.df_raw["out_prncp_inv"] = yeojohnson_out_prncp_inv
        t=sns.histplot(yeojohnson_out_prncp_inv, label="Skewness: %.2f"%(yeojohnson_out_prncp_inv.skew()))
        t.legend() # Result of Yeo-Johnson method: 0.53 (lowest result)
        visuals.show_skewness_out_prncp_inv()
        
    def transform_skew_total_payment(self):
        self.df_raw["total_payment"] = self.df_raw.total_payment**(1/2) # Result of Square Root transformation: 0.374
        visuals.show_skewness_total_payment()
        
    def transform_skew_total_rec_prncp(self):
        self.df_raw["total_rec_prncp"] = self.df_raw.total_rec_prncp**(1/2) # Result of Square Root transformation: 0.369
        visuals.show_skewness_total_rec_prncp()
        
        
skew_transform = DataFrameSkewTransform(df_raw)

skew_transform.transform_skew_loan_amount()

skew_transform.transform_skew_funded_amount()

skew_transform.transform_skew_funded_amount_inv()

skew_transform.transform_skew_instalment()

skew_transform.transform_skew_annual_inc()

skew_transform.transform_skew_delinq_2yrs()

skew_transform.transform_skew_inq_last_6mths()

skew_transform.transform_skew_open_accounts()

skew_transform.transform_skew_total_accounts()

skew_transform.transform_skew_out_prncp()

skew_transform.transform_skew_out_prncp_inv()

skew_transform.transform_skew_total_payment()

skew_transform.transform_skew_total_rec_prncp()
    
# %%
df_raw.info()

# %%
plot.figure(figsize=(10, 5))
sns.histplot(df_raw["loan_amount"], bins=40, kde=False)
plot.title("Histogram of loan amount")
plot.xlabel('£ (pounds)')
plot.ylabel('Frequency')
plot.show()

# %%
import seaborn as sns

sns.violinplot(data=df_raw, y="loan_amount")
sns.despine()

# %%

column_of_interest = 'loan_amount'
plot.figure(figsize=(10, 6)) 
sns.boxplot(x=df_raw[column_of_interest], color='lightblue', showfliers=False)
sns.stripplot(x=df_raw[column_of_interest], color='black', size=4, jitter=True)
plot.title(f'Box plot for {column_of_interest}') 
plot.xlabel('') 
plot.show()

# %%
column_of_interest = 'loan_amount'
df_raw.boxplot(column=column_of_interest)
plot.show()
# %%
column_of_interest = 'funded_amount_inv'
df_raw.boxplot(column=column_of_interest)
plot.show()
# %%
Q1 = df_raw['loan_amount'].quantile(0.25)
Q3 = df_raw['loan_amount'].quantile(0.75)
IQR = Q3 - Q1

threshold = 1.5
outliers = df_raw[(df_raw['loan_amount'] < Q1 - threshold * IQR) | (df_raw['loan_amount'] > Q3 + threshold * IQR)]

#print(IQR)
print(outliers)










































































































# %%
df_raw["loan_amount"].hist(bins=50)
print(f"Skew of loan_amount column is {df_raw['loan_amount'].skew()}") #skew: 0.805... moderate positive skew
print("This indicates a moderate positive skew.")
# %%
df_raw["loan_amount"] = df_raw.loan_amount**(1/2) #Square Root Transformation is working: 0.165
# %%--------




df_raw["funded_amount"].hist(bins=50)
print(f"Skew of funded_amount column is {df_raw['funded_amount'].skew()}") #skew: 0.869... moderate positive skew
print("This indicates a moderate positive skew")
# %%
df_raw["funded_amount"] = df_raw.funded_amount**(1/2) #Square Root Transformation is working: 0.180
#%% -----




#---------------------------------------------------------------------------
df_raw["funded_amount_inv"].hist(bins=50)
print(f"Skew of funded_amount_inv column is {df_raw['funded_amount_inv'].skew()}") #skew: 0.813..moderate positive skew
print("This indicates a moderate positive skew")
# %%
df_raw["funded_amount_inv"] = df_raw.funded_amount_inv**(1/2) #Square Root Transformation is working: 0.072
#%%-----------------------------------------------------------------------------





#------------------------------ no change needed for this column
df_raw["int_rate"].hist(bins=50)
print(f"Skew of int_rate column is {df_raw['int_rate'].skew()}") #skew: 0.456...distr is approx symmetric
print("This indicates that a distribution is approx. symmetric.") # no need to change anything ----
# %% -----------------------------------





df_raw["instalment"].hist(bins=50)
print(f"Skew of instalment column is {df_raw['instalment'].skew()}") #skew: 0.996...moderate positive skew
print("This indicates a moderate positive skew.")
# %%
df_raw["instalment"] = df_raw.instalment**(1/2) #Square Root Transformation is working: 0.246
# %%-------------------------------------------------------------------



import numpy as np 

df_raw["annual_inc"].hist(bins=50)
print(f"Skew of annual_inc column is {df_raw['annual_inc'].skew()}") #skew: 8.71...strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["annual_inc"] = df_raw["annual_inc"].map(lambda i: np.log(i) if i>0 else 0)
t=sns.histplot(df_raw["annual_inc"],label="Skewness: %.2f"%(df_raw["annual_inc"].skew()))
t.legend() # log transformation working: 0.14
# %%----------------------------------------------------------




#-----------------------------------no need to change
df_raw["dti"].hist(bins=50)
print(f"Skew of dti column is {df_raw['dti'].skew()}") #skew: 0.189...distr is approx.symmetric
print("This indicates that a distribution is approx. symmetric.") # no need to transform
# %%-------------------------------------------------------






df_raw["delinq_2yrs"].hist(bins=50)
print(f"Skew of delinq_2yrs column is {df_raw['delinq_2yrs'].skew()}") #skew: 5.37...strong positive skew
print("This indicates a strong positive skew.")
# %%
from scipy.stats import yeojohnson 
from scipy import stats

yeojohnson_delinq_2yrs = df_raw["delinq_2yrs"]
yeojohnson_delinq_2yrs = stats.yeojohnson(yeojohnson_delinq_2yrs)
yeojohnson_delinq_2yrs = pd.Series(yeojohnson_delinq_2yrs[0])
df_raw["delinq_2yrs"] = yeojohnson_delinq_2yrs
t=sns.histplot(yeojohnson_delinq_2yrs, label="Skewness: %.2f"%(yeojohnson_delinq_2yrs.skew()))
t.legend() # trialing Yeo-Johnson transformation skewness went down to 1.87...not perfect yet.
# %%
df_raw["delinq_2yrs"] = df_raw.delinq_2yrs**(1/2) # Result of Root Square: 2.36...not good enough!
# %%
df_raw["delinq_2yrs"] = df_raw["delinq_2yrs"].map(lambda i: np.log(i) if i>0 else 0)
t=sns.histplot(df_raw["delinq_2yrs"],label="Skewness: %.2f"%(df_raw["delinq_2yrs"].skew()))
t.legend() # trialing log transformation: 5.41... useless!
# %%
from scipy import stats
boxcox_delinq_2yrs = df_raw["delinq_2yrs"]
boxcox_delinq_2yrs = stats.boxcox(boxcox_delinq_2yrs)
boxcox_delinq_2yrs = pd.Series(boxcox_delinq_2yrs[0])
t=sns.histplot(boxcox_delinq_2yrs, label="Skewness: %.2f"%(boxcox_delinq_2yrs.skew()))
t.legend() # box cox transformation: DATA MUST BE POSITIVE...CANT USE THIS METHOD. 
# %%--------







df_raw["inq_last_6mths"].hist(bins=50)
print(f"Skew of inq_last_6mths column is {df_raw['inq_last_6mths'].skew()}") #skew: 3.25...strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["inq_last_6mths"] = df_raw["inq_last_6mths"].map(lambda i: np.log(i) if i>0 else 0)
t=sns.histplot(df_raw["inq_last_6mths"],label="Skewness: %.2f"%(df_raw["inq_last_6mths"].skew()))
t.legend() # Results of log transformation: 1.97. not good enough....
# %%
df_raw["inq_last_6mths"] = df_raw.inq_last_6mths**(1/2) # Result of Root Square transformation: 0.553 very close but still not good enough!
# %%
yeojohnson_inq_last_6mths = df_raw["inq_last_6mths"]
yeojohnson_inq_last_6mths = stats.yeojohnson(yeojohnson_inq_last_6mths)
yeojohnson_inq_last_6mths = pd.Series(yeojohnson_inq_last_6mths[0])
t=sns.histplot(yeojohnson_inq_last_6mths, label="Skewness: %.2f"%(yeojohnson_inq_last_6mths.skew()))
t.legend() # yeojohnson transformation works: 0.25 skew now I just need to make it permanent on the data!
# %%
df_raw["inq_last_6mths"] = stats.yeojohnson(df_raw["inq_last_6mths"])
df_raw["inq_last_6mths"] = pd.Series(df_raw["inq_last_6mths"][0])
t=sns.histplot(df_raw["inq_last_6mths"], label="Skewness: %.2f"%(df_raw["inq_last_6mths"].skew()))
t.legend()
# %%
yeojohnson_inq_last_6mths = df_raw["inq_last_6mths"]
yeojohnson_inq_last_6mths = stats.yeojohnson(yeojohnson_inq_last_6mths)
yeojohnson_inq_last_6mths = pd.Series(yeojohnson_inq_last_6mths[0])
df_raw["inq_last_6mths"] = yeojohnson_inq_last_6mths
t=sns.histplot(yeojohnson_inq_last_6mths, label="Skewness: %.2f"%(yeojohnson_inq_last_6mths.skew()))
t.legend() # permanent way? 0.25
# %%-------------------------------------------------





df_raw["open_accounts"].hist(bins=50)
print(f"Skew of open_accounts column is {df_raw['open_accounts'].skew()}") #skew: 1.059...strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["open_accounts"] = df_raw["open_accounts"].map(lambda i: np.log(i) if i>0 else 0)
t=sns.histplot(df_raw["open_accounts"],label="Skewness: %.2f"%(df_raw["open_accounts"].skew()))
t.legend() #Result of lof transformation: -0.47
# %%------------------------------------------






df_raw["total_accounts"].hist(bins=50)
print(f"Skew of total_accounts column is {df_raw['total_accounts'].skew()}") #skew: 0.779...moderate positive skew
print("This indicates a moderate positive skew.")
# %%
df_raw["total_accounts"] = df_raw.total_accounts**(1/2) #Result of Square Root Transformation is working: 0.109
# %%-------------------------------------------




df_raw["out_prncp"].hist(bins=50)
print(f"Skew of out_prncp column is {df_raw['out_prncp'].skew()}") #skew: 2.35..strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["out_prncp"] = df_raw.out_prncp**(1/2) #Results of Root Square transformation: 1.225... not good enough!
# %%
df_raw["out_prncp"] = df_raw["out_prncp"].map(lambda i: np.log(i) if i>0 else 0)
t=sns.histplot(df_raw["out_prncp"],label="Skewness: %.2f"%(df_raw["out_prncp"].skew()))
t.legend() #trialing log transformation: 0.58..very close, but still NOT GOOD ENOUGH!
# %%
from scipy import stats
boxcox_out_prncp = df_raw["out_prncp"]
boxcox_out_prncp = stats.boxcox(boxcox_out_prncp)
boxcox_out_prncp = pd.Series(boxcox_out_prncp[0])
t=sns.histplot(boxcox_out_prncp, label="Skewness: %.2f"%(boxcox_out_prncp.skew()))
t.legend() # boxcox transformation:  DATA MUST BE POSITIVE...METHOD CANNOT BE USED. 
# %%
yeojohnson_out_prncp = df_raw["out_prncp"]
yeojohnson_out_prncp = stats.yeojohnson(yeojohnson_out_prncp)
yeojohnson_out_prncp = pd.Series(yeojohnson_out_prncp[0])
df_raw["out_prncp"] = yeojohnson_out_prncp
t=sns.histplot(yeojohnson_out_prncp, label="Skewness: %.2f"%(yeojohnson_out_prncp.skew()))
t.legend() # trialing yeojohnson method: 0.53 very close, but still not good enough
# %%-------------------------------------------



df_raw["out_prncp_inv"].hist(bins=50)
print(f"Skew of out_prncp_inv column is {df_raw['out_prncp_inv'].skew()}") #skew: 2.354... strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["out_prncp_inv"] = df_raw.out_prncp_inv**(1/2) #Results of Root Square transformation: 1.225... not good enough!
# %%
df_raw["out_prncp_inv"] = df_raw["out_prncp_inv"].map(lambda i: np.log(i) if i>0 else 0)
t=sns.histplot(df_raw["out_prncp_inv"],label="Skewness: %.2f"%(df_raw["out_prncp_inv"].skew()))
t.legend() # trialing log transformation: 0.58...close but still not good enough!
# %%
from scipy import stats
boxcox_out_prncp_inv = df_raw["out_prncp_inv"]
boxcox_out_prncp_inv = stats.boxcox(boxcox_out_prncp_inv)
boxcox_out_prncp_inv = pd.Series(boxcox_out_prncp_inv[0])
t=sns.histplot(boxcox_out_prncp_inv, label="Skewness: %.2f"%(boxcox_out_prncp_inv.skew()))
t.legend() # boxcox transformation: DATA MUST BE POSITIVE, CANT USE THIS METHOD
# %%
yeojohnson_out_prncp_inv = df_raw["out_prncp_inv"]
yeojohnson_out_prncp_inv = stats.yeojohnson(yeojohnson_out_prncp_inv)
yeojohnson_out_prncp_inv = pd.Series(yeojohnson_out_prncp_inv[0])
df_raw["out_prncp_inv"] = yeojohnson_out_prncp_inv
t=sns.histplot(yeojohnson_out_prncp_inv, label="Skewness: %.2f"%(yeojohnson_out_prncp_inv.skew()))
t.legend() # trialing yeojohnson method: 0.53.. close but still not good enough!
# %%------------------------------------





df_raw["total_payment"].hist(bins=50)
print(f"Skew of total_payment column is {df_raw['total_payment'].skew()}") #skew: 1.269..strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["total_payment"] = df_raw.total_payment**(1/2) #Square Root transformation is working: 0.374
# %%----------------------------------------------------




df_raw["total_rec_prncp"].hist(bins=50)
print(f"Skew of total_rec_prncp column is {df_raw['total_rec_prncp'].skew()}") #skew: 1.2626...strong positive skew
print("This indicates a strong positive skew.")
# %%
df_raw["total_rec_prncp"] = df_raw.total_rec_prncp**(1/2) #Squared Root transformation is working: 0.369
# %%
