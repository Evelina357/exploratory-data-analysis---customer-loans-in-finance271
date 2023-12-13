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
        print("\n")
        
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
        
    def test_normal_distribution(self):    
        nd_columns = ["loan_amount", "funded_amount", "funded_amount_inv", "int_rate", "instalment", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_accounts", "total_accounts", "out_prncp", "out_prncp_inv", "total_payment", "total_payment_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_amount", "collections_12_mths_ex_med"]
        for colm in nd_columns:
            data = self.df_raw[colm]
            stat, p = normaltest(data, nan_policy='omit')
            print("Statistics=%.3f, p=%.3f" % (stat, p), "\n")
            qq_plot = qqplot(self.df_raw[colm], scale=1, line='q')
            plot.title(f"The distribution of {colm} values")
            plot.show()
            print(f'\nThe median of {colm} is {df_raw[colm].median()}')
            print(f'The mean of {colm} is {df_raw[colm].mean()}')
        
    def show_disc_prob_distr(self):
        dpd_columns = ["term", "grade", "sub_grade", "employment_length", "home_ownership", "verification_status", "loan_status", "payment_plan", "purpose", "policy_code", "application_type"]
        for clmn in dpd_columns:
            plot.rc("axes.spines", top=False, right=False)
            probs = self.df_raw[clmn].value_counts(normalize=True)
            dpd=sns.barplot(y=probs.index, x=probs.values, color='b')
            plot.xlabel("Values")
            plot.ylabel("Probability")
            plot.title(f"Discrete Probability Distribution of {clmn}")
            plot.show()
            print("\nThe value counts in  ")
            print(self.df_raw[clmn].value_counts())
            print(f'\nThe mode of {clmn} column is {self.df_raw[clmn].mode()[0]}')
            
    def show_skewness(self):
        skew_columns = ["loan_amount", "funded_amount", "funded_amount_inv", "int_rate", "instalment", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_accounts", "total_accounts", "out_prncp", "out_prncp_inv", "total_payment", "total_rec_prncp"]
        for sk_colm in skew_columns:
            self.df_raw[sk_colm].hist(bins=50)
            print(f"\nSkew of {sk_colm} column is {self.df_raw[sk_colm].skew()}", "\n")
            
    def show_cont_data_outliers(self):
        Cont_data_columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_amount', 'collections_12_mths_ex_med']
        for colm in Cont_data_columns:
            column_of_interest = colm
            self.df_raw.boxplot(column=column_of_interest)
            plot.show()
            
    #to see categorical data outliers, you just need to call show_dpd = visuals.show_disc_prob_distr()   . 
        
visuals = Plotter(df_raw)

null_values = visuals.show_null_values()

test_nd = visuals.test_normal_distribution()

show_dpd = visuals.show_disc_prob_distr()

show_skew = visuals.show_skewness()

show_cont_outliers = visuals.show_cont_data_outliers()

# %%
class DataFrameNanTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def impute_with_median(self):
        median_columns = ["funded_amount", "int_rate", "collections_12_mths_ex_med"]
        for column in median_columns:
            self.df_raw[column] = self.df_raw[column].fillna(self.df_raw[column].median())
            print(self.df_raw[column].info(), "\n")
        
    def impute_with_mode(self):
        mode_columns = ["term", "employment_length"]
        for mod_column in mode_columns:
            self.df_raw[mod_column] = self.df_raw[mod_column].fillna(self.df_raw[mod_column].mode()[0])
            print(self.df_raw[mod_column].info(), "\n")
    
    def drop_rows_last_payment_date(self):
        self.df_raw.dropna(how="any", subset=["last_payment_date"], inplace=True)
        print(self.df_raw["last_payment_date"].info(), "\n")
        
    def drop_rows_last_credit_pull_date(self):
        self.df_raw.dropna(how="any", subset=["last_credit_pull_date"], inplace=True)
        print(self.df_raw["last_credit_pull_date"].info(), "\n")
    
    def drop_columns(self):
        self.df_raw = self.df_raw.drop(columns=["mths_since_last_delinq", "mths_since_last_record", "next_payment_date", "mths_since_last_major_derog"], inplace=True)
        
        
df_transform = DataFrameNanTransform(df_raw)

med_columns = df_transform.impute_with_median()

mod_columns = df_transform.impute_with_mode()

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
        
    def transform_skew_square_root(self):
        columns = ["loan_amount", "funded_amount", "funded_amount_inv", "instalment", "total_accounts", "total_payment", "total_rec_prncp"]
        for clm in columns:
            self.df_raw[clm] = self.df_raw[clm]**(1/2)
            
    def transform_skew_log(self):
        log_columns = ["annual_inc", "open_accounts"]
        for colm in log_columns:
            self.df_raw[colm] = self.df_raw[colm].map(lambda i: np.log(i) if i>0 else 0)
            t=sns.histplot(self.df_raw[colm],label="Skewness: %.2f"%(self.df_raw[colm].skew()))
            t.legend()
            
    def transform_skew_yeojohnson(self):
        yeo_columns = ["delinq_2yrs", "inq_last_6mths", "out_prncp", "out_prncp_inv"]
        for colmn in yeo_columns:
            yeojohnson_colmn = self.df_raw[colmn]
            yeojohnson_colmn = stats.yeojohnson(yeojohnson_colmn)
            yeojohnson_colmn = pd.Series(yeojohnson_colmn[0])
            self.df_raw[colmn] = yeojohnson_colmn
            t=sns.histplot(yeojohnson_colmn, label="Skewness: %.2f"%(yeojohnson_colmn.skew()))
            t.legend()
        
    #Result of square Root Transformation for "loan_amount": 0.165
        
    # Result of Square Root Transformation for "funded_amount": 0.180
        
    # Result of square Root Transformation for "funded_amount_inv": 0.072
        
    # Result of square Root Transformation for "instalment": 0.246
        
    # Result for log transformation for "annual_inc": 0.14
        
    #  of Yeo-Johnson transformation for "delinq_2yrs": 1.87(lowest result in comparison to other methods).
        
    # Result of Yeo-Johnson transformation for "inq_last_6mths": 0.25
        
    # Result of a log transformation for "open_accounts": -0.47
        
    # Result of a Square Root Transformation for "total_accounts": 0.109
        
    # Result of Yeo-Johnson method for "out_prncp": 0.53 (lowest result)
        
    # Result of Yeo-Johnson method for "out_prncp_inv": 0.53 (lowest result)
        
    # Result of Square Root transformation for "total_payment": 0.374
        
    # Result of Square Root transformation for "total_rec_prncp": 0.369
        
        
skew_transform = DataFrameSkewTransform(df_raw)

transform_square_root = skew_transform.transform_skew_square_root()

transform_log = skew_transform.transform_skew_log()

transform_yeo = skew_transform.transform_skew_yeojohnson()

show_skew = visuals.show_skewness()
    
# %%

class DataFrameOutliersTransform:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def categ_data_outliers_transform(self):
        self.df_raw["home_ownership"].replace({"None": "Other"}, inplace=True)
        self.df_raw["loan_status"].replace({"Does not meet the credit policy. Status:Charged Off": "Charged Off"}, inplace=True)
        self.df_raw["loan_status"].replace({"Does not meet the credit policy. Status:Fully Paid": "Fully Paid"}, inplace=True)
        self.df_raw = self.df_raw.drop(self.df_raw[self.df_raw['payment_plan'] == 'y'].index)
        visuals.show_disc_prob_distr()
        
    def iqr_outliers_removal(self):
        iqr_columns = ["funded_amount_inv", "delinq_2yrs", "total_rec_prncp", "collections_12_mths_ex_med", "int_rate", "instalment", "dti", "total_payment", "total_payment_inv"]
        for iqr_colm in iqr_columns:
            Q1 = df_raw[iqr_colm].quantile(0.25)
            Q3 = df_raw[iqr_colm].quantile(0.75)
            IQR = Q3 - Q1
            threshold = 1.5
            df_raw.loc[df_raw[(df_raw[iqr_colm] < Q1 - threshold * IQR) | (df_raw[iqr_colm] > Q3 + threshold * IQR)]]
            #outliers = df_raw[(df_raw[iqr_colm] < Q1 - threshold * IQR) | (df_raw[iqr_colm] > Q3 + threshold * IQR)]
            self.df_raw[iqr_colm] = self.df_raw[iqr_colm].drop(list(outliers.index))
            #column_of_interest = iqr_colm       #graphic check if worked.
            #self.df_raw.boxplot(column=column_of_interest)
            #plot.show()
    def flooring_capping_outliers_removal(self):
        floor_cap_columns = ["annual_inc", "open_accounts", "total_accounts"]
        for fl_cp_column in floor_cap_columns:
            flooring = self.df_raw[fl_cp_column].quantile(0.10)
            capping = self.df_raw[fl_cp_column].quantile(0.90)
            print(f'The flooring for the lower values is: {flooring}')
            print(f'The capping for the higher values is: {capping}')
            self.df_raw[fl_cp_column] = self.df_raw.where(self.df_raw[fl_cp_column] <flooring, flooring,self.df_raw[fl_cp_column])
            self.df_raw[fl_cp_column] = self.df_raw.where(self.df_raw[fl_cp_column] >capping, capping,self.df_raw[fl_cp_column])
            #self.df_raw[fl_cp_column] = np.where(self.df_raw[fl_cp_column] <flooring, flooring,self.df_raw[fl_cp_column])
            #self.df_raw[fl_cp_column] = np.where(self.df_raw[fl_cp_column] >capping, capping,self.df_raw[fl_cp_column])
            #column_of_interest = fl_cp_column     #graphic check if worked.
            #self.df_raw.boxplot(column=column_of_interest)
            #plot.show()
            
    def transform_outliers_median(self):
        outliers_median = ["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_amount"]
        for out_med_column in outliers_median:
            median = self.df_raw[out_med_column].quantile(0.50)
            high_percentile = self.df_raw[out_med_column].quantile(0.95)
            print(f'The median value of {out_med_column} is {median}') 
            print(f'The 95th percentile value of {out_med_column} is {high_percentile}') 
            self.df_raw[out_med_column] = self.df_raw.where(self.df_raw[out_med_column] > high_percentile, median, self.df_raw[out_med_column])
            #self.df_raw[out_med_column] = np.where(self.df_raw[out_med_column] > high_percentile, median, self.df_raw[out_med_column])
            column_of_interest = out_med_column
            df_raw.boxplot(column=column_of_interest)
            plot.show()

outl_transform = DataFrameOutliersTransform(df_raw)

outl_transform.categ_data_outliers_transform()

outl_transform.iqr_outliers_removal()

outl_transform.flooring_capping_outliers_removal()

outl_transform.transform_outliers_median()

show_cont_outliers = visuals.show_cont_data_outliers() #show graphs to check if outliers were successfully removed.
# %%
df_raw.info()

show_cont_outliers = visuals.show_cont_data_outliers()

show_dpd = visuals.show_disc_prob_distr()



















# drafts below................................................................................................


# %%
plot.figure(figsize=(10, 5))
sns.histplot(df_raw["annual_inc"], bins=40, kde=False)
plot.title("Histogram of annual_inc")
plot.xlabel('£ (pounds)')
plot.ylabel('Frequency')
plot.show()

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
Q1 = df_raw['int_rate'].quantile(0.25)
Q3 = df_raw['int_rate'].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (25th percentile): {Q3}")
print(f"IQR: {IQR}")

threshold = 1.5
outliers = df_raw['int_rate'][(df_raw['int_rate'] < (Q1 - threshold * IQR)) | (df_raw['int_rate'] > (Q3 + threshold * IQR))]

print("Outliers:")
print(outliers)

df_raw["int_rate"].value_counts()[25.99]

# %%
Q1 = df_raw['funded_amount_inv'].quantile(0.25)
Q3 = df_raw['funded_amount_inv'].quantile(0.75)
IQR = Q3 - Q1

threshold = 1.5
outliers = df_raw[(df_raw['funded_amount_inv'] < Q1 - threshold * IQR) | (df_raw['funded_amount_inv'] > Q3 + threshold * IQR)]
# %%

df_raw["funded_amount_inv"] = df_raw["funded_amount_inv"].drop(outliers.index)
# %%


#graphing each continious data column to see the outliers. 

Cont_data_columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_amount', 'collections_12_mths_ex_med']
for colm in Cont_data_columns:
    column_of_interest = colm
    df_raw.boxplot(column=column_of_interest)
    plot.show()

# %%
df_raw.shape
# %%
show_cont_outliers = visuals.show_cont_data_outliers()
# no outliers in "loan_amount", "funded_amount", "inq_last_6mths", "out_prncp", "out_prncp_inv".
# %%















# Removing outliers for columns that do not have many outliers ("funded_amount_inv", "delinq_2yrs", "total_rec_prncp", "collections_12_mths_ex_med")

Q1 = df_raw['funded_amount_inv'].quantile(0.25)
Q3 = df_raw['funded_amount_inv'].quantile(0.75)
IQR = Q3 - Q1

threshold = 1.5
outliers = df_raw[(df_raw['funded_amount_inv'] < Q1 - threshold * IQR) | (df_raw['funded_amount_inv'] > Q3 + threshold * IQR)]


df_raw["funded_amount_inv"] = df_raw["funded_amount_inv"].drop(outliers.index)
# %%
#Plotting how it looks like after removing outliers:
column_of_interest = 'funded_amount_inv'
df_raw.boxplot(column=column_of_interest)
plot.show()
# %%
iqr_columns = ["funded_amount_inv", "delinq_2yrs", "total_rec_prncp", "collections_12_mths_ex_med"]
for iqr_colm in iqr_columns:
    Q1 = df_raw[iqr_colm].quantile(0.25)
    Q3 = df_raw[iqr_colm].quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5
    outliers = df_raw[(df_raw[iqr_colm] < Q1 - threshold * IQR) | (df_raw[iqr_colm] > Q3 + threshold * IQR)]
    df_raw[iqr_colm] = df_raw[iqr_colm].drop(outliers.index)
    column_of_interest = iqr_colm
    df_raw.boxplot(column=column_of_interest)
    plot.show()
    
# %%
# Quantile-based flooring and capping (for "annual_inc", "open_accounts", "total_accounts")
floor_cap_columns = ["annual_inc", "open_accounts", "total_accounts"]
for fl_cp_column in floor_cap_columns:
    flooring = df_raw[fl_cp_column].quantile(0.10)
    capping = df_raw[fl_cp_column].quantile(0.90)
    print(f'The flooring for the lower values is: {flooring}')
    print(f'The capping for the higher values is: {capping}')
    df_raw[fl_cp_column] = np.where(df_raw[fl_cp_column] <flooring, flooring,df_raw[fl_cp_column])
    df_raw[fl_cp_column] = np.where(df_raw[fl_cp_column] >capping, capping,df_raw[fl_cp_column])
    column_of_interest = fl_cp_column
    df_raw.boxplot(column=column_of_interest)
    plot.show()




# %%
flooring = df_raw["annual_inc"].quantile(0.10)
capping = df_raw["annual_inc"].quantile(0.90)
print(f'The flooring for the lower values is: {df_raw["annual_inc"].quantile(0.10)}')
print(f'The capping for the higher values is: {df_raw["annual_inc"].quantile(0.90)}')

df_raw["annual_inc"] = np.where(df_raw["annual_inc"] <flooring, flooring,df_raw['annual_inc'])
df_raw["annual_inc"] = np.where(df_raw["annual_inc"] >capping, capping,df_raw['annual_inc'])
print(df_raw['annual_inc'].skew())

column_of_interest = "annual_inc"
df_raw.boxplot(column=column_of_interest)
plot.show()

# %%
df_raw.info()
df_raw.shape

# %% test how outlier removal worked:


column_of_interest = "annual_inc"
df_raw.boxplot(column=column_of_interest)
plot.show()
# %%
df_raw["last_payment_amount"].value_counts().sum
# %%
plot.figure(figsize=(10, 5))
sns.histplot(df_raw["annual_inc"], bins=40, kde=False)
plot.title("Histogram of annual_inc")
plot.xlabel('£ (pounds)')
plot.ylabel('Frequency')
plot.show()

# %%
outliers_median = ["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_amount"]
for out_med_column in outliers_median:
    median = df_raw[out_med_column].quantile(0.50)
    high_percentile = df_raw[out_med_column].quantile(0.95)
    print(f'The median value of {out_med_column} is {median}') 
    print(f'The 95th percentile value of {out_med_column} is {high_percentile}') 
    df_raw[out_med_column] = np.where(df_raw[out_med_column] > high_percentile, median, df_raw[out_med_column])
    column_of_interest = out_med_column
    df_raw.boxplot(column=column_of_interest)
    plot.show()  # is it changing the graphs? look at the skews maybe?


































































































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
