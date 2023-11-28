#%%
from sqlalchemy import create_engine
import pandas as pd
import yaml

def load_credentials_file():
    with open('credentials.yaml', 'r') as f:
        cred = yaml.safe_load(f)
    return cred

credentials = load_credentials_file()
'''
Creates a function which loads the credentials.yaml file. 

Credentials.yaml file stores the AWS RDS database credentials (ignored by GitHub). The load_credentials_file() function opens 
the yaml file that is stored locally and returns the data dictionary, contained within 'cred' variable. Finally, the 
'credentials' variable is created which purpose is to call the function.

'''

#%%
class RDSDatabaseConnector:
    '''
    This class uses the information in the credentials.yaml file to connect to the remote database.

    Attributes:
        credentials (dict): contains credentials (information needed to connect to the local AWS RDS database) which is presented
        in the dictionary format. 
    '''    
    def __init__(self, credentials):
        self.credentials = credentials
        
    def create_engine(self):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.credentials.get("RDS_HOST")
        USER = self.credentials.get("RDS_USER")
        PASSWORD = self.credentials.get("RDS_PASSWORD")
        DATABASE = self.credentials.get("RDS_DATABASE")
        PORT = self.credentials.get("RDS_PORT")
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        return engine
    '''
    This method of the class creates an engine which is used to connect to the remote dabatase.
    
    The values for HOST, USER, PASSWORD, DATABASE and PORT are extracted from the credentials.yaml file while the values
    of DATABASE_TYPE and DBAPI are provided in the method. This information is input to create an engine connecting
    to the specified AWS RDS database.
    '''
    def extract_RDS_data(self, engine):
        data = pd.read_sql_table("loan_payments", engine)
        return data
    '''
    This method of the class extracts the data from the AWS RDS remote database. 
    
    The data is extracted, formatted into Pandas DataFrame and returned as 'data' variable. 
    '''
        
db_connect = RDSDatabaseConnector(credentials)

engine = db_connect.create_engine()

df = db_connect.extract_RDS_data(engine)

'''
Three instances of the class are created to access the extracted data.
'''

print(df.head())

'''
Tests whether the data is successfully extracted and can be accessed by printing top rows of the database.
'''
#%%
def save_to_csv():
    df.to_csv("C:/Users/eveli/ai_core/EDA/loan_payments.csv")
    
save_to_csv()

'''
This function is used to save the extracted data locally in csv format.

When save_to_csv() function is called, the data extracted from the remote RDS database is saved locally in csv format
and names "loan_payments".
'''
# %%

def load_data():
    loan_payments = pd.read_csv("C:/Users/eveli/ai_core/EDA/loan_payments.csv")
    return loan_payments
lp_df = load_data()
'''
The function loads the "loan_payments.csv" database from the local computer.

lp_df calls the function which returns the locally stored database in a table format. 
'''

lp_df

lp_df.shape  #Shows that the table has 54231 rows and 44 rows. 

lp_df.describe()

'''
Various commands to explore the dataset.

lp_df: calls the database.
lp_df.shape: calls for the information of the database shape.
lp_df.describe(): calls for the summary of the database that includes information such as count, mean, max, min etc.
'''

# %%
