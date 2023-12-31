#%%
from sqlalchemy import create_engine
import pandas as pd
import yaml

def load_credentials_file():
    with open('credentials.yaml', 'r') as f:
        cred = yaml.safe_load(f)
    return cred

credentials = load_credentials_file()
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
