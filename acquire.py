import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire Titanic Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df and returns the df.
    '''
    # Create SQL query.
    sql_query =  """
    SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, fips, lotsizesquarefeet, poolcnt, 
    regionidzip, yearbuilt, latitude, longitude, taxvaluedollarcnt, garagetotalsqft, regionidcounty, regionidcity, taxamount          transactiondate FROM properties_2017 AS prop17
    JOIN predictions_2017 AS pred17 ON prop17.parcelid = pred17.parcelid
    WHERE transactiondate >= '2017-01-01' AND transactiondate <= '2017-12-31'
    AND propertylandusetypeid = 261;
    """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df

