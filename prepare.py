import os

from pydataset import data
# import our own acquire module

import acquire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")



def split_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(
        df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(
        train, test_size=.3, random_state=123, stratify=train.survived)
    return train, validate, test

def wrangle_zillow(df):
    """
    This function will clean zillow to ensure that all fields are not null and bathrooms, bedrooms,
    lot_size and tax value are positive numbers.
    
    """
    # Dropping rows with 'NaN' from certain columns
    df=df.dropna(subset = ['calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'regionidzip', 'yearbuilt', 'taxvaluedollarcnt', 'regionidcounty'])
    # Replacing rest of 'NaN' with 0 for other columns
    df = df.fillna(0)
    # Deleting rows with bedrooms, bathrooms, total_living_area, lot_size, tax_value, garage_area count 0 or negative, since it makes no sense
    df = df[df.bedroomcnt > 0]
    df = df[df.bathroomcnt > 0]
    df = df[df.lotsizesquarefeet > 0]
    df = df[df.taxvaluedollarcnt > 0]
    return df

def get_histogram(df):
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = ['bedrooms', 'bathrooms', 'total_living_area', 'tax_value', 'garage_area']

    for i, col in enumerate(cols):
    
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
    
        # Create subplot.
        plt.subplot(1, 5, plot_number)
    
        # Title with column name.
        plt.title(col)
    
        # Display histogram for column.
        df[col].hist(bins= 5, edgecolor='black')
    

    
        # Hide gridlines.
        plt.grid(False)
        
        
        plt.figure(figsize=(16,4))

      
                
        
def get_boxplot(df):        
    # List of columns
    cols = [col for col in df.columns if col not in ['year_built', 'fips', 'lot_size', 'pools', 'zip_code', 'latitude',                                                    'longitude', 'transactiondate', 'regionidcounty', 'regionidcity']]
    plt.figure(figsize=(16, 20))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[cols])
    

        # Hide gridlines.
        plt.grid(False)

    plt.show()
    

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'total_living_area', 'garage_area'])
    
    # get distributions of numeric data
    get_histogram(df)
    get_boxplot(df)
    
    
    # train/validate/test split
    # train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    # train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
          
    
    return df 



def wrangle_zillow_split(df):
    '''Acquire and prepare data from Zillow database for explore'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test




def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df








