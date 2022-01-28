import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DBSCAN import
from sklearn.cluster import DBSCAN

# Scaler import
from sklearn.preprocessing import MinMaxScaler

import env


#################################################################

def get_curriculum_logs():
    filename = "curriculum-access.csv"
    # This function give us DataFrame of curriculum_logs

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=False)
    else:
        # read the SQL query into a dataframe
        url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/curriculum_logs'
        query = '''
        SELECT date,
               path as endpoint,
               user_id,
               cohort_id,
               ip as source_ip
        FROM logs;
        '''
        df = pd.read_sql(query, url)

        # Write that dataframe to disk for later.
        df.to_csv(filename, index = False)

        return df 
    
    
#################################################################

def anomalies_curriculum_access():
    # acquire data using the above function
    df = get_curriculum_logs()
    
    # This function give us DataFrame of anomalies in curriculum_access

    # convert date to a pandas datetime format and set as index
    df.date = pd.to_datetime(df.date)
    df = df.set_index(df.date)

    page_views = df.groupby(['user_id'])['endpoint'].agg(['count', 'nunique'])

    # create the scaler
    scaler = MinMaxScaler().fit(page_views)
    # use the scaler
    page_views_scaled_array = scaler.transform(page_views)

    # construct DBSCAN object
    dbsc = DBSCAN(eps = 0.1, min_samples=4).fit(page_views_scaled_array)

    # Now, let's add the scaled value columns back onto the dataframe
    columns = list(page_views.columns)
    scaled_columns = ["scaled_" + column for column in columns]

    # Create a dataframe containing the scaled values
    scaled_df = pd.DataFrame(page_views_scaled_array, columns=scaled_columns, index=page_views.index)

    # Merge the scaled and non-scaled values into one dataframe
    page_views = page_views.merge(scaled_df, left_index=True, right_index=True)
    
    # defining labels
    labels = dbsc.labels_

    #add labels back to the dataframe
    page_views['labels'] = labels

    # anomalies 

    anomalies_df = page_views[page_views.labels==-1]

    return anomalies_df

#################################################################

def anomalies_customers():
    # This function give us DataFrame of anomalies in customers
    
    # acquire data using the above function
    df = pd.read_sql(sql, url, index_col="customer_id")

    # selecting frozen and Grocery
    df =  df[['Grocery', 'Frozen']]

    # create the scaler
    scaler = MinMaxScaler().fit(df)
    # use the scaler
    df_scaled_array = scaler.transform(df)

    # construct DBSCAN object
    dbsc = DBSCAN(eps = 0.1, min_samples=4).fit(df_scaled_array)

    # Now, let's add the scaled value columns back onto the dataframe
    columns = list(df.columns)
    scaled_columns = ["scaled_" + column for column in columns]

    # Create a dataframe containing the scaled values
    scaled_df = pd.DataFrame(df_scaled_array, columns=scaled_columns, index=df.index)

    # Merge the scaled and non-scaled values into one dataframe
    df = df.merge(scaled_df, left_index=True, right_index=True)
    
    # defining labels
    labels = dbsc.labels_

    #add labels back to the dataframe
    df['labels'] = labels

    # anomalies 

    anomalies_df = df[df.labels==-1]

    return anomalies_df

#################################################################

def anomalies_zillow(col1, col2, ep_value, min_sample_value):
    
    # This function give us DataFrame of anomalies in zillow
    # acquire data using the above function
    df= acquire.new_zillow_data()
    
    # preparing df
    df = prepare.wrangle_zillow(df)

    # selecting col2 and col1
    df =  df[[col1, col2]]

    # create the scaler
    scaler = MinMaxScaler().fit(df)
    # use the scaler
    df_scaled_array = scaler.transform(df)

    # construct DBSCAN object
    dbsc = DBSCAN(eps = ep_value, min_samples=min_sample_value).fit(df_scaled_array)

    # Now, let's add the scaled value columns back onto the dataframe
    columns = list(df.columns)
    scaled_columns = ["scaled_" + column for column in columns]

    # Create a dataframe containing the scaled values
    scaled_df = pd.DataFrame(df_scaled_array, columns=scaled_columns, index=df.index)

    # Merge the scaled and non-scaled values into one dataframe
    df = df.merge(scaled_df, left_index=True, right_index=True)
    
    # defining labels
    labels = dbsc.labels_

    #add labels back to the dataframe
    df['labels'] = labels

    # anomalies 

    anomalies_df = df[df.labels==-1]

    return anomalies_df