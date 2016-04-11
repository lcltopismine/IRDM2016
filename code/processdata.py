# interpreter: python 2.7

import pandas as pd
from datetime import datetime
import numpy as np

datafolder = '../data/'

# import data
loadhistory = pd.read_csv(datafolder + 'Load_history.csv', parse_dates=[[1, 2, 3]], thousands=',')
temphistory = pd.read_csv(datafolder + 'temperature_history.csv', parse_dates=[[1, 2, 3]])

# unpivot
loadhistory=pd.melt(loadhistory, id_vars=['year_month_day', 'zone_id'], var_name='hour')
temphistory=pd.melt(temphistory, id_vars=['year_month_day', 'station_id'], var_name='hour')

# drop rows where value is NaN
loadhistory.dropna(inplace=True)
temphistory.dropna(inplace=True)

# build datetime
def buildDateTime(df):
    df.hour = df.hour.str.replace('h', '')
    df.hour = pd.to_timedelta(df.hour.astype(int) - 1, unit='h')
    df['datetime'] = df.year_month_day + df.hour
    return df

loadhistory = buildDateTime(loadhistory)
temphistory = buildDateTime(temphistory)

# drop and reorder colums
loadhistory = loadhistory[['datetime', 'zone_id', 'value']]
temphistory = temphistory[['datetime', 'station_id', 'value']]
loadhistory = loadhistory.sort_values(by=['zone_id', 'datetime'], ascending=[True, True])

# Add categorical time variables
def addTimeDateCategories(df):
    df['summer'] = df['datetime'].dt.month.isin(range(4, 10))
    df['hour'] = df['datetime'].dt.hour
    return df

loadhistory = addTimeDateCategories(loadhistory)
temphistory = addTimeDateCategories(temphistory)

# print loadhistory
# print temphistory

# write out to csv
for i in range(1,21):
    subset = loadhistory[loadhistory.zone_id == i]
    filename = 'load_history_processed_zone_%s.csv'%i
    subset.to_csv(filename)
temphistory.to_csv(datafolder + 'temperature_history_processed.csv')

