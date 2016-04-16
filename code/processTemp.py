import pandas as pd
from datetime import datetime
import numpy as np

# import data
temphistory = pd.read_csv('../data/temperature_history.csv', parse_dates=[[1, 2, 3]])

# unpivot
temphistory=pd.melt(temphistory, id_vars=['year_month_day', 'station_id'], var_name='hour')

# drop rows where value is NaN
temphistory.dropna(inplace=True)

# build datetime
def buildDateTime(df):
    df.hour = df.hour.str.replace('h', '')
    df.hour = pd.to_timedelta(df.hour.astype(int) - 1, unit='h')
    df['datetime'] = df.year_month_day + df.hour
    print 'datetime'
    print df['datetime']
    return df

temphistory = buildDateTime(temphistory)

# drop and reorder colums
temphistory = temphistory[['datetime', 'station_id', 'value']]
temphistory = temphistory.sort_values(by=['station_id', 'datetime'], ascending=[True, True])

# write out to csv    
for i in range(1,12):
    subset = temphistory[temphistory.station_id == i]
    filename = '../data/outputs/temp_history_processed_station_%s.csv'%i
    subset.to_csv(filename)    
