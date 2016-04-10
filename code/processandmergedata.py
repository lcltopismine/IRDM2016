# interpreter: python 2.7

import pandas as pd
from datetime import datetime
import numpy as np

# file locations
datafoldername = '../data/'
loadfilename_train = 'Load_history.csv'
tempfilename_train = 'temperature_history.csv'
loadfilename_test = 'Load_solution.csv'
tempfilename_test = 'temperature_solution.csv'


def process_load_data(filename):

    # import data
    df = pd.read_csv(filename, parse_dates=[[1, 2, 3]], thousands=',')

    # unpivot
    df = pd.melt(df, id_vars=['year_month_day', 'zone_id'], var_name='hour')

    # drop rows where value is NaN
    df.dropna(inplace=True)

    # create datetime col
    df.hour = df.hour.str.replace('h', '')
    df.hour = pd.to_timedelta(df.hour.astype(int) - 1, unit='h')
    df['datetime'] = df.year_month_day + df.hour

    # drop and reorder columns
    df = df[['datetime', 'zone_id', 'value']]

    print df
    return df


def process_temp_data(filename):

    # import data
    df = pd.read_csv(filename, parse_dates=[[1, 2, 3]])

    # unpivot
    df = pd.melt(df, id_vars=['year_month_day', 'station_id'], var_name='hour')

    df.dropna(inplace=True)

    # create datetime col
    df.hour = df.hour.str.replace('h', '')
    df.hour = pd.to_timedelta(df.hour.astype(int) - 1, unit='h')
    df['datetime'] = df.year_month_day + df.hour

    df = df[['datetime', 'station_id', 'value']]

    # pivot temps by station
    dfpivot = df.pivot(index='datetime', columns='station_id', values='value')
    dfpivot.columns.name = None
    dfpivot.rename(columns=lambda x: 'tempstn_'+str(x), inplace=True)
    dfpivot.reset_index(inplace=True)

    print dfpivot

    return dfpivot


# Add categorical time variables
def addTimeDateCategories(df):
    df['summer'] = df['datetime'].dt.month.isin(range(4, 10))
    df['hour'] = df['datetime'].dt.hour
    return df


def load_to_df(loadfilename, tempfilename):
    load = process_load_data(loadfilename)
    temp = process_temp_data(tempfilename)

    df = load.merge(temp, on='datetime', how='left')

    return df


def main():

    print 'load training data'
    load = process_load_data(datafoldername+loadfilename_train)
    temp = process_temp_data(datafoldername+tempfilename_train)
    X_train_df = load.merge(temp, on='datetime', how='left')


    print 'load test data'
    load_test = process_load_data(datafoldername+loadfilename_train)
    temp_test = process_temp_data(datafoldername+tempfilename_train)

    # Todo: - Some of the temp data is already provided in training - need to merge from both.
    X_test_df = load_test(temp_test, on='datetime', how='left')

    # write out to csv
    # loadhistory.to_csv(datafolder + 'load_history_processed.csv')
    # temphistory.to_csv(datafolder + 'temperature_history_processed.csv')

if __name__ == "__main__": main()
