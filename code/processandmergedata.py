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
outputfoldername = datafoldername+'output/'
outputfilename_train = 'train_processed.csv'
outputfilename_test = 'test_processed.csv'


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

    return dfpivot


# Add categorical time variables
def addTimeDateCategories(df):
    df['summer'] = df['datetime'].dt.month.isin(range(4, 10))
    df['hour'] = df['datetime'].dt.hour
    return df


def main():

    print 'process load training data'
    load = process_load_data(datafoldername+loadfilename_train)

    print 'process temp training data'
    temp = process_temp_data(datafoldername+tempfilename_train)

    print 'merge training data'
    X_train_df = load.merge(temp, on='datetime', how='left')


    print 'process load test data'
    load_test = process_load_data(datafoldername+loadfilename_train)

    print 'process temp test data'
    temp_test = process_temp_data(datafoldername+tempfilename_train)

    # Some of the temp data is already provided in training - need to merge from both.
    # append temp data
    print 'concat temp train and test data'
    temp_all = pd.concat([temp, temp_test])

    print 'merge test data'
    X_test_df = load_test.merge(temp_all, on='datetime', how='left')

    print 'save train data'
    X_train_df.to_csv(outputfoldername + 'train_processed.csv')

    print 'save test data'
    X_test_df.to_csv(outputfoldername + 'test_processed.csv')

if __name__ == "__main__": main()
