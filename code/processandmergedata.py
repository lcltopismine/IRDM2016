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
holidayfilename = 'Holiday_Processed.csv'


def process_load_data(filename):

    # import data
    df = pd.read_csv(filename, parse_dates=[[1, 2, 3]], thousands=',')

    # unpivot
    df = pd.melt(df, id_vars=['year_month_day', 'zone_id'], var_name='hour')

    # drop rows where value is NaN
    df.dropna(inplace=True)

    # drop where zoneid = 21 [this is just a total row]
    df = df[df.zone_id != 21]

    # create datetime col
    df.hour = df.hour.str.replace('h', '')
    df.hour = pd.to_timedelta(df.hour.astype(int) - 1, unit='h')
    df['datetime'] = df.year_month_day + df.hour

    # drop and reorder columns
    df = df[['datetime', 'zone_id', 'value']].copy()

    # add weights
    df['weight'] = 1
    # increase weight on future predictions - where datetime > 2008-06-30 05:30
    predictions_start_datetime = datetime(2008, 6, 30, 05, 30, 0)
    df.loc[df['datetime'] > predictions_start_datetime, 'weight'] = 8

    # add trend variable [incremental number of hours]
    trend_start_datetime = datetime(2004, 1, 1, 0, 0, 0)
    df['trend'] = (df.datetime - trend_start_datetime) / np.timedelta64(1, 'h') + 1

    return df


def process_temp_data(filename):

    # import data
    df = pd.read_csv(filename, parse_dates=[[1, 2, 3]])

    # unpivot
    df = pd.melt(df, id_vars=['year_month_day', 'station_id'], var_name='hour')

    # drop where value missing
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

def process_holiday_data(filename):
    df = pd.read_csv(filename, parse_dates=[0])
    df['holiday']=1
    df = df[['HolidayDate', 'holiday']]
    df.columns = ['datetime', 'holiday']
    return df


# Add categorical time variables
def add_timedate_categories(df):
    df['summer'] = df['datetime'].dt.month.isin(range(4, 10))
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    return df

def get_data():

    print 'get load training data'
    load = process_load_data(datafoldername+loadfilename_train)

    print 'get load test data'
    load_test = process_load_data(datafoldername+loadfilename_test)

    print 'get temp training data'
    temp = process_temp_data(datafoldername+tempfilename_train)

    # I'm not sure if we should use test temp data for building or evaluating our models.
    # but incorporated for now.

    print 'process temp test data'
    temp_test = process_temp_data(datafoldername+tempfilename_test)

    print 'concat temp train and test data'
    temp = pd.concat([temp, temp_test])

    print 'merge training load data with temp data'
    X_train_df = load.merge(temp, on='datetime', how='left')

    print 'merge test load data with temps'
    X_test_df = load_test.merge(temp, on='datetime', how='left')

    print 'get holiday data'
    holidays = process_holiday_data(datafoldername+holidayfilename)

    print 'merge holiday dates on train'
    X_train_df = X_train_df.merge(holidays, on='datetime', how='left')
    X_train_df['holiday'].fillna(0, inplace=True)

    print 'merge holiday dates on test'
    X_test_df = X_test_df.merge(holidays, on='datetime', how='left')
    X_test_df['holiday'].fillna(0, inplace=True)

    print 'add datetime categorical variables on train'
    X_train_df = add_timedate_categories(X_train_df)

    print 'add datetime categorical variables on test'
    X_test_df = add_timedate_categories(X_test_df)

    return X_train_df, X_test_df


def save_data_csv(df, filename):
    df.to_csv(filename, index=False, date_format='%Y-%m-%d %H:%M:%S')


def main():

    print 'get data'
    X_train_df, X_test_df = get_data()

    print 'save train data'
    save_data_csv(X_train_df, outputfoldername + 'train_processed.csv')

    print 'save test data'
    save_data_csv(X_test_df, outputfoldername + 'test_processed.csv')

    print 'also save train data split by zoneid'
    for i in range(1, 21):
        print 'zoneid = %s' % i
        subset = X_train_df[X_train_df.zone_id == i]
        filename = 'train_processed_zone_%s.csv' % i
        save_data_csv(subset, outputfoldername + filename)

if __name__ == "__main__": main()
