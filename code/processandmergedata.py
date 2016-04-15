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
arimafilename = 'arimaTemp.csv'

outputfoldername = datafoldername+'output/'
outputfilename_train = 'train_processed.csv'
outputfilename_test = 'test_processed.csv'
holidayfilename = 'Holiday_Processed.csv'


def get_data(use_ARIMA_estimates=False):

    print 'get data from files'
    load = process_load_data(datafoldername+loadfilename_train)
    load_test = process_load_data(datafoldername+loadfilename_test)
    temp = process_temp_data(datafoldername+tempfilename_train)
    holidays = process_holiday_data(datafoldername+holidayfilename)

    print 'merge load with temp data'
    X_train_df = load.merge(temp, on='datetime', how='left')
    X_test_df = load_test.merge(temp, on='datetime', how='left')

    print 'estimate missing temps'
    # find rows with missing temperatures
    missingtemp = X_test_df[X_test_df.isnull().any(axis=1)][['datetime', 'zone_id']].copy()
    # calculate estimates for missing periods
    if use_ARIMA_estimates:
        # use preprocessed arima estimates
        estimatedtemps = process_arima_temp_data(datafoldername+arimafilename)
    else:
        # use means of historical temps at same day/time.
        estimatedtemps = get_estimated_temps(missingtemp[['datetime']].drop_duplicates(), temp)

    # merge estimates against missing rows, and use to update original dataframe in place
    replacementtemps = missingtemp.merge(estimatedtemps, left_on='datetime', right_on='datetime', how='left')
    replace_unknown_temps(X_test_df, replacementtemps)

    print 'merge in holiday dates'
    X_train_df = X_train_df.merge(holidays, on='datetime', how='left')
    X_train_df['holiday'].fillna(0, inplace=True)
    X_test_df = X_test_df.merge(holidays, on='datetime', how='left')
    X_test_df['holiday'].fillna(0, inplace=True)

    print 'add datetime categorical variables'
    X_train_df = add_datetime_categories(X_train_df)
    X_test_df = add_datetime_categories(X_test_df)

    return X_train_df, X_test_df


def process_load_data(filename):

    # import data
    df = pd.read_csv(filename, parse_dates=[[1, 2, 3]], thousands=',')

    # unpivot
    df = pd.melt(df, id_vars=['year_month_day', 'zone_id'], var_name='hour')

    # drop rows where value is NaN
    df.dropna(inplace=True)

    # drop where zoneid = 21 [this is just a total row, that occurs in solution data only]
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

    df = df.sort_values(by=['zone_id', 'datetime'], ascending=[True, True])

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

    dfpivot = dfpivot.sort_values(by='datetime', ascending=True)

    return dfpivot


def process_arima_temp_data(filename):

    # import data
    df = pd.read_csv(filename, index_col=False, parse_dates=[1])
    # column ordering
    df = df[['datetime', 'station_id', 'value']]

    # pivot temps by station
    dfpivot = df.pivot(index='datetime', columns='station_id', values='value')
    dfpivot.columns.name = None
    dfpivot.rename(columns=lambda x: 'tempstn_'+str(x), inplace=True)
    dfpivot.reset_index(inplace=True)

    dfpivot = dfpivot.sort_values(by='datetime', ascending=True)

    return dfpivot


def process_holiday_data(filename):
    df = pd.read_csv(filename, parse_dates=[0])
    df['holiday'] = 1
    df = df[['HolidayDate', 'holiday']]
    df.columns = ['datetime', 'holiday']
    return df


# Add categorical time variables
def add_datetime_categories(df):
    df['summer'] = df['datetime'].dt.month.isin(range(4, 10))
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    return df


def save_data_csv(df, filename):
    df.to_csv(outputfoldername+filename, index=False, date_format='%Y-%m-%d %H:%M:%S')


def get_mean_temps(temphistory):
    # calculate average temperatures at given day and hour over previous years.
    temphistory.set_index(['datetime'], inplace=True)
    day_hour_means = temphistory.groupby(lambda x: (x.month, x.day, x.hour)).mean()
    return day_hour_means


def replace_unknown_temps(to_update, estimates):
    # updates a dataframe to replace nan values with estimates from another df
    # NB operates inplace.
    to_update.set_index(['datetime', 'zone_id'], inplace=True)
    estimates.set_index(['datetime', 'zone_id'], inplace=True)
    to_update.update(estimates)
    to_update.reset_index(inplace=True)


def get_estimated_temps(missingtemp_periods, temp_history):
    # calculates average temperature values from provided history, and provides against specified missing periods
    avgtemps = get_mean_temps(temp_history)
    missingtemp_periods['month_day_hour'] = zip(missingtemp_periods.datetime.dt.month,
                                                missingtemp_periods.datetime.dt.day,
                                                missingtemp_periods.datetime.dt.hour)
    estimatedtemps = missingtemp_periods.merge(avgtemps, left_on='month_day_hour', right_index=True, how='left')

    estimatedtemps.drop('month_day_hour', axis=1, inplace=True)

    return estimatedtemps

def main():

    print 'get data'
    X_train_df, X_test_df = get_data()

    print 'save train data'
    save_data_csv(X_train_df, 'train_processed.csv')

    print 'save test data'
    save_data_csv(X_test_df, 'test_processed.csv')

    print 'also save train data split by zoneid'
    for i in range(1, 21):
        print 'zoneid = %s' % i
        subset = X_train_df[X_train_df.zone_id == i]
        filename = 'train_processed_zone_%s.csv' % i
        save_data_csv(subset, filename)

    print 'save processed temperature data'
    temp = process_temp_data(datafoldername+tempfilename_train)
    save_data_csv(temp, 'tempdata_processed.csv')

if __name__ == "__main__":  main()
