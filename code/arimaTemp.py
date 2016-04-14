from __future__ import division
import sys
import pandas as pd
import numpy as np
import pyper as pr
import matplotlib.pylab as plt 
from statsmodels.tsa.stattools import adfuller, acf, pacf_ols, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, date, timedelta
from arima import arima
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

#Predicts temperatures from the 2008-07-01 to the 2008-07-07 with arima model
def arimaTemp(filename):
    
    datetimes = np.zeros(0)

    r = pr.R(RCMD="C:\\Program Files\\R\\R-3.1.2\\bin\\R", use_numpy=True, use_pandas=True)
    
    datetimes = np.arange('2008-07-01 00:00:00','2008-07-08 00:00:00',dtype='datetime64[h]')

    for j in range(1,12):

        data = pd.read_csv('../data/output/temp_history_processed_station_%s.csv'%j, parse_dates='datetime')  
     
        subts = data["value"]
        print 'Predictions for zone %s'%j    
        results = arima(subts,r)       
       
        results = pd.DataFrame(results, columns=['value'])
        
        results['datetime'] = datetimes
        results['station_id'] = j
    
        results.to_csv(filename, index=False, date_format='%Y-%m-%d %H:%M:%S', mode='a')  
        
arimaTemp('../data/output/arimaTemp.csv')      
     
