from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
from statsmodels.tsa.stattools import adfuller, acf, pacf_ols, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

#Test the stationarity of the time series by plotting the rolling mean and rolling standard deviation
#and by performing the Dickey-Fuller test
def test_stationarity(ts):
    
    #Determing rolling statistics

    rolmean = ts.rolling(window=24,center=False).mean()
    rolstd = ts.rolling(window=24,center=False).std()
    
    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

#Decompose the time series as trend, seaonality and residuals and plot the results    
def decompose(ts_log):
    decomposition = seasonal_decompose(ts_log)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()    
    plt.show()

#Return the differenced time series values and plot the result    
def difference(ts_log):
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    #plt.plot(ts_log_diff)    
    #plt.show()
    return ts_log_diff

#Plot the ACF and PACF on the differenced values in order the determine the ARIMA parameters (p,q)    
def plotACFandPACF(ts_log_diff):
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf_ols(ts_log_diff, nlags=20)
    
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()    
    plt.show()

#Perform ARIMA predictions, get the predictions back to scale and plots the results    
def arima(subts,p,q,start,end):
    
    ts_log = np.log(subts) 
    ts_log_diff = difference(ts_log)
    
    #Computes the ARIMA model 
    model = ARIMA(ts_log, order=(p, 1, q))  
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))    
    plt.show()
    
    #Compute ARIMA predictions for existing values
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    
    #Compute ARIMA predictions for forcasted values
    predictions = results_ARIMA.predict(start=start,end=end,typ='linear')
    predictionsSeries = pd.Series(predictions, copy=True)
    #print predictionsSeries   
    #print predictions_ARIMA_diff.tail(20)
    predictions_ARIMA_diff = predictions_ARIMA_diff.append(predictionsSeries, verify_integrity=True)
    
    #Compute cumulative sums
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    #print predictions_ARIMA_diff_cumsum    

    #Add cumulatives sums back to base number
    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=predictions_ARIMA_diff.index)

    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    #print predictions_ARIMA_log

    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(subts)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-subts)**2)/len(subts)))  
    plt.show()
    

if __name__ == '__main__':
    
    missingRanges = [['2005-03-06 00:00:00','2005-03-12 23:00:00'],['2005-06-20 00:00:00','2005-06-26 23:00:00'],
                     ['2005-09-10 00:00:00','2005-09-10 23:00:00'],['2005-12-25 00:00:00','2005-12-31 23:00:00'],
                     ['2006-02-13 00:00:00','2006-02-19 23:00:00'],['2006-05-25 00:00:00','2006-05-31 23:00:00'],
                     ['2006-08-02 00:00:00','2006-08-08 23:00:00'],['2006-11-22 00:00:00','2006-11-28 23:00:00'],
                     ['2008-06-30 06:00:00','2008-07-07 23:00:00']]
    
    for j in range(1,21):
        
        data = pd.read_csv('../data/output/train_processed_zone_%s.csv'%j, parse_dates='datetime', index_col='datetime')
        ts = data["value"]        
        print 'Predictions for zone %s'%j
        
        for i in range(0,9):
            print 'Predicitons for %s to %s'%(missingRanges[i][0],missingRanges[i][1]) 
            
            if i == 0:
                subts = ts[:'2005-03-06']
            else:
                subts = ts[missingRanges[i-1][1]:missingRanges[i][0]]
                
            plt.plot(subts)
            plt.show()
            ts_log = np.log(subts)   
            ts_log_diff = difference(ts_log)
            test_stationarity(ts_log_diff)
            plotACFandPACF(ts_log_diff)
            results = arima(subts,1,2,missingRanges[i][0],missingRanges[i][1])       
