import pandas as pd
from datetime import datetime
from processandmergedata import save_data_csv

def WRMSE(zoneresults, saveresults=False, modelname=''):
    """
    Calculates Weighted Root Mean Square Error
    :param zoneresults: pandas dataframe including the columns 'datetime', 'zone_id', 'weight', 'value', 'prediction'
    :return: WRMSE scalar
    """

    # calculate errors - system results are just sum over all zones.
    zoneresults['error'] = zoneresults.value - zoneresults.prediction
    sysresults = zoneresults.groupby('datetime')[['weight', 'value', 'prediction', 'error']].sum()
    sysresults.reset_index(inplace=True)
    # calculate square errors
    zoneresults['square_error'] = zoneresults.error ** 2
    sysresults['square_error'] = sysresults.error ** 2

    # apply weights
    zoneresults['weighted_square_error'] = zoneresults.weight * zoneresults.square_error
    sysresults['weighted_square_error'] = sysresults.weight * sysresults.square_error

    # calculate WRMSE
    total_weighted_square_error = zoneresults.weighted_square_error.sum() + sysresults.weighted_square_error.sum()
    total_weights = zoneresults.weight.sum() + sysresults.weight.sum()
    wrmse = (total_weighted_square_error / total_weights) ** 0.5

    if saveresults:
        timestamp = datetime.now().strftime("%Y%M%d-%H%M%S")
        filenamestem = modelname+'-'+timestamp
        print 'saving predictions and errors with filestem: ' + filenamestem
        save_data_csv(zoneresults, filenamestem+'_zoneresults.csv')
        save_data_csv(sysresults, filenamestem+'_sysresults.csv')


    return wrmse
