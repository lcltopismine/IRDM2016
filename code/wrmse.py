import pandas as pd

def WRMSE(zoneresults):
    """
    Calculates Weighted Root Mean Square Error
    :param zoneresults: pandas dataframe including the columns 'datetime', 'zone_id', 'weight', 'value', 'prediction'
    :return: WRMSE scalar
    """

    # calculate errors - system results are just sum over all zones.
    zoneresults['error'] = zoneresults.value - zoneresults.prediction
    sysresults = zoneresults.groupby('datetime')[['weight', 'error']].sum()

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

    return wrmse
