1) review related work and provide a solution (existing or new) - 2) properly evaluate using suitable metrics
- 3) write a clear manual, report, and upload the code to github with a clear indication for IRDM 2016 group project at UCL


REGIONAL DATA - A hierarchical load forecasting problem
This dataset 

20 energy usage zones
11 weather stations (locations unknown relative to zones)


Observations
patterns amongst zones can be different.  Different customer classes exist (residential, business, industry).  Different responses to temperature / day of week / time of day / season
Electricity provider can also transfer load from one zone to another, on either a short term or long term basis.

Issues
missing data; need to backcast and also forecast.  Can use full history to backcast.  Consider using backcasts to feed back into forecasts?
Can use US holiday data



At the regional / zonal level, aggregate energy usage incorporates demand from a wide range of sources.  A portion will be accounted for by the many individual households in a region, perhaps usage patterns of the type observed in the micro dataset.  Additionally there will be industrial, commercial and large public sector users, with different patterns of usage.  The allocation of energy across the different zones represented in the dataset is also an unknown: load patterns may be effected by infrastructural changes to the energy network itself, or by operational decisions made by the energy supplier.  In this setting it is not practical to model / forecast energy usage from the micro level of individual usage patterns.  Instead we will seek to identify means of predicting from within the aggregate time series, or from other exogenous variables where we can identify some predictive power on overall energy usage levels.




DATA EXPLORATION
We conducted initial data exploration using Tableau, a product designed for quick visualisation of large datasets.  We found this a useful tool to efficiently identify patterns and anomalies within the data, providing useful insights for the design of features to consider for future model development.


ZONAL LOAD DATA
