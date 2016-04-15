THINGS WE’VE DONE
ARIMA model of load and temperature
Benchmark regression model
Gradient boosting model
Neural Nets model

CODING THINGS TO DO
Collect results from each model
[for benchmark, gradient, nn] - rerun with arima temp forecasts and compare results
Tidy up and ensure commented.  Note any dependencies.
Remove any unused code

THE MANUAL - all to contribute.
README.MD only
List dependencies; List steps to run each model

THE REPORT
Latex.
Intro - Artemis
 - brief discussion of what GEFCom2012 is

Design Challenges
1. temperature forecasting      Megan
2. feature selection						Artemis
- what evidence that different factors are important - stuff from poster?
- what features used
- time series features [day of week etc]
holidays
discussion in Tao paper
3. evaluation methods						Rupert
discussion of evaluation methods root mean square; weighted.  Rupert

Modelling Approaches
Introduction								Rupert

For each of the below, cover:
- design choices
- implementation methods
- interpretation of feature importance [incl. charts?]
- results

1. 	Arima		(load & temp)		Megame
2. 	Regression					 		Rupert
3. 	Gradient Boosting    		Artemis
4. 	Neural Nets					 		Rupert

5. Compare results between models - charts? tables?
6. Conclusions - commentary on strengths/weaknesses of different approaches; future work.   

7. References   -  All to compile
Tao paper
IRDM competition paper
..




THINGS WE NEED TO COVER
1) clear description of your systems/algorithms
2) Motivations of design choices
3) Sensible manipulation of the data
   AND new findings/conclusions from our analysis
4) Appropriate choice of evaluation measures


TIME SERIES PROJECT
How to identify patterns in time series is critical for many businesses including finance and e-commerce
 - This project aims to implement and test various time series techniques for forecasting. Possible topics:
 - Using deep learning & neural networks reading1, reading2, reading3 - Using regression. reading1, reading2
 - Datasets: Energy dataset; Climate data; UCI • Analyse and report your results:

1) review related work and provide a solution (existing or new)
2) properly evaluate using suitable metrics
3) write a clear manual, report, and upload the code to github with a clear indication for IRDM 2016 group project at UCL

REGIONAL DATA - A hierarchical load forecasting problem
20 energy usage zones
11 weather stations (locations unknown relative to zones)

Observations
patterns amongst zones can be different.
Different customer classes exist (residential, business, industry).
Different responses to temperature / day of week / time of day / season
Electricity provider can also transfer load from one zone to another, on either a short term or long term basis.

Issues
missing data; need to backcast and also forecast.  Can use full history to backcast.
Consider using backcasts to feed back into forecasts?
Can use US holiday data