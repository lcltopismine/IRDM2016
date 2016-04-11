Evaluation

The forecasting accuracy will be evaluated by weighted root mean square error.
Greater weight is placed on forecasting (where temperature data is missing).

The weights are assigned as following:

Each hour of the 8 backcasted weeks at zonal level: 1;

Each hour of the 8 backcasted weeks at system level: 20;

Each hour of the 1 forecasted week at zonal level: 8;

Each hour of the 1 forecasted week at system level: 160;

Details of weight assignment are shown in data file: weights.csv.


Evaluation formula for RootMeanSquare is given here:
https://www.kaggle.com/wiki/RootMeanSquaredError

Python code:
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y, y_pred)**0.5

R:
RMSE <- sqrt(mean((y-y_pred)^2))


BUT note that this project uses WEIGHTED rmse.
code is given here:
https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting/forums/t/2591/error-metric-suggestions/13986#post13986

WRMS =sqrt( ( 1/W) * ( 1*sum(hourly zone backcast error^2) + 20*sum(hourly system backcast error^2)  +
                                    8*sum(hourly zone forecast error^2) + 160*sum(hourly system forecast error^2))
                                    
There is also weights.csv containing a mapping of weights - we could alternatively add this in to processing data?
