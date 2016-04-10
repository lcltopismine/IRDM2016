Evaluation

The forecasting accuracy will be evaluated by weighted root mean square error.
Greater weight is placed on forecasting (where temperature data is missing).

The weights are assigned as following:

Each hour of the 8 backcasted weeks at zonal level: 1;

Each hour of the 8 backcasted weeks at system level: 20;

Each hour of the 1 forecasted week at zonal level: 8;

Each hour of the 1 forecasted week at system level: 160;

Details of weight assignment are shown in data file: weights.csv.


Evaluation formula is given here:
https://www.kaggle.com/wiki/RootMeanSquaredError

Python code:
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y, y_pred)**0.5