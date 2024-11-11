## DEVELOPED BY: PRAVEEN S
## REGISTER NO: 212222240078
## DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/OnionTimeSeries - Sheet1.csv')

# Convert 'datesold' to datetime and set as index
data['date'] = pd.to_datetime(data['Date'])
data.set_index('date', inplace=True)

# Plot the time series data
plt.plot(data.index, data['Min'])
plt.xlabel('Date')
plt.ylabel('Price of Onion')
plt.title('Price Time Series')
plt.show()

# Function to check stationarity using ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Min'])

# Plot ACF and PACF to determine SARIMA parameters
plot_acf(data['Min'])
plt.show()
plot_pacf(data['Min'])
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Min'][:train_size], data['Min'][train_size:]

# Define and fit the SARIMA model on training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Min')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()


```
### OUTPUT:

![Untitled](https://github.com/user-attachments/assets/076e92f7-777b-4669-8abf-8fc25dcfac7a)

![image](https://github.com/user-attachments/assets/14f60888-1b68-49e4-8a82-8c8d8fa9bb91)

![image](https://github.com/user-attachments/assets/58c1c8ff-12ed-4532-93fc-0db294e5a0f8)

![image](https://github.com/user-attachments/assets/fd79fb3c-7bc3-4bad-a3b8-4ae7cb51321c)

![Untitled](https://github.com/user-attachments/assets/0225a8dc-091d-4cbc-8d6b-a3534487f2cf)


### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
