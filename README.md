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
![Untitled](https://github.com/user-attachments/assets/c45f9645-6020-407a-8446-e15d07b63aec)

![image](https://github.com/user-attachments/assets/414a785e-9520-418a-ba2b-ed799f9739c9)

![image](https://github.com/user-attachments/assets/7913288d-9905-428a-8bc8-f8f0202979f8)

![image](https://github.com/user-attachments/assets/2908b55f-6658-43eb-8ffa-327507eed602)

![Untitled](https://github.com/user-attachments/assets/af65fe4b-7ee3-440c-b951-c61676e3235e)


### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
