import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential, save_model, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
from plot_and_output import generate_plot, print_metrics
import tensorflow as tf
import backtrader as bt
from backtest import run_backtest


tf.config.run_functions_eagerly(True)


def collect_data(symbol):
    # Initialize the TimeSeries class with your Alpha Vantage API key
    ts = TimeSeries(key='', output_format='pandas')

    try:
        # Get daily adjusted stock data
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

        # Select the columns we're interested in
        data = data[['1. open', '2. high', '3. low', '4. close']]
    except ValueError:
        print(f"Error: Invalid stock ticker '{symbol}'")
        return None

    return data

def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Filter out data before 2023
    data = data[data.index.year >= 2022]

    # Normalize the data to a range of 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data, scaler


# Example usage:

# Collect data
symbol = input("Enter a stock ticker: ")
data = collect_data(symbol)

if data is not None:
    # Preprocess data
    data, scaler = preprocess_data(data)

    # Print the preprocessed data
    print(data)


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 3])  # '4. close' is at index 3
    return np.array(X), np.array(Y)

def create_model(look_back, features):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(look_back, features)))  # Increase number of LSTM units
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))  # Add another LSTM layer
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))  # Increase number of LSTM units
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Define look_back period
look_back = 80  # Number of previous time steps to use as input variables to predict the next time period

# Prepare the dataset
X, Y = create_dataset(data.values, look_back)

# Split data into training, validation, and test sets (70% training, 15% validation, 15% testing)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[0:train_size], X[train_size:train_size+val_size], X[train_size+val_size:len(X)]
Y_train, Y_val, Y_test = Y[0:train_size], Y[train_size:train_size+val_size], Y[train_size+val_size:len(Y)]

# Create and fit the LSTM network
features = data.shape[1]
model = create_model(look_back, features)
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_val, Y_val))  # Increase batch size
# After training the model
model.save('my_model.h5')  # saves the model weights along with the architecture

# Generate predictions
train_predict = model.predict(X_train)
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)


# Calculate root mean squared error
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
val_rmse = np.sqrt(mean_squared_error(Y_val, val_predict))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))


# Save the weights
print("Saving model weights...")
model.save_weights('weights.h5')
print("Done!")

# Save the scaler
print("Saving scaler...")
joblib.dump(scaler, 'scaler.pkl')
print("Done!")

print_metrics(train_rmse, val_rmse, test_rmse)





# Create a new scaler for 'close' prices only
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit_transform(data[['4. close']])

print("Saving close scaler...")
joblib.dump(scaler, 'close_scaler.pkl')
print("Done!")



# Inverse transform the 'close' prices
test_predict_actual = close_scaler.inverse_transform(test_predict)
Y_test_actual = close_scaler.inverse_transform(Y_test.reshape(-1, 1))

# Assuming Y_test_actual is a numpy array
Y_test_actual = np.squeeze(Y_test_actual)

# Generate date index for predictions
dates = data.index[train_size+val_size+look_back+1:train_size+val_size+look_back+1+len(Y_test_actual)]


generate_plot(dates, Y_test_actual, test_predict_actual, symbol)

results = run_backtest(dates, Y_test_actual, test_predict_actual)

# Print out the trade statistics
trade_analyzer = results[0].analyzers.tradeanalyzer.get_analysis()
for k, v in trade_analyzer.items():
    print(f'{k}: {v}')
