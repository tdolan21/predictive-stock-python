import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l1, l2
from keras.layers import TimeDistributed
import yfinance as yf
from datetime import datetime

# Define the ticker symbol
tickerSymbol = input("Please enter the ticker symbol: ")

# Download the historical data
tickerData = yf.Ticker(tickerSymbol)
df_original = tickerData.history(period='1d', start='2010-1-1', end=datetime.today().strftime('%Y-%m-%d'))
df_original.to_csv('stock_data.csv')

# Preprocess the data
df = df_original['Close'].values
df = df.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[0:train_size, :], df[train_size:len(df), :]

# Create a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, len(train)):
    X_train.append(train[i-60:i, 0])
    y_train.append(train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()

# Add an LSTM layer with L1 regularization and dropout
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l1(0.01)))
model.add(Dropout(0.3))

# Add another LSTM layer with L2 regularization and dropout
model.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))

# Add another LSTM layer with dropout
model.add(LSTM(units=50))
model.add(Dropout(0.3))

# Add a Dense layer
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Test the model
inputs = df[train_size - 60:]  # Use data from the end of the training set to the end of the entire dataset
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, len(inputs)):  # Start from 60 to create sequences of 60 data points
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Create a new dataframe for the predicted values
predicted_df = pd.DataFrame(predicted_stock_price[-len(test):], index=df_original.index[train_size:], columns=['Predicted Price'])




# Plot the results
fig, axs = plt.subplots(2, figsize=(14,10))

# Plot the real stock price
axs[0].plot(df_original.index[-5:], df_original['Close'][-5:], color='red', label='Real Stock Price')
axs[0].set_title('Real Stock Price for ' + tickerSymbol)
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Stock Price (in $)')
axs[0].legend()
axs[0].grid(True)

# Plot the predicted stock price
axs[1].plot(predicted_df.index[-5:], predicted_df['Predicted Price'][-5:], color='blue', label='Predicted Stock Price')
axs[1].set_title('Predicted Stock Price for ' + tickerSymbol)
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Stock Price (in $)')
axs[1].legend()
axs[1].grid(True)

# Format the x-axis to display the dates more clearly
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()

plt.tight_layout()
plt.show()



# Calculate metrics
actual_values = df_original['Close'][train_size+60:train_size+60+len(predicted_stock_price)]
predicted_stock_price = predicted_stock_price[:len(actual_values)]

predicted_stock_price = predicted_stock_price.flatten()
rmse = np.sqrt(mean_squared_error(actual_values, predicted_stock_price))
mae = mean_absolute_error(actual_values, predicted_stock_price)
r2 = r2_score(actual_values, predicted_stock_price)




print("Length of actual values: ", len(actual_values))
print("Length of predicted values: ", len(predicted_stock_price))


# Print metrics
print("Evaluation metrics for the trained model:")
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("R-squared (R2 ): {:.2f}".format(r2))
