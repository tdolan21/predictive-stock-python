import matplotlib.pyplot as plt


def generate_plot(dates, Y_test_actual, test_predict_actual, symbol):
    plt.figure(figsize=(14,5))
    plt.plot(dates, Y_test_actual, color='blue', label='Actual closing price')
    plt.plot(dates, test_predict_actual, color='red', label='Predicted closing price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

def print_metrics(train_rmse, val_rmse, test_rmse):
    print("**************************************")
    print(f'Train RMSE: {train_rmse}')
    print(f'Validation RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')
    print("**************************************")
