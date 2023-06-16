# Stock Price Predictor

This project uses a Long Short-Term Memory (LSTM) model to predict the closing price of a given stock using historical data.

## Dependencies

The project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- keras
  

You can install these dependencies using pip:

bash
pip install numpy pandas matplotlib scikit-learn keras

## Usage

To use the stock price predictor, you need to run the main.py script. When prompted, enter the ticker symbol of the stock you want to predict.

python main.py

The script will download the historical data for the given stock, train the LSTM model, and then use the model to predict the closing price of the stock. The real and predicted prices are plotted on a graph for visual comparison.

The script also calculates and prints the following evaluation metrics for the trained model:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R-squared (R2)

## Note

The model is trained on 80% of the available data and tested on the remaining 20%. The data is normalized before being fed into the model. The model architecture consists of three LSTM layers with dropout and L1/L2 regularization, followed by a Dense layer.
