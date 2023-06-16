# backtest.py
import backtrader as bt
import pandas as pd
import numpy as np

class LSTMStrategy(bt.Strategy):
    def __init__(self):
        self.data_predicted = self.datas[1]

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        price_diff = self.data_predicted - self.data.close
        if price_diff > 0.05:  # Change this threshold to suit your needs
            self.log('BUY CREATE, %.2f' % self.data.close[0])
            self.order_target_percent(target=1.0)
        elif price_diff < -0.05:  # Change this threshold to suit your needs
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.order_target_percent(target=-1.0)


def run_backtest(dates, Y_test_actual, test_predict_actual):
    # Create a new DataFrame for the backtest
    price_data = pd.DataFrame(Y_test_actual, index=dates, columns=['close'])
    predicted_data = pd.DataFrame(test_predict_actual.flatten(), index=dates, columns=['close'])

    # Create a Cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # Add the strategy
    cerebro.addstrategy(LSTMStrategy)

    # Create data feeds
    price_data_feed = bt.feeds.PandasData(dataname=price_data)
    predicted_data_feed = bt.feeds.PandasData(dataname=predicted_data)

    # Add the data feeds to Cerebro
    cerebro.adddata(price_data_feed)
    cerebro.adddata(predicted_data_feed)
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Run the backtest
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    return results
