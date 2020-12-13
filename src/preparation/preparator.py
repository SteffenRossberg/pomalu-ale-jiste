import numpy as np


class DataPreparator:

    @staticmethod
    def normalize_data(data):
        max_value = data.max()
        min_value = data.min()
        data = (data - min_value) / (max_value - min_value)
        return np.array(data.tolist())

    @staticmethod
    def calculate_signals(quotes):

        def __apply_buy(row):
            return (row['close']
                    if row['buy'] > 0 and (row['sell_tmp'] / row['buy_tmp']) - 1.0 > 0.01
                    else np.nan)

        def __apply_sell(row):
            return (row['close']
                    if row['sell'] > 0 and (row['sell_tmp'] / row['buy_tmp']) - 1.0 > 0.01
                    else np.nan)

        def __apply_initial_buy(row):
            return (row['close']
                    if row['prev_close'] > row['close'] and row['close'] < row['next_close']
                    else np.nan)

        def __apply_initial_sell(row):
            return (row['close']
                    if row['prev_close'] < row['close'] and row['close'] > row['next_close']
                    else np.nan)

        quotes = quotes.copy()
        quotes['prev_close'] = quotes['close'].shift(1)
        quotes['next_close'] = quotes['close'].shift(-1)
        quotes['buy'] = quotes.apply(__apply_initial_buy, axis=1)
        quotes['sell'] = quotes.apply(__apply_initial_sell, axis=1)
        for _ in range(0, 2):
            quotes['buy_tmp'] = quotes['buy'].fillna(method='bfill')
            quotes['sell_tmp'] = quotes['sell'].fillna(method='ffill')
            quotes['buy'] = quotes.apply(__apply_buy, axis=1)
            quotes['sell'] = quotes.apply(__apply_sell, axis=1)
        return quotes[['buy', 'sell']]

    @staticmethod
    def calculate_windows(quotes, days=5, normalize=True):

        def do_not_normalize(window):
            return window

        def build_window(row):
            normalize_action = DataPreparator.normalize_data if normalize else do_not_normalize
            index = row['index']
            start = index - days + 1
            stop = index + 1
            window = (np.array([np.zeros((days, len(columns)))], dtype=np.float32)
                      if index < days - 1
                      else normalize_action(np.array([quotes[start:stop][columns].values], dtype=np.float32)))
            return window

        quotes = quotes.copy()
        columns = ['open', 'high', 'low', 'close']
        quotes['index'] = range(len(quotes))
        windows = quotes.apply(build_window, axis=1)
        return windows

    @staticmethod
    def filter_windows_by_signal(quotes, signal_column, window_column='window'):
        windows = quotes[quotes[signal_column] > 0][window_column].values
        windows = windows.tolist()
        windows = np.array(windows)
        return windows
