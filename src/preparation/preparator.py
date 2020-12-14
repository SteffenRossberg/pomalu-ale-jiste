import numpy as np
import time
import threading
from src.preparation.msethread import MseThread


class DataPreparator:

    @staticmethod
    def normalize_data(data):
        max_value = data.max()
        min_value = data.min()
        data = (data - min_value) / (max_value - min_value)
        return np.array(data.tolist())

    @staticmethod
    def calculate_mse(x, y):
        diff = x - y
        diff = np.squeeze(diff, axis=0)
        square = diff * diff
        mse = np.mean(square, axis=(2, 3))
        return mse

    @staticmethod
    def find_samples(data, sample_threshold, match_threshold):
        x = data
        all_indices = None
        steps = 2000
        threads = []

        for i in range(0, len(x), steps):
            x_part = x[i:i + (steps if len(x) - i > steps else len(x) - i)]
            for j in range(0, len(x), steps):
                y_part = x[j:j + (steps if len(x) - j > steps else len(x) - j)]
                thread = MseThread(x_part, y_part, i, j, match_threshold)
                threads.append(thread)

        for thread in threads:
            while len([t1 for t1 in filter(lambda t2: t2.isAlive(), threads)]) >= 8:
                time.sleep(0.01)
            thread.start()

        for thread in threads:
            thread.join()
            all_indices = thread.indices if all_indices is None else np.concatenate((all_indices, thread.indices))

        buckets = [[] for _ in range(len(x))]
        for i in range(len(all_indices)):
            index0 = all_indices[i][0]
            index1 = all_indices[i][1]
            buckets[index0].append([data[index1][0]])
        all_samples = []
        for bucket in filter(lambda b: len(b) > sample_threshold, buckets):
            all_samples += bucket
        result = np.array(all_samples, dtype=np.float32)
        return result

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
