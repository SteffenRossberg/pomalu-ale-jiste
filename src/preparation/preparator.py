import numpy as np
import torch
import pandas as pd
import os
import json
from imblearn.over_sampling import SMOTE, ADASYN


class DataPreparator:

    @staticmethod
    def prepare_frames(provider, days=5, start_date='2000-01-01', end_date='2015-12-31'):
        frames_path = f'data/eod/{start_date}.{end_date}/rl_frames.json'
        if not os.path.exists(frames_path):
            frames = []
            for ticker, company in provider.tickers.items():
                print(f'Loading {ticker:5} - {company} ...')
                quotes = provider.load(ticker, start_date, end_date)
                if quotes is None:
                    continue
                quotes['window'] = DataPreparator.calculate_windows(quotes, days, normalize=True)
                quotes['last_days'] = DataPreparator.calculate_last_days(quotes, days, normalize=True)
                quotes[['buy', 'sell']] = DataPreparator.calculate_signals(quotes)
                frames.append({
                    'ticker': ticker,
                    'company': company,
                    'dates': quotes['date'].dt.strftime('%Y-%m-%d').values[days:].tolist(),
                    'windows': [window.tolist() for window in quotes['window'].values[days:]],
                    'prices': quotes['close'].values[days:].tolist(),
                    'buy_signals': quotes['buy'].values[days:].tolist(),
                    'sell_signals': quotes['sell'].values[days:].tolist(),
                    'buys': quotes['buy'].fillna(method='ffill').values[days:].tolist(),
                    'sells': quotes['sell'].fillna(method='ffill').values[days:].tolist(),
                    'last_days': [last_days for last_days in quotes['last_days'].values[days:]]
                })
            with open(frames_path, 'w') as outfile:
                print(f'Saving {frames_path} ...')
                json.dump(frames, outfile, indent=4)
        with open(frames_path, 'r') as infile:
            print(f'Loading {frames_path} ...')
            frames = json.load(infile)
        return {frame['ticker']: frame for frame in frames}

    @staticmethod
    def prepare_all_quotes(
            provider,
            days=5,
            start_date='2000-01-01',
            end_date='2015-12-31',
            tickers=None,
            intra_day=False):
        quotes_path = f'data/eod/{start_date}.{end_date}/all_quotes.h5'
        tickers_path = f'data/eod/{start_date}.{end_date}/all_tickers.json'
        if intra_day:
            quotes_path = f'data/intra_day/{start_date}.{end_date}/all_quotes.h5'
            tickers_path = f'data/intra_day/{start_date}.{end_date}/all_tickers.json'
        if not os.path.exists(quotes_path):
            if tickers is None:
                tickers = provider.tickers
            all_quotes = None
            all_tickers = {}
            for ticker, company in tickers.items():
                print(f'Load {company} ...')
                if intra_day:
                    quotes = provider.load_intra_day(ticker, start_date, end_date)
                else:
                    quotes = provider.load(ticker, start_date, end_date)
                if quotes is None:
                    continue
                quotes[f'{ticker}_window'] = DataPreparator.calculate_windows(quotes, days=days, normalize=True)
                quotes[f'{ticker}_last_days'] = DataPreparator.calculate_last_days(quotes, days=days, normalize=True)
                quotes = quotes.rename(columns={
                    'close': f'{ticker}_close'
                })
                quotes = quotes[['date', f'{ticker}_window', f'{ticker}_last_days', f'{ticker}_close']].copy()
                if all_quotes is None:
                    all_quotes = quotes
                else:
                    all_quotes = pd.merge(all_quotes, quotes, how='outer', on='date')
                all_tickers[ticker] = company
            all_quotes.to_hdf(quotes_path, key='all_quotes')
            with open(tickers_path, 'w') as outfile:
                print(f'Saving {tickers_path} ...')
                json.dump(all_tickers, outfile, indent=4)
        all_quotes = pd.read_hdf(quotes_path, key='all_quotes')
        with open(tickers_path, 'r') as infile:
            print(f'Loading {tickers_path} ...')
            all_tickers = json.load(infile)
        return all_quotes, all_tickers

    @staticmethod
    def calculate_changes(quotes):
        columns = ['open', 'high', 'low', 'close']
        return [changes.tolist() for changes in quotes[columns].pct_change(1).values]

    @staticmethod
    def over_sample(buys, sells, nones):
        features = np.concatenate((buys, sells, nones), axis=0)
        labels = np.array([1 for _ in range(len(buys))] +
                          [2 for _ in range(len(sells))] +
                          [0 for _ in range(len(nones))], dtype=np.int)
        all_features = features.reshape(
            features.shape[0],
            features.shape[1] * features.shape[2] * features.shape[3])
        sampled_features, sampled_labels = SMOTE().fit_resample(all_features, labels)
        sampled_features = sampled_features.reshape(
            sampled_features.shape[0],
            features.shape[1],
            features.shape[2],
            features.shape[3])
        sampled_buys = np.array([sampled_features[i]
                                 for i in range(len(sampled_features))
                                 if sampled_labels[i] == 1],
                                dtype=np.float32)
        sampled_sells = np.array([sampled_features[i]
                                  for i in range(len(sampled_features))
                                  if sampled_labels[i] == 2],
                                 dtype=np.float32)
        sampled_nones = np.array([sampled_features[i]
                                  for i in range(len(sampled_features))
                                  if sampled_labels[i] == 0],
                                 dtype=np.float32)
        return sampled_buys, sampled_sells, sampled_nones

    @staticmethod
    def prepare_samples(
            provider,
            days=5,
            start_date='2000-01-01',
            end_date='2015-12-31',
            sample_threshold=2,
            sample_match_threshold=0.003,
            buy_sell_match_threshold=0.002,
            filter_match_threshold=0.001,
            device="cpu"):
        """
        Prepares categorized samples of stock price windows

        Parameters
        ----------
        provider : DataProvider
            DataProvider to retrieve stock prices
        days : int, optional
            size of time frame to form the samples, default: 5
        start_date : str, optional
            start date to generate samples from, e.g. '2000-12-31', default: '2000-01-01'
        end_date : str, optional
            end date to generate samples from, e.g. '2000-12-31', default: '2015-12-31'
        sample_threshold : int, optional
            how often should sample detected to decide it is a valid sample or not, default: 2
        sample_match_threshold : float, optional
            upper limit to decide, it is a valid sample or not, default: 0.003
        buy_sell_match_threshold : float, optional
            upper limit to decide, it is same sample of buy and sell or not, default: 0.002
        filter_match_threshold : float, optional
            upper limit to decide, it is a buy/sell sample and not a none signaled sample, default: 0.001
        device : str
            Device to use for calculation, cpu or cuda, default: cpu

        Returns
        -------
        buy_samples, sell_samples, none_samples
        """

        all_buys = None
        all_sells = None
        all_none = None
        samples_path = f'data/eod/{start_date}.{end_date}/samples.npz'
        if not os.path.exists(samples_path):
            for ticker, company in provider.tickers.items():
                # get data
                quotes = provider.load(ticker, start_date, end_date)
                if quotes is None:
                    continue
                # prepare data
                quotes[['buy', 'sell']] = DataPreparator.calculate_signals(quotes)
                quotes['window'] = DataPreparator.calculate_windows(quotes, days=days, normalize=True)
                buys = DataPreparator.filter_windows_by_signal(quotes, 'buy', 'window')
                sells = DataPreparator.filter_windows_by_signal(quotes, 'sell', 'window')
                none = DataPreparator.filter_windows_without_signal(quotes, 'window', days=days)
                print(f'{ticker:5} - {company:40} - buys: {np.shape(buys)} - sells: {np.shape(sells)}')

                all_buys = buys if all_buys is None else np.concatenate((all_buys, buys))
                all_sells = sells if all_sells is None else np.concatenate((all_sells, sells))
                all_none = none if all_none is None else np.concatenate((all_none, none))

            print(f'Total: buys: {np.shape(all_buys)} - sells: {np.shape(all_sells)}')
            unique_buys, unique_sells = DataPreparator.extract_unique_samples(
                device,
                all_buys,
                all_sells,
                match_threshold=buy_sell_match_threshold)
            print(f'Unique: buys: {np.shape(unique_buys)} - sells: {np.shape(unique_sells)}')
            sample_buys = DataPreparator.find_samples(
                device,
                unique_buys,
                sample_threshold=sample_threshold,
                match_threshold=sample_match_threshold)
            sample_sells = DataPreparator.find_samples(
                device,
                unique_sells,
                sample_threshold=sample_threshold,
                match_threshold=sample_match_threshold)
            print(f'Samples: buys: {np.shape(sample_buys)} - sells: {np.shape(sample_sells)}')
            buys, _ = DataPreparator.extract_unique_samples(
                device,
                sample_buys,
                all_none,
                match_threshold=filter_match_threshold,
                extract_both=False)
            sells, _ = DataPreparator.extract_unique_samples(
                device,
                sample_sells,
                all_none,
                match_threshold=filter_match_threshold,
                extract_both=False)
            print(f'Unique samples: buys: {np.shape(buys)} - sells: {np.shape(sells)}')
            np.savez_compressed(samples_path, buys=buys, sells=sells, none=all_none)
        samples_file = np.load(samples_path)
        buy_samples = samples_file['buys']
        sell_samples = samples_file['sells']
        none_samples = samples_file['none']
        return buy_samples, sell_samples, none_samples

    @staticmethod
    def normalize_data(data):
        max_value = data.max()
        min_value = data.min()
        data = ((data - min_value) / (max_value - min_value)
                if max_value - min_value != 0.0
                else np.zeros(data.shape, dtype=np.float32))
        return data

    @staticmethod
    def calculate_mse(x, y):
        diff = x - y
        diff = np.squeeze(diff, axis=0)
        square = diff * diff
        mse = np.mean(square, axis=(2, 3))
        return mse

    @staticmethod
    def extract_unique_samples(device, x, y, match_threshold, extract_both=True):
        def extract(all_matches, samples, match_index):
            matched_indices = set([match[match_index] for match in all_matches])
            unique_samples = [samples[i] for i in range(len(samples)) if i not in matched_indices]
            return np.array(unique_samples, dtype=np.float32)

        matches = DataPreparator.find_matches_by_mse(x, y, match_threshold, device)
        filtered_x = extract(matches, x, 0)
        filtered_y = None if not extract_both else extract(matches, y, 1)
        return filtered_x, filtered_y

    @staticmethod
    def find_samples(device, data, sample_threshold, match_threshold):
        matches = DataPreparator.find_matches_by_mse(data, data, match_threshold, device)
        buckets = [[] for _ in range(len(data))]
        for i in range(len(matches)):
            index0 = matches[i][0]
            index1 = matches[i][1]
            buckets[index0].append([data[index1][0]])
        all_samples = []
        for bucket in filter(lambda b: len(b) > sample_threshold, buckets):
            all_samples += bucket
        result = np.array(all_samples, dtype=np.float32)
        return result

    # noinspection PyUnusedLocal
    @staticmethod
    def find_matches_by_mse(x, y, mse_match_threshold, device):
        all_indices = None
        steps = 5000
        for i in range(0, len(x), steps):
            x_part = None
            x_part = x[i:i + (steps if len(x) - i > steps else len(x) - i)]
            for j in range(0, len(y), steps):
                y_part = None
                diff = None
                square = None
                mse = None
                compare = None
                indices = None
                y_part = y[j:j + (steps if len(y) - j > steps else len(y) - j)]
                x_part = torch.tensor(x_part).to(device)
                y_part = torch.transpose(torch.tensor(y_part), 0, 1).to(device)
                diff = x_part - y_part
                square = diff * diff
                mse = torch.mean(square, dim=(2, 3))
                compare = torch.tensor(mse < mse_match_threshold, dtype=torch.bool, device=device)
                indices = torch.nonzero(compare) + torch.tensor([i, j]).to(device)
                indices = indices.detach().cpu().numpy()
                all_indices = indices if all_indices is None else np.concatenate((all_indices, indices))
        return all_indices

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
    def calculate_last_days(quotes, days=5, normalize=True):
        last_days = DataPreparator.calculate_windows(quotes, days, normalize, ['close'])
        last_days = [window.squeeze().tolist() for window in last_days]
        return last_days

    @staticmethod
    def calculate_windows(quotes, days=5, normalize=True, columns=None):

        def do_not_normalize(window):
            return window

        normalize_action = DataPreparator.normalize_data if normalize else do_not_normalize
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
        quotes = quotes.copy()
        windows = [(np.array([np.zeros((days, len(columns)))], dtype=np.float32)
                    if win.shape[0] < days
                    else normalize_action(np.array([win.values.tolist()], dtype=np.float32)))
                   for win in quotes[columns].rolling(days)]
        return windows

    @staticmethod
    def filter_windows_by_signal(quotes, signal_column, window_column='window'):
        windows = quotes[quotes[signal_column] > 0][window_column].values
        windows = windows.tolist()
        windows = np.array(windows, dtype=np.float32)
        return windows

    @staticmethod
    def filter_windows_without_signal(quotes, window_column='window', days=5):
        non_signal_filter = ~(quotes['buy'] > 0) & ~(quotes['sell'] > 0)
        windows = quotes[non_signal_filter][window_column][days:].values
        windows = windows.tolist()
        windows = np.array(windows, dtype=np.float32)
        return windows
