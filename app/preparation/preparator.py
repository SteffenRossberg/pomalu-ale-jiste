import numpy as np
import torch
import pandas as pd
import os
import json
from imblearn.over_sampling import SMOTE
from app.environment.dataprovider import DataProvider


class DataPreparator:

    @classmethod
    def prepare_all_quotes(
            cls,
            provider: DataProvider,
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
            all_quotes = None
            all_tickers = {}
            columns = ['open', 'high', 'low', 'close']
            tickers = tickers if tickers is not None else provider.tickers.keys()
            for ticker in tickers:
                company = provider.tickers[ticker]
                print(f'Load {company} ...')
                if intra_day:
                    quotes = provider.load_intra_day(ticker, start_date, end_date)
                else:
                    quotes = provider.load(ticker, start_date, end_date)
                if quotes is None:
                    continue
                quotes[f'{ticker}_window'] = \
                    cls.calculate_windows(
                        quotes,
                        days=days,
                        normalize=True,
                        columns=columns,
                        adjust=provider.adjust_prices)
                quotes = quotes.rename(columns={'adj_close': f'{ticker}_close'})
                columns = ['date', f'{ticker}_window', f'{ticker}_close']
                quotes = quotes[columns].copy()
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

    @classmethod
    def prepare_samples(
            cls,
            provider,
            days=5,
            start_date='2000-01-01',
            end_date='2015-12-31',
            sample_threshold=2,
            sample_match_threshold=0.003,
            buy_sell_match_threshold=0.002,
            filter_match_threshold=0.001,
            tickers=None,
            device='cpu'):
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
        tickers : array, optional
            tickers to sample, default: None
        device : str
            Device to use for calculation, cpu or cuda, default: cpu

        Returns
        -------
        buy_samples, sell_samples, none_samples
        """

        all_buys = None
        all_sells = None
        all_none = None
        columns = ['open', 'high', 'low', 'close']
        samples_path = f'data/eod/{start_date}.{end_date}/samples.npz'
        if not os.path.exists(samples_path):
            tickers = tickers if tickers is not None else provider.tickers.keys()
            for ticker in tickers:
                company = provider.tickers[ticker]
                # get data
                quotes = provider.load(ticker, start_date, end_date)
                if quotes is None:
                    continue
                # prepare data
                quotes[['buy', 'sell']] = cls.calculate_signals(quotes)
                quotes['window'] = \
                    cls.calculate_windows(
                        quotes,
                        days=days,
                        normalize=True,
                        columns=columns,
                        adjust=provider.adjust_prices)
                buys = cls.filter_windows_by_signal(quotes, days, 'buy', 'window')
                sells = cls.filter_windows_by_signal(quotes, days, 'sell', 'window')
                none = cls.filter_windows_without_signal(quotes, days, 'window')
                print(f'{ticker:5} - {company:40} - buys: {np.shape(buys)} - sells: {np.shape(sells)}')
                if len(buys) > 0:
                    all_buys = buys if all_buys is None else np.concatenate((all_buys, buys))
                if len(sells) > 0:
                    all_sells = sells if all_sells is None else np.concatenate((all_sells, sells))
                if len(none) > 0:
                    all_none = none if all_none is None else np.concatenate((all_none, none))

            print(f'Total: buys: {np.shape(all_buys)} - sells: {np.shape(all_sells)}')
            unique_buys, unique_sells = cls.extract_unique_samples(
                device,
                all_buys,
                all_sells,
                match_threshold=buy_sell_match_threshold)
            print(f'Unique: buys: {np.shape(unique_buys)} - sells: {np.shape(unique_sells)}')
            sample_buys = cls.find_samples(
                device,
                unique_buys,
                sample_threshold=sample_threshold,
                match_threshold=sample_match_threshold)
            sample_sells = cls.find_samples(
                device,
                unique_sells,
                sample_threshold=sample_threshold,
                match_threshold=sample_match_threshold)
            print(f'Samples: buys: {np.shape(sample_buys)} - sells: {np.shape(sample_sells)}')
            buys, _ = cls.extract_unique_samples(
                device,
                sample_buys,
                all_none,
                match_threshold=filter_match_threshold,
                extract_both=False)
            sells, _ = cls.extract_unique_samples(
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
    def calculate_changes(quotes):
        columns = ['adj_open', 'adj_high', 'adj_low', 'adj_close']
        return [changes.tolist() for changes in quotes[columns].pct_change(1).values]

    @staticmethod
    def over_sample(buys, sells, nones, seed):
        features = np.concatenate((buys, sells, nones), axis=0)
        labels = np.array([1 for _ in range(len(buys))] +
                          [2 for _ in range(len(sells))] +
                          [0 for _ in range(len(nones))], dtype=np.int)
        all_features = features.reshape(
            features.shape[0],
            features.shape[1] * features.shape[2] * features.shape[3])
        sampled_features, sampled_labels = SMOTE(random_state=seed).fit_resample(all_features, labels)
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

    @classmethod
    def extract_unique_samples(cls, device, x, y, match_threshold, extract_both=True):
        def extract(all_matches, samples, match_index):
            matched_indices = set([match[match_index] for match in all_matches])
            unique_samples = [samples[i] for i in range(len(samples)) if i not in matched_indices]
            return np.array(unique_samples, dtype=np.float32)

        matches = cls.find_matches_by_mse(x, y, match_threshold, device)
        filtered_x = extract(matches, x, 0)
        filtered_y = None if not extract_both else extract(matches, y, 1)
        return filtered_x, filtered_y

    @classmethod
    def find_samples(cls, device, data, sample_threshold, match_threshold):
        matches = cls.find_matches_by_mse(data, data, match_threshold, device)
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
                x_calc = torch.tensor(x_part).to(device)
                y_calc = torch.transpose(torch.tensor(y_part), 0, 1).to(device)
                diff = x_calc - y_calc
                square = diff * diff
                mse = torch.mean(square, dim=(2, 3))
                compare = torch.le(mse, mse_match_threshold)
                indices = torch.nonzero(compare) + torch.tensor([i, j]).to(device)
                indices = indices.detach().cpu().numpy()
                all_indices = indices if all_indices is None else np.concatenate((all_indices, indices))
        return all_indices

    @classmethod
    def calculate_signals(cls, quotes):
        quotes = quotes.copy()
        quotes['index'] = range(len(quotes))
        quotes['min'] = quotes['adj_close'].rolling(10, center=True).min()
        quotes['max'] = quotes['adj_close'].rolling(10, center=True).max()
        quotes['sell_index'] = quotes.apply(lambda r: r['index'] if r['max'] == r['adj_close'] else np.nan, axis=1)
        quotes['buy_index'] = quotes.apply(lambda r: r['index'] if r['min'] == r['adj_close'] else np.nan, axis=1)
        cls.fill_na(quotes)
        quotes['sell_index'] = quotes.apply(
            lambda r: (r['index']
                       if quotes[int(r['sell_start']):int(r['sell_end']) + 1]['adj_close'].max() == r['adj_close']
                       else np.nan),
            axis=1)
        quotes['buy_index'] = quotes.apply(
            lambda r: (r['index']
                       if quotes[int(r['buy_start']):int(r['buy_end']) + 1]['adj_close'].min() == r['adj_close']
                       else np.nan),
            axis=1)
        cls.fill_na(quotes)
        quotes['sell'] = quotes.apply(
            lambda r: (r['adj_close']
                       if (quotes[int(r['buy_start']):int(r['buy_end']) + 1]['adj_close'].max() == r['adj_close'] and
                           quotes[int(r['buy_start']):int(r['buy_end']) + 1]['sell_index'].max() > 0)
                       else np.nan),
            axis=1)
        quotes['buy'] = quotes.apply(
            lambda r: (r['adj_close']
                       if (quotes[int(r['sell_start']):int(r['sell_end']) + 1]['adj_close'].min() == r['adj_close'] and
                           quotes[int(r['sell_start']):int(r['sell_end']) + 1]['buy_index'].max() > 0)
                       else np.nan),
            axis=1)

        quotes['tmp_sell'] = quotes['sell'].fillna(method='bfill')
        quotes['tmp_buy'] = quotes['buy'].fillna(method='ffill')
        quotes['buy'] = quotes.apply(
            lambda r: (r['buy'] if r['buy'] > 0.0 and (r['tmp_sell'] / r['tmp_buy']) - 1.0 > 0.05 else np.nan),
            axis=1
        )
        quotes['sell'] = quotes.apply(
            lambda r: (r['sell'] if r['sell'] > 0.0 and (r['tmp_sell'] / r['tmp_buy']) - 1.0 > 0.05 else np.nan),
            axis=1
        )

        return quotes[['buy', 'sell']]

    @staticmethod
    def fill_na(quotes):
        quotes['sell_start'] = quotes['sell_index'].fillna(method='ffill').fillna(0)
        quotes['buy_start'] = quotes['buy_index'].fillna(method='ffill').fillna(0)
        quotes['sell_end'] = quotes['sell_index'].fillna(method='bfill').fillna(quotes['index'].max())
        quotes['buy_end'] = quotes['buy_index'].fillna(method='bfill').fillna(quotes['index'].max())

    @classmethod
    def calculate_windows(cls, quotes, days=5, normalize=True, columns=None, adjust=None):
        columns = ['adj_open', 'adj_high', 'adj_low', 'adj_close'] if columns is None else columns
        normalize_data = cls.normalize_data if normalize else (lambda window: window[columns].values)
        adjust_data = adjust if adjust is not None else (lambda window: window[columns].values)
        quotes = quotes.copy()
        windows = [(np.array([np.zeros((days, len(columns)))], dtype=np.float32)
                    if win.shape[0] < days
                    else normalize_data(np.array([adjust_data(win).tolist()], dtype=np.float32)))
                   for win in quotes.rolling(days)]
        return windows

    @staticmethod
    def filter_windows_by_signal(quotes, days, signal_column, window_column='window'):
        windows = quotes[quotes[signal_column] > 0][window_column][days:].values
        windows = windows.tolist()
        windows = np.array(windows, dtype=np.float32)
        return windows

    @staticmethod
    def filter_windows_without_signal(quotes, days, window_column='window'):
        non_signal_filter = ~(quotes['buy'] > 0) & ~(quotes['sell'] > 0)
        windows = quotes[non_signal_filter][window_column][days:].values
        windows = windows.tolist()
        windows = np.array(windows, dtype=np.float32)
        return windows
