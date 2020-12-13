import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator

if __name__ != "__main__":
    exit(0)

# get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--apikey", required=True, type=str, help="API key for Tiingo webservice")
args = parser.parse_args()

# init stuff
provider = DataProvider(args.apikey)

# test to get data from provider and prepare it
for ticker, company in provider.tickers.items():
    # get data
    quotes = provider.load(ticker, '2011-01-01', '2015-12-31')
    if quotes is None:
        continue
    quotes.set_index(keys=['date'], drop=False, inplace=True)

    # prepare data
    quotes[['buy', 'sell']] = DataPreparator.calculate_signals(quotes)
    quotes['window'] = DataPreparator.calculate_windows(quotes, days=5, normalize=True)
    buys = DataPreparator.filter_windows_by_signal(quotes, 'buy', 'window')
    sells = DataPreparator.filter_windows_by_signal(quotes, 'sell', 'window')
    print(f'{ticker:5} - {company:40} - buys: {np.shape(buys)} - sells: {np.shape(sells)}')

    # do visually something
    ax = quotes['close'].plot(title=company)
    quotes[['buy', 'sell']].plot(ax=ax, marker='o')
    plt.show()
    plt.close()
