import argparse
import matplotlib.pyplot as plt
from src.environment.dataprovider import DataProvider


if __name__ != "__main__":
    exit(0)

# get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--apikey", required=True, type=str, help="API key for Tiingo webservice")
args = parser.parse_args()

# init stuff
provider = DataProvider(args.apikey)

# simple test to get data from provider
for ticker, company in provider.tickers.items():
    quotes = provider.load(ticker, '2010-01-01', '2010-12-31')
    if quotes is None:
        continue
    quotes.set_index(keys=['date'], drop=False, inplace=True)
    quotes[['open', 'high', 'low', 'close']].plot(title=company)
    plt.show()
    plt.close()
