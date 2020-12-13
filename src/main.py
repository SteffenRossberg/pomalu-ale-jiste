import argparse
import matplotlib.pyplot as plt
from src.environment.dataprovider import DataProvider


if __name__ != "__main__":
    exit(0)
parser = argparse.ArgumentParser()
parser.add_argument("--apikey", required=True, type=str, help="API key for Tiingo webservice")
args = parser.parse_args()
api_key = args.apikey

provider = DataProvider(api_key)
msft = provider.load('MSFT', '2010-01-01', '2010-12-31')
msft.set_index(keys=['date'], drop=False, inplace=True)
msft[['open', 'high', 'low', 'close']].plot()
plt.show()
plt.close()
