import argparse
import numpy as np
import torch
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.training.gym import Gym


if __name__ != "__main__":
    exit(0)

# get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--apikey", required=True, type=str, help="API key for Tiingo webservice")
args = parser.parse_args()

# init stuff
days = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print("Create data provider ...")
provider = DataProvider(args.apikey)
print("Create agent manager ...")
manager = NetManager(device)
print("Create gym ...")
gym = Gym(manager)
print("Create buyer ...")
buyer, buyer_optimizer = manager.create_auto_encoder(days)
buyer_loss = manager.load_net('buyer.autoencoder', buyer, buyer_optimizer)
print("Create seller ...")
seller, seller_optimizer = manager.create_auto_encoder(days)
seller_loss = manager.load_net('seller.autoencoder', seller, seller_optimizer)

# prepare buyer, seller auto encoder train data
all_buys = None
all_sells = None
for ticker, company in provider.tickers.items():
    # get data
    quotes = provider.load(ticker, '2000-01-01', '2015-12-31')
    if quotes is None:
        continue
    # prepare data
    quotes[['buy', 'sell']] = DataPreparator.calculate_signals(quotes)
    quotes['window'] = DataPreparator.calculate_windows(quotes, days=days, normalize=True)
    buys = DataPreparator.filter_windows_by_signal(quotes, 'buy', 'window')
    sells = DataPreparator.filter_windows_by_signal(quotes, 'sell', 'window')
    print(f'{ticker:5} - {company:40} - buys: {np.shape(buys)} - sells: {np.shape(sells)}')
    if all_buys is None:
        all_buys = buys
    else:
        all_buys = np.concatenate((all_buys, buys))
    if all_sells is None:
        all_sells = sells
    else:
        all_sells = np.concatenate((all_sells, sells))
print(f'Total: buys: {np.shape(all_buys)} - sells: {np.shape(all_sells)}')

# train buyer
gym.train_auto_encoder('buyer', buyer, buyer_optimizer, all_buys, buyer_loss)
# train seller
gym.train_auto_encoder('seller', seller, seller_optimizer, all_sells, seller_loss)
