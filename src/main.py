import argparse
import numpy as np
import torch
import os
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
print("Create classifier ...")
classifier, classifier_optimizer = manager.create_classifier(buyer, seller)
classifier_loss = manager.load_net('trader.classifier', classifier, classifier_optimizer)

# prepare buyer, seller auto encoder train data
all_buys = None
all_sells = None
all_none = None
samples_path = 'data/samples.npz'
if not os.path.exists(samples_path):
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
        none = DataPreparator.filter_windows_without_signal(quotes, 'window', days=days)
        print(f'{ticker:5} - {company:40} - buys: {np.shape(buys)} - sells: {np.shape(sells)}')
        if all_buys is None:
            all_buys = buys
        else:
            all_buys = np.concatenate((all_buys, buys))
        if all_sells is None:
            all_sells = sells
        else:
            all_sells = np.concatenate((all_sells, sells))
        if all_none is None:
            all_none = none
        else:
            all_none = np.concatenate((all_none, none))
    print(f'Total: buys: {np.shape(all_buys)} - sells: {np.shape(all_sells)}')
    all_buys = DataPreparator.find_samples(all_buys, sample_threshold=5, match_threshold=0.002)
    all_sells = DataPreparator.find_samples(all_sells, sample_threshold=5, match_threshold=0.002)
    print(f'Filtered: buys: {np.shape(all_buys)} - sells: {np.shape(all_sells)}')
    np.savez_compressed(samples_path, all_buys=all_buys, all_sells=all_sells, all_none=all_none)
samples_file = np.load(samples_path)
all_buys = samples_file['all_buys']
all_sells = samples_file['all_sells']
all_none = samples_file['all_none']

# train buyer auto encoder
gym.train_auto_encoder('buyer', buyer, buyer_optimizer, all_buys, buyer_loss)

# train seller auto encoder
gym.train_auto_encoder('seller', seller, seller_optimizer, all_sells, seller_loss)

# prepare signaled features and labels
classifier_features = np.concatenate((all_buys, all_sells))
classifier_labels = [1 for _ in range(len(all_buys))] + [2 for _ in range(len(all_sells))]
classifier_labels = np.array(classifier_labels)

# prepare none signaled features and labels
classifier_none_features = all_none
classifier_none_labels = [0 for _ in range(len(all_none))]
classifier_none_labels = np.array(classifier_none_labels)

# deactivate trained auto encoders parameters
for parameter in classifier.buyer_auto_encoder.parameters():
    parameter.requires_grad = False
for parameter in classifier.seller_auto_encoder.parameters():
    parameter.requires_grad = False

# train trader classifier
gym.train_classifier('trader', classifier, classifier_optimizer,
                     classifier_features, classifier_labels,
                     classifier_none_features, classifier_none_labels,
                     classifier_loss, max_steps=1000)
