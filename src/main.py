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
    unique_buys, unique_sells = DataPreparator.extract_unique_samples(all_buys, all_sells, match_threshold=0.002)
    print(f'Unique: buys: {np.shape(unique_buys)} - sells: {np.shape(unique_sells)}')
    buys = DataPreparator.find_samples(unique_buys, sample_threshold=5, match_threshold=0.002)
    sells = DataPreparator.find_samples(unique_sells, sample_threshold=5, match_threshold=0.002)
    print(f'Filtered: buys: {np.shape(buys)} - sells: {np.shape(sells)}')
    np.savez_compressed(samples_path, buys=buys, sells=sells, none=all_none)
samples_file = np.load(samples_path)
buys = samples_file['buys']
sells = samples_file['sells']
all_none = samples_file['none']

# train buyer auto encoder
gym.train_auto_encoder('buyer', buyer, buyer_optimizer, buys, buyer_loss)

# train seller auto encoder
gym.train_auto_encoder('seller', seller, seller_optimizer, sells, seller_loss)

# prepare signaled features and labels
classifier_features = np.concatenate((buys, sells))
classifier_labels = [1 for _ in range(len(buys))] + [2 for _ in range(len(sells))]
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

start_capital = 50_000.0
capital = start_capital
for ticker, company in provider.tickers.items():
    quotes = provider.load(ticker, '2016-01-01', '2020-12-31')
    quotes['window'] = DataPreparator.calculate_windows(quotes, days=days, normalize=True)
    if quotes is None:
        continue
    count = 0
    price = 0.0
    for index, row in quotes.iterrows():
        window = row['window']
        if np.sum(window) == 0:
            continue
        features = torch.from_numpy(window).float().reshape(1, 1, days, 4).to(device)
        prediction = classifier(features).cpu().detach().numpy()
        action = np.argmax(prediction)
        price = row['close']
        if action == 1:
            count = int(capital / price)
            if count > 0:
                capital -= 1.0
                capital -= count * price
                print(f'{row["date"]} - {ticker:5} - buy {count} x ${price:.2f} = $ {count * price:.2f}')
        elif action == 2 and count > 0:
            capital -= 1.0
            capital += count * price
            print(f'{row["date"]} - {ticker:5} - sell {count} x ${price:.2f} = $ {count * price:.2f}')
    if count > 0:
        capital -= 1.0
        capital += count * price
    print(f'{ticker:5} - {company:40} - ${capital:.2f}')
