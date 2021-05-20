import argparse
import torch
import os
from datetime import datetime
from app.environment.dataprovider import DataProvider
from app.preparation.preparator import DataPreparator
from app.networks.manager import NetManager
from app.training.gym import Gym
from app.utility.logger import Logger
import prometheus_client
import numpy as np

if __name__ != "__main__":
    exit(0)

prometheus_client.start_http_server(5000, '0.0.0.0')

# Let's train data of 13 years from 01/01/2000 to 12/31/2012
train_start_date = '2000-01-01'
train_end_date = '2012-12-31'
train_tickers = None

# Use the last 5 days as a time frame for sampling, forecasting and trading
sample_days = 5
seed = 1234567890
deterministic = True
today = datetime.now()
run_id = "current"


# get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--apikey",
                    required=True,
                    type=str,
                    help="API key for Tiingo webservice")
parser.add_argument("--train_buyer",
                    required=False,
                    type=int,
                    default=0,
                    help="Train buyer detectors (sample auto encoders)")
parser.add_argument("--train_seller",
                    required=False,
                    type=int,
                    default=0,
                    help="Train seller detectors (sample auto encoders)")
parser.add_argument("--train_classifier",
                    required=False,
                    type=int,
                    default=0,
                    help="Train classifier (sample classifier)")
args = parser.parse_args()

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


print("Create data provider ...")
provider = DataProvider(args.apikey)

print("Create agent manager ...")
manager = NetManager(device, seed, deterministic, data_directory=f'data/networks/{run_id}')

print("Create gym ...")
gym = Gym(manager)

print("Create trader ...")
trader = manager.create_trader(sample_days)
buyer_optimizer, buyer_result = manager.create_buyer_optimizer(trader)
seller_optimizer, seller_result = manager.create_seller_optimizer(trader)
classifier_optimizer, classifier_result = manager.create_classifier_optimizer(trader)

buyer_optimizer, buyer_result = \
    manager.load(
        'buyer',
        trader.buyer,
        buyer_optimizer,
        trader.reset_buyer,
        lambda: manager.create_buyer_optimizer(trader),
        buyer_result)

seller_optimizer, seller_result = \
    manager.load(
        'seller',
        trader.seller,
        seller_optimizer,
        trader.reset_seller,
        lambda: manager.create_seller_optimizer(trader),
        seller_result)

classifier_optimizer, classifier_result = \
    manager.load(
        'classifier',
        trader.classifier,
        classifier_optimizer,
        trader.reset_classifier,
        lambda: manager.create_classifier_optimizer(trader),
        classifier_result)

print("Prepare samples ...")
raw_buy_samples, raw_sell_samples, raw_none_samples = \
    DataPreparator.prepare_samples(
        provider,
        days=sample_days,
        start_date=train_start_date,
        end_date=train_end_date,
        tickers=train_tickers,
        device=device)

print("Resample ...")
buy_samples, sell_samples, none_samples = \
    DataPreparator.over_sample(
        raw_buy_samples,
        raw_sell_samples,
        raw_none_samples,
        seed)

os.makedirs(f'data/networks/{run_id}', exist_ok=True)

Logger.log(run_id, f"Id: {run_id}")

if args.train_buyer > 0:
    buyer_features = np.array(buy_samples, dtype=np.float32)
    print("Train buyer samples detector ...")
    buyer_result = gym.train_auto_encoder(
        'buyer',
        trader.buyer,
        buyer_optimizer,
        buyer_features,
        buyer_result,
        max_epochs=500,
        max_steps=10)
    print("Reload trader with best training result after training ...")
    buyer_optimizer, buyer_result = manager.load(
        'buyer',
        trader.buyer,
        buyer_optimizer,
        trader.reset_buyer,
        lambda: manager.create_buyer_optimizer(trader),
        buyer_result)
    Logger.log(run_id, f"buyer.auto.encoder: {buyer_result:.7f}")

if args.train_seller > 0:
    seller_features = np.array(sell_samples, dtype=np.float32)
    print("Train seller samples detector ...")
    seller_result = gym.train_auto_encoder(
        'seller',
        trader.seller,
        seller_optimizer,
        seller_features,
        seller_result,
        max_epochs=500,
        max_steps=10)
    print("Reload trader with best training result after training ...")
    seller_optimizer, seller_result = manager.load(
        'seller',
        trader.seller,
        seller_optimizer,
        trader.reset_seller,
        lambda: manager.create_seller_optimizer(trader),
        seller_result)
    Logger.log(run_id, f"seller.auto.encoder: {seller_result:.7f}")

if args.train_classifier > 0:
    classifier_features = []
    classifier_labels = []
    for i in range(np.max((len(buy_samples), len(sell_samples), len(none_samples)))):
        if i < len(buy_samples):
            classifier_features.append(buy_samples[i])
            classifier_labels.append(1)
        if i < len(sell_samples):
            classifier_features.append(sell_samples[i])
            classifier_labels.append(2)
        if i < len(none_samples):
            classifier_features.append(none_samples[i])
            classifier_labels.append(0)
    classifier_features = np.array(classifier_features, dtype=np.float32)
    classifier_labels = np.array(classifier_labels, dtype=np.int32)
    print("Train classifier ...")
    classifier_result = gym.train_classifier(
        'classifier',
        trader,
        classifier_optimizer,
        classifier_features,
        classifier_labels,
        classifier_result,
        max_epochs=500,
        max_steps=20)
    print("Reload trader with best training result after training ...")
    classifier_optimizer, seller_result = manager.load(
        'classifier',
        trader.classifier,
        classifier_optimizer,
        trader.reset_classifier,
        lambda: manager.create_classifier_optimizer(trader),
        classifier_result)
    Logger.log(run_id, f"classifier: {classifier_result:.7f}")
