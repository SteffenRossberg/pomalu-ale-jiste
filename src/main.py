import argparse
import numpy as np
import torch
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.training.gym import Gym
from src.trading.trader import Trader
from src.environment.stock_exchange import StockExchange


if __name__ != "__main__":
    exit(0)

# Let's train data of 16 years from 01/01/2000 to 12/31/2015
train_start_date = '2000-01-01'
train_end_date = '2015-12-31'
# Let's trade unseen data of 5 years from 01/01/2016 to 12/31/2020.
trader_start_date = '2016-01-01'
trader_end_date = '2020-12-31'
trader_start_capital = 50_000.0
trader_capital_gains_tax = 25.0
trader_solidarity_surcharge = 5.5
# Use the last 5 days as a time frame for sampling, forecasting and trading
sample_days = 5

# get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--apikey",
                    required=True,
                    type=str,
                    help="API key for Tiingo webservice")
parser.add_argument("--train_detectors",
                    required=False,
                    type=int,
                    default=1,
                    help="Train buyer/seller detectors (sample auto encoders)")
parser.add_argument("--train_classifier",
                    required=False,
                    type=int,
                    default=1,
                    help="Train classifier")
parser.add_argument("--train_decision_maker",
                    required=False,
                    type=int,
                    default=1,
                    help="Train decision maker")
args = parser.parse_args()
# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Create data provider ...")
provider = DataProvider(args.apikey)

print("Create agent manager ...")
manager = NetManager(device)

print("Create gym ...")
gym = Gym(manager)

print("Create buyer auto encoder ...")
buyer, buyer_optimizer = manager.create_auto_encoder(sample_days)
buyer_loss = manager.load_net('buyer.autoencoder', buyer, buyer_optimizer)

print("Create seller auto encoder ...")
seller, seller_optimizer = manager.create_auto_encoder(sample_days)
seller_loss = manager.load_net('seller.autoencoder', seller, seller_optimizer)

print("Create classifier ...")
classifier, classifier_optimizer = manager.create_classifier(buyer, seller)
classifier_loss = manager.load_net('trader.classifier', classifier, classifier_optimizer)

print("Create decision maker ...")
# decision_maker, decision_optimizer = manager.create_decision_maker(classifier, state_size=sample_days + 1)
decision_maker, decision_optimizer = manager.create_decision_maker(classifier, state_size=3)
best_mean_val = manager.load_net('trader.decision_maker', decision_maker, decision_optimizer, -100.0)

print("Create trader ...")
trader = Trader(trader_start_capital,
                trader_capital_gains_tax,
                trader_solidarity_surcharge,
                provider,
                device,
                sample_days)

print("Prepare samples ...")
buy_samples, sell_samples, none_samples = DataPreparator.prepare_samples(provider,
                                                                         days=sample_days,
                                                                         start_date=train_start_date,
                                                                         end_date=train_end_date)

if args.train_detectors > 0:
    print("Train buyer samples detector ...")
    gym.train_auto_encoder('buyer', buyer, buyer_optimizer, buy_samples, buyer_loss)

    print("Train seller samples detector ...")
    gym.train_auto_encoder('seller', seller, seller_optimizer, sell_samples, seller_loss)

    print("Reload buyer samples detector with best training result after training ...")
    manager.load_net('buyer.autoencoder', buyer, buyer_optimizer)

    print("Reload seller samples detector with best training result after training ...")
    manager.load_net('seller.autoencoder', seller, seller_optimizer)

if args.train_classifier > 0:
    print("Deactivate buyer samples detector parameters ...")
    for parameter in classifier.buyer_auto_encoder.parameters():
        parameter.requires_grad = False
    print("Deactivate seller samples detector parameters ...")
    for parameter in classifier.seller_auto_encoder.parameters():
        parameter.requires_grad = False

    print("Prepare signed features and labels for trader classifier ...")
    classifier_features = np.concatenate((buy_samples, sell_samples))
    classifier_labels = [1 for _ in range(len(buy_samples))] + [2 for _ in range(len(sell_samples))]
    classifier_labels = np.array(classifier_labels)

    print("Prepare unsigned features and labels for trader classifier ...")
    classifier_none_features = none_samples
    classifier_none_labels = [0 for _ in range(len(none_samples))]
    classifier_none_labels = np.array(classifier_none_labels)

    print("Train trader classifier ...")
    gym.train_classifier('trader', classifier, classifier_optimizer,
                         classifier_features, classifier_labels,
                         classifier_none_features, classifier_none_labels,
                         classifier_loss, max_steps=1000)

    print("Reload classifier with best training result after training ...")
    manager.load_net('trader.classifier', classifier, classifier_optimizer)

if args.train_decision_maker > 0:
    print("Deactivate buyer samples detector parameters ...")
    for parameter in classifier.buyer_auto_encoder.parameters():
        parameter.requires_grad = False

    print("Deactivate seller samples detector parameters ...")
    for parameter in classifier.seller_auto_encoder.parameters():
        parameter.requires_grad = False

    print("Deactivate classifier parameters ...")
    for parameter in classifier.parameters():
        parameter.requires_grad = False

    print("Prepare stock exchange environment ...")
    stock_exchange = StockExchange.from_provider(provider,
                                                 sample_days,
                                                 train_start_date,
                                                 train_end_date,
                                                 reset_on_close=False)

    print("Train decision maker ...")
    gym.train_decision_maker('trader', decision_maker, decision_optimizer, best_mean_val, stock_exchange)

    print("Reload decision maker with best training result after training ...")
    manager.load_net('trader.decision_maker', decision_maker, decision_optimizer)

print(f"Trade all stocks from {trader_start_date} to {trader_end_date} ...")
trader.trade(decision_maker, trader_start_date, trader_end_date, False, provider.tickers)

print(f"Buy and hold all stocks from {trader_start_date} to {trader_end_date} ...")
trader.buy_and_hold(trader_start_date, trader_end_date, False, provider.tickers)

print(f"Trade DOW30 stocks from {trader_start_date} to {trader_end_date} ...")
trader.trade(decision_maker, trader_start_date, trader_end_date, False, provider.dow30_tickers)

print(f"Buy and hold DOW30 stocks from {trader_start_date} to {trader_end_date} ...")
trader.buy_and_hold(trader_start_date, trader_end_date, False, provider.dow30_tickers)
