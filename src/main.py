import argparse
import numpy as np
import torch
import os
from datetime import datetime
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.training.gym import Gym
from src.trading.trader import Trader
from src.environment.stock_exchange import StockExchange
from src.environment.enums import TrainingLevels
from src.utility.logger import Logger
from prometheus_client import start_http_server

if __name__ != "__main__":
    exit(0)

start_http_server(5000, '0.0.0.0')

# Let's train data of 16 years from 01/01/2000 to 12/31/2015
train_start_date = '2000-01-01'
train_end_date = '2015-12-31'
# Let's trade unseen data of 5 years from 01/01/2016 to 12/31/2020.
trader_start_date = '2019-01-01'
trader_end_date = '2020-12-31'
trader_start_capital = 50_000.0
trader_order_fee = 1.0
trader_capital_gains_tax = 25.0
trader_solidarity_surcharge = 5.5
# Use the last 5 days as a time frame for sampling, forecasting and trading
sample_days = 5
today = datetime.now()
# run_id = f"{today:%Y%m%d.%H%M%S}"
run_id = "20210106.224436"

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
manager = NetManager(device, data_directory=f'data/networks/{run_id}')

print("Create gym ...")
gym = Gym(manager)

print("Create buyer auto encoder ...")
buyer, buyer_optimizer = manager.create_auto_encoder(sample_days)
buyer_loss = manager.load_net('buyer.auto.encoder', buyer, buyer_optimizer)

print("Create seller auto encoder ...")
seller, seller_optimizer = manager.create_auto_encoder(sample_days)
seller_loss = manager.load_net('seller.auto.encoder', seller, seller_optimizer)

print("Create classifier ...")
classifier, classifier_optimizer = manager.create_classifier(buyer, seller)
classifier_loss = manager.load_net('trader.classifier', classifier, classifier_optimizer)

print("Create decision maker ...")
decision_maker, decision_optimizer = manager.create_decision_maker(classifier, state_size=7)
best_mean_val = manager.load_net('trader.decision.maker', decision_maker, decision_optimizer, -100.0)

print("Prepare samples ...")
buy_samples, sell_samples, none_samples = DataPreparator.prepare_samples(provider,
                                                                         days=sample_days,
                                                                         start_date=train_start_date,
                                                                         end_date=train_end_date)

print("Prepare quotes ...")
all_quotes, all_tickers = DataPreparator.prepare_all_quotes(provider,
                                                            sample_days,
                                                            trader_start_date,
                                                            trader_end_date,
                                                            provider.tickers,
                                                            True)


print("Create trader ...")
trader = Trader(
    provider,
    decision_maker,
    len(provider.tickers),
    int(len(provider.tickers) / 3),
    all_quotes,
    all_tickers,
    trader_start_date,
    trader_end_date,
    trader_start_capital,
    trader_order_fee,
    trader_capital_gains_tax,
    trader_solidarity_surcharge,
    device,
    sample_days)


def train(train_id, train_detectors, train_classifier, train_decision_maker):
    global buyer_loss, seller_loss, classifier_loss, best_mean_val

    os.makedirs(f'data/networks/{train_id}', exist_ok=True)

    Logger.log(train_id, f"Id: {train_id}")

    if train_detectors > 0:
        print("Train buyer samples detector ...")
        gym.train_auto_encoder(
            'buyer',
            buyer,
            buyer_optimizer,
            buy_samples,
            buyer_loss)
        print("Reload buyer samples detector with best training result after training ...")
        buyer_loss = manager.load_net('buyer.auto.encoder', buyer, buyer_optimizer)
        Logger.log(train_id, f"buyer.auto.encoder: {buyer_loss:.7f}")
        classifier.buyer_auto_encoder = buyer
        decision_maker.classifier.buyer_auto_encoder = buyer

        print("Train seller samples detector ...")
        gym.train_auto_encoder(
            'seller',
            seller,
            seller_optimizer,
            sell_samples,
            seller_loss)
        print("Reload seller samples detector with best training result after training ...")
        seller_loss = manager.load_net('seller.auto.encoder', seller, seller_optimizer)
        Logger.log(train_id, f"seller.auto.encoder: {seller_loss:.7f}")
        classifier.seller_auto_encoder = seller
        decision_maker.classifier.seller_auto_encoder = seller

    if train_classifier > 0:
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
        gym.train_classifier(
            'trader',
            classifier,
            classifier_optimizer,
            classifier_features,
            classifier_labels,
            classifier_none_features,
            classifier_none_labels,
            classifier_loss,
            max_steps=20)
        print("Reload classifier with best training result after training ...")
        classifier_loss = manager.load_net('trader.classifier', classifier, classifier_optimizer)
        Logger.log(train_id, f"trader.classifier: {classifier_loss:.7f}")
        decision_maker.classifier = classifier

    if train_decision_maker > 0:
        print("Deactivate buyer samples detector parameters ...")
        for parameter in decision_maker.classifier.buyer_auto_encoder.parameters():
            parameter.requires_grad = False

        print("Deactivate seller samples detector parameters ...")
        for parameter in decision_maker.classifier.seller_auto_encoder.parameters():
            parameter.requires_grad = False

        print("Deactivate classifier parameters ...")
        for parameter in decision_maker.classifier.parameters():
            parameter.requires_grad = False

        reset_on_close = False
        training_level = TrainingLevels.Skip | TrainingLevels.Buy | TrainingLevels.Hold | TrainingLevels.Sell
        print("Prepare stock exchange environment ...")
        stock_exchange = StockExchange.from_provider(provider,
                                                     sample_days,
                                                     train_start_date,
                                                     train_end_date,
                                                     reset_on_close=reset_on_close)
        print(f"Train decision maker {str(training_level)} (reset on close = {reset_on_close}) ...")
        stock_exchange.train_level = training_level
        gym.train_decision_maker(
            'trader',
            decision_maker,
            decision_optimizer,
            best_mean_val,
            stock_exchange,
            stop_predicate=lambda mean_value: mean_value > 220.0,
            stop_on_count=5)
        print("Reload decision maker with best training result after training ...")
        best_mean_val = manager.load_net('trader.decision.maker', decision_maker, decision_optimizer)
        print(f"Seeds: {stock_exchange.seeds}")
        Logger.log(train_id, f"Train Level: {str(training_level)}")
        Logger.log(train_id, f"Reset On Close: {reset_on_close}")
        Logger.log(train_id, f"Stock Exchange Seeds: {stock_exchange.seeds}")
        Logger.log(train_id, f"Trader Best mean value: {best_mean_val:.7f}")


train(run_id, args.train_detectors, args.train_classifier, args.train_decision_maker)

print(f"Best mean value: {best_mean_val:.7f}")
trader.trade(run_id, True)
