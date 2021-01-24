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
trader_start_date = '2016-01-01'
trader_end_date = '2020-12-31'
trader_decision_maker = 'trader.decision.maker'
trade_intra_day = False
trader_start_capital = 50_000.0
trader_order_fee = 1.0
trader_capital_gains_tax = 25.0
trader_solidarity_surcharge = 5.5
# Use the last 5 days as a time frame for sampling, forecasting and trading
sample_days = 5
today = datetime.now()
# run_id = f"{today:%Y%m%d.%H%M%S}"
run_id = "current"

# get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--apikey",
                    required=True,
                    type=str,
                    help="API key for Tiingo webservice")
parser.add_argument("--train_detectors",
                    required=False,
                    type=int,
                    default=0,
                    help="Train buyer/seller detectors (sample auto encoders)")
parser.add_argument("--train_classifier",
                    required=False,
                    type=int,
                    default=0,
                    help="Train classifier")
parser.add_argument("--train_decision_maker",
                    required=False,
                    type=int,
                    default=0,
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
                                                            trade_intra_day)

print("Prepare frames ...")
frames = DataPreparator.prepare_frames(provider, sample_days, train_start_date, train_end_date)

print("Create trader ...")
trader = Trader(
    provider,
    decision_maker,
    len(provider.tickers),
    10,  # int(len(provider.tickers) / 3),
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
        all_windows = []
        all_labels = []
        for ticker in frames.keys():
            buy_windows = [
                frames[ticker]['windows'][i]
                for i in range(len(frames[ticker]['windows']))
                if frames[ticker]['buy_signals'][i] > 0.0
            ]
            all_windows += buy_windows
            all_labels += [1 for _ in range(len(buy_windows))]
            sell_windows = [
                frames[ticker]['windows'][i]
                for i in range(len(frames[ticker]['windows']))
                if frames[ticker]['sell_signals'][i] > 0.0
            ]
            all_windows += sell_windows
            all_labels += [2 for _ in range(len(sell_windows))]
            none_windows = [
                frames[ticker]['windows'][i]
                for i in range(len(frames[ticker]['windows']))
                if not frames[ticker]['buy_signals'][i] > 0.0 and not frames[ticker]['sell_signals'][i] > 0.0
            ]
            signals_count = len(buy_windows) + len(sell_windows)
            all_windows += none_windows[:signals_count]
            all_labels += [0 for _ in range(signals_count)]
        all_windows = np.array(all_windows, dtype=np.float32)
        all_labels = np.array(all_labels)

        print("Train trader classifier ...")
        gym.train_classifier(
            'trader',
            classifier,
            classifier_optimizer,
            all_windows,
            all_labels,
            classifier_loss,
            max_steps=20)
        print("Reload classifier with best training result after training ...")
        classifier_loss = manager.load_net('trader.classifier', classifier, classifier_optimizer)
        Logger.log(train_id, f"trader.classifier: {classifier_loss:.7f}")
        decision_maker.classifier = classifier

    if train_decision_maker > 0:

        print("Prepare stock exchange environment ...")
        training_level = TrainingLevels.Skip | TrainingLevels.Buy | TrainingLevels.Hold | TrainingLevels.Sell
        train_stock_exchange = StockExchange.from_provider(
            provider,
            sample_days,
            train_start_date,
            train_end_date,
            random_offset_on_reset=False,
            reset_on_close=False,
            seed=1234567890)
        validation_stock_exchange = StockExchange.from_provider(
            provider,
            sample_days,
            train_start_date,
            train_end_date,
            random_offset_on_reset=True,
            reset_on_close=True,
            seed=1234567890)
        print(f"Train decision maker {str(training_level)} ...")
        train_stock_exchange.train_level = training_level
        validation_stock_exchange.train_level = training_level
        gym.train_decision_maker(
            'trader',
            decision_maker,
            decision_optimizer,
            best_mean_val,
            train_stock_exchange,
            validation_stock_exchange)
        print("Reload decision maker with best training result after training ...")
        best_mean_val = manager.load_net('trader.decision.maker', decision_maker, decision_optimizer)
        Logger.log(train_id, f"Train Level: {str(training_level)}")
        Logger.log(train_id, f"Trader Best mean value: {best_mean_val:.7f}")

    manager.save_net(f'buyer.auto.encoder', buyer, buyer_optimizer, loss=buyer_loss)
    manager.save_net(f'buyer.encoder', buyer.encoder, loss=buyer_loss)
    manager.save_net(f'buyer.decoder', buyer.decoder, loss=buyer_loss)
    manager.save_net(f'seller.auto.encoder', seller, seller_optimizer, loss=seller_loss)
    manager.save_net(f'seller.encoder', seller.encoder, loss=seller_loss)
    manager.save_net(f'seller.decoder', seller.decoder, loss=seller_loss)
    manager.save_net(f'trader.classifier', classifier, classifier_optimizer, loss=classifier_loss)
    manager.save_net(f'trader.decision.maker', decision_maker, decision_optimizer, loss=best_mean_val)


train(run_id, args.train_detectors, args.train_classifier, args.train_decision_maker)
if trader_decision_maker is not None:
    best_mean_val = manager.load_net(trader_decision_maker, decision_maker, decision_optimizer, -100.0)

trader.trade(run_id, trade_intra_day)
