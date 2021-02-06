import argparse
import torch
import os
from datetime import datetime
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.training.gym import Gym
from src.trading.game import Game
from src.environment.stock_exchange import StockExchange
from src.environment.enums import TrainingLevels
from src.utility.logger import Logger
import prometheus_client
import numpy as np

if __name__ != "__main__":
    exit(0)

prometheus_client.start_http_server(5000, '0.0.0.0')

# Let's train data of 16 years from 01/01/2000 to 12/31/2015
train_start_date = '2000-01-01'
train_end_date = '2012-12-31'
validation_start_date = '2013-01-01'
validation_end_date = '2015-12-31'

# Let's trade unseen data of 5 years from 01/01/2016 to 12/31/2020.
trader_start_date = '2016-01-01'
trader_end_date = '2020-12-31'
trade_intra_day = False
trader_start_capital = 50_000.0
trader_max_limit_positions = 35
trader_order_fee = 1.0
trader_capital_gains_tax = 25.0
trader_solidarity_surcharge = 5.5

# Use the last 5 days as a time frame for sampling, forecasting and trading
sample_days = 5
seed = 1234567890
deterministic = True
today = datetime.now()
# run_id = f"{today:%Y%m%d.%H%M%S}"
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
manager = NetManager(device, seed, deterministic, data_directory=f'data/networks/{run_id}')

print("Create gym ...")
gym = Gym(manager)

print("Create trader ...")
trader = manager.create_trader(sample_days, state_size=7)
buyer_optimizer, buyer_result = manager.create_buyer_optimizer(trader)
seller_optimizer, seller_result = manager.create_seller_optimizer(trader)
classifier_optimizer, classifier_result = manager.create_classifier_optimizer(trader)
decision_maker_optimizer, decision_maker_result = manager.create_decision_maker_optimizer(trader)

buyer_optimizer, buyer_result = manager.load(
    'buyer',
    trader.buyer,
    buyer_optimizer,
    trader.reset_buyer,
    lambda: manager.create_buyer_optimizer(trader),
    buyer_result)

seller_optimizer, seller_result = manager.load(
    'seller',
    trader.seller,
    seller_optimizer,
    trader.reset_seller,
    lambda: manager.create_seller_optimizer(trader),
    seller_result)

classifier_optimizer, classifier_result = manager.load(
    'classifier',
    trader.classifier,
    classifier_optimizer,
    trader.reset_classifier,
    lambda: manager.create_classifier_optimizer(trader),
    classifier_result)

decision_maker_optimizer, decision_maker_result = manager.load(
    'decision_maker',
    trader.decision_maker,
    decision_maker_optimizer,
    trader.reset_decision_maker,
    lambda: manager.create_decision_maker_optimizer(trader),
    decision_maker_result)

print("Prepare samples ...")
buy_samples, sell_samples, none_samples = DataPreparator.prepare_samples(
    provider,
    days=sample_days,
    start_date=train_start_date,
    end_date=train_end_date,
    device=device)

print("Resample ...")
buy_samples, sell_samples, none_samples = DataPreparator.over_sample(buy_samples, sell_samples, none_samples)

print("Prepare quotes ...")
all_quotes, all_tickers = DataPreparator.prepare_all_quotes(
    provider,
    sample_days,
    trader_start_date,
    trader_end_date,
    provider.tickers,
    trade_intra_day)

print("Prepare frames ...")
frames = DataPreparator.prepare_frames(provider, sample_days, train_start_date, train_end_date)

print("Create game ...")
game = Game(
    provider,
    trader,
    len(provider.tickers),
    trader_max_limit_positions,
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


def train(train_id, train_buyer, train_seller, train_classifier, train_decision_maker):
    global buyer_optimizer, seller_optimizer, classifier_optimizer, decision_maker_optimizer
    global buyer_result, seller_result, classifier_result, decision_maker_result

    os.makedirs(f'data/networks/{train_id}', exist_ok=True)

    Logger.log(train_id, f"Id: {train_id}")

    if train_buyer > 0:
        print("Train buyer samples detector ...")
        buyer_result = gym.train_auto_encoder(
            'buyer',
            trader.buyer,
            buyer_optimizer,
            buy_samples,
            buyer_result,
            max_steps=50)
        print("Reload trader with best training result after training ...")
        buyer_optimizer, buyer_result = manager.load(
            'buyer',
            trader.buyer,
            buyer_optimizer,
            trader.reset_buyer,
            lambda: manager.create_buyer_optimizer(trader),
            buyer_result)
        Logger.log(train_id, f"buyer.auto.encoder: {buyer_result:.7f}")

    if train_seller > 0:
        print("Train seller samples detector ...")
        seller_result = gym.train_auto_encoder(
            'seller',
            trader.seller,
            seller_optimizer,
            sell_samples,
            seller_result,
            max_steps=50)
        print("Reload trader with best training result after training ...")
        seller_optimizer, seller_result = manager.load(
            'seller',
            trader.seller,
            seller_optimizer,
            trader.reset_seller,
            lambda: manager.create_seller_optimizer(trader),
            seller_result)
        Logger.log(train_id, f"seller.auto.encoder: {seller_result:.7f}")

    if train_classifier > 0:
        print("Train classifier ...")
        classifier_features = np.concatenate((buy_samples, sell_samples, none_samples), axis=0)
        classifier_labels = np.array(
            [1 for _ in range((len(buy_samples)))] +
            [2 for _ in range((len(sell_samples)))] +
            [0 for _ in range((len(none_samples)))],
            dtype=np.int)
        classifier_result = gym.train_classifier(
            'classifier',
            trader,
            classifier_optimizer,
            classifier_features,
            classifier_labels,
            classifier_result,
            max_steps=20)
        print("Reload trader with best training result after training ...")
        classifier_optimizer, seller_result = manager.load(
            'classifier',
            trader.classifier,
            classifier_optimizer,
            trader.reset_classifier,
            lambda: manager.create_classifier_optimizer(trader),
            classifier_result)
        Logger.log(train_id, f"classifier: {classifier_result:.7f}")

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
            seed=seed)
        validation_stock_exchange = StockExchange.from_provider(
            provider,
            sample_days,
            validation_start_date,
            validation_end_date,
            random_offset_on_reset=True,
            reset_on_close=True,
            seed=seed)
        print(f"Train decision maker {str(training_level)} ...")
        train_stock_exchange.train_level = training_level
        validation_stock_exchange.train_level = training_level
        decision_maker_result = gym.train_trader(
            'decision_maker',
            trader,
            decision_maker_optimizer,
            decision_maker_result,
            train_stock_exchange,
            validation_stock_exchange)
        print("Reload trader with best training result after training ...")
        decision_maker_optimizer, decision_maker_result = manager.load(
            'decision_maker',
            trader.decision_maker,
            decision_maker_optimizer,
            trader.reset_decision_maker,
            lambda: manager.create_decision_maker_optimizer(trader),
            decision_maker_result)
        Logger.log(train_id, f"decision_maker: {decision_maker_result:.7f}")
        Logger.log(train_id, f"Train Level: {str(training_level)}")
        Logger.log(train_id, f"Trader Best mean value: {decision_maker_result:.7f}")


train(run_id, args.train_buyer, args.train_seller, args.train_classifier, args.train_decision_maker)

game.trade(run_id, trade_intra_day)
