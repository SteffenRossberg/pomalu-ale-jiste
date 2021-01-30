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
from prometheus_client import start_http_server

if __name__ != "__main__":
    exit(0)

start_http_server(5000, '0.0.0.0')

# Let's train data of 16 years from 01/01/2000 to 12/31/2015
train_start_date = '2000-01-01'
train_end_date = '2012-12-31'
validation_start_date = '2013-01-01'
validation_end_date = '2015-12-31'
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
parser.add_argument("--train_trader",
                    required=False,
                    type=int,
                    default=0,
                    help="Train trader")
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

print("Create trader ...")
trader = manager.create_trader(sample_days, state_size=7)
manager.load_trader('trader', trader)

print("Create optimizers ...")
optimizers, results = manager.create_optimizers(trader)
results = manager.load_optimizers('trader', optimizers, results)

print("Prepare samples ...")
buy_samples, sell_samples, none_samples = DataPreparator.prepare_samples(
    provider,
    days=sample_days,
    start_date=train_start_date,
    end_date=train_end_date,
    device=device)

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


def train(train_id, train_buyer, train_seller, train_trader):
    global results

    os.makedirs(f'data/networks/{train_id}', exist_ok=True)

    Logger.log(train_id, f"Id: {train_id}")

    while train_buyer > 0:
        print("Train buyer samples detector ...")
        gym.train_auto_encoder(
            'buyer',
            trader,
            trader.buyer,
            optimizers,
            buy_samples,
            results,
            max_steps=50)
        print("Reload trader with best training result after training ...")
        manager.load_trader('trader', trader)
        results = manager.load_optimizers('trader', optimizers, results)
        Logger.log(train_id, f"buyer.auto.encoder: {results['buyer']:.7f}")
        if results['buyer'] <= 0.09:
            break
        # trader.reset_buyer(device)
        optimizers['buyer'], results['buyer'] = manager.create_buyer_optimizer(trader)

    while train_seller > 0:
        print("Train seller samples detector ...")
        gym.train_auto_encoder(
            'seller',
            trader,
            trader.seller,
            optimizers,
            sell_samples,
            results,
            max_steps=50)
        print("Reload trader with best training result after training ...")
        manager.load_trader('trader', trader)
        results = manager.load_optimizers('trader', optimizers, results)
        Logger.log(train_id, f"seller.auto.encoder: {results['seller']:.7f}")
        if results['seller'] <= 0.09:
            break
        # trader.reset_seller(device)
        optimizers['seller'], results['seller'] = manager.create_seller_optimizer(trader)

    if train_trader > 0:
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
            validation_start_date,
            validation_end_date,
            random_offset_on_reset=True,
            reset_on_close=True,
            seed=1234567890)
        print(f"Train decision maker {str(training_level)} ...")
        train_stock_exchange.train_level = training_level
        validation_stock_exchange.train_level = training_level
        gym.train_trader(
            'decision_maker',
            trader,
            optimizers,
            results,
            train_stock_exchange,
            validation_stock_exchange)
        print("Reload trader with best training result after training ...")
        manager.load_trader('trader', trader)
        results = manager.load_optimizers('trader', optimizers, results)
        Logger.log(train_id, f"Train Level: {str(training_level)}")
        Logger.log(train_id, f"Trader Best mean value: {results['decision_maker']:.7f}")


train(run_id, args.train_buyer, args.train_seller, args.train_trader)

if trader_decision_maker is not None:
    manager.load_trader(trader_decision_maker, trader)

game.trade(run_id, trade_intra_day)
