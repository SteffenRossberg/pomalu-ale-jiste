import argparse
import torch
from datetime import datetime
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.trading.game import Game

if __name__ != "__main__":
    exit(0)

torch.multiprocessing.set_start_method('spawn')

# Let's trade unseen data of 5 years from 01/01/2016 to 12/31/2020.
trader_start_date = '2016-01-01'
trader_end_date = '2020-12-31'
trade_intra_day = False
trader_start_capital = 50_000.0
trader_max_limit_positions = 10  # 35
trader_order_fee = 1.0
trader_capital_gains_tax = 25.0
trader_solidarity_surcharge = 5.5
trade_tickers = None

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
args = parser.parse_args()

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("Create data provider ...")
provider = DataProvider(args.apikey)

print("Create agent manager ...")
manager = NetManager(device, seed, deterministic, data_directory=f'data/networks/{run_id}')

print("Create trader ...")
trader = manager.create_trader(sample_days)
buyer_optimizer, buyer_result = manager.create_buyer_optimizer(trader)
seller_optimizer, seller_result = manager.create_seller_optimizer(trader)
classifier_optimizer, classifier_result = manager.create_classifier_optimizer(trader)

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

print("Prepare quotes ...")
all_quotes, all_tickers = DataPreparator.prepare_all_quotes(
    provider,
    sample_days,
    trader_start_date,
    trader_end_date,
    trade_tickers,
    trade_intra_day)

print("Create game ...")
game = Game(
    provider,
    trader,
    len(trade_tickers if trade_tickers is not None else provider.tickers.keys()),
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

game.trade(run_id, trade_intra_day)
