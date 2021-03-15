import argparse
import torch
from datetime import datetime
from app.environment.dataprovider import DataProvider
from app.preparation.preparator import DataPreparator
from app.networks.manager import NetManager
from app.trading.game import Game

if __name__ != "__main__":
    exit(0)

# Let's trade unseen data of 5 years from 01/01/2016 to 12/31/2020.
start_date = '2016-01-01'
end_date = '2020-12-31'
start_capital = 50_000.0
max_limit_positions = 5
buy_and_hold = False
profit_taking_threshold = 0
order_fee = 1.0
spread_in_percent = 0.8
capital_gains_tax_in_percent = 25.0
solidarity_surcharge_in_percent = 5.5
tickers = ['AAPL', 'ADBE', 'AMD', 'AMZN', 'ATVI', 'BIDU', 'CRM', 'EA', 'EBAY', 'FB', 'IBM', 'MSFT', 'NFLX',
           'NVDA', 'ORCL', 'PYPL', 'QCOM', 'SNE', 'TWTR', 'VRSN']
use_intra_day = False

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
    start_date,
    end_date,
    tickers,
    use_intra_day)

print("Create game ...")
game = Game(
    provider,
    trader,
    len(tickers if tickers is not None else provider.tickers.keys()),
    max_limit_positions,
    all_quotes,
    tickers,
    start_date,
    end_date,
    start_capital,
    spread_in_percent,
    order_fee,
    capital_gains_tax_in_percent,
    solidarity_surcharge_in_percent,
    device,
    sample_days)

game.trade(
    run_id,
    profit_taking_threshold=profit_taking_threshold,
    buy_and_hold=buy_and_hold,
    intra_day=use_intra_day)
