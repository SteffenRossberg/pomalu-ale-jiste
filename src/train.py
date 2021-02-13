import argparse
import torch
import os
from datetime import datetime
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.training.gym import Gym
from src.environment.stock_exchange import StockExchange
from src.environment.enums import TrainingLevels
from src.utility.logger import Logger
import prometheus_client
import numpy as np


prometheus_client.start_http_server(5000, '0.0.0.0')

# Let's train data of 16 years from 01/01/2000 to 12/31/2015
train_start_date = '2000-01-01'
train_end_date = '2012-12-31'
validation_start_date = '2013-01-01'
validation_end_date = '2015-12-31'
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
parser.add_argument("--train_decision_maker",
                    required=False,
                    type=int,
                    default=0,
                    help="Train decision maker")
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
decision_maker_optimizer, decision_maker_result = manager.create_decision_maker_optimizer(trader)

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

decision_maker_optimizer, decision_maker_result = \
    manager.load(
        'decision_maker',
        trader.decision_maker,
        decision_maker_optimizer,
        trader.reset_decision_maker,
        lambda: manager.create_decision_maker_optimizer(trader),
        decision_maker_result)

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

print("Prepare frames ...")
frames = DataPreparator.prepare_frames(provider, sample_days, train_start_date, train_end_date)


os.makedirs(f'data/networks/{run_id}', exist_ok=True)

Logger.log(run_id, f"Id: {run_id}")

if args.train_buyer > 0:
    buyer_features = np.array(buy_samples, dtype=np.float32)
    manager.init_seed(seed, deterministic)
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
    manager.init_seed(seed, deterministic)
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
    classifier_labels = np.array(classifier_labels, dtype=np.int)
    manager.init_seed(seed, deterministic)
    print("Train classifier ...")
    classifier_result = gym.train_classifier(
        'classifier',
        trader,
        classifier_optimizer,
        classifier_features,
        classifier_labels,
        classifier_result,
        max_epochs=500,
        max_steps=10)
    print("Reload trader with best training result after training ...")
    classifier_optimizer, seller_result = manager.load(
        'classifier',
        trader.classifier,
        classifier_optimizer,
        trader.reset_classifier,
        lambda: manager.create_classifier_optimizer(trader),
        classifier_result)
    Logger.log(run_id, f"classifier: {classifier_result:.7f}")

if args.train_decision_maker > 0:
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
        tickers=[ticker for ticker in provider.tickers.keys()][:20],
        random_offset_on_reset=False,
        reset_on_close=False,
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
    Logger.log(run_id, f"decision_maker: {decision_maker_result:.7f}")
    Logger.log(run_id, f"Train Level: {str(training_level)}")
    Logger.log(run_id, f"Trader Best mean value: {decision_maker_result:.7f}")

# classifier_features = []
# classifier_labels = []
# for i in range(len(buy_samples)):
#     classifier_features.append(buy_samples[i])
#     classifier_features.append(sell_samples[i])
#     classifier_features.append(none_samples[i])
#     classifier_labels.append(1)
#     classifier_labels.append(2)
#     classifier_labels.append(0)
# classifier_features = np.array(classifier_features, dtype=np.float32)
# classifier_features = classifier_features.reshape(classifier_features.shape[0], sample_days * 4)
# classifier_labels = np.array(classifier_labels, dtype=np.int)
#
#
# autoPyTorch = AutoNetClassification("full_cs")
# autoPyTorch.print_help()
# autoPyTorch.fit(
#     classifier_features,
#     classifier_labels,
#     log_level='debug',
#     max_runtime=10_000_000,
#     min_budget=10,
#     max_budget=20,
#     budget_type='epochs',
#     use_pynisher=False,
#     over_sampling_methods=['none'])

# for row, index in all_quotes.iterrows():
#     window = row['MSFT_window'].values
#     y_pred = autoPyTorch.predict(window)
