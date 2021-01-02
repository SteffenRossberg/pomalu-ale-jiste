import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.environment.dataprovider import DataProvider
from src.preparation.preparator import DataPreparator
from src.networks.manager import NetManager
from src.training.gym import Gym
from src.trading.trader import Trader
from src.environment.stock_exchange import StockExchange
from src.environment.enums import TrainingLevels
from datetime import datetime

if __name__ != "__main__":
    exit(0)

# Let's train data of 16 years from 01/01/2000 to 12/31/2015
train_start_date = '2000-01-01'
train_end_date = '2015-12-31'
# Let's trade unseen data of 5 years from 01/01/2016 to 12/31/2020.
trader_start_date = '2016-01-01'
trader_end_date = '2020-12-31'
trader_start_capital = 50_000.0
trader_order_fee = 1.0
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
                trader_order_fee,
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
    while True:
        print("Train buyer samples detector ...")
        gym.train_auto_encoder('buyer', buyer, buyer_optimizer, buy_samples, buyer_loss)
        print("Reload buyer samples detector with best training result after training ...")
        buyer_loss = manager.load_net('buyer.autoencoder', buyer, buyer_optimizer)
        if buyer_loss < 0.003:
            break
        # train result is not good enough, simply re-create a new buyer and try it again...
        buyer, buyer_optimizer = manager.create_auto_encoder(sample_days)
        buyer_loss = 100.0

    while True:
        print("Train seller samples detector ...")
        gym.train_auto_encoder('seller', seller, seller_optimizer, sell_samples, seller_loss)
        print("Reload seller samples detector with best training result after training ...")
        seller_loss = manager.load_net('seller.autoencoder', seller, seller_optimizer)
        if seller_loss < 0.003:
            break
        # train result is not good enough, simply re-create a new seller and try it again...
        seller, seller_optimizer = manager.create_auto_encoder(sample_days)
        seller_loss = 100.0

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

    while True:
        print("Train trader classifier ...")
        gym.train_classifier('trader', classifier, classifier_optimizer,
                             classifier_features, classifier_labels,
                             classifier_none_features, classifier_none_labels,
                             classifier_loss, max_steps=1000)
        print("Reload classifier with best training result after training ...")
        classifier_loss = manager.load_net('trader.classifier', classifier, classifier_optimizer)
        if classifier_loss < 0.4:
            break
        # train result is not good enough, simply re-create a new classifier and try it again...
        classifier, classifier_optimizer = manager.create_classifier(buyer, seller)
        classifier_loss = 100.0

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

    stock_exchange.train_level = 0
    for train_level in [
        TrainingLevels.Skip | TrainingLevels.Buy | TrainingLevels.Hold | TrainingLevels.Sell
    ]:
        print(f"Train decision maker at level {train_level} ...")
        stock_exchange.train_level = train_level
        gym.train_decision_maker('trader', decision_maker, decision_optimizer, best_mean_val, stock_exchange)
        print("Reload decision maker with best training result after training ...")
        best_mean_val = manager.load_net('trader.decision_maker', decision_maker, decision_optimizer)

all_quotes, all_tickers = DataPreparator.prepare_all_quotes(provider,
                                                            sample_days,
                                                            trader_start_date,
                                                            trader_end_date,
                                                            provider.tickers)
result = ''
max_all_positions = len(provider.tickers)
max_limit_all_positions = int(len(provider.tickers) / 3)
max_dow_positions = len(provider.dow30_tickers)
max_limit_dow_positions = int(len(provider.dow30_tickers) / 3)

print(f"Trade limited all stocks from {trader_start_date} to {trader_end_date} ...")
message, limit_all_investments, limit_all_gain_loss = \
    trader.trade(
        decision_maker,
        all_quotes,
        all_tickers,
        False,
        provider.tickers,
        max_positions=max_limit_all_positions)
result += f'\nTrade Portfolio (max {int(len(provider.tickers) / 3)} stocks): {message}'

print(f"Trade all stocks from {trader_start_date} to {trader_end_date} ...")
message, all_investments, all_gain_loss = \
    trader.trade(
        decision_maker,
        all_quotes,
        all_tickers,
        True,
        provider.tickers,
        max_positions=max_all_positions)
result += f'\nTrade All ({len(provider.tickers)} stocks): {message}'

print(f"Buy and hold all stocks from {trader_start_date} to {trader_end_date} ...")
message = \
    trader.buy_and_hold(
        all_quotes,
        all_tickers,
        False,
        provider.tickers)
result += f'\nBuy % Hold All ({len(provider.tickers)} stocks): {message}'

print(result)

index_ticker = 'URTH'
index_title = provider.etf_tickers[index_ticker]
compare_index = provider.load(index_ticker, trader_start_date, trader_end_date, True)

all_title = f'All stocks ({max_all_positions} positions)'
limit_all_title = f'All stocks (max. {max_limit_all_positions} positions at once)'
gain_loss_all_title = f'Return all stocks ({max_all_positions} positions)'
gain_loss_limit_all_title = f'Return all stocks (max. {max_limit_all_positions} positions at once)'

length = (len(compare_index)
          if len(compare_index) < len(all_investments)
          else len(all_investments))

resulting_frame = pd.DataFrame(
    data={
        'index': range(length),
        'date': np.array(compare_index['date'].values[-length:]),
        index_title: np.array(compare_index['close'].values[-length:]),
        all_title: np.array(all_investments[-length:]),
        limit_all_title: np.array(limit_all_investments[-length:]),
        gain_loss_all_title: np.array(all_gain_loss[-length:]) + trader_start_capital,
        gain_loss_limit_all_title: np.array(limit_all_gain_loss[-length:]) + trader_start_capital
    })

all_columns = [
    index_title,
    all_title,
    limit_all_title,
    gain_loss_all_title,
    gain_loss_limit_all_title
]
changes_columns = [f'Change {column}' for column in all_columns]
for column in all_columns:
    change_column = f'Change {column}'
    resulting_frame[change_column] = resulting_frame[column].pct_change(1).fillna(0.0) * 100.0
    resulting_frame[column] = \
        resulting_frame.apply(
            lambda row: np.sum(resulting_frame[change_column].values[0:int(row['index'] + 1)]),
            axis=1)

resulting_frame.set_index(resulting_frame['date'], inplace=True)

fig, ax = plt.subplots(nrows=2)

investment_columns = [
    all_title,
    limit_all_title
]
resulting_frame[index_title].plot.area(ax=ax[0], stacked=False)
resulting_frame[investment_columns].plot(
    ax=ax[0],
    figsize=(20, 10),
    linewidth=2,
    colormap='Spectral',
    title=f'Investment vs {index_title}')

gain_loss_columns = [
    gain_loss_all_title,
    gain_loss_limit_all_title
]
resulting_frame[index_title].plot.area(ax=ax[1], stacked=False)
resulting_frame[gain_loss_columns].plot(
    ax=ax[1],
    figsize=(20, 10),
    linewidth=2,
    colormap='Spectral',
    title=f'Portfolio Changes vs {index_title}')

plt.show()
plt.close()

today = datetime.now()
resulting_frame[gain_loss_columns].copy().to_csv(f'data/trading.gain_loss.{today:%Y%m%d.%H%M%S}.csv')
