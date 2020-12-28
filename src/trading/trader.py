import torch
import numpy as np
from src.preparation.preparator import DataPreparator


class Trader:

    def __init__(self, capital, capital_gains_tax, solidarity_surcharge, stock_exchange, device, days=5):
        self.start_capital = capital
        self.capital_gains_tax = capital_gains_tax
        self.solidarity_surcharge = solidarity_surcharge
        self.stock_exchange = stock_exchange
        self.device = device
        self.days = days

    def trade(self, agent, start_date='2016-01-01', end_date='2020-12-31', report_each_trade=True):
        tax_rate = self.capital_gains_tax / 100.0
        tax_rate *= self.solidarity_surcharge / 100.0 + 1.0
        for ticker, company in self.stock_exchange.tickers.items():
            capital = self.start_capital
            quotes = self.stock_exchange.load(ticker, start_date, end_date)
            if quotes is None:
                continue
            quotes['window'] = DataPreparator.calculate_windows(quotes, days=self.days, normalize=True)
            quotes['last_days'] = DataPreparator.calculate_last_days(quotes, days=self.days, normalize=True)
            count = 0
            price = 0.0
            buy_price = 0.0
            for index, row in quotes.iterrows():
                window = row['window']
                last_day_position = row['last_days'][-1]
                if np.sum(window) == 0:
                    continue
                price_window = np.array(window, dtype=np.float32).flatten()
                day_yield = quotes['close'][index] / quotes['close'][index - 1] - 1.0
                has_stocks = 1.0 if count > 0 else 0.0
                state = np.array([day_yield, has_stocks, last_day_position], dtype=np.float32)
                features = np.append(price_window, state)
                features = torch.tensor(features, dtype=torch.float32).reshape(1, len(features)).to(self.device)
                prediction = agent(features).cpu().detach().numpy()
                action = np.argmax(prediction)
                price = row['close']
                if action == 1 and count == 0 and capital > price + 1.0:
                    count = int(capital / price)
                    if count > 0:
                        buy_price = price
                        capital -= 1.0
                        capital -= count * price
                        message = f'{row["date"]} - {ticker:5} - buy  '
                        message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                        self.report(message, report_each_trade)
                elif action == 2 and count > 0:
                    capital -= 1.0
                    capital += count * price
                    earnings = (count * price) - (count * buy_price)
                    if earnings > 0.0:
                        tax = earnings * tax_rate
                        capital -= tax
                    message = f'{row["date"]} - {ticker:5} - sell '
                    message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                    self.report(message, report_each_trade)
                    count = 0
            if count > 0:
                capital -= 1.0
                capital += count * price
                earnings = (count * price) - (count * buy_price)
                if earnings > 0.0:
                    tax = earnings * tax_rate
                    capital -= tax
            message = f'{ticker:5} {company:40} '
            message += f'${self.start_capital:10.2f} => ${capital:10.2f} = ${capital - self.start_capital:10.2f}'
            message = f'{ticker};{company};'
            message += f'{self.start_capital:.2f};{capital:.2f};{capital - self.start_capital:.2f}'
            self.report(message, True)

    @staticmethod
    def report(message, verbose):
        if not verbose:
            return
        print(message)
