import torch
import numpy as np


class Trader:

    def __init__(self, capital, capital_gains_tax, solidarity_surcharge, stock_exchange, device, days=5):
        self.start_capital = capital
        self.capital_gains_tax = capital_gains_tax
        self.solidarity_surcharge = solidarity_surcharge
        self.stock_exchange = stock_exchange
        self.device = device
        self.days = days
        self.tax_rate = self.capital_gains_tax / 100.0
        self.tax_rate *= self.solidarity_surcharge / 100.0 + 1.0

    def trade(self,
              agent,
              quotes,
              all_tickers,
              report_each_trade=True,
              tickers=None):
        if tickers is None:
            tickers = self.stock_exchange.tickers
        tickers = {ticker: self.stock_exchange.tickers[ticker] for ticker in tickers.keys() if ticker in all_tickers}
        capital = 0.0
        start_investment = self.start_capital / len(tickers)
        result_csv_content = 'ticker;company;start_capital;end_capital;earnings'
        for ticker, company in tickers.items():
            investment = start_investment
            count = 0
            price = 0.0
            buy_price = 0.0
            for index, row in quotes.iterrows():
                window = row[f'{ticker}_window']
                if (window is np.nan or
                        row[f'{ticker}_last_days'] is np.nan or
                        window is None or
                        row[f'{ticker}_last_days'] is None or
                        np.sum(window) == 0):
                    continue
                last_day_position = row[f'{ticker}_last_days'][-1]
                price_window = np.array(window, dtype=np.float32).flatten()
                day_yield = quotes[f'{ticker}_close'][index] / quotes[f'{ticker}_close'][index - 1] - 1.0
                has_stocks = 1.0 if count > 0 else 0.0
                state = np.array([day_yield, has_stocks, last_day_position], dtype=np.float32)
                features = np.append(price_window, state)
                features = torch.tensor(features, dtype=torch.float32).reshape(1, len(features)).to(self.device)
                prediction = agent(features).cpu().detach().numpy()
                action = np.argmax(prediction)
                price = row[f'{ticker}_close']
                if action == 1 and count == 0 and investment > price + 1.0:
                    count = int(investment / price)
                    if count > 0:
                        buy_price = price
                        investment -= 1.0
                        investment -= count * price
                        message = f'{row["date"]} - {ticker:5} - buy  '
                        message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                        self.report(message, report_each_trade)
                elif action == 2 and count > 0:
                    investment -= 1.0
                    investment += count * price
                    earnings = (count * price) - (count * buy_price)
                    if earnings > 0.0:
                        tax = earnings * self.tax_rate
                        investment -= tax
                    message = f'{row["date"]} - {ticker:5} - sell '
                    message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                    self.report(message, report_each_trade)
                    count = 0
            if count > 0:
                investment -= 1.0
                investment += count * price
                earnings = (count * price) - (count * buy_price)
                if earnings > 0.0:
                    tax = earnings * self.tax_rate
                    investment -= tax
            capital += investment
            message = f'{ticker:5} {company:40} '
            message += f'${start_investment:10.2f} => ${investment:10.2f} = ${investment - start_investment:10.2f}'
            self.report(message, True)
            message = f'{ticker};{company};'
            message += f'{start_investment:.2f};{investment:.2f};{investment - start_investment:.2f}'
            result_csv_content += f'\n{message}'
        message = f'Total '
        message += f'${self.start_capital:10.2f} => ${capital:10.2f} = ${capital - self.start_capital:10.2f}'
        self.report(message, True)
        return message, None

    def buy_and_hold(self,
                     quotes,
                     all_tickers,
                     report_each_trade=True,
                     tickers=None):
        if tickers is None:
            tickers = self.stock_exchange.tickers
        tickers = {ticker: self.stock_exchange.tickers[ticker] for ticker in tickers.keys() if ticker in all_tickers}
        start_investment = self.start_capital / len(tickers)
        capital = 0.0
        for ticker, company in tickers.items():
            investment = start_investment
            ticker_quotes = quotes[quotes[f'{ticker}_close'] > 0.0]
            row = ticker_quotes.iloc[0]
            price = row[f'{ticker}_close']
            count = int(investment / price)
            if count > 0:
                buy_price = price
                investment -= 1.0
                investment -= count * price
                message = f'Buy & Hold - {row["date"]} - {ticker:5} - buy  '
                message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                self.report(message, report_each_trade)

                row = ticker_quotes.iloc[-1]
                price = row[f'{ticker}_close']
                investment -= 1.0
                investment += count * price
                earnings = (count * price) - (count * buy_price)
                if earnings > 0.0:
                    tax = earnings * self.tax_rate
                    investment -= tax
                message = f'Buy & Hold - {row["date"]} - {ticker:5} - sell '
                message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                self.report(message, report_each_trade)

                capital += investment
                message = f'Buy & Hold - {ticker:5} {company:40} '
                message += f'${start_investment:10.2f} => ${investment:10.2f} = ${investment - start_investment:10.2f}'
                self.report(message, True)
        message = f'Buy & Hold - Total '
        message += f'${self.start_capital:10.2f} => ${capital:10.2f} = ${capital - self.start_capital:10.2f}'
        self.report(message, True)
        return message, None

    def trade_concurrent(self,
                         agent,
                         quotes,
                         all_tickers,
                         report_each_trade=True,
                         tickers=None,
                         max_positions=5):
        if tickers is None:
            tickers = self.stock_exchange.tickers
        tickers = {ticker: self.stock_exchange.tickers[ticker] for ticker in tickers.keys() if ticker in all_tickers}
        investment = self.start_capital
        portfolio = {}
        total_earnings = []
        total_earning = 0.0
        for index, row in quotes.iterrows():
            actions = self.calculate_actions(agent, tickers, portfolio, quotes, row, index)
            if actions is None:
                total_earnings.append(total_earning)
                continue
            for ticker, action in actions.items():
                if ticker not in portfolio or not portfolio[ticker]['count'] > 0:
                    continue
                price = row[f'{ticker}_close']
                if action['index'] == 2:
                    count = portfolio[ticker]['count']
                    buy_price = portfolio[ticker]['buy_price']
                    investment -= 1.0
                    investment += count * price
                    earnings = (count * price) - (count * buy_price)
                    if earnings > 0.0:
                        tax = earnings * self.tax_rate
                        investment -= tax
                        earnings -= tax
                    total_earning += earnings
                    message = f'{row["date"]} - {ticker:5} - sell '
                    message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                    self.report(message, report_each_trade)
                    del portfolio[ticker]
            total_earnings.append(total_earning)
            possible_position_count = max_positions - len(portfolio)
            if possible_position_count <= 0:
                continue
            for ticker in sorted(actions.keys(), key=lambda t: actions[t]['value'], reverse=True):
                if len(portfolio) >= max_positions:
                    break
                if ticker in portfolio:
                    continue
                action = actions[ticker]
                if action['index'] != 1:
                    continue
                price = row[f'{ticker}_close']
                possible_position_investment = investment / possible_position_count
                if possible_position_investment > price + 1.0:
                    count = int(possible_position_investment / price)
                    if count > 0:
                        portfolio[ticker] = {'buy_price': price, 'count': count}
                        investment -= 1.0
                        investment -= count * price
                        message = f'{row["date"]} - {ticker:5} - buy  '
                        message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                        self.report(message, report_each_trade)

        row = quotes.iloc[len(quotes) - 1]
        for ticker, position in portfolio.items():
            price = row[f'{ticker}_close']
            count = portfolio[ticker]['count']
            buy_price = portfolio[ticker]['buy_price']
            investment -= 1.0
            investment += count * price
            earnings = (count * price) - (count * buy_price)
            if earnings > 0.0:
                tax = earnings * self.tax_rate
                investment -= tax
                earnings -= tax
            total_earning += earnings
            message = f'{row["date"]} - {ticker:5} - sell '
            message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f} -> clear positions'
            self.report(message, report_each_trade)
        total_earnings.append(total_earning)

        message = f'Concurrent Total '
        message += f'${self.start_capital:10.2f} => ${investment:10.2f} = ${investment - self.start_capital:10.2f}'
        self.report(message, True)
        return message, total_earnings

    def calculate_actions(self, agent, tickers, portfolio, quotes, row, index):
        features, eval_tickers = self.calculate_features(tickers, portfolio, quotes, row, index)
        if eval_tickers is None:
            return None
        prediction = agent(features).cpu().detach().numpy()
        action_indexes = np.argmax(prediction, axis=1)
        action_values = np.amax(prediction, axis=1)
        actions = {
            eval_tickers[i]: {
                'index': action_indexes[i],
                'value': action_values[i]
            }
            for i in range(len(eval_tickers))
        }
        return actions

    def calculate_features(self, tickers, portfolio, quotes, row, index):
        evaluated_tickers = []
        feature_batch = []
        for ticker in tickers.keys():
            window = row[f'{ticker}_window']
            if (window is np.nan or
                    row[f'{ticker}_last_days'] is np.nan or
                    window is None or
                    row[f'{ticker}_last_days'] is None or
                    np.sum(window) == 0.0):
                continue
            last_day_position = row[f'{ticker}_last_days'][-1]
            price_window = np.array(window, dtype=np.float32).flatten()
            day_yield = quotes[f'{ticker}_close'][index] / quotes[f'{ticker}_close'][index - 1] - 1.0
            has_stocks = 1.0 if ticker in portfolio and portfolio[ticker]['count'] > 0 else 0.0
            state = np.array([day_yield, has_stocks, last_day_position], dtype=np.float32)
            features = np.append(price_window, state)
            feature_batch.append(features)
            evaluated_tickers.append(ticker)
        if len(evaluated_tickers) == 0:
            return None, None
        feature_batch = torch.tensor(feature_batch, dtype=torch.float32)
        feature_batch = feature_batch.reshape(feature_batch.shape[0], feature_batch.shape[-1])
        feature_batch = feature_batch.to(self.device)
        return feature_batch, evaluated_tickers

    @staticmethod
    def report(message, verbose):
        if not verbose:
            return
        print(message)
