import torch
import numpy as np


class Trader:

    def __init__(self, capital, order_fee, capital_gains_tax, solidarity_surcharge, stock_exchange, device, days=5):
        self.start_capital = capital
        self.capital_gains_tax = capital_gains_tax
        self.solidarity_surcharge = solidarity_surcharge
        self.stock_exchange = stock_exchange
        self.device = device
        self.days = days
        self.tax_rate = self.capital_gains_tax / 100.0
        self.tax_rate *= self.solidarity_surcharge / 100.0 + 1.0
        self.order_fee = order_fee

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
            count = int((investment - self.order_fee) / price)
            if count > 0:
                buy_price = price
                investment -= self.order_fee
                investment -= count * price
                message = f'Buy & Hold - {row["date"]} - {ticker:5} - buy  '
                message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                self.report(message, report_each_trade)

                row = ticker_quotes.iloc[-1]
                price = row[f'{ticker}_close']
                investment -= self.order_fee
                investment += count * price
                earnings = (count * price) - (count * buy_price)
                if earnings > 0.0:
                    tax = earnings * self.tax_rate
                    investment -= tax
                message = f'Buy & Hold - {row["date"]} - {ticker:5} - sell '
                message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                self.report(message, report_each_trade)

                capital += investment
        message = f'Buy & Hold - Total '
        message += f'${self.start_capital:10.2f} => ${capital:10.2f} = ${capital - self.start_capital:10.2f}'
        self.report(message, True)
        return message

    def trade(self,
              agent,
              quotes,
              all_tickers,
              report_each_trade=True,
              tickers=None,
              max_positions=None):
        if tickers is None:
            tickers = self.stock_exchange.tickers
        tickers = {ticker: self.stock_exchange.tickers[ticker] for ticker in tickers.keys() if ticker in all_tickers}
        if max_positions is None:
            max_positions = len(tickers)
        investment = self.start_capital
        portfolio = {}
        investments = []
        gain_loss = []
        total_gain_loss = 0.0

        for index, row in quotes.iterrows():
            actions = self.calculate_actions(agent, tickers, portfolio, quotes, row, index)
            if actions is not None:
                for ticker in tickers.keys():
                    if row[f'{ticker}_close'] > 0.0 and ticker in portfolio:
                        portfolio[ticker]['last_price'] = row[f'{ticker}_close']
                for ticker, action in actions.items():
                    if ticker not in portfolio or not portfolio[ticker]['count'] > 0:
                        continue
                    price = row[f'{ticker}_close']
                    if not price > 0.0:
                        price = portfolio[ticker]['last_price']
                        action['index'] = 2
                    if action['index'] == 2:
                        count = portfolio[ticker]['count']
                        buy_price = portfolio[ticker]['buy_price']
                        investment -= self.order_fee
                        investment += count * price
                        earnings = (count * price) - (count * buy_price)
                        if earnings > 0.0:
                            tax = earnings * self.tax_rate
                            investment -= tax
                            earnings -= tax
                        total_gain_loss += earnings
                        message = f'{row["date"]} - {ticker:5} - sell '
                        message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                        self.report(message, report_each_trade)
                        del portfolio[ticker]
                possible_position_count = max_positions - len(portfolio)
                if possible_position_count > 0:

                    def action_filter(t):
                        return t not in portfolio and actions[t]['index'] == 1

                    def action_sort(t):
                        return actions[t]['value']

                    for ticker in sorted(filter(action_filter, actions.keys()), key=action_sort, reverse=True):
                        if len(portfolio) >= max_positions:
                            break
                        if ticker in portfolio:
                            continue
                        action = actions[ticker]
                        if action['index'] != 1:
                            continue
                        price = row[f'{ticker}_close']
                        possible_position_investment = investment / possible_position_count
                        if possible_position_investment > price + self.order_fee:
                            count = int(possible_position_investment / price)
                            if count > 0:
                                portfolio[ticker] = {'buy_price': price, 'count': count}
                                investment -= self.order_fee
                                investment -= count * price
                                message = f'{row["date"]} - {ticker:5} - buy  '
                                message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f}'
                                self.report(message, report_each_trade)
            new_investment = self.calculate_current_investment(investment, portfolio, row)
            investments.append(new_investment)
            gain_loss.append(total_gain_loss)

        row = quotes.iloc[len(quotes) - 1]
        for ticker, position in portfolio.items():
            price = row[f'{ticker}_close']
            if not price > 0.0:
                price = position['last_price']
            count = portfolio[ticker]['count']
            buy_price = portfolio[ticker]['buy_price']
            investment -= self.order_fee
            investment += count * price
            earnings = (count * price) - (count * buy_price)
            if earnings > 0.0:
                tax = earnings * self.tax_rate
                investment -= tax
                earnings -= tax
            message = f'{row["date"]} - {ticker:5} - sell '
            message += f'{count:5} x ${price:7.2f} = ${count * price:10.2f} -> clear positions'
            self.report(message, report_each_trade)

        investments.append(investment)
        gain_loss.append(total_gain_loss)
        message = f'Total '
        message += f'${self.start_capital:10.2f} => ${investment:10.2f} = ${investment - self.start_capital:10.2f}'
        self.report(message, True)

        return message, investments, gain_loss

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
                'value': action_values[i],
                'predictions': prediction[i]
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

    def calculate_current_investment(self, investment, portfolio, row):
        for ticker, position in portfolio.items():
            price = row[f'{ticker}_close']
            if not price > 0.0:
                price = portfolio[ticker]['last_price']
            investment += position['count'] * price
            investment -= self.order_fee
            earnings = (position['count'] * price) - (position['count'] * position['buy_price'])
            if earnings > 0.0:
                tax = earnings * self.tax_rate
                investment -= tax
                earnings -= tax
        return investment

    @staticmethod
    def report(message, verbose):
        if not verbose:
            return
        print(message)
