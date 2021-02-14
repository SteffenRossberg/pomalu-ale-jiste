import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.figure import Figure
from src.utility.logger import Logger
from src.environment.enums import Actions


class Game:

    def __init__(
            self,
            stock_exchange,
            decision_maker,
            max_positions,
            max_limit_positions,
            all_quotes,
            all_tickers,
            start_date,
            end_date,
            capital,
            order_fee,
            capital_gains_tax,
            solidarity_surcharge,
            device,
            days=5):
        self.stock_exchange = stock_exchange
        self.decision_maker = decision_maker
        self.max_positions = max_positions
        self.max_limit_positions = max_limit_positions
        self.all_quotes = all_quotes
        self.all_tickers = all_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.start_capital = capital
        self.capital_gains_tax = capital_gains_tax
        self.solidarity_surcharge = solidarity_surcharge
        self.device = device
        self.days = days
        self.tax_rate = self.capital_gains_tax / 100.0
        self.tax_rate *= self.solidarity_surcharge / 100.0 + 1.0
        self.order_fee = order_fee

    def trade(
            self,
            run_id,
            intraday=False,
            decision_maker=None):

        if decision_maker is None:
            decision_maker = self.decision_maker

        result = ''

        print(f"Trade limited all stocks from {self.start_date} to {self.end_date} ...")
        message, limit_all_investments, limit_all_gain_loss = \
            self._trade(
                decision_maker,
                self.all_quotes,
                self.all_tickers,
                True,
                self.stock_exchange.tickers,
                max_positions=self.max_limit_positions)
        result += f'\nTrade Portfolio (max {self.max_limit_positions} stocks): {message}'

        print(f"Trade all stocks from {self.start_date} to {self.end_date} ...")
        message, all_investments, all_gain_loss = \
            self._trade(
                decision_maker,
                self.all_quotes,
                self.all_tickers,
                True,
                self.stock_exchange.tickers,
                max_positions=self.max_positions)
        result += f'\nTrade All ({self.max_positions} stocks): {message}'

        print(f"Buy and hold all stocks from {self.start_date} to {self.end_date} ...")
        message = \
            self._buy_and_hold(
                self.all_quotes,
                self.all_tickers,
                False,
                self.stock_exchange.tickers)
        result += f'\nBuy % Hold All ({len(self.stock_exchange.tickers)} stocks): {message}'

        print(result)
        Logger.log(run_id, result)

        index_ticker = 'URTH'
        index_title = self.stock_exchange.etf_tickers[index_ticker]
        if intraday:
            compare_index = self.stock_exchange.load_intra_day(index_ticker, self.start_date, self.end_date, True)
        else:
            compare_index = self.stock_exchange.load(index_ticker, self.start_date, self.end_date, True)

        all_title = f'All stocks ({self.max_positions} positions)'
        limit_all_title = f'All stocks (max. {self.max_limit_positions} positions at once)'
        gain_loss_all_title = f'Return all stocks ({self.max_positions} positions)'
        gain_loss_limit_all_title = f'Return all stocks (max. {self.max_limit_positions} positions at once)'

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
                gain_loss_all_title: np.array(all_gain_loss[-length:]) + self.start_capital,
                gain_loss_limit_all_title: np.array(limit_all_gain_loss[-length:]) + self.start_capital
            })

        all_columns = [
            index_title,
            all_title,
            limit_all_title,
            gain_loss_all_title,
            gain_loss_limit_all_title
        ]
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
            title=f'Portfolio Changes vs {index_title}')

        text_color = '#cfcfcf'
        rc('font', weight='bold')
        rc('text', color=text_color)
        fig.patch.set_facecolor('#1d1d1d')
        for current_ax in ax:
            x_min = resulting_frame.index.min()
            x_max = resulting_frame.index.max()
            current_ax.set_xlim(x_min, x_max)
            current_ax.grid(which='major', axis='both')
            current_ax.spines['bottom'].set_color(text_color)
            current_ax.spines['top'].set_color(text_color)
            current_ax.spines['right'].set_color(text_color)
            current_ax.spines['left'].set_color(text_color)
            current_ax.tick_params(axis='both', which='major', colors=text_color, labelsize='large', grid_alpha=0.2)
            current_ax.set_facecolor('#1f1f1f')
            current_ax.xaxis.label.set_visible(False)
            current_ax.yaxis.label.set_visible(False)
            current_ax.title.set_color(text_color)
            current_ax.title.set_weight('bold')
            legend = current_ax.legend(
                facecolor='#333333',
                framealpha=0.4,
                ncol=1)
            for text in legend.get_texts():
                text.set_color(text_color)

        results = resulting_frame[gain_loss_columns].copy()
        results.to_csv(f'data/{run_id}.trading.gain_loss.csv')

        plt.savefig(f'data/{run_id}.chart.png')
        plt.show()
        plt.close()

    def _buy_and_hold(
            self,
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

    def _trade(
            self,
            agent,
            quotes,
            all_tickers,
            report_each_trade=True,
            tickers=None,
            max_positions=None):

        tickers = self.stock_exchange.tickers if tickers is None else tickers
        tickers = {ticker: self.stock_exchange.tickers[ticker] for ticker in tickers.keys() if ticker in all_tickers}
        max_positions = len(tickers) if max_positions is None else max_positions
        investment = self.start_capital
        portfolio = {}
        investments = []
        gain_loss = []
        total_gain_loss = 0.0

        for index, row in quotes.iterrows():
            actions = self.calculate_actions(agent, tickers, portfolio, quotes, row, index)
            if actions is not None:
                self.update_last_prices(row, portfolio)
                # self.prepare_buy_and_hold(actions)
                self.prepare_profit_taking(row, portfolio, actions, 20.0)
                investment, earnings = self.sell(row, investment, portfolio, actions, report_each_trade)
                investment = self.buy(row, investment, portfolio, actions, max_positions)
                total_gain_loss += earnings

            new_investment = self.calculate_current_investment(investment, portfolio, row)
            investments.append(new_investment)
            gain_loss.append(total_gain_loss)

        investment = self.start_capital + total_gain_loss
        investments.append(investment)
        gain_loss.append(total_gain_loss)

        message = f'Total '
        message += f'${self.start_capital:10.2f} => ${investment:10.2f} = ${total_gain_loss:10.2f}'
        self.report(message, True)

        return message, investments, gain_loss

    def buy(self, row, investment, portfolio, actions, max_positions):

        def action_filter(t):
            return t not in portfolio and actions[t]['index'] == Actions.Buy

        def action_sort(t):
            return actions[t]['value']

        possible_position_count = max_positions - len(portfolio)
        if possible_position_count <= 0:
            return investment
        tickers = [ticker for ticker in sorted(filter(action_filter, actions.keys()), key=action_sort, reverse=True)]
        tickers = tickers[:possible_position_count]
        possible_investment = investment / possible_position_count
        for ticker in tickers:
            price = row[f'{ticker}_close']
            if possible_investment < price + self.order_fee:
                continue
            investment -= self.order_fee
            count = int((possible_investment - self.order_fee) / price)
            investment -= count * price
            portfolio[ticker] = {
                'buy_date': row['date'],
                'buy_price': price,
                'count': count,
                'last_price': price
            }
        return investment

    def sell(self, row, investment, portfolio, actions, report_each_trade, clear_positions=False):
        total_earnings = 0.0
        for ticker, action in actions.items():
            if ticker not in portfolio or not portfolio[ticker]['count'] > 0:
                continue
            price = row[f'{ticker}_close']
            if not price > 0.0:
                price = portfolio[ticker]['last_price']
                action['index'] = Actions.Sell
            if action['index'] == Actions.Sell:
                count = portfolio[ticker]['count']
                buy_price = portfolio[ticker]['buy_price']
                investment -= self.order_fee
                investment += count * price
                earnings = (count * price) - (count * buy_price)
                if earnings > 0.0:
                    tax = earnings * self.tax_rate
                    investment -= tax
                    earnings -= tax
                total_earnings += earnings
                message = f'{portfolio[ticker]["buy_date"]} - {row["date"]} - {ticker:5} - '
                message += f'${buy_price:.2f} -> ${price:.2f}{" ... clear positions" if clear_positions else ""}'
                self.report(message, report_each_trade)
                del portfolio[ticker]
        return investment, total_earnings

    @classmethod
    def prepare_buy_and_hold(cls, actions):
        for ticker in actions.keys():
            if actions[ticker]['index'] == Actions.Sell:
                actions[ticker] = {'index': Actions.SkipOrHold}

    def prepare_profit_taking(self, row, portfolio, actions, threshold=5.0):
        for ticker in portfolio.keys():
            position = portfolio[ticker]
            if self.calculate_position_gain_loss(ticker, position, row) >= threshold:
                actions[ticker] = {'index': Actions.Sell}

    @classmethod
    def update_last_prices(cls, row, portfolio):
        for ticker in portfolio.keys():
            if row[f'{ticker}_close'] > 0.0:
                portfolio[ticker]['last_price'] = row[f'{ticker}_close']

    def calculate_position_gain_loss(
            self,
            ticker,
            position,
            row):
        price = row[f'{ticker}_close']
        if not price > 0.0:
            price = position['last_price']
        count = position['count']
        buy_price = position['buy_price']
        buy_in = count * buy_price
        sell_out = count * price
        earnings = sell_out - buy_in
        if earnings > 0.0:
            earnings -= earnings * self.tax_rate
        gain_loss = earnings
        gain_loss -= self.order_fee
        returns = (((gain_loss + buy_in) / buy_in) - 1.0) * 100.0
        return returns

    def calculate_actions(
            self,
            agent,
            tickers,
            portfolio,
            quotes,
            row,
            index):
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

    def calculate_features(
            self,
            tickers,
            portfolio,
            quotes,
            row,
            index):
        evaluated_tickers = []
        feature_batch = []
        for ticker in tickers.keys():
            window = row[f'{ticker}_window']
            if (window is np.nan or
                    np.isnan(window).any() or
                    row[f'{ticker}_last_days'] is np.nan or
                    np.isnan(row[f'{ticker}_last_days']).any() or
                    window is None or
                    row[f'{ticker}_last_days'] is None or
                    np.sum(window) == 0.0):
                continue
            day_yield = quotes[f'{ticker}_close'][index] / quotes[f'{ticker}_close'][index - 1] - 1.0
            has_stocks = 1.0 if ticker in portfolio and portfolio[ticker]['count'] > 0 else 0.0
            price_window = np.array(window, dtype=np.float32).flatten()
            features = np.append(price_window, np.array([day_yield, has_stocks], dtype=np.float32))
            feature_batch.append(features)
            evaluated_tickers.append(ticker)
        if len(evaluated_tickers) == 0:
            return None, None
        feature_batch = torch.tensor(feature_batch, dtype=torch.float32)
        feature_batch = feature_batch.reshape(feature_batch.shape[0], feature_batch.shape[-1])
        feature_batch = feature_batch.to(self.device)
        return feature_batch, evaluated_tickers

    def calculate_current_investment(
            self,
            investment,
            portfolio,
            row):
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
    def report(
            message,
            verbose):
        if not verbose:
            return
        print(message)
