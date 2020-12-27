import numpy as np
from src.environment.actions import Actions


class PortfolioState:

    def __init__(self,
                 start_investment,
                 trading_fees,
                 tax_rate,
                 reset_on_close):
        self.trading_fees = trading_fees
        self.tax_rate = tax_rate
        self.reset_on_close = reset_on_close
        self._frame = None
        self._offset = None
        self._start_investment = start_investment
        self._investment = self._start_investment
        self._stock_count = 0
        self._buy_price = 0.0

    @property
    def investment(self):
        return self._investment

    @property
    def stock_count(self):
        return self._stock_count

    @property
    def buy_price(self):
        return self._buy_price

    @property
    def offset(self):
        return self._offset

    @property
    def start_investment(self):
        return self._start_investment

    @property
    def current_price(self):
        return self._frame['prices'][self._offset]

    @property
    def current_date(self):
        return self._frame['dates'][self._offset]

    @property
    def shape(self):
        return len(self._frame['windows'][0]) * 4 + 1 + 1,

    def encode(self):
        investment = self._stock_count * self._frame['prices'][self._offset] + self._investment
        encoded = np.array(self._frame['windows'][self._offset], dtype=np.float32).flatten()
        encoded = np.append(encoded, [
            investment / self._start_investment - 1.0,
            self._frame['last_days'][self._offset][-1]])
        return encoded

    def reset(self, frame, offset):
        self._frame = frame
        self._offset = offset
        self._stock_count = 0
        self._buy_price = 0.0
        self._investment = self._start_investment

    def step(self, action):
        reward = 0.0
        done = False
        price = self._frame['prices'][self._offset]
        if action == Actions.Buy and self._stock_count == 0:
            count = int((self._investment - self.trading_fees) / price)
            if count > 0:
                self._investment -= self.trading_fees
                self._investment -= count * price
                self._stock_count = count
                self._buy_price = price
                reward -= 100.0 * (self.trading_fees / (count * price))
        elif action == Actions.Sell and self._stock_count > 0:
            done |= self.reset_on_close
            earnings = (price * self._stock_count) - (self._buy_price * self._stock_count)
            if earnings > 0.0:
                earnings *= self.tax_rate
            self._investment += earnings
            self._stock_count = 0
            self._buy_price = 0.0
            reward -= 100.0 * (self.trading_fees / (self._stock_count * price))
            reward += 100.0 * ((price - self._buy_price) / self._buy_price)

        self._offset += 1
        done |= self._offset >= len(self._frame['windows'])
        return reward, done
