import numpy as np
from src.environment.actions import Actions


class PortfolioState:

    def __init__(self,
                 days,
                 start_investment,
                 trading_fees,
                 tax_rate,
                 reset_on_close):
        self._days = days
        self._start_investment = start_investment
        self.trading_fees = trading_fees
        self.tax_rate = tax_rate
        self.reset_on_close = reset_on_close
        self._frame = None
        self._offset = None
        self._investment = self._start_investment
        self._stock_count = 0
        self._buy_price = 0.0
        self._sell_price = 0.0
        self._top_price = 0.0
        self._bottom_price = 0.0
        self.__train_level = 1

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
        return self._days * 4 + 1 + 1 + 1,

    @property
    def train_level(self) -> int:
        return self.__train_level

    @train_level.setter
    def train_level(self, value: int):
        self.__train_level = value

    def encode(self):
        day_yield = self._frame['prices'][self._offset] / self._frame['prices'][self._offset - 1] - 1.0
        has_stocks = 1.0 if self._stock_count > 0 else 0.0
        last_day_position = self._frame['last_days'][self._offset][-1]
        state = np.array([day_yield, has_stocks, last_day_position], dtype=np.float32)
        price_window = np.array(self._frame['windows'][self._offset], dtype=np.float32).flatten()
        encoded = np.append(price_window, state)
        return encoded

    def reset(self, frame, offset):
        self._frame = frame
        self._investment = self._start_investment
        self._offset = offset
        self._stock_count = 0
        self._buy_price = 0.0
        self._sell_price = self._frame['prices'][self._offset]
        self._top_price = self._frame['prices'][self._offset]
        self._bottom_price = self._frame['prices'][self._offset]

    def step(self, action):
        reward = 0.0
        done = False
        price = self._frame['prices'][self._offset]
        if action == Actions.Sell:
            if self._stock_count > 0:
                reward -= 100.0 * (self.trading_fees / (self._stock_count * price))
                done |= self.reset_on_close
                earnings = (price * self._stock_count) - (self._buy_price * self._stock_count)
                if earnings > 0.0:
                    earnings *= 1.0 - self.tax_rate
                if self.__train_level >= 1:
                    reward += 100.0 * (earnings / (self._buy_price * self._stock_count))
                self._investment += self._buy_price * self._stock_count
                self._investment += earnings
                self._stock_count = 0
                self._buy_price = 0.0
                self._sell_price = price
                self._top_price = price
                self._bottom_price = 0.0
        elif action == Actions.Buy:
            if self._stock_count == 0:
                count = int((self._investment - self.trading_fees) / price)
                if count > 0:
                    reward -= 100.0 * (self.trading_fees / (count * price))
                    if self.__train_level >= 2:
                        reward += 100.0 * ((self._top_price - price) / price)
                    self._investment -= self.trading_fees
                    self._investment -= count * price
                    self._stock_count = count
                    self._buy_price = price
                    self._sell_price = 0.0
                    self._top_price = 0.0
                    self._bottom_price = price
        elif self._stock_count == 0 and self.__train_level >= 3:
            if self._top_price > price:
                reward += 100.0 * ((self._top_price - price) / price)
            else:
                reward -= 100.0 * ((price - self._top_price) / self._top_price)
                self._top_price = price
        elif self._stock_count > 0 and self.__train_level >= 4:
            if price > self._bottom_price:
                reward += 100.0 * ((price - self._bottom_price) / self._bottom_price)
            else:
                reward -= 100.0 * ((self._bottom_price - price) / price)
                self._bottom_price = price

        self._offset += 1
        done |= self._offset >= len(self._frame['windows']) - self._days
        return reward, done
