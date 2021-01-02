import numpy as np
from src.environment.enums import Actions, TrainingLevels


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
        self._trade_days = 0
        self._buy_price = 0.0
        self._top_price = 0.0
        self.__train_level = TrainingLevels.Buy

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
    def train_level(self) -> TrainingLevels:
        return self.__train_level

    @train_level.setter
    def train_level(self, value: TrainingLevels):
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
        self._top_price = self._frame['prices'][self._offset]

    def step(self, action):
        reward = 0.0
        done = False
        price = self._frame['prices'][self._offset]
        count = int((self._investment - self.trading_fees) / price)
        if action == Actions.Buy and self._stock_count == 0 and count > 0:
            reward -= (self.trading_fees / (count * price)) * 100.0
            if self.__train_level & TrainingLevels.Buy == TrainingLevels.Buy:
                reward += self.calculate_reward(self._top_price, price)
            self._investment -= self.trading_fees
            self._investment -= count * price
            self._stock_count = count
            self._buy_price = price
            self._top_price = price
            self._trade_days = 1
        elif action == Actions.Sell and self._stock_count > 0:
            reward -= (self.trading_fees / (self._stock_count * price)) * 100.0
            if self.__train_level & TrainingLevels.Sell == TrainingLevels.Sell:
                reward += self.calculate_reward(price, self._buy_price, self._trade_days)
            self._investment -= self.trading_fees
            self._investment += count * price
            self._investment -= ((count * price) - (count * self._buy_price)) * (self.tax_rate if reward > 0.0 else 1.0)
            self._trade_days = 0
            self._stock_count = 0
            self._buy_price = 0.0
            self._top_price = price
            done |= self.reset_on_close
        elif action == Actions.SkipOrHold and self._stock_count == 0:
            self._trade_days += 1
            if self.__train_level & TrainingLevels.Skip == TrainingLevels.Skip:
                reward += self.calculate_reward(price, self._top_price, 1)
            self._top_price = price if self._top_price < price else self._top_price
        elif action == Actions.SkipOrHold and self._stock_count > 0:
            self._trade_days += 1
            if self.__train_level & TrainingLevels.Hold == TrainingLevels.Hold:
                reward += self.calculate_reward(price, self._buy_price, 1)
        self._offset += 1
        done |= self._offset >= len(self._frame['windows']) - self._days
        return reward, done

    def calculate_reward(self, price, lower_price, factor=2):
        returns = ((price / lower_price) - 1.0)
        returns *= 1.0 - (self.tax_rate if returns > 0.0 else 0.0)
        returns *= factor
        reward = returns * 100.0
        return reward
