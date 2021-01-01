import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
from src.environment.actions import Actions
from src.preparation.preparator import DataPreparator
from src.environment.portfolio_state import PortfolioState

DEFAULT_START_INVESTMENT = 50_000.0
CAPITAL_GAINS_TAX = 25.0
SOLIDARITY_SURCHARGE = 5.5
DEFAULT_TRADING_FEES = 1.0
DEFAULT_TAX_RATE = (CAPITAL_GAINS_TAX * (SOLIDARITY_SURCHARGE / 100.0 + 1.0)) / 100.0


class StockExchange(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 frames,
                 days,
                 reset_on_close=True,
                 random_offset_on_reset=True,
                 start_investment=DEFAULT_START_INVESTMENT,
                 trading_fees=DEFAULT_TRADING_FEES,
                 tax_rate=DEFAULT_TAX_RATE):
        self._frames = frames
        self._state = PortfolioState(days, start_investment, trading_fees, tax_rate, reset_on_close)
        self._ticker = None
        self.np_random = None
        self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_offset_on_reset = random_offset_on_reset
        self.seed()

    @property
    def train_level(self) -> int:
        return self._state.train_level

    @train_level.setter
    def train_level(self, value: int):
        self._state.train_level = value

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._ticker = self.np_random.choice(list(self._frames.keys()))
        frame = self._frames[self._ticker]
        if self.random_offset_on_reset:
            offset = self.np_random.choice(len(frame['windows']) - len(frame['windows'][0]))
        else:
            offset = len(frame['windows'][0])
        self._state.reset(frame, offset)
        return self._state.encode()

    def step(self, action_index):
        action = Actions(action_index)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            'ticker': self._ticker,
            'company': self._frames[self._ticker]['company'],
            'investment': self._state.investment,
            'current_date': self._state.current_date,
            'current_price': self._state.current_price,
            'offset': self._state.offset,
            'stock_count': self._state.stock_count,
            'buy_price': self._state.buy_price
        }
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_provider(cls, provider, days, start_date, end_date, **kwargs):
        frames = DataPreparator.prepare_rl_frames(provider, days, start_date, end_date)
        return StockExchange(frames, days, **kwargs)
