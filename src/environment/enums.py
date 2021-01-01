import enum


class Actions(enum.Enum):
    SkipOrHold = 0
    Buy = 1
    Sell = 2


class TrainingLevels(enum.IntEnum):
    Buy = 1
    BuySell = 2
    SkipBuyHoldSell = 3
