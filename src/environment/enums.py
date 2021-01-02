import enum


class Actions(enum.IntEnum):
    SkipOrHold = 0
    Buy = 1
    Sell = 2


class TrainingLevels(enum.IntFlag):
    Buy = 1
    Sell = 2
    Hold = 4
    Skip = 8
