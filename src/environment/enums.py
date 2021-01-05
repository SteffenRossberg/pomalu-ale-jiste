import enum


class Actions(enum.IntEnum):
    SkipOrHold = 0
    Buy = 1
    Sell = 2


class TrainingLevels(enum.IntFlag):
    Skip = 8
    Buy = 4
    Hold = 2
    Sell = 1
