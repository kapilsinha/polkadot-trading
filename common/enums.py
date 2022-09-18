from enum import IntEnum


class Direction(IntEnum):
    FORWARD = 1 # Give token0, get token1
    REVERSE = 2 # Give token1, get token1

class DataRowType(IntEnum):
    """
    These are wordy names but I think they capture the idea pretty well
    """
    END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT = 1
    ON_UPDATE_TOKEN_PAIR_SNAPSHOT = 2
    END_OF_BLOCK_TOKEN_SNAPSHOT = 3
    ON_UPDATE_TOKEN_SNAPSHOT = 4
    SWAP_TXN = 5

class AlphaDirection(IntEnum):
    BULLISH = 1 # Expect price to increase
    BEARISH = 2 # Expect price to decrease
