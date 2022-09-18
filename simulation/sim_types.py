from copy import copy
from common.enums import Direction

from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, List


@dataclass(frozen=True)
class Order:
    '''
    We won’t bother with deadline, amountInMax, amountOutMin, etc.
    Pretend there is no slippage
    '''
    order_id: int
    amount_in: float
    path: List[str] # list of token addresses

    # These are not required for the sim logic but provide useful metadata
    block_num: int
    last_txn_index: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Trade:
    order_id: int
    trade_id: int
    pair_address: str
    # delta is the change in the amount of tokens in the liquidity pool
    # delta is positive if it deposits tokens into the liquidity pool, negative if it takes them out
    amount0_delta: float
    amount1_delta: float


class Token:
    def __init__(self, token_address, usd_value):
        self.token_address = token_address
        self._usd_value = usd_value

    def get_usd_value(self):
        return self._usd_value


class TokenPair:
    def __init__(self, pair_address, token0, token1):
        self.pair_address = pair_address
        self.token0_address = token0
        self.token1_address = token1
        self.reserve0 = None
        self.reserve1 = None

    def set_reserves(self, reserve0, reserve1):
        self.reserve0 = reserve0
        self.reserve1 = reserve1

    def quote_no_fees(self, direction: Direction, amount_in=1) -> float:
        '''
        Hacky alert! This will be called from SOR on token pairs that have reserve0 = reserve1 = None because
        the token pair has not yet been created (i.e. set_reserves has not been called) but the graph has been
        instantiated with all token pairs. Hence we return zero to indicate this token pair yields nothing.
        '''
        if self.reserve0 is None or self.reserve1 is None:
            return 0
        r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
        rate = amount_in * r1 / r0 if r0 > 0 and r1 > 0 else 0
        return rate

    def quote_with_fees(self, direction: Direction, amount_in=1) -> float:
        if self.reserve0 is None or self.reserve1 is None:
            return 0
        r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
        amount_out = (amount_in * .9975 * r1) / (r0 + amount_in * .9975)
        return amount_out

    def inverse_quote_no_fees(self, direction: Direction, amount_out: float) -> float:
        assert(self.reserve0 is not None and self.reserve1 is not None and self.reserve0 > 0 and self.reserve1 > 0)
        r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
        return amount_out * r0 / r1 # same as quote_no_fees in the opposite direction
    
    def inverse_quote_with_fees(self, direction: Direction, amount_out: float):
        assert(self.reserve0 is not None and self.reserve1 is not None and self.reserve0 > 0 and self.reserve1 > 0)
        r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
        amount_in = (400/399) * amount_out * r0 / (r1 - amount_out)
        return amount_in

    def compute_trade_for_target_forward_quote(self, target_forward_quote: float):
        cur_quote = self.quote_no_fees(Direction.FORWARD)
        if target_forward_quote == cur_quote:
            return None, 0, 0
        if target_forward_quote < cur_quote:
            # Forward direction trade decreases the quote (makes token1 scarcer, token0 more abundant), vice versa
            direction = Direction.FORWARD
            r0, r1 = self.reserve0, self.reserve1
            rate = target_forward_quote
        else:
            direction = Direction.REVERSE
            r0, r1 = self.reserve1, self.reserve0
            rate = 1 / target_forward_quote
        
        # This baby was computed with the help of Wolfram Alpha
        amount_in = (np.sqrt(r0 * (rate * r0 + 638400 * r1) / rate) - 799 * r0) / 798
        amount_out = self.quote_with_fees(direction, amount_in)
        return direction, amount_in, amount_out

    def __repr__(self):
        return f'TokenPair(pair_address={self.pair_address}, reserve0={self.reserve0}, reserve1={self.reserve1})'


if __name__ == '__main__':
    from copy import copy

    pair = TokenPair('0xabcd', 'token0', 'token1')
    pair.set_reserves(1e3, 2e3)
    print(pair.quote_no_fees(Direction.FORWARD, 100)) # 200
    print(pair.quote_no_fees(Direction.REVERSE, 100)) # 50
    
    print(pair.quote_with_fees(Direction.FORWARD, 100)) # ~181
    print(pair.quote_with_fees(Direction.REVERSE, 100)) # ~47

    amount_in = 100
    amount_out = pair.quote_no_fees(Direction.FORWARD, amount_in) # 200
    computed_amount_in = pair.inverse_quote_no_fees(Direction.FORWARD, amount_out) # 100
    assert(amount_in == computed_amount_in)

    amount_in = 100
    amount_out = pair.quote_with_fees(Direction.FORWARD, amount_in) # ~181
    computed_amount_in = pair.inverse_quote_with_fees(Direction.FORWARD, amount_out) # 99.9999
    assert(abs(amount_in - computed_amount_in) < 1e-12)

    trade_direction, trade_amount_in, trade_amount_out = pair.compute_trade_for_target_forward_quote(1)
    print(trade_direction, trade_amount_in, trade_amount_out)
    p2 = copy(pair)
    if trade_direction == Direction.FORWARD:
        p2.set_reserves(pair.reserve0 + trade_amount_in, pair.reserve1 - trade_amount_out)
    else:
        p2.set_reserves(pair.reserve0 - trade_amount_out, pair.reserve1 + trade_amount_in)
    print(p2.quote_no_fees(Direction.FORWARD), p2, '\n')

    trade_direction, trade_amount_in, trade_amount_out = pair.compute_trade_for_target_forward_quote(1.5)
    print(trade_direction, trade_amount_in, trade_amount_out)
    p2 = copy(pair)
    if trade_direction == Direction.FORWARD:
        p2.set_reserves(pair.reserve0 + trade_amount_in, pair.reserve1 - trade_amount_out)
    else:
        p2.set_reserves(pair.reserve0 - trade_amount_out, pair.reserve1 + trade_amount_in)
    print(p2.quote_no_fees(Direction.FORWARD), p2)

    trade_direction, trade_amount_in, trade_amount_out = pair.compute_trade_for_target_forward_quote(2.5)
    print(trade_direction, trade_amount_in, trade_amount_out, '\n')
    p2 = copy(pair)
    if trade_direction == Direction.FORWARD:
        p2.set_reserves(pair.reserve0 + trade_amount_in, pair.reserve1 - trade_amount_out)
    else:
        p2.set_reserves(pair.reserve0 - trade_amount_out, pair.reserve1 + trade_amount_in)
    print(p2.quote_no_fees(Direction.FORWARD), p2)
