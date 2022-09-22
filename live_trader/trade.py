from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    order_id: int
    txn_hash: str
    is_success: bool # whether the txn was rejected
    # delta is the change in the amount of tokens in the liquidity pool
    # delta is positive if it deposits tokens into the liquidity pool, negative if it takes them out
    amount0_delta: int
    amount1_delta: int
