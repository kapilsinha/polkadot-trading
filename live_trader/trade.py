from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    order_id: int
    txn_hash: str
    is_success: bool # whether the txn was rejected
