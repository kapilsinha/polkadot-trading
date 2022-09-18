from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Order:
    '''
    Designed to fit the interface for swapExactTokensForTokens
    (https://github.com/stellaswap/core/blob/master/amm/StellaSwapV2Router02.sol#L635)
    '''
    amount_in: int
    amount_out_min: int
    path: List[str] # list of token addresses
    to: str # address to which the out-token goes
    deadline: int # timestamp by which this order must execute
    
    order_id: int
    metadata: Dict[str, Any]
