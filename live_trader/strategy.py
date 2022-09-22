from live_trader.state_manager import StateManager
from live_trader.order import Order
from live_trader.trade import Trade

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Strategy(ABC):
    @abstractmethod
    def on_state_update(self, state_manager: StateManager) -> List[Order]:
        raise NotImplementedError()

    @abstractmethod
    def on_event(self, state_manager: StateManager, event_data: Dict[str, Any]) -> List[Order]:
        raise NotImplementedError()

    @abstractmethod
    def on_our_trades(self, state_manager: StateManager, new_trades: List[Trade]):
        raise NotImplementedError()
