from common.enums import AlphaDirection

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

@dataclass(frozen=True)
class TriggerConfig:
    name: str
    price_delta_bps: Optional[float]
    alpha_bps: Optional[float]

    def __post_init__(self):
        if self.price_delta_bps is None and self.alpha_bps is None:
            raise ValueError(f'Trigger {self.name}: At least one of price_delta_bps or alpha_bps must be non-null')

    @staticmethod
    def create_from_dict(d: Dict[str, Any]):
        return TriggerConfig(
            name=d['name'],
            price_delta_bps=d.get('price_delta_bps'),
            alpha_bps=d.get('alpha_bps'),
        )

class CloseTrigger:
    def __init__(self, config: TriggerConfig, open_alpha_direction: AlphaDirection, ref_price: float):
        self.name = config.name

        # Trigger if val is INSIDE the (lo, hi) range
        if config.alpha_bps is None:
            self.alpha_bps_trigger_range = (float('-inf'), float('inf'))
        elif open_alpha_direction == AlphaDirection.BULLISH:
            self.alpha_bps_trigger_range = (float('-inf'), config.alpha_bps)
        elif open_alpha_direction == AlphaDirection.BEARISH:
            self.alpha_bps_trigger_range = (-config.alpha_bps, float('inf'))
        else:
            raise ValueError('Should never reach here')

        # Note that if config.price_delta_bps is negative, it is likely a stop loss trigger,
        # so we invert the trigger interval. A switch statement would have been nice
        if config.price_delta_bps is None:
            self.price_trigger_range = (float('-inf'), float('inf'))
        elif open_alpha_direction == AlphaDirection.BULLISH and config.price_delta_bps > 0:
            price_multiplier = (1 + config.price_delta_bps / 10_000)
            self.price_trigger_range = (ref_price * price_multiplier, float('inf'))
        elif open_alpha_direction == AlphaDirection.BULLISH and config.price_delta_bps < 0:
            price_multiplier = (1 + config.price_delta_bps / 10_000)
            self.price_trigger_range = (float('-inf'), ref_price * price_multiplier)
        elif open_alpha_direction == AlphaDirection.BEARISH and config.price_delta_bps > 0:
            price_multiplier = (1 - config.price_delta_bps / 10_000)
            self.price_trigger_range = (float('-inf'), ref_price * price_multiplier)
        elif open_alpha_direction == AlphaDirection.BEARISH and config.price_delta_bps < 0:
            price_multiplier = (1 - config.price_delta_bps / 10_000)
            self.price_trigger_range = (ref_price * price_multiplier, float('inf'))
        else:
            raise ValueError('Should never reach here')
            
    """
    Returns true if all the conditions are met, else false
    """
    def should_close(self, alpha_bps, cur_price):
        return all((
            self.price_trigger_range[0]     <= cur_price <= self.price_trigger_range[1],
            self.alpha_bps_trigger_range[0] <= alpha_bps <= self.alpha_bps_trigger_range[1],
        ))

class TriggerContainer:
    def __init__(self, config):
        self.open_trigger_cfg = TriggerConfig.create_from_dict(config['trigger_open_order'])
        self.close_trigger_cfgs = [
            TriggerConfig.create_from_dict(cfg) for cfg in config['trigger_close_order']
        ]
        self.close_triggers = None
        self.price_at_open_position = None

    def should_open_position_and_direction(self, alpha_bps: float):
        if -self.open_trigger_cfg.alpha_bps < alpha_bps < self.open_trigger_cfg.alpha_bps:
            return False, None
        if alpha_bps < -self.open_trigger_cfg.alpha_bps:
            return True, AlphaDirection.BEARISH
        if alpha_bps > self.open_trigger_cfg.alpha_bps:
            return True, AlphaDirection.BULLISH
        raise ValueError('Should never reach here')

    def create_close_triggers(self, open_alpha_direction: AlphaDirection, price_at_open_position: float):
        self.close_triggers = [
            CloseTrigger(cfg, open_alpha_direction, price_at_open_position) for cfg in self.close_trigger_cfgs
        ]
        self.price_at_open_position = price_at_open_position

    def should_close_position(self, alpha_bps, price):
        if self.close_triggers is None:
            raise ValueError('Close triggers have not been set, so should_close_position should not be called')
        val = [t.name for t in self.close_triggers if t.should_close(alpha_bps, price)]
        if len(val) > 0:
            price_delta_bps = 10_000 * (price - self.price_at_open_position) / self.price_at_open_position
            logging.info(f'We should close our position at '
                         f'alpha_bps={alpha_bps}, price_delta_bps={price_delta_bps}, '
                         f'price={price}, price_at_open={self.price_at_open_position}. '
                         f'The following triggers fired: {val}')
            return True
        return False

    def clear_close_triggers(self):
        self.close_triggers = None
        self.price_at_open_position = None
