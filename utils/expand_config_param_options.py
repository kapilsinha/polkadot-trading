from copy import deepcopy
import sys
from typing import Any, Dict, List
import yaml

def zero_pad(num, length):
    s = str(num)
    assert(len(s) <= length)
    return '0' * (length - len(s)) + s

def explode_binance_alpha_config(filename='sim_cfg.yaml', out_filename=None):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    binance_alpha_config = config['strategy']['binance_alpha']
    
    out_config = deepcopy(config)
    out_config['strategy']['binance_alpha'] = {}
    for name, substrat in binance_alpha_config.items():
        exploded_configs = _explode_config(substrat)
        print(name, len(exploded_configs))
        num_length = len(str(len(exploded_configs)))
        for i, c in enumerate(exploded_configs):
            out_config['strategy']['binance_alpha'][f'{name}{zero_pad(i + 1, num_length)}'] = c
    
    out_filename = out_filename or filename.replace('.yaml', '_expanded.yaml')
    with open(out_filename, 'w') as outfile:
        yaml.dump(out_config, outfile)
    print(f'Outputted generated file to {out_filename}')
    return config

def _explode_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k, v in config.items():
        if type(v) is dict:
            exploded_subconfigs = _explode_config(v)
            if len(exploded_subconfigs) > 1:
                exploded_configs = []
                for exploded_subconfig in exploded_subconfigs:
                    c = deepcopy(config)
                    c[k] = exploded_subconfig
                    exploded_configs += _explode_config(c)
                return exploded_configs
        if type(v) is list:
            for i, x in enumerate(v):
                exploded_subconfigs = _explode_config(x)
                if len(exploded_subconfigs) > 1:
                    exploded_configs = []
                    for exploded_subconfig in exploded_subconfigs:
                        c = deepcopy(config)
                        if type(exploded_subconfig) is dict and 'remove' in exploded_subconfig.values():
                            del c[k][i]
                        else:
                            c[k][i] = exploded_subconfig
                        exploded_configs += _explode_config(c)
                    return exploded_configs
        if type(v) is str and '|' in v:
            exploded_configs = []
            values = [x.strip() if x.strip() == 'remove' else eval(x) for x in v.split('|')]
            for individual_value in values:
                c = deepcopy(config)
                c[k] = individual_value
                exploded_configs += _explode_config(c)
            return exploded_configs
    return [config]


if __name__ == '__main__':
    filename = sys.argv[1]
    config = explode_binance_alpha_config(filename)

