from collections import defaultdict
from glob import glob
import numpy as np
import os
import re
import sys
import yaml


# Quick bash command to sort PnLs of multiple configs
# for file in out/*; do tail $file | grep cumulative_exec2 | cut -c35- | tr -d '\n' && echo ' ->' $file; done | sort

def prettify(x):
    if type(x) == str:
        return x
    if type(x) == float:
        return round(x, 3)
    if type(x) == list:
        return [prettify(y) for y in x]
    if type(x) == dict or type(x) == defaultdict:
        # return '\n'.join([f'{k}: {prettify(v)}' for k, v in x.items()])
        return {k: prettify(v) for k, v in x.items()}
    raise TypeError('Unsupported argument type to prettify')

def analyze_file(filename):
    total_pnl_regex = re.compile('.*Total PnL \(compared to doing nothing\): \$(\d+.\d\d).*')
    order_regex = re.compile(".*orders = .*'trigger_name': '(.*?)'.*")
    trade_regex = re.compile('.*trades =\[(.*?)\].*') # success if group(1) has text, fail if empty
    exec2_pnl_regex = re.compile('.*cumulative_exec2_pnl=\$(-?\d+.\d+).*')
    trade_counter_regex = re.compile('.*trade_id=(\d+).*is_success=(True|False)')

    trigger_counter = defaultdict(int)
    prev_cumulative_exec2_pnl = 0
    cumulative_exec2_pnl = 0
    exec2_pnls = [] # measured from closed position to closed position
    close_trigger_to_exec2_pnl = defaultdict(list)

    with open(filename, 'r') as f:
        x = f.readlines()
        total_pnl = float(total_pnl_regex.match(x[-2]).group(1))

        did_last_trade_open_position = False
        did_last_trade_close_position = False
        last_trigger = ''
        num_failed_trades = 0
        num_successful_trades = 0
        num_trades = 0
        for line in x:
            order_match = order_regex.match(line)
            if order_match:
                trigger_name = order_match.group(1)
                trigger_counter[trigger_name] += 1
                last_trigger = trigger_name

            trade_match = trade_regex.match(line)
            if trade_match:
                is_success = trade_match.group(1) != ''
                if is_success:
                    # an opposite-side open order sets both the below to True
                    did_last_trade_close_position = did_last_trade_open_position
                    did_last_trade_open_position = (last_trigger == 'price_movement_predictor')
            
            exec2_pnl_match = exec2_pnl_regex.match(line)
            if exec2_pnl_match and did_last_trade_close_position:
                cumulative_exec2_pnl = float(exec2_pnl_match.group(1))
                exec2_pnl = cumulative_exec2_pnl - prev_cumulative_exec2_pnl
                prev_cumulative_exec2_pnl = cumulative_exec2_pnl
                exec2_pnls.append(exec2_pnl)
                close_trigger_to_exec2_pnl[last_trigger].append(exec2_pnl)

            trade_counter_match = trade_counter_regex.match(line)
            if trade_counter_match:
                num_trades = int(trade_counter_match.group(1))
                num_successful_trades += trade_counter_match.group(2) == 'True'
                num_failed_trades += trade_counter_match.group(2) == 'False'
        
    print('Execution PnL list (measured close-to-close):', prettify(exec2_pnls))
    print('Sorted execution PnL list (measured close-to-close):', prettify(sorted(exec2_pnls)))
    print('Close trigger to PnL list:', prettify(close_trigger_to_exec2_pnl))
    print(f'Successful trades: {num_successful_trades} ({num_trades} attempted trades, {num_failed_trades} orders rejected)')
    print('Trigger counter:', dict(trigger_counter))
    print('Close trigger to average PnL:',
        prettify({k: f'avg={sum(v) / len(v):0.3f}, total={sum(v):0.2f}' for k, v in close_trigger_to_exec2_pnl.items()}))
    print('Total PnL:', total_pnl)
    print('Total PnL (adj. for gas fees):', prettify(total_pnl - 0.005 * num_trades))


def analyze_dir(dir, expanded_sim_cfg_filename):
    '''
    Expects that dir/{strategy}.out matches {strategy} in the config
    We could do something more sophisticated that parses the un-expanded config to find which
    params are 'exploded' but we just hard-code for sake of speed.
    '''
    with open(expanded_sim_cfg_filename, 'r') as f:
        config = yaml.safe_load(f)['strategy']['binance_alpha']

    total_pnl_regex = re.compile('.*Total PnL \(compared to doing nothing\): \$(\d+.\d\d).*')
    file_to_pnl = {}
    field_to_pnl = defaultdict(list)
    files = glob(f'{dir}/*.out')
    for filename in sorted(files):
        with open(filename, 'r') as f: 
            x = f.readlines()
            total_pnl = float(total_pnl_regex.match(x[-2]).group(1))
            file_to_pnl[filename] = total_pnl
        
        strategy_name = os.path.split(filename)[1].replace('.out', '')
        strategy_config = config[strategy_name]
        val = (total_pnl, strategy_name)
        field_to_pnl[('allow_opposite_side_order_to_close', strategy_config['allow_opposite_side_order_to_close'])].append(val)
        open_triggers = strategy_config['trigger_open_order']
        field_to_pnl[(open_triggers['name'], f'alpha_bps={open_triggers["alpha_bps"]}')].append(val)
        close_triggers = strategy_config['trigger_close_order']
        if 'alpha_reversal' not in [x['name'] for x in close_triggers]:
            field_to_pnl[('alpha_reversal', 'removed')].append(val)
        for c in close_triggers:
            field_to_pnl[(c['name'], f'alpha={c.get("alpha_bps")}', 'price_delta_bps={c.get("price_delta_bps")}')].append(val)

    # print(field_to_pnl)
    for field, vals in sorted(field_to_pnl.items()):
        pnls = [x[0] for x in vals]
        percentiles = list(np.percentile(pnls, [0, 1, 2, 5, 25, 50, 75, 95, 98, 99, 100]))
        print(field, sum(pnls) / len(pnls), percentiles)

if __name__ == '__main__':
    path = sys.argv[1]
    if os.path.isfile(path):
        analyze_file(path)
    elif os.path.isdir(path):
        sim_cfg_filename = sys.argv[2]
        analyze_dir(path, sim_cfg_filename)
    else:
        raise ValueError('Path is not a directory or file')
