from collections import defaultdict
import re
import sys

def prettify(x):
    if type(x) == float:
        return round(x, 3)
    if type(x) == list:
        return [prettify(y) for y in x]
    if type(x) == dict or type(x) == defaultdict:
        return {k: prettify(v) for k, v in x.items()}
    raise TypeError('Unsupported argument type to prettify')

total_pnl_regex = re.compile('.*Total PnL \(compared to doing nothing\): \$(\d+.\d\d).*')
trigger_regex = re.compile(".*'trigger_name': '(.*?)'.*")
exec2_pnl_regex = re.compile('.*cumulative_exec2_pnl=\$(-?\d+.\d+).*')
trade_regex = re.compile('.*trade_id=(\d+).*is_success=(True|False)')

trigger_counter = defaultdict(int)
prev_cumulative_exec2_pnl = 0
cumulative_exec2_pnl = 0
exec2_pnls = [] # measured from closed position to closed position
close_trigger_to_exec2_pnl = defaultdict(list)

filename = sys.argv[1]
with open(filename, 'r') as f:
    x = f.readlines()
    total_pnl = float(total_pnl_regex.match(x[-2]).group(1))

    did_last_trade_close_position = False
    last_trigger = ''
    num_failed_trades = 0
    num_successful_trades = 0
    num_trades = 0
    for line in x:
        trigger_match = trigger_regex.match(line)
        if trigger_match is not None:
            trigger_name = trigger_match.group(1)
            trigger_counter[trigger_name] += 1
            did_last_trade_close_position = trigger_name != 'price_movement_predictor'
            last_trigger = trigger_name
        
        exec2_pnl_match = exec2_pnl_regex.match(line)
        if exec2_pnl_match is not None and did_last_trade_close_position:
            cumulative_exec2_pnl = float(exec2_pnl_match.group(1))
            exec2_pnl = cumulative_exec2_pnl - prev_cumulative_exec2_pnl
            prev_cumulative_exec2_pnl = cumulative_exec2_pnl
            exec2_pnls.append(exec2_pnl)
            close_trigger_to_exec2_pnl[last_trigger].append(exec2_pnl)

        trade_match = trade_regex.match(line)
        if trade_match is not None:
            num_trades = int(trade_match.group(1))
            num_successful_trades += trade_match.group(2) == 'True'
            num_failed_trades += trade_match.group(2) == 'False'
    
#    num_trades = sum(trigger_counter.values())

print('Execution PnL list (measured close-to-close):', prettify(exec2_pnls))
print('Close trigger to PnL list:', prettify(close_trigger_to_exec2_pnl))
print(f'Attempted trades: {num_trades} ({num_successful_trades} successful trades, {num_failed_trades} orders rejected)')
print('Trigger counter:', dict(trigger_counter))
print('Close trigger to average PnL:', prettify({k: sum(v) / len(v) for k, v in close_trigger_to_exec2_pnl.items()}))
print('Total PnL:', total_pnl)
print('Total PnL (adj. for gas fees):', prettify(total_pnl - 0.005 * num_trades))
