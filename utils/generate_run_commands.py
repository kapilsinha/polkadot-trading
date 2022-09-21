from math import ceil


def gen_commands(num):
    cmds = []
    padding = len(str(num))
    for i in range(1, num + 1):
        length = len(str(i))
        padded_i = '0' * (padding - length) + str(i)
        cmds.append(f'time python -m simulation.binance_alpha_strategy wglmr_xcdot{padded_i} &> out/wglmr_xcdot{padded_i}.out')
    num_processes = 18
    n = ceil(len(cmds) / num_processes)
    chunked_cmds = [cmds[i:i + n] for i in range(0, len(cmds), n)]
    serial_cmds = [f"({' && '.join(c)}) &" for c in chunked_cmds]
    print('\n'.join(serial_cmds))

gen_commands(1920)