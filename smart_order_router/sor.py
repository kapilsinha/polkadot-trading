from common.enums import Direction
from smart_order_router.graph import Graph

MAX_DEPTH = 4

"""
Return best quote, path
"""
def single_sor_no_fees(graph: Graph, start_token: str, dest_token: str, amount_in=1, max_depth: int = MAX_DEPTH):
    return _sor_helper(graph, start_token, dest_token, lambda direction, pair, amount_in: pair.quote_no_fees(direction, amount_in), amount_in, max_depth)
 
def single_sor_with_fees(graph: Graph, start_token: str, dest_token: str, amount_in=1, max_depth: int = MAX_DEPTH):
    return _sor_helper(graph, start_token, dest_token, lambda direction, pair, amount_in: pair.quote_with_fees(direction, amount_in), amount_in, max_depth)

def _sor_helper(graph, start_token, dest_token, quote_func, amount_in, max_depth):
    def dfs_helper(cur_token, cur_depth, cur_quote, cur_path, best_quote, best_path):
        if cur_token == dest_token and (cur_quote > best_quote or (cur_quote == best_quote and len(cur_path) < len(best_path))):
            best_quote = cur_quote
            best_path = cur_path.copy()

        if cur_depth < max_depth:
            for pair in graph.adj[cur_token]:
                if cur_token == pair.token0_address:
                    next_token = pair.token1_address
                    new_quote = quote_func(Direction.FORWARD, pair, cur_quote)
                elif cur_token == pair.token1_address:
                    next_token = pair.token0_address
                    new_quote = quote_func(Direction.REVERSE, pair, cur_quote)
                else:
                    raise ValueError(f'Invalid graph adjacency list for token {cur_token}, pair {pair}')
                cur_path.append(pair)
                best_quote, best_path = dfs_helper(next_token, cur_depth + 1, new_quote, cur_path, best_quote, best_path)
                del cur_path[-1]
        
        return best_quote, best_path
    
    best_quote, best_path = dfs_helper(start_token, 0, amount_in, [], float('-inf'), [])
    return best_quote if best_quote > 0 else None, best_path


if __name__ == '__main__':
    
    class TokenPair:
        def __init__(self, pair_address, token0, token1, reserve0, reserve1):
            self.pair_address = pair_address
            self.token0_address = token0
            self.token1_address = token1
            self.reserve0 = reserve0
            self.reserve1 = reserve1
        
        def quote_no_fees(self, direction, amount_in=1):
            r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
            rate = amount_in * r1 / r0 if r0 > 0 and r1 > 0 else None
            return rate
        
        def quote_with_fees(self, direction, amount_in=1):
            r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
            amount_out = (amount_in * .9975 * r1) / (r0 + amount_in * .9975)
            return amount_out

        def __repr__(self):
            return self.pair_address
     
    pairs = [
        TokenPair('pair1', 'token1', 'token2', 100, 200),
        TokenPair('pair2', 'token3', 'token2', 100, 200),
        TokenPair('pair3', 'token3', 'token4', 100, 200),
        # TokenPair('pair3', 'token1', 'token2', 100, 100),
        TokenPair('pair4', 'token4', 'token2', 100, 200),
        TokenPair('pair5', 'token4', 'token6', 100, 200),
        TokenPair('pair6', 'token5', 'token7', 100, 200),
        TokenPair('pair7', 'token5', 'token7', 99, 201),
        # TokenPair('pair6', 'token5', 'token5', 100, 200),
    ]
    graph = Graph(pairs)
    print(graph)
    
    print('token1 -> token2 no fees:', single_sor_no_fees(graph, 'token1', 'token2'))
    print('token1 -> token3 no fees:', single_sor_no_fees(graph, 'token1', 'token3'))
    print('token1 -> token4 no fees:', single_sor_no_fees(graph, 'token1', 'token4'))
    print('token1 -> token5 no fees:', single_sor_no_fees(graph, 'token1', 'token5'))
    print('token5 -> token7 no fees:', single_sor_no_fees(graph, 'token5', 'token7'))

    print('token1 -> token2 w fees:', single_sor_with_fees(graph, 'token1', 'token2'))
    print('token1 -> token3 w fees:', single_sor_with_fees(graph, 'token1', 'token3'))
    print('token1 -> token4 w fees:', single_sor_with_fees(graph, 'token1', 'token4', amount_in=25))
    print('token1 -> token5 w fees:', single_sor_with_fees(graph, 'token1', 'token5'))
    print('token5 -> token7 w fees:', single_sor_with_fees(graph, 'token5', 'token7')) 
