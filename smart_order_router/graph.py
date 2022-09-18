class Graph:
    def __init__(self, token_pairs):
        tokens = set([pair.token0_address for pair in token_pairs]).union(
            set([pair.token1_address for pair in token_pairs]))

        # {token: adjacent edges} to store graph
        self.adj = {t: [] for t in tokens}
        for pair in token_pairs:
            # In theory there can be liquidity pools with the same token on both sides, but we disallow it
            if pair.token0_address == pair.token1_address:
                raise ValueError(f'token0 == token1 == True for pair {pair}: we expect liquidity pools to have two different tokens')
            self.adj[pair.token0_address].append(pair)
            self.adj[pair.token1_address].append(pair)

    def get_tokens(self):
        return self.adj.keys()

    def __repr__(self):
        return '\n'.join([f'Token {token}: adjacent edges = {adj_pairs}' for token, adj_pairs in sorted(self.adj.items())])
