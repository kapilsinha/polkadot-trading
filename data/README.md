# Data dumps

## Directory structure
```
$ tree | grep -v zip | grep -v feather
.
├── binance_history
│   └── eth_usdt
│       └── raw # zipped files by date
├── README.md
├── stellaswap_liquid_pairs.csv # Manually noted down the ~20 most liquid pairs (excluding stable pairs) from https://analytics.stellaswap.com/ on 2022-08-10.
├── stellaswap_metadata_snapshot.csv # Contains the output of `stellaswap_snapshot.py`
├── stellaswap_txn_history # Contains the output of `stellaswap_save_data.py`, broken in 10,000 block intervals.
│   ├── all
│   ├── end_of_block_token # filters to keep only END_OF_BLOCK_TOKEN_SNAPSHOT rows
│   └── end_of_block_token_pair # filters to keep only END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT rows
└── token_metadata.txt # Manually noted down information on the ~25 top tokens that are part of the above liquid pairs on 2022-09-15.
```
