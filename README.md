# Moonbeam (Polkadot parachain) trading

## Requirements and installing packages
Please use Python 3.10. Websockets behavior has changed as of Python 3.10,
so I've pointed to a beta version of the Web3 library to resolve conflicting dependencies.
Note: Using a virtualenv is recommended to avoid version conflicts. 
```
source .venv/bin/activate
pip3 install -r requirements.txt
deactivate
```

### Special instructions for running on chainhub.me
Nix is painful, so please just run the following to create and enter the virtualenv (this will install packages from requirements.txt).
```
nix-shell venv.nix
```

## Run scripts
Run scripts from this folder instead of from within any subdirectory
```
(.venv) $ python -m smart_order_router.sor
```
You may need to set some env vars before running the live trader script (namely, PRIVATE\_KEY for your wallet)

# Misc notes
We use public RPC endpoints here. Eventually when we "productionize", we should migrate off of these (https://docs.moonbeam.network/builders/get-started/endpoints/#endpoint-providers)
