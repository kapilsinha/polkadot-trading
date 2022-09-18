from live_trader.order import Order

import logging
import os
from typing import Any, Dict, List
import web3

logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)


class OrderSender:
    def __init__(self, router, web3, glmr_token_ref, config):
        self.router = router
        self.web3 = web3
        self.glmr_token_ref = glmr_token_ref

        self.wallet_address = config['wallet_address']
        self.should_send_orders = config['should_send_orders']
        self.min_glmr_balance = config['min_glmr_balance']
        self.max_txn_fee_proportion_of_pnl = config['max_txn_fee_proportion_of_pnl']
        self.max_fee_per_gas_gwei = config['max_fee_per_gas_gwei']
        self.private_key = os.environ.get('PRIVATE_KEY')
        self._validate_wallet()

    def _validate_wallet(self):
        from eth_account import Account
        from eth_account.signers.local import LocalAccount
        from web3.middleware import construct_sign_and_send_raw_middleware

        assert self.private_key is not None, 'You must set PRIVATE_KEY environment variable'
        assert self.private_key.startswith('0x'), 'Private key must start with 0x hex prefix'

        account: LocalAccount = Account.from_key(self.private_key)
        self.web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))
        assert(self.wallet_address == account.address)

    def send_orders_blocking(self, orders: List[Order]):
        if not self.should_send_orders:
            logging.warning('NOT sending out orders because should_send_orders is False')
            return
        glmr_balance = self.web3.eth.get_balance(self.wallet_address) * 1e-18
        if glmr_balance < self.min_glmr_balance:
            logging.warning(f'NOT sending out orders because GLMR balance ({glmr_balance}) is below '
                            f'threshold of {self.min_glmr_balance}. Add more GLMR to your wallet!')
            return
        txn_hashes = [self._send_order(order) for order in orders]
        # We block on the transaction receipts being mined so that we do not have multiple
        # waves of pending transactions. This obviously will be a multi-second delay but
        # this should be acceptable as we wait for the next block of txns anyway
        # in our trading loop. If/when we move to parsing pending txns, this may require some
        # TODO: revisiting
        logging.warning(f'Sending out orders: {txn_hashes}...')
        for txn_hash in txn_hashes:
            try:
                self.web3.eth.wait_for_transaction_receipt(txn_hash, timeout=300)
            except web3.exceptions.TimeExhausted as e:
                logging.error(f'Timed out waiting for txn receipt: {e}. '
                              f'The txn will likely be included in a later block and rejected, so we move on.')
        logging.warning(f'Finished sending orders: {txn_hashes}')

    def _send_order(self, order: Order):
        nonce = self.web3.eth.get_transaction_count(self.wallet_address)
        txn = self.router.functions.swapExactTokensForTokens(
            order.amount_in,
            order.amount_out_min,
            order.path,
            order.to,
            order.deadline,
        ).build_transaction({
            # must specify 'from' or you'll get a ValueError:
            # {'code': -32603, 'message': 'execution fatal: Module(ModuleError [...])'}
            'from': self.wallet_address,
            'nonce': nonce,
            'maxFeePerGas': int(200 * 1e9),
            'maxPriorityFeePerGas': int(3 * 1e9),
        })
        txn = self._update_txn_gas_fees(txn, order)
        signed_txn = self.web3.eth.account.sign_transaction(txn, private_key=self.private_key)
        return self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)

    def _update_txn_gas_fees(self, txn: Dict[str, Any], order: Order):
        """
        max $ transaction fee = gas limit (max amount gas) * max fee per gas (GLMR / gas) * glmr_usd_value ($ / GLMR)
        """
        if 'expected_pnl' not in order.metadata:
            return txn
        max_usd_txn_fee = order.metadata['expected_pnl'] * self.max_txn_fee_proportion_of_pnl
        max_fee_per_gas = min(
            int(max_usd_txn_fee / (txn['gas'] * self.glmr_token_ref.get_usd_value())),
            int(self.max_fee_per_gas_gwei * 1e9),
        )
        txn.update({
            'maxFeePerGas': max_fee_per_gas,
            'maxPriorityFeePerGas': max_fee_per_gas,
        })
        return txn        
