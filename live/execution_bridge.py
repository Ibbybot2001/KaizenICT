"""
TRADERSPOST EXECUTION BRIDGE (CENTRALIZED)
------------------------------------------
Handles production order execution via TradersPost webhooks.
Includes hourly rate limiting and MNQ-specific payload formatting.
"""

import requests
import time
import logging

logger = logging.getLogger("ExecutionBridge")

class TradersPostBroker:
    def __init__(self):
        self.webhook_url = "https://webhooks.traderspost.io/trading/webhook/333a4649-cb77-43bd-8281-7b62d03d013c/4a88f97c0e65ff1447113015e5e74109"
        self.request_history = []  # List of timestamps
        self.MAX_REQUESTS_PER_HOUR = 30 # Defensive default
        logger.info(f"🚀 TradersPostBroker initialized. Limit: {self.MAX_REQUESTS_PER_HOUR}/hr.")

    def _check_rate_limit(self) -> bool:
        """Prune old requests and check if under limit."""
        now = time.time()
        one_hour_ago = now - 3600
        self.request_history = [t for t in self.request_history if t > one_hour_ago]
        return len(self.request_history) < self.MAX_REQUESTS_PER_HOUR

    def execute_order(self, pool_id, direction, size, sl, tp):
        """
        Sends a POST request to TradersPost Webhook.
        Payload structure (MNQ/Futures): { ticker, action, quantity, stopLoss, takeProfit, sentiment, timeInForce }
        """
        if not self._check_rate_limit():
            logger.critical(f"🛑 [TP-LIMIT] Rate limit reached ({len(self.request_history)}/hr). Blocking order for {pool_id}!")
            return "RATE_LIMIT_EXCEEDED"

        self.request_history.append(time.time())
        
        # Directions: TradersPost expects "buy"/"sell" and "bullish"/"bearish"
        action_str = "buy" if direction.name == "LONG" else "sell"
        sentiment_str = "bullish" if direction.name == "LONG" else "bearish"
        
        payload = {
            "ticker": "MNQ",
            "action": action_str,
            "quantity": int(size),
            "timeInForce": "day",
            "sentiment": sentiment_str,
            "orderType": "market",
            "stopLoss": {
                "stopPrice": float(round(sl, 2))
            },
            "takeProfit": {
                "limitPrice": float(round(tp, 2))
            }
        }

        logger.info(f"📡 [TP-SEND] {action_str.upper()} {size} MNQ | SL: {payload['stopLoss']['stopPrice']} | TP: {payload['takeProfit']['limitPrice']} | Pool: {pool_id}")

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ [TP-OK] Order accepted by TradersPost. Response: {response.text}")
                return "TP-OK"
            else:
                logger.error(f"❌ [TP-FAIL] Webhook error {response.status_code}: {response.text}")
                return f"ERROR-{response.status_code}"
                
        except Exception as e:
            logger.error(f"❌ [TP-CRIT] Failed to send webhook: {e}")
            return "CRITICAL_ERROR"
