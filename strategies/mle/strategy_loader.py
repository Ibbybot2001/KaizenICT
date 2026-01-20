import json
import os
import logging
from datetime import datetime

logger = logging.getLogger("StrategyLoader")

# Default Fallback Application Config
# These values are used if the JSON file is missing or a strategy ID is not found.
DEFAULTS = {
    "GOLDEN_DEFAULT": {
        "sl": 5.0,
        "tp": 40.0,
        "description": "Legacy Default"
    },
    "ASIA_Hybrid": {
        "sl": 5.0,
        "tp": 40.0,
        "description": "Asia Session Hybrid (Fallback)"
    },
    "IB_Hybrid": {
        "sl": 5.0,
        "tp": 40.0,
        "description": "IB Session Hybrid (Fallback)"
    }
}

STRATEGY_FILE_PATH = r"C:\Users\CEO\ICT reinforcement\overnight_results\validated_strategies.json"

class StrategyLoader:
    _instance = None
    _strategies = {}
    _last_load_time = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyLoader, cls).__new__(cls)
            cls._instance._load_strategies()
        return cls._instance

    def _load_strategies(self):
        """Loads strategies from the JSON file."""
        print(f"DEBUG: Loading strategies from {STRATEGY_FILE_PATH}")
        try:
            if os.path.exists(STRATEGY_FILE_PATH):
                with open(STRATEGY_FILE_PATH, 'r') as f:
                    data = json.load(f)
                    
                # Parse depending on JSON structure (List vs Dict)
                # Assuming the miner outputs a list of dicts, we index by 'name' or 'id'
                if isinstance(data, list):
                    for item in data:
                        # Fallback: Use 'session' as ID if name/id missing
                        s_id = item.get('name') or item.get('id')
                        if not s_id and item.get('session'):
                            # Map session to Bridge Strategy ID format
                            sess = item.get('session').upper()
                            if sess == 'ASIA': s_id = 'ASIA_Hybrid'
                            elif sess == 'IB': s_id = 'IB_Hybrid'
                            elif sess == 'LONDON': s_id = 'LONDON_Strategy'
                            else: s_id = f"{sess.upper()}_Strategy"
                            
                            # Debug Log
                            print(f"DEBUG: Mapped session '{item.get('session')}' -> ID '{s_id}'")
                        
                        if s_id:
                            self._strategies[s_id] = {
                                "sl": float(item.get('stop_loss', item.get('sl', 5.0))),
                                "tp": float(item.get('take_profit', item.get('tp', 40.0))),
                                # Full genome for live trading parity
                                "body": float(item.get('body', 0.0)),
                                "wick": float(item.get('wick', 100.0)),  # % max
                                "fvg": float(item.get('fvg', 0.0)),
                                "vol": float(item.get('vol', 0.0)),
                                "description": f"Mined {item.get('date', 'Unknown')}"
                            }
                elif isinstance(data, dict):
                     self._strategies.update(data)
                     
                logger.info(f"✅ Loaded {len(self._strategies)} strategies from {STRATEGY_FILE_PATH}")
                print(f"DEBUG: Loaded Keys: {list(self._strategies.keys())}")
                self._last_load_time = datetime.now()
            else:
                logger.warning(f"⚠️ Strategy file not found at {STRATEGY_FILE_PATH}. Using defaults.")
                print(f"DEBUG: File NOT FOUND at {STRATEGY_FILE_PATH}")
                self._strategies = DEFAULTS.copy()
        except Exception as e:
            logger.error(f"❌ Failed to load strategies: {e}")
            print(f"DEBUG: Exception {e}")
            self._strategies = DEFAULTS.copy()

    def get_config(self, strategy_id):
        """
        Returns the configuration for a given strategy ID.
        Falls back to defaults if not found.
        """
        # Reload if empty (or could add Logic to reload every hour)
        if not self._strategies:
            self._load_strategies()

        config = self._strategies.get(strategy_id)
        if not config:
            # Try fuzzy match or fallback
            if "ASIA" in strategy_id: return DEFAULTS["ASIA_Hybrid"]
            if "IB" in strategy_id: return DEFAULTS["IB_Hybrid"]
            return DEFAULTS["GOLDEN_DEFAULT"]
            
        return config

# Singleton Accessor
loader = StrategyLoader()

def get_strategy_config(strategy_id):
    return loader.get_config(strategy_id)
