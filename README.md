# PJ/ICT Production Trading Engine (V2.1 - Sentinel Edition)

## 🛡️ MISSION CRITICAL: THE PRODUCTION STACK
This repository contains an institutional-grade, fully automated trading system for MNQ (Nasdaq-100 Micros). It implements the validated PJ/ICT concept through a multi-layered, self-healing execution environment.

---

## 🗺️ SYSTEM ARCHITECTURE & FILE MAP

### 🏗️ CORE EXECUTION (PROJECT ROOT)
| File / Folder | Purpose |
| :--- | :--- |
| `sentinel_watchdog.py` | **Master Supervisor**. Monitors all processes and heartbeats. Auto-restarts system on failure. |
| `run_live_engine.py` | **The Brain**. Consumes tick data, executes the Pool FSM, and manages Risk/PnL. |
| `ibkr_bridge.py` | **The Heart**. 1Hz unthrottled data feed from TWS/Gateway. Includes forced heartbeat. |
| `live_dashboard.py` | **The eyes**. Real-time Console UI for monitoring PnL, Trades, and Data Fidelity. |
| `dashboard_logger.py` | **The Scribe**. Handles 1Hz micro-batched logging to Google Sheets. |
| `LAUNCH_PJ_ENGINE.bat` | **Desktop Entrance**. The one-click launcher that activates the Sentinel. |

### 🧬 SYSTEM INTERNALS (`/live`)
| File | Responsibility |
| :--- | :--- |
| `live/pool_fsm.py` | **Logic Gate**. Manages the state transition for Liquidity Pools (DEFINED -> SWEEP -> RECLAIM). |
| `live/risk_guard.py` | **The Shield**. Enforces Max Daily Loss, Max Trades, and $2/pt Multiplier. |
| `live/bar_builder.py` | **Integrity**. Builds immutable 1-minute bars from raw ticks with zero lookahead bias. |
| `live/execution_bridge.py`| **The Arm**. Communicates with the TradersPost API for multi-broker execution. |

### 🧪 STRATEGY & RESEARCH (`/strategies/mle`)
| File | Importance |
| :--- | :--- |
| `strategies/mle/phase16_pj_engine.py` | **Concept Truth**. The original backtester where the current PJ Model was proven. |
| `strategies/mle/titan_search.py` | **The Miner**. GPU-accelerated brute-force search engine for strategy discovery. |
| `strategies/mle/phase16a_validation.py`| **The Auditor**. Out-of-sample (70/30) validation script for proving edge. |

### 📂 LOGS & DATA
- `/data/GOLDEN_DATA`: High-fidelity historical Parquet files used for strategy mining.
- `sentinel.log`: Diagnostic history of system restarts and health checks.
- `live_ticks.csv`: Raw, unthrottled local tick record.

---

## 🚀 LAUNCH PROTOCOL (PRODUCTION)
1.  **Configure Credentials**: Ensure `service_account.json` (Google Sheets) and TradersPost webhooks are set.
2.  **Open TWS/Gateway**: Ensure Interactive Brokers is logged in on Port 7496.
3.  **Run Launcher**: Double-click `LAUNCH_PJ_ENGINE.bat` on the Desktop.
4.  **Observe Sentinel**: The Sentinel will open 3 separate windows (Bridge, Engine, Dashboard).
5.  **Audit Data**: Check the "Data Fidelity Audit" on the Dashboard to confirm a 1:1 match between IBKR and Google Sheets.

---

## 🛠️ CORE FEATURES
- **1Hz Sync Mode**: Micro-batches all ticks every 1.1s for perfect Google Sheets consistency without quota blocks.
- **Forced Heartbeat**: Guarantees a clinical 1Hz stream even during zero-volume market periods.
- **Fidelity Audit**: Real-time measurement of raw data vs. logged data to ensure 0.0% data loss.
- **Sentinel Auto-Recovery**: System detects stalls or crashes and repairs itself within 15 seconds.

---
**Status**: `CERTIFIED PRODUCTION V2.1`  
**Concept**: `PJ/ICT RECLAIM SNIPER`
