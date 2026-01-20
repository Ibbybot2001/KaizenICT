# 🦅 PJ/ICT Titan Trading Engine (v2.5 - Institutional Grade)

[![System Status](https://img.shields.io/badge/Status-CERTIFIED_PRODUCTION-green?style=for-the-badge&logo=opsgenie)](https://github.com/pesosz/antigravity-auto-accept)
[![Engine](https://img.shields.io/badge/Logic-TITAN_HYBRID_V17-blue?style=for-the-badge)](https://github.com)
[![Hardware](https://img.shields.io/badge/Discovery-RTX_4080_GPU-orange?style=for-the-badge)](https://github.com)

An institutional-grade, fully autonomous trading suite for **MNQ** (Nasdaq-100 Micro Futures), designed for 24/7 uptime and high-fidelity execution auditability.

---

## 🎯 MISSION VISION
We are building a **High-Frequency Institutional Trading Swarm**. The system is designed to autonomously discover, validate, and execute ICT-based liquidity strategies on the Nasdaq-100 (MNQ). 

**The Core Concept**: Rather than relying on a single stagnant strategy, we use a GPU-accelerated "Genetic Miner" to constantly evolve new filters (Kill Zones, Day-of-Week, Momentum, Level Sweeps) and graduate the survivors into a 100% uptime, self-healing live execution environment.

---

## 🛠️ THE PRODUCTION STACK
This repository contains a multi-layered, self-healing execution environment for MNQ (Nasdaq-100 Micros). 

---

## 🏗️ 4-Layer Architecture

The system is built on a "Brutal Realism" philosophy, moving from raw market data to validated alpha in four distinct stages:

### Layer 1: Data Fidelity & Persistence (`ibkr_bridge.py`)
*   **The Heart**: 1Hz unthrottled data feed from IBKR TWS/Gateway.
*   **Persistent Audit**: Millisecond-level tick probe via `trades_audit.csv` ensuring zero-lookahead and "fidelity-locked" entries.
*   **Forced Heartbeat**: Clinical sync frequency maintained even through zero-volume market periods.

### Layer 2: Uptime & Self-Healing (`sentinel_watchdog.py`)
*   **The Supervisor**: Monitors all sub-processes with 15-second stall detection.
*   **Clean-State Recovery**: Proactive process killing of orphaned instances to prevent data conflicts.
*   **Desktop Entrance**: `LAUNCH_PJ_ENGINE.bat` for one-click activation.

### Layer 3: Telemetry & Dashboard (`dashboard_logger.py`)
*   **The Scribe**: Micro-batched logging (54 req/min) to Google Sheets, optimized to stay under API quotas.
*   **JSON Integrity**: Native Python type enforcement (NumPy-to-Int/Float) to prevent serialization crashes during day rollovers.
*   **Timezone Lock**: America/New_York timestamps applied across all diagnostic files.

### Layer 4: HOD (Hardware Optimized Discovery) (`overnight_runner.py`)
*   **The Miner**: Brute-force genetic research suite leveraging the RTX 4080 GPU.
*   **Randomized Validation**: 70/30 Train/Test month splitting to avoid temporal overfitting.
*   **Brutal Scaling**: 11 distinct research phases covering Kill Zones, Volatility Regimes, and Level Sweeps.

---

## 🚀 Launch Protocols

1.  **Verify TWS**: Log in to Interactive Brokers TWS or Gateway (Port 7496).
2.  **Activation**: Execute `LAUNCH_PJ_ENGINE.bat`.
3.  **Audit**: Monitor `sentinel.log` for process heartbeat and `trades_audit.csv` for trade execution.

---

## 📊 Performance Benchmarks (Hybrid V17)
*   **Profit Factor**: 1.88 - 2.73 (Validated)
*   **Logic**: IB/Asia Liquidity Sweep + FVG Reclaim + SMA 200 Filter
*   **Frequency**: 2-6 Trades/Day (Selective Sniper Mode)

---

## 📁 Critical Files Map

| Path | Purpose |
| :--- | :--- |
| `ibkr_bridge.py` | Primary execution bot (Market Hours + Strategy). |
| `sentinel_watchdog.py` | Supervisory heartbeat and auto-restart logic. |
| `overnight_runner.py` | GPU-accelerated research and mining suite. |
| `dashboard_logger.py` | Google Sheets and Local CSV logging. |
| `trades_audit.csv` | Millisecond-level trade lifecycle audit log. |
| `strategies/mle/` | Core research logic and backtesting engines. |

---

## 🛡️ Reliability Features
*   **Tick-Level SL/TP Probe**: Actively monitors price every tick to detect breaches before the next 1-min bar.
*   **Settlement Cooldown**: 30s buffer between orders to prevent race conditions.
*   **Signature Tracking**: Prevents redundant execution on the same signal timestamp.

**Last Certified Audit:** 2026-01-20  
**Version:** `PRODUCTION_TITAN_V2.5`
