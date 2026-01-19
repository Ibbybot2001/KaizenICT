# 📊 KaizenICT Data Logging Map
**Status**: Production Active
**Last Updated**: January 2026

This document defines the authoritative data flow for all system logs, ensuring 100% observability of every request, tick, and trade.

## 🔗 Destination: Central Google Sheet
**URL**: [KaizenICT Production Dashboard](https://docs.google.com/spreadsheets/d/1hcE1sdBSbuk2stouI_79ajYlp5JwbBOavFT0-fNLcjw/edit)
**Auth**: via `service_account.json`

---

## 1. Raw Market Data (Ticks)
Every single price update from IBKR is captured, buffered, and logged.

- **Source Code**: `dashboard_logger.py` -> `log_tick()`
- **Capture Frequency**: Real-time (buffered 1.1s for API safety)
- **Destination Sheet**: `RawTicks`
- **Columns**: `Timestamp`, `Price`, `Size`
- **Purpose**: Forensic audit of latency and slippage.

## 2. Market State (1-Minute Bars)
Aggregated OHLCV data and calculated metrics tailored for the strategy.

- **Source Code**: `dashboard_logger.py` -> `log_min_data()`
- **Trigger**: End of every 1-minute candle
- **Destination Sheet**: `OneMinuteData`
- **Columns**:
  - `Timestamp`
  - `Price` (Close)
  - `Action` (Signal/Wait)
  - `Details` (Context)
  - `RelVol` (Relative Volume)
  - `WickRatio`
  - `BodyTicks`
  - `DonchianHigh/Low`
  - `DistHigh/Low` (Distance to channels)
  - `Status`
- **Purpose**: Strategy decision verification and signal auditing.

## 3. Trade Execution Log
Complete lifecycle of every trade, from signal to result.

- **Source Code**: `dashboard_logger.py` -> `log_trade()`
- **Trigger**: Entry Order Sent, Stop Loss Hit, Take Profit Hit
- **Destination Sheet**: `TradeLog`
- **Columns**:
  - `Timestamp`
  - `PoolID` (e.g., ONL, PDH)
  - `Direction` (LONG/SHORT)
  - `Entry`, `StopLoss`, `TakeProfit`
  - `Status` (FILLED, CLOSED, STOPPED)
  - `PnL_Pts` (Points)
  - `PnL_USD` (Points * $2 MNQ)
  - `Setup Metrics` (Wick, Vol, Body, ZScore at entry)
- **Purpose**: P&L tracking and edge validation.

---

## 4. System Health & Errors
Operational logs for debugging and uptime monitoring.

- **Source Code**: Standard Python `logging` module
- **Local File**: `output/logs/system.log` (Rotated daily)
- **Console**: Stdout (Visible in Terminal)
- **Content**: Connection drops, restart events, API errors (429s), and critical exceptions.

## 📁 Key File Locations
| Component | File Path |
| :--- | :--- |
| **Logger Logic** | `C:/Users/CEO/ICT reinforcement/dashboard_logger.py` |
| **Credentials** | `C:/Users/CEO/ICT reinforcement/service_account.json` |
| **Raw Data** | `C:/Users/CEO/ICT reinforcement/data/` |
| **Strategy Logs** | `C:/Users/CEO/ICT reinforcement/output/` |

---
**✅ COMPLIANCE**: This logging architecture ensures that **every request** and **every market event** is permanently recorded for forensic reconstruction.
