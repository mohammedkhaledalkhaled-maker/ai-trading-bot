# AI Trading Bot

AI-powered cryptocurrency trading bot for real-time market analysis, trade signal generation, and performance tracking.

## Features
- Machine learning model using XGBoost
- Real-time crypto market scanning
- Binance market data integration via CCXT
- Telegram bot for alerts and notifications
- Backtesting and parameter optimization
- SQLite database for trade tracking and analytics
- Multi-timeframe analysis

## Tech Stack
- Python
- XGBoost
- CCXT
- SQLite
- Telegram Bot API

## Project Overview
This project analyzes cryptocurrency market data using technical indicators and machine learning to identify high-probability trading opportunities. It includes real-time scanning, alerting, backtesting, and performance analysis.

## Files
- `bot.py` — main bot logic
- `db.py` — database management
- `analytics.py` — performance analytics
- `backtest_all.py` — backtesting engine
- `train_tp_sl_model_1h_bot_features.py` — model training
- `ai_model_tp_sl_bot_1h.json` — trained model

## Notes
Sensitive files such as database files, API keys, and environment variables are excluded from the repository.
