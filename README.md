# CondVest — Financial Analytics App

A light, focused Streamlit dashboard that helps you decide **what to buy**
based on strategy-driven picks, with live technical-indicator analysis on any
US equity.

## Features

- **Market pulse rail** — line charts for S&P 500, NASDAQ, Dow Jones at a glance.
- **Symbol search** — search any US common stock; instant candlestick chart with
  industry, market cap, and last price/change.
- **Live indicators** — overlay up to 4 indicators at once:
  SMA(20/50/200), EMA(20/50), VWAP, Bollinger Bands, RSI(14), MACD.
- **Picks of the Day** — pick a date (default = latest scan date) and a strategy
  (`vegas_channel_short_term`, `golden_cross`, …) and see ranked picks with
  pick price, current price, and **return % since pick**.
- Click any pick to load it into the main chart.

## Data sources

| Table              | Used for                                       |
| ------------------ | ---------------------------------------------- |
| `raw_ohlcv`        | OHLCV bars for charts and indicators           |
| `stock_picks`      | Strategy picks (date, symbol, price, rank, …) |
| `symbol_metadata`  | Company name, industry, market cap, exchange   |

> Return % is computed as `(latest_close - pick_price) / pick_price`. To wire
> in your custom return SQL procedure, replace the `latest_close` CTE inside
> `load_picks()` in `app.py`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add a `.streamlit/secrets.toml` with your Postgres credentials:

```toml
[postgres]
host           = "..."
port           = 5432
db_name_postgres = "condvest"
user           = "..."
password       = "..."
```
