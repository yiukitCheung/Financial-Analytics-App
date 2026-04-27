# TradLyte — Financial Analytics App

A light, focused Streamlit dashboard that helps you decide **what to buy**
based on strategy-driven picks, with live technical-indicator analysis on any
US equity. All data is loaded over HTTPS from the serving API — there is no
direct database access.

## Features

- **Market Pulse rail** — line charts for S&P 500, NASDAQ, Dow Jones at a glance.
- **Symbol search** — search any US common stock from the cached universe;
  instant candlestick chart with name, industry, market cap, and last price/change.
- **Live indicators** — overlay up to 4 indicators at once:
  SMA(20/50/200), EMA(20/50), VWAP, Bollinger Bands, RSI(14), MACD.
- **Picks of the Day** — pick a date (default = latest scan date discovered via
  `/picks/today`) and a strategy, see ranked picks with pick price, current
  price, and **return-to-date** straight from `/picks/{date}/returns`.
- Click any pick to load it into the main chart.

## Data sources

All endpoints come from the serving API. Base URL is configured in
`.streamlit/secrets.toml`.

| Endpoint                                    | Used for                                   |
| ------------------------------------------- | ------------------------------------------ |
| `GET /health`                               | Connectivity check                         |
| `GET /screener/quotes?type=CS`              | Symbol-search universe (paginated)         |
| `GET /market/ohlcv/{symbol}`                | Candlestick bars + index series            |
| `GET /market/quote/{symbol}`                | Per-symbol metadata fallback               |
| `GET /picks/today`                          | Discover latest scan date + strategies     |
| `GET /picks/{scan_date}/returns?horizons=…` | Picks panel (price, close_now, return)     |

> The picks panel reads `return_to_date` directly from
> `/picks/{scan_date}/returns` — no client-side return math.

## Configuration

Edit `.streamlit/secrets.toml`:

```toml
[api]
base_url = "***REDACTED***"
api_key  = "<YOUR_SERVING_API_KEY>"
timeout  = 15
```

Auth header `x-api-key` is sent automatically on every request except `/health`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
