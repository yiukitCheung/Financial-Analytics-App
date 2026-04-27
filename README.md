# TradLyte — Financial Analytics App

A light, focused Streamlit dashboard for understanding **what to buy, and why**.
TradLyte combines strategy-driven daily picks with on-demand technical-indicator
analysis on any US equity, presented in a clean dark-mode UI.

## What it does

- **Market Pulse** rail of intraday-percent line charts for major equity indices
  and commodities, so you always know the broader tape at a glance.
- **Symbol search + chart** — type any US stock and get an instant candlestick
  chart with company name, industry, market cap, and live price/change.
- **Live technical indicators** — overlay up to 4 at a time, including
  SMAs, EMAs (8 / 13 / 20 / 50 / 144 / 169 — the Vegas Channel set), VWAP,
  Bollinger Bands, RSI(14), and MACD. All computed in-browser, no extra calls.
- **Range presets** — flip between YTD / 1Y / 3Y / 5Y on the main chart.
- **Picks of the Day** — pick a date and a strategy and see the top-ranked
  signals with pick price, current price, and live return-to-date. Click any
  pick to load it into the main chart.

## Tech stack

- [Streamlit](https://streamlit.io/) for the UI
- [Plotly](https://plotly.com/python/) for the charts
- pandas / numpy for the indicator engine
- A backing data API (configured privately) for symbols, OHLCV, picks, and
  return calculations

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configuration lives in a private `.streamlit/secrets.toml` (gitignored) — get
the values from the project owner.

## Status

Personal / research project. Not investment advice.
