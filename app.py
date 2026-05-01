"""TradLyte — Financial Analytics App

A light, focused dashboard for discovering what to buy based on saved
strategy picks, with live technical-indicator analysis on any US equity.

All data is loaded via the serving HTTP API (see [api] in secrets.toml).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # pragma: no cover — fallback for older Streamlit
    def add_script_run_ctx(thread=None, ctx=None):  # type: ignore[no-redef]
        return None

    def get_script_run_ctx():  # type: ignore[no-redef]
        return None


def _parallel(jobs: dict[str, tuple]) -> dict[str, Any]:
    """Run independent loaders concurrently with Streamlit ctx propagated.

    `jobs` maps result-key -> (callable, args, kwargs). Returns key -> result
    or the raised Exception, never raises.
    """
    ctx = get_script_run_ctx()

    def _init():
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs))),
                            initializer=_init) as ex:
        futures = {k: ex.submit(fn, *args, **(kwargs or {}))
                   for k, (fn, args, kwargs) in jobs.items()}
        for k, f in futures.items():
            try:
                results[k] = f.result()
            except Exception as e:  # noqa: BLE001 — capture per-job
                results[k] = e
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradLyte — Financial Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme + CSS
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#00d4aa"
ACCENT_RED = "#ff5267"
BG = "#0a0e1a"
CARD = "#121826"
BORDER = "rgba(255,255,255,0.06)"
MUTED = "rgba(255,255,255,0.55)"

STYLES = f"""
<style>
#MainMenu, footer, header {{ visibility: hidden; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: radial-gradient(1200px 600px at 0% 0%, #0f1626 0%, {BG} 55%) fixed;
    color: #e7ebf3;
}}
.block-container {{ padding: 1.4rem 2.2rem 3rem; max-width: 1500px; }}

/* Brand header */
.brand-row {{ display:flex; align-items:center; gap:14px; margin-bottom: 4px; }}
.brand-title {{
    font-size: 26px; font-weight: 700; letter-spacing: -0.01em;
    background: linear-gradient(90deg, #ffffff 0%, {ACCENT} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.brand-tag {{
    color: {MUTED}; font-size: 13px; font-weight: 500;
    border-left: 1px solid {BORDER}; padding-left: 12px; margin-left: 4px;
}}
.brand-sub {{ color: {MUTED}; font-size: 13.5px; line-height: 1.55; max-width: 880px; margin-top: 6px; }}

/* Section labels */
.eyebrow {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
    color: {MUTED}; text-transform: uppercase; margin: 0 0 8px 2px;
}}
.rail-group {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.16em;
    color: rgba(255,255,255,0.36); text-transform: uppercase;
    margin: 12px 0 4px 2px;
}}

/* Section header (used by Picks of the Day and similar) */
.section-head {{ margin: 4px 0 16px; }}
.section-title {{
    display: inline-flex; align-items: center; gap: 10px;
    font-size: 22px; font-weight: 700; letter-spacing: -0.01em;
    color: #ffffff;
}}
.section-title::before {{
    content: ""; width: 4px; height: 22px; border-radius: 3px;
    background: linear-gradient(180deg, {ACCENT} 0%, #00a583 100%);
    display: inline-block;
}}
.section-sub {{
    color: {MUTED}; font-size: 13.5px; line-height: 1.55;
    max-width: 880px; margin: 6px 0 0 14px;
}}

/* Range preset radio → segmented-control look */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    flex-direction: row !important; gap: 0 !important;
    justify-content: flex-end;
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 3px; width: fit-content;
    margin-left: auto;
}}
div[data-testid="stRadio"] label {{
    padding: 4px 14px !important;
    border-radius: 7px !important;
    font-size: 12px !important; font-weight: 600 !important;
    color: {MUTED} !important; cursor: pointer;
    transition: all .12s ease;
}}
div[data-testid="stRadio"] label:hover {{ color: #e7ebf3 !important; }}
div[data-testid="stRadio"] label > div:first-child {{ display: none !important; }}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: rgba(0,212,170,0.14) !important;
    color: {ACCENT} !important;
}}

/* Card */
.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 14px 16px;
}}

/* Symbol meta strip */
.meta-strip {{
    display:flex; flex-wrap:wrap; gap: 10px 22px; align-items:baseline;
    padding: 4px 2px 12px;
}}
.meta-strip .sym {{
    font-size: 28px; font-weight: 700; letter-spacing: -0.01em; color: #fff;
}}
.meta-strip .name {{ color: {MUTED}; font-size: 14px; font-weight: 500; }}
.meta-strip .pill {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
    padding: 3px 9px; border-radius: 999px;
    background: rgba(255,255,255,0.04); color: {MUTED};
    border: 1px solid {BORDER};
}}
.meta-strip .price {{
    font-size: 22px; font-weight: 700; color: #fff; margin-left: auto;
}}
.meta-strip .delta-pos {{ color: {ACCENT}; font-weight:600; font-size: 14px; margin-left: 8px; }}
.meta-strip .delta-neg {{ color: {ACCENT_RED}; font-weight:600; font-size: 14px; margin-left: 8px; }}

/* Streamlit widget polish */
div[data-baseweb="select"] > div {{
    background: {CARD} !important; border-color: {BORDER} !important;
    border-radius: 10px !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
    background: rgba(0,212,170,0.12) !important;
    border: 1px solid rgba(0,212,170,0.35) !important;
    color: {ACCENT} !important;
}}
div[data-testid="stDateInput"] > div > div {{
    background: {CARD} !important; border-color: {BORDER} !important;
    border-radius: 10px !important;
}}

/* Buttons */
div[data-testid="stButton"] > button {{
    background: transparent;
    border: 1px solid {BORDER};
    color: #e7ebf3;
    border-radius: 10px;
    font-weight: 600;
    font-size: 13px;
    padding: 6px 14px;
    transition: all .15s ease;
    width: 100%;
}}
div[data-testid="stButton"] > button:hover {{
    background: rgba(0,212,170,0.10);
    border-color: {ACCENT};
    color: {ACCENT};
}}

/* Picks table */
.picks-table {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
.picks-table th {{
    text-align: left; font-size: 11px; font-weight: 700;
    letter-spacing: 0.10em; text-transform: uppercase;
    color: {MUTED}; padding: 10px 14px; border-bottom: 1px solid {BORDER};
}}
.picks-table td {{
    padding: 12px 14px; font-size: 13.5px; color: #e7ebf3;
    border-bottom: 1px solid {BORDER};
}}
.picks-table tr:last-child td {{ border-bottom: none; }}
.picks-table .sym-cell {{ font-weight: 700; color: #fff; letter-spacing: 0.02em; }}
.picks-table .industry {{ color: {MUTED}; font-size: 12px; }}
.picks-table .pos {{ color: {ACCENT}; font-weight: 600; }}
.picks-table .neg {{ color: {ACCENT_RED}; font-weight: 600; }}
.picks-table .num {{ font-variant-numeric: tabular-nums; }}
.picks-table .rank {{
    display:inline-block; min-width: 22px; text-align:center;
    padding: 2px 6px; border-radius: 6px; font-size: 11px; font-weight: 700;
    background: rgba(255,255,255,0.04); color: {MUTED};
}}

/* Divider */
hr {{ margin: 1.1rem 0; border: none; border-top: 1px solid {BORDER}; }}

/* Caption */
.muted {{ color: {MUTED}; font-size: 12.5px; }}
</style>
"""

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, system-ui, sans-serif", color="#e7ebf3"),
)

# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────
class APIError(RuntimeError):
    pass


@st.cache_resource
def get_session() -> requests.Session:
    """Single keep-alive HTTP session for all API calls."""
    cfg = st.secrets["api"]
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    if cfg.get("api_key"):
        s.headers["x-api-key"] = cfg["api_key"]
    return s


def _api_get(path: str, params: Optional[dict] = None) -> dict:
    cfg = st.secrets["api"]
    base = cfg["base_url"].rstrip("/")
    url = f"{base}{path}"
    timeout = float(cfg.get("timeout", 15))
    try:
        r = get_session().get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise APIError(f"Network error calling {path}: {e}") from e
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {}).get("message") or r.text[:300]
        except Exception:
            err = r.text[:300]
        hint = ""
        if r.status_code == 403:
            hint = "  (hint: API key may be required — check [api].api_key in .streamlit/secrets.toml)"
        elif r.status_code == 404:
            hint = "  (hint: path or symbol not found — check API base_url and arguments)"
        raise APIError(f"{r.status_code} {path}: {err}{hint}")
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (API-backed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def api_health() -> dict:
    return _api_get("/health")


@st.cache_data(ttl=3600, show_spinner=False)
def load_top_symbols(limit: int = 10) -> pd.DataFrame:
    """Top-N common stocks by market cap, used as the quick-pick dropdown.

    Single small screener call (was 500 rows; now 10 by default) so cold-start
    is fast. Anything the user types in the search bar is fetched on demand
    via `/market/ohlcv/{symbol}` and `/market/quote/{symbol}` (both cached).
    """
    payload = _api_get(
        "/screener/quotes",
        params={
            "type": "CS",
            "limit": min(max(limit, 1), 50),
            "offset": 0,
            "sort": "marketcap:desc",
        },
    )
    data = payload.get("data", []) or []
    if not data:
        return pd.DataFrame(columns=["symbol", "name", "industry", "market_cap",
                                     "primary_exchange"])
    df = pd.DataFrame(data)
    keep = ["symbol", "name", "industry", "market_cap", "primary_exchange"]
    return df[[c for c in keep if c in df.columns]].drop_duplicates("symbol")


@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    """Daily OHLCV for the most recent `days` calendar days, ascending.

    Uses an explicit `[start_date, end_date]` window. Without a window the API
    interprets `sort=asc&limit=N` as "the oldest N rows", which yields years-old
    data for any range > a few weeks. Sending the window guarantees the chart
    ends on the latest trading day.
    """
    today = date.today()
    start = today - timedelta(days=max(days, 1))
    payload = _api_get(
        f"/market/ohlcv/{symbol}",
        params={
            "interval": "1d",
            "start_date": start.isoformat(),
            "end_date": today.isoformat(),
            "limit": 2000,
            "sort": "asc",
        },
    )
    data = payload.get("data", []) or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df.get("trading_date", df.get("timestamp")))
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=300, show_spinner=False)
def load_index(symbol: str, days: int = 90) -> pd.DataFrame:
    df = load_ohlcv(symbol, days=days)
    if df.empty:
        return df
    return df[["date", "close"]]


@st.cache_data(ttl=600, show_spinner=False)
def load_symbol_meta(symbol: str) -> dict:
    """Per-symbol metadata. Tries the cached top-symbols list first as a free
    hit, otherwise calls /market/quote/{symbol}.
    """
    try:
        top = load_top_symbols()
        hit = top.loc[top["symbol"] == symbol] if not top.empty else top
        if not hit.empty:
            row = hit.iloc[0].to_dict()
            return {
                "name": row.get("name"),
                "industry": row.get("industry"),
                "marketcap": row.get("market_cap"),
                "primary_exchange": row.get("primary_exchange"),
            }
    except APIError:
        pass
    try:
        payload = _api_get(f"/market/quote/{symbol}")
        d = payload.get("data") or {}
        return {
            "name": d.get("name"),
            "industry": d.get("industry"),
            "marketcap": d.get("market_cap"),
            "primary_exchange": d.get("primary_exchange"),
        }
    except APIError:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_picks_today(limit: int = 100) -> dict:
    """Returns the raw `/picks/today` payload (data + meta)."""
    return _api_get("/picks/today", params={"limit": limit})


@st.cache_data(ttl=300, show_spinner=False)
def load_picks_returns(scan_dt: date, horizons: str = "1,5,21") -> dict:
    """Returns the raw `/picks/{scan_date}/returns` payload (data + meta)."""
    return _api_get(
        f"/picks/{scan_dt.isoformat()}/returns",
        params={"horizons": horizons},
    )


def _enrich_picks(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Attach name + industry + market_cap to each pick via parallel
    /market/quote lookups. `load_symbol_meta` is cached and tries the cached
    top-symbols list as a free fast-path before hitting the API.
    """
    if picks_df.empty:
        return picks_df

    syms = picks_df["symbol"].tolist()
    metas: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(syms)))) as ex:
        for sym, meta in zip(syms, ex.map(load_symbol_meta, syms)):
            metas[sym] = meta or {}

    out = picks_df.copy()
    out["name"] = [metas.get(s, {}).get("name") for s in syms]
    out["industry"] = [metas.get(s, {}).get("industry") for s in syms]
    out["marketcap"] = [metas.get(s, {}).get("marketcap") for s in syms]
    return out


def load_picks(scan_dt: date, strategy: Optional[str], limit: int = 25) -> pd.DataFrame:
    """Picks for a date, optionally filtered by strategy, with name/industry
    enriched and `return_pct` derived from the API's `return_to_date`.
    """
    payload = load_picks_returns(scan_dt)
    rows: list[dict[str, Any]] = payload.get("data", []) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if strategy:
        df = df[df["strategy_name"] == strategy]
    if df.empty:
        return df
    df = df.sort_values("rank").head(limit).reset_index(drop=True)

    df["pick_price"] = pd.to_numeric(df.get("pick_price"), errors="coerce")
    df["current_price"] = pd.to_numeric(df.get("close_now"), errors="coerce")
    df["return_pct"] = pd.to_numeric(df.get("return_to_date"), errors="coerce") * 100.0

    if "metadata" in df.columns:
        df["confidence"] = df["metadata"].apply(
            lambda m: (m or {}).get("ranking_score") if isinstance(m, dict) else None
        )
    else:
        df["confidence"] = np.nan
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df = _enrich_picks(df)
    return df[[
        "rank", "symbol", "signal", "strategy_name", "pick_price", "current_price",
        "return_pct", "confidence", "name", "industry", "marketcap",
    ]]


def discover_pick_context() -> tuple[Optional[date], list[str]]:
    """Use /picks/today to find the latest scan_date and known strategies."""
    try:
        payload = load_picks_today(limit=200)
    except APIError:
        return None, []
    meta = payload.get("meta") or {}
    rows = payload.get("data") or []
    scan_dt: Optional[date] = None
    sd = meta.get("scan_date")
    if sd:
        try:
            scan_dt = datetime.strptime(sd, "%Y-%m-%d").date()
        except ValueError:
            scan_dt = None
    strategies = sorted({r.get("strategy_name") for r in rows if r.get("strategy_name")})
    return scan_dt, strategies


# ─────────────────────────────────────────────────────────────────────────────
# Indicators (live, in-memory)
# ─────────────────────────────────────────────────────────────────────────────
INDICATOR_TYPES = {
    # name -> (kind, color)  kind: "overlay" or "subplot_rsi" or "subplot_macd" or "overlay_band"
    # SMAs
    "SMA 20":          ("overlay",      "#42a5f5"),
    "SMA 50":          ("overlay",      "#ab47bc"),
    "SMA 200":         ("overlay",      "#ff7043"),
    # Short EMAs
    "EMA 8":           ("overlay",      "#66bb6a"),
    "EMA 13":          ("overlay",      "#c0ca33"),
    "EMA 20":          ("overlay",      "#26c6da"),
    "EMA 50":          ("overlay",      "#ffca28"),
    # Long EMAs (Vegas channel)
    "EMA 144":         ("overlay",      "#e53935"),
    "EMA 169":         ("overlay",      "#7e57c2"),
    # Other overlays
    "VWAP":            ("overlay",      "#ec407a"),
    "Bollinger Bands": ("overlay_band", "#90caf9"),
    # Subplots
    "RSI 14":          ("subplot_rsi",  "#ffca28"),
    "MACD":            ("subplot_macd", "#26c6da"),
}
INDICATOR_NAMES = list(INDICATOR_TYPES.keys())
MAX_INDICATORS = 4

# ── Chart time-range presets ─────────────────────────────────────────────────
RANGE_OPTIONS = ["YTD", "1Y", "3Y", "5Y"]


def _range_to_days(label: str) -> int:
    """Map a preset label to a calendar-day window for /market/ohlcv?limit=…"""
    today = date.today()
    if label == "YTD":
        return max((today - date(today.year, 1, 1)).days, 30)
    if label == "1Y":
        return 365
    if label == "3Y":
        return 365 * 3
    if label == "5Y":
        return 365 * 5
    return 365


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def compute_indicator(df: pd.DataFrame, name: str) -> dict:
    close = df["close"]
    out: dict = {"name": name, "kind": INDICATOR_TYPES[name][0], "color": INDICATOR_TYPES[name][1]}
    if name == "SMA 20":
        out["series"] = _sma(close, 20)
    elif name == "SMA 50":
        out["series"] = _sma(close, 50)
    elif name == "SMA 200":
        out["series"] = _sma(close, 200)
    elif name == "EMA 8":
        out["series"] = _ema(close, 8)
    elif name == "EMA 13":
        out["series"] = _ema(close, 13)
    elif name == "EMA 20":
        out["series"] = _ema(close, 20)
    elif name == "EMA 50":
        out["series"] = _ema(close, 50)
    elif name == "EMA 144":
        out["series"] = _ema(close, 144)
    elif name == "EMA 169":
        out["series"] = _ema(close, 169)
    elif name == "VWAP":
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        out["series"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    elif name == "Bollinger Bands":
        mid = _sma(close, 20)
        std = close.rolling(20, min_periods=1).std()
        out["mid"], out["upper"], out["lower"] = mid, mid + 2 * std, mid - 2 * std
    elif name == "RSI 14":
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        out["series"] = 100 - (100 / (1 + rs))
    elif name == "MACD":
        ema12, ema26 = _ema(close, 12), _ema(close, 26)
        macd = ema12 - ema26
        signal = _ema(macd, 9)
        out["macd"], out["signal"], out["hist"] = macd, signal, macd - signal
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def make_price_chart(df: pd.DataFrame, symbol: str, indicators: list[str]) -> go.Figure:
    computed = [compute_indicator(df, n) for n in indicators]
    has_rsi = any(c["kind"] == "subplot_rsi" for c in computed)
    has_macd = any(c["kind"] == "subplot_macd" for c in computed)

    rows = 1 + int(has_rsi) + int(has_macd)
    heights = [0.66] + [0.17] * (rows - 1) if rows > 1 else [1.0]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=heights,
    )

    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=ACCENT, decreasing_line_color=ACCENT_RED,
        increasing_fillcolor=ACCENT, decreasing_fillcolor=ACCENT_RED,
        name=symbol, showlegend=False,
    ), row=1, col=1)

    for c in computed:
        if c["kind"] == "overlay":
            fig.add_trace(go.Scatter(
                x=df["date"], y=c["series"], mode="lines",
                line=dict(color=c["color"], width=1.4),
                name=c["name"],
            ), row=1, col=1)
        elif c["kind"] == "overlay_band":
            fig.add_trace(go.Scatter(
                x=df["date"], y=c["upper"], mode="lines",
                line=dict(color=c["color"], width=1, dash="dot"),
                name=f"{c['name']} U", showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df["date"], y=c["lower"], mode="lines",
                line=dict(color=c["color"], width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(144,202,249,0.07)",
                name=c["name"],
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df["date"], y=c["mid"], mode="lines",
                line=dict(color=c["color"], width=1.2),
                name=f"{c['name']} M", showlegend=False,
            ), row=1, col=1)

    sub_row = 1
    if has_rsi:
        sub_row += 1
        rsi = next(c for c in computed if c["kind"] == "subplot_rsi")
        fig.add_trace(go.Scatter(
            x=df["date"], y=rsi["series"], mode="lines",
            line=dict(color=rsi["color"], width=1.4), name="RSI 14",
        ), row=sub_row, col=1)
        for lvl, dash in [(70, "dot"), (30, "dot")]:
            fig.add_hline(y=lvl, line=dict(color="rgba(255,255,255,0.18)", width=1, dash=dash),
                          row=sub_row, col=1)
        fig.update_yaxes(range=[0, 100], row=sub_row, col=1, title_text="RSI",
                         title_font=dict(size=10, color=MUTED))
    if has_macd:
        sub_row += 1
        m = next(c for c in computed if c["kind"] == "subplot_macd")
        colors = np.where(m["hist"] >= 0, ACCENT, ACCENT_RED)
        fig.add_trace(go.Bar(x=df["date"], y=m["hist"], marker_color=colors,
                             name="MACD Hist", showlegend=False, opacity=0.55),
                      row=sub_row, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=m["macd"], mode="lines",
                                 line=dict(color=m["color"], width=1.4), name="MACD"),
                      row=sub_row, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=m["signal"], mode="lines",
                                 line=dict(color="#ffca28", width=1.2, dash="dot"), name="Signal"),
                      row=sub_row, col=1)
        fig.update_yaxes(title_text="MACD", row=sub_row, col=1,
                         title_font=dict(size=10, color=MUTED))

    fig.update_layout(
        height=520 if rows == 1 else 560 + 90 * (rows - 1),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", showline=False, zeroline=False)
    return fig


def make_index_line(df: pd.DataFrame, name: str, color: str) -> go.Figure:
    df = df.copy()
    df["pct"] = (df["close"] / df["close"].iloc[0] - 1) * 100.0
    last = df["pct"].iloc[-1]
    pos = last >= 0
    sign = "+" if pos else ""
    val_color = ACCENT if pos else ACCENT_RED
    fill_color = "rgba(0,212,170,0.10)" if pos else "rgba(255,82,103,0.10)"
    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["pct"], mode="lines",
        line=dict(color=color, width=1.6),
        fill="tozeroy", fillcolor=fill_color,
        hovertemplate="%{x|%b %d}<br>%{y:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=110,
        title=dict(
            text=f"<span style='color:#cfd6e4;font-weight:600'>{name}</span>"
                 f" &nbsp;<b style='color:{val_color}'>{sign}{last:.2f}%</b>",
            font=dict(size=12), x=0, xanchor="left",
        ),
        margin=dict(l=2, r=2, t=24, b=2),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, zeroline=False),
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_marketcap(mc: Optional[float]) -> str:
    if mc is None or pd.isna(mc):
        return "—"
    mc = float(mc)
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.2f}M"
    return f"${mc:,.0f}"


def render_meta_strip(symbol: str, meta: dict, df: pd.DataFrame) -> None:
    name = meta.get("name") or "—"
    industry = meta.get("industry") or "—"
    exch = meta.get("primary_exchange") or "—"
    mcap = _fmt_marketcap(meta.get("marketcap"))
    if df.empty or len(df) < 2:
        price_html = ""
    else:
        last = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]
        change = last - prev
        pct = change / prev * 100 if prev else 0
        cls = "delta-pos" if change >= 0 else "delta-neg"
        sign = "+" if change >= 0 else ""
        price_html = (
            f'<div class="price">${last:,.2f}'
            f'<span class="{cls}">{sign}{change:,.2f} ({sign}{pct:.2f}%)</span></div>'
        )
    st.markdown(
        f"""
        <div class="meta-strip">
          <span class="sym">{symbol}</span>
          <span class="name">{name}</span>
          <span class="pill">{exch}</span>
          <span class="pill">{industry}</span>
          <span class="pill">Mkt Cap {mcap}</span>
          {price_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_picks_table(picks: pd.DataFrame, scan_dt: date) -> None:
    if picks.empty:
        st.markdown(
            f'<div class="card"><span class="muted">'
            f'No picks found for {scan_dt:%b %d, %Y}.</span></div>',
            unsafe_allow_html=True,
        )
        return
    rows_html = []
    for _, r in picks.iterrows():
        ret = r["return_pct"]
        if pd.isna(ret):
            ret_html = '<span class="muted num">—</span>'
        else:
            cls = "pos" if ret >= 0 else "neg"
            sign = "+" if ret >= 0 else ""
            ret_html = f'<span class="{cls} num">{sign}{ret:.2f}%</span>'
        cur = r["current_price"]
        cur_html = f"${cur:,.2f}" if pd.notna(cur) else "—"
        conf = r["confidence"]
        conf_html = f"{conf*100:.1f}%" if pd.notna(conf) else "—"
        name = (r["name"] or "")[:42]
        industry = (r["industry"] or "—")[:32]
        rows_html.append(
            f"<tr>"
            f"<td><span class='rank'>#{int(r['rank'])}</span></td>"
            f"<td><div class='sym-cell'>{r['symbol']}</div>"
            f"<div class='industry'>{name}</div></td>"
            f"<td class='industry'>{industry}</td>"
            f"<td class='num'>${r['pick_price']:,.2f}</td>"
            f"<td class='num'>{cur_html}</td>"
            f"<td>{ret_html}</td>"
            f"<td class='num industry'>{conf_html}</td>"
            f"</tr>"
        )
    table_html = f"""
    <div class="card" style="padding:4px 4px;">
    <table class="picks-table">
      <thead><tr>
        <th></th><th>Symbol</th><th>Industry</th>
        <th>Pick Price</th><th>Current</th><th>Return</th><th>Confidence</th>
      </tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    cols = st.columns(min(len(picks), 8))
    for i, (_, r) in enumerate(picks.iterrows()):
        if i >= len(cols):
            break
        # Same symbol can appear under multiple strategies on the same date,
        # so the key must include row index + strategy + rank to stay unique.
        strat = (r.get("strategy_name") or "any")
        key = f"open_{i}_{r['symbol']}_{strat}_{int(r['rank'])}_{scan_dt}"
        with cols[i]:
            if st.button(f"Open {r['symbol']}", key=key):
                st.session_state.symbol = r["symbol"]
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(STYLES, unsafe_allow_html=True)

    if "symbol" not in st.session_state:
        st.session_state.symbol = "AAPL"
    if "indicators" not in st.session_state:
        st.session_state.indicators = ["EMA 8", "EMA 13", "EMA 144", "EMA 169"]
    if "chart_range" not in st.session_state:
        st.session_state.chart_range = "1Y"

    # ── Brand header ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-title">TradLyte</div>
          <div class="brand-tag">Financial Analytics</div>
        </div>
        <div class="brand-sub">
          Discover what to buy, backed by strategy-driven picks. Search any US equity,
          overlay live technical indicators, and review the day's top signals with
          real-time return tracking from the moment they were picked.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Concurrent preload: indices + universe + main OHLCV + picks_today ────
    symbol = st.session_state.symbol
    chart_days = _range_to_days(st.session_state.chart_range)

    MARKET_PULSE = [
        ("Indices", [
            ("SPY",  "S&P 500",   ACCENT),
            ("QQQ",  "NASDAQ",    "#42a5f5"),
            ("DJIA", "Dow Jones", "#ab47bc"),
        ]),
        ("Commodities", [
            ("GLD", "Gold",      "#ffd54f"),
            ("SLV", "Silver",    "#bdbdbd"),
            ("USO", "Crude Oil", "#ff8a65"),
        ]),
    ]
    pulse_symbols = [s for _, items in MARKET_PULSE for (s, _, _) in items]

    with st.spinner("Loading market data…"):
        preload = _parallel({
            "top_symbols": (load_top_symbols, (), {"limit": 10}),
            "ohlcv":       (load_ohlcv, (symbol, chart_days), None),
            "picks_today": (load_picks_today, (), {"limit": 200}),
            **{f"idx:{s}": (load_index, (s, 90), None) for s in pulse_symbols},
        })

    def _df_or_empty(key: str) -> pd.DataFrame:
        v = preload.get(key)
        return v if isinstance(v, pd.DataFrame) else pd.DataFrame()

    # ── Top: Market pulse (left rail) + main analytics (right) ───────────────
    rail, main_col = st.columns([1, 4], gap="large")

    with rail:
        st.markdown('<p class="eyebrow">Market Pulse</p>', unsafe_allow_html=True)
        for group_label, items in MARKET_PULSE:
            st.markdown(f'<p class="rail-group">{group_label}</p>',
                        unsafe_allow_html=True)
            for sym, label, color in items:
                idx_df = _df_or_empty(f"idx:{sym}")
                if idx_df.empty:
                    st.markdown(f'<span class="muted">{label}: no data</span>',
                                unsafe_allow_html=True)
                    continue
                st.plotly_chart(make_index_line(idx_df, label, color),
                                use_container_width=True, key=f"idx_{sym}",
                                config={"displayModeBar": False})

    with main_col:
        top_symbols_df = _df_or_empty("top_symbols")
        top_symbols = (top_symbols_df["symbol"].tolist()
                       if not top_symbols_df.empty else [])

        def _on_search_submit():
            val = (st.session_state.get("sym_search") or "").strip().upper()
            if val and val != st.session_state.symbol:
                st.session_state.symbol = val

        def _on_quick_pick():
            v = st.session_state.get("sym_quick")
            if v and v != st.session_state.symbol:
                st.session_state.symbol = v

        c_search, c_quick, c_ind = st.columns([1.4, 1.2, 2.4], gap="medium")
        with c_search:
            st.text_input(
                "Search symbol",
                value="",
                placeholder="Type a ticker (e.g. AAPL) and press Enter",
                key="sym_search",
                on_change=_on_search_submit,
                help="Press Enter to load. Any US ticker works.",
            )
        with c_quick:
            quick_options = list(top_symbols)
            if st.session_state.symbol in quick_options:
                quick_index = quick_options.index(st.session_state.symbol)
            else:
                quick_index = 0
            st.selectbox(
                "Quick pick — top 10",
                options=quick_options or [st.session_state.symbol],
                index=quick_index,
                key="sym_quick",
                on_change=_on_quick_pick,
                help="Top 10 US common stocks by market cap.",
            )
        with c_ind:
            picked = st.multiselect(
                f"Indicators (max {MAX_INDICATORS})",
                INDICATOR_NAMES,
                default=st.session_state.indicators,
                key="ind_select",
                max_selections=MAX_INDICATORS,
            )
            st.session_state.indicators = picked

        # OHLCV was prefetched in parallel for the *current* range; refetch only
        # if the symbol or range has since changed (cache makes this cheap).
        df = _df_or_empty("ohlcv")
        if df.empty or st.session_state.symbol != symbol:
            try:
                df = load_ohlcv(st.session_state.symbol, days=chart_days)
            except APIError as e:
                df = pd.DataFrame()
                st.error(f"Couldn't load **{st.session_state.symbol}** — {e}")
        symbol = st.session_state.symbol
        try:
            meta = load_symbol_meta(symbol)
        except APIError:
            meta = {}
        render_meta_strip(symbol, meta, df)

        # Range presets — sit just above the chart, right-aligned visually
        _, c_range = st.columns([3, 2], gap="small")
        with c_range:
            new_range = st.radio(
                "Range",
                options=RANGE_OPTIONS,
                index=RANGE_OPTIONS.index(st.session_state.chart_range),
                horizontal=True,
                label_visibility="collapsed",
                key="chart_range_radio",
            )
            if new_range != st.session_state.chart_range:
                st.session_state.chart_range = new_range
                st.rerun()

        if df.empty:
            st.warning(f"No price data found for **{symbol}**.")
        else:
            st.plotly_chart(
                make_price_chart(df, symbol, st.session_state.indicators),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Bottom: Picks of the Day ─────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">Picks of the Day</div>
          <div class="section-sub">
            The day's top-ranked signals from each strategy, with the price at
            the moment they were picked and the live return-to-date so you can
            see how the call is playing out. Pick a date or strategy to dig in.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_dt, strategies = discover_pick_context()
    if latest_dt is None and not strategies:
        st.markdown('<span class="muted">No picks available yet.</span>',
                    unsafe_allow_html=True)
        return

    default_dt = latest_dt or date.today()
    strategy_options = ["All strategies"] + strategies
    topn_options = [10, 25, 50, 100]

    c_date, c_strat, c_count = st.columns([1.2, 1.5, 1], gap="medium")
    with c_date:
        scan_dt = st.date_input(
            "Pick date",
            value=default_dt,
            max_value=date.today(),
            format="YYYY-MM-DD",
            key="pick_date",
            help=f"Latest scan: {latest_dt.isoformat()}" if latest_dt else None,
        )
    with c_strat:
        strategy_label = st.selectbox(
            "Strategy",
            options=strategy_options,
            index=0,
            format_func=lambda s: s if s == "All strategies" else s.replace("_", " ").title(),
            key="pick_strategy",
        )
        strategy = None if strategy_label == "All strategies" else strategy_label
    with c_count:
        top_n = st.selectbox(
            "Top N", topn_options, index=topn_options.index(10), key="pick_topn",
        )

    picks = load_picks(scan_dt, strategy, limit=top_n)

    if not picks.empty:
        avg_ret = picks["return_pct"].dropna().mean()
        winners = (picks["return_pct"].dropna() > 0).sum()
        total = picks["return_pct"].notna().sum()
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        m1.markdown(
            f'<div class="card"><div class="eyebrow" style="margin-bottom:4px">Picks</div>'
            f'<div style="font-size:22px;font-weight:700">{len(picks)}</div></div>',
            unsafe_allow_html=True,
        )
        if pd.notna(avg_ret):
            sign = "+" if avg_ret >= 0 else ""
            color = ACCENT if avg_ret >= 0 else ACCENT_RED
        else:
            sign, color = "", "#fff"
            avg_ret = 0
        m2.markdown(
            f'<div class="card"><div class="eyebrow" style="margin-bottom:4px">Avg Return</div>'
            f'<div style="font-size:22px;font-weight:700;color:{color}">'
            f'{sign}{avg_ret:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
        win_rate = (winners / total * 100) if total else 0
        m3.markdown(
            f'<div class="card"><div class="eyebrow" style="margin-bottom:4px">Win Rate</div>'
            f'<div style="font-size:22px;font-weight:700">{win_rate:.0f}%'
            f'<span style="font-size:13px;color:{MUTED};font-weight:500"> &nbsp;{winners}/{total}</span></div></div>',
            unsafe_allow_html=True,
        )
        days_held = (date.today() - scan_dt).days
        m4.markdown(
            f'<div class="card"><div class="eyebrow" style="margin-bottom:4px">Days Held</div>'
            f'<div style="font-size:22px;font-weight:700">{days_held}'
            f'<span style="font-size:13px;color:{MUTED};font-weight:500"> &nbsp;days</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    render_picks_table(picks, scan_dt)


if __name__ == "__main__":
    main()
