"""CondVest — Financial Analytics App

A light, focused dashboard for discovering what to buy based on saved
strategy picks, with live technical-indicator analysis on any US equity.
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from plotly.subplots import make_subplots
from psycopg2.extras import RealDictCursor

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CondVest — Financial Analytics",
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
# DB
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_pg():
    cfg = st.secrets["postgres"]
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["db_name_postgres"],
        user=cfg["user"],
        password=cfg["password"],
        sslmode="require",
    )
    conn.autocommit = True
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_symbols() -> list[str]:
    with get_pg().cursor() as cur:
        cur.execute(
            "SELECT DISTINCT symbol FROM symbol_metadata WHERE type='CS' ORDER BY symbol"
        )
        return [r[0] for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    since = datetime.now() - timedelta(days=days)
    with get_pg().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT timestamp AS date, open, high, low, close, volume
            FROM raw_ohlcv
            WHERE symbol = %s AND timestamp >= %s AND interval = '1d'
            ORDER BY timestamp ASC
            """,
            (symbol, since),
        )
        df = pd.DataFrame(cur.fetchall())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


@st.cache_data(ttl=300)
def load_index(symbol: str, days: int = 90) -> pd.DataFrame:
    since = datetime.now() - timedelta(days=days)
    with get_pg().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT timestamp AS date, close
            FROM raw_ohlcv
            WHERE symbol = %s AND timestamp >= %s AND interval = '1d'
            ORDER BY timestamp ASC
            """,
            (symbol, since),
        )
        df = pd.DataFrame(cur.fetchall())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    return df


@st.cache_data(ttl=600)
def load_symbol_meta(symbol: str) -> dict:
    with get_pg().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT name, industry, marketcap, primary_exchange
            FROM symbol_metadata WHERE symbol = %s LIMIT 1
            """,
            (symbol,),
        )
        row = cur.fetchone()
        return dict(row) if row else {}


@st.cache_data(ttl=300)
def load_pick_dates(limit: int = 60) -> list[date]:
    with get_pg().cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT scan_date FROM stock_picks
            ORDER BY scan_date DESC LIMIT %s
            """,
            (limit,),
        )
        return [r[0] for r in cur.fetchall()]


@st.cache_data(ttl=600)
def load_strategies() -> list[str]:
    with get_pg().cursor() as cur:
        cur.execute("SELECT DISTINCT strategy_name FROM stock_picks ORDER BY strategy_name")
        return [r[0] for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_picks(scan_dt: date, strategy: str, limit: int = 25) -> pd.DataFrame:
    """Picks for a date + strategy, joined with symbol metadata + latest close.

    Return calc note:  uses stored `price` (pick-day price) vs latest close from
    raw_ohlcv. To swap in your SQL procedure later, replace the `latest_close`
    CTE with: `LEFT JOIN <your_proc>(symbol, scan_date) ...`.
    """
    sql = """
        WITH p AS (
            SELECT symbol, strategy_name, signal, price, confidence, rank, scan_date
            FROM stock_picks
            WHERE scan_date = %(d)s AND strategy_name = %(s)s
            ORDER BY rank ASC
            LIMIT %(lim)s
        ),
        latest_close AS (
            SELECT DISTINCT ON (symbol) symbol, close, timestamp
            FROM raw_ohlcv
            WHERE interval='1d' AND symbol IN (SELECT symbol FROM p)
            ORDER BY symbol, timestamp DESC
        )
        SELECT
            p.rank, p.symbol, p.signal, p.price::float AS pick_price,
            p.confidence::float AS confidence,
            COALESCE(lc.close::float, NULL) AS current_price,
            sm.name, sm.industry, sm.marketcap
        FROM p
        LEFT JOIN latest_close lc ON lc.symbol = p.symbol
        LEFT JOIN symbol_metadata sm ON sm.symbol = p.symbol
        ORDER BY p.rank ASC
    """
    with get_pg().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, {"d": scan_dt, "s": strategy, "lim": limit})
        df = pd.DataFrame(cur.fetchall())
    if df.empty:
        return df
    df["return_pct"] = np.where(
        df["current_price"].notna() & (df["pick_price"] > 0),
        (df["current_price"] - df["pick_price"]) / df["pick_price"] * 100.0,
        np.nan,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Indicators (live, in-memory)
# ─────────────────────────────────────────────────────────────────────────────
INDICATOR_TYPES = {
    # name -> (kind, color)  kind: "overlay" or "subplot_rsi" or "subplot_macd" or "overlay_band"
    "SMA 20":          ("overlay",      "#42a5f5"),
    "SMA 50":          ("overlay",      "#ab47bc"),
    "SMA 200":         ("overlay",      "#ff7043"),
    "EMA 20":          ("overlay",      "#26c6da"),
    "EMA 50":          ("overlay",      "#ffca28"),
    "VWAP":            ("overlay",      "#ec407a"),
    "Bollinger Bands": ("overlay_band", "#90caf9"),
    "RSI 14":          ("subplot_rsi",  "#ffca28"),
    "MACD":            ("subplot_macd", "#26c6da"),
}
INDICATOR_NAMES = list(INDICATOR_TYPES.keys())
MAX_INDICATORS = 4


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
    elif name == "EMA 20":
        out["series"] = _ema(close, 20)
    elif name == "EMA 50":
        out["series"] = _ema(close, 50)
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
        with cols[i]:
            if st.button(f"Open {r['symbol']}", key=f"open_{r['symbol']}_{scan_dt}"):
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
        st.session_state.indicators = ["SMA 20", "SMA 50"]

    # ── Brand header ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-title">CondVest</div>
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

    # ── Top: Market pulse (left rail) + main analytics (right) ───────────────
    rail, main_col = st.columns([1, 4], gap="large")

    with rail:
        st.markdown('<p class="eyebrow">Market Pulse</p>', unsafe_allow_html=True)
        for sym, label, color in [
            ("^GSPC", "S&P 500",    ACCENT),
            ("^IXIC", "NASDAQ",     "#42a5f5"),
            ("^DJI",  "Dow Jones",  "#ab47bc"),
        ]:
            idx_df = load_index(sym)
            if idx_df.empty:
                st.markdown(f'<span class="muted">{label}: no data</span>',
                            unsafe_allow_html=True)
                continue
            st.plotly_chart(make_index_line(idx_df, label, color),
                            use_container_width=True, key=f"idx_{sym}",
                            config={"displayModeBar": False})

    with main_col:
        # Search + indicator controls
        c_search, c_ind = st.columns([2, 3], gap="medium")
        with c_search:
            symbols = load_symbols()
            if st.session_state.symbol not in symbols:
                symbols = [st.session_state.symbol] + symbols
            idx = symbols.index(st.session_state.symbol)
            chosen = st.selectbox(
                "Search symbol",
                symbols,
                index=idx,
                key="sym_select",
                help="Type to search any US common stock",
            )
            if chosen != st.session_state.symbol:
                st.session_state.symbol = chosen
                st.rerun()
        with c_ind:
            picked = st.multiselect(
                f"Indicators (max {MAX_INDICATORS})",
                INDICATOR_NAMES,
                default=st.session_state.indicators,
                key="ind_select",
                max_selections=MAX_INDICATORS,
            )
            st.session_state.indicators = picked

        symbol = st.session_state.symbol
        df = load_ohlcv(symbol, days=365)
        meta = load_symbol_meta(symbol)
        render_meta_strip(symbol, meta, df)

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
    st.markdown('<p class="eyebrow">Picks of the Day</p>', unsafe_allow_html=True)

    pick_dates = load_pick_dates()
    strategies = load_strategies()
    if not pick_dates or not strategies:
        st.markdown('<span class="muted">No picks available yet.</span>',
                    unsafe_allow_html=True)
        return

    c_date, c_strat, c_count = st.columns([1.2, 1.5, 1], gap="medium")
    with c_date:
        scan_dt = st.selectbox(
            "Pick date",
            options=pick_dates,
            index=0,
            format_func=lambda d: d.strftime("%a, %b %d, %Y") + (
                "  • latest" if d == pick_dates[0] else ""
            ),
            key="pick_date",
        )
    with c_strat:
        strategy = st.selectbox(
            "Strategy",
            options=strategies,
            index=0,
            format_func=lambda s: s.replace("_", " ").title(),
            key="pick_strategy",
        )
    with c_count:
        top_n = st.selectbox("Top N", [10, 25, 50, 100], index=1, key="pick_topn")

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
