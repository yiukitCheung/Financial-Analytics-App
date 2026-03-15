import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CondVest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Connections ───────────────────────────────────────────────────────────────
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

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_symbols() -> list:
    with get_pg().cursor() as cur:
        cur.execute(
            "SELECT DISTINCT symbol FROM symbol_metadata WHERE type = 'CS' ORDER BY symbol"
        )
        return [row[0] for row in cur.fetchall()]

@st.cache_data(ttl=300)
def load_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    since = datetime.now() - timedelta(days=days)
    with get_pg().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT timestamp AS date, open, high, low, close
            FROM raw_ohlcv
            WHERE symbol = %s AND timestamp >= %s AND interval = '1d'
            ORDER BY timestamp ASC
            """,
            (symbol, since),
        )
        df = pd.DataFrame(cur.fetchall())
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
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
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=600)
def load_picks() -> list:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    with get_pg().cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT symbol
            FROM stock_picks
            WHERE created_at >= %s
            ORDER BY symbol
            """,
            (today,),)
        return [row[0] for row in cur.fetchall()]

# ── Chart builders ────────────────────────────────────────────────────────────
def make_candle(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        name=symbol
    ))
    fig.update_layout(
        height=500,
        template="plotly_dark",
        title=dict(text=symbol, font=dict(size=22, color="#ffffff")),
        xaxis_rangeslider_visible=False,
        xaxis_title=None,
        yaxis_title="Price (USD)",
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", showline=False)
    return fig

# fill colour map keyed by line colour
_FILLS = {
    "#26a69a": "rgba(38,166,154,0.10)",
    "#42a5f5": "rgba(66,165,245,0.10)",
    "#ab47bc": "rgba(171,71,188,0.10)",
    "#ff7043": "rgba(255,112,67,0.10)",
}

def make_index_line(df: pd.DataFrame, name: str, color: str) -> go.Figure:
    df = df.copy()
    df["pct"] = (df["close"] / df["close"].iloc[0] - 1) * 100
    last = df["pct"].iloc[-1]
    sign = "+" if last >= 0 else ""
    val_color = "#26a69a" if last >= 0 else "#ef5350"

    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["pct"],
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=_FILLS.get(color, "rgba(255,255,255,0.05)"),
    ))
    fig.update_layout(
        height=115,
        template="plotly_dark",
        title=dict(
            text=f"{name} &nbsp;<b style='color:{val_color}'>{sign}{last:.1f}%</b>",
            font=dict(size=12)
        ),
        margin=dict(l=0, r=0, t=28, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── CSS ───────────────────────────────────────────────────────────────────────
STYLES = """
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 0; }

/* Pick buttons */
div[data-testid="stButton"] > button {
    background: #0e1117;
    border: 1px solid #26a69a;
    color: #26a69a;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    padding: 4px 10px;
    transition: all .18s ease;
}
div[data-testid="stButton"] > button:hover {
    background: #26a69a;
    color: #0e1117;
}

/* Divider thinning */
hr { margin: 0.6rem 0; border-color: rgba(255,255,255,0.08); }

/* Market pulse header */
.pulse-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    margin-bottom: 2px;
}

/* Picks row label */
.picks-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    margin-bottom: 4px;
}
</style>
"""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(STYLES, unsafe_allow_html=True)

    # Session state
    if "symbol" not in st.session_state:
        st.session_state.symbol = "AAPL"

    # ── Header: logo + search ─────────────────────────────────────────────────
    logo_col, search_col = st.columns([1, 6])
    with logo_col:
        st.image("logo.png", width=110)
    with search_col:
        symbols = load_symbols()
        safe_idx = symbols.index(st.session_state.symbol) \
            if st.session_state.symbol in symbols else 0
        chosen = st.selectbox(
            "symbol", symbols,
            index=safe_idx,
            label_visibility="collapsed",
            key="sym_select"
        )
        if chosen != st.session_state.symbol:
            st.session_state.symbol = chosen

    st.divider()

    # ── Middle: candlestick (left) + market pulse (right) ────────────────────
    chart_col, index_col = st.columns([3, 1], gap="medium")

    with chart_col:
        df = load_ohlcv(st.session_state.symbol)
        if df.empty:
            st.warning(f"No price data found for **{st.session_state.symbol}**.")
        else:
            st.plotly_chart(
                make_candle(df, st.session_state.symbol),
                use_container_width=True
            )

    with index_col:
        st.markdown('<p class="pulse-header">Market Pulse</p>',
                    unsafe_allow_html=True)
        for sym, label, color in [
            ("^GSPC", "S&P 500",      "#26a69a"),
            ("^IXIC", "NASDAQ",       "#42a5f5"),
            ("^DJI",  "Dow Jones",    "#ab47bc"),
            ("^RUT",  "Russell 2000", "#ff7043"),
        ]:
            idx_df = load_index(sym)
            if not idx_df.empty:
                st.plotly_chart(
                    make_index_line(idx_df, label, color),
                    use_container_width=True,
                    key=f"idx_{sym}"
                )

    st.divider()

    # ── Bottom: picks of the day ──────────────────────────────────────────────
    st.markdown('<p class="picks-label">📌 Stock Picks of the Day</p>',
                unsafe_allow_html=True)
    picks = load_picks()
    if not picks:
        st.caption("No picks yet today — check back after market open.")
    else:
        cols = st.columns(min(len(picks), 12))
        for i, pick in enumerate(picks[:12]):
            with cols[i]:
                if st.button(pick, key=f"pick_{pick}"):
                    st.session_state.symbol = pick
                    st.rerun()


if __name__ == "__main__":
    main()
