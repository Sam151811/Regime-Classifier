"""
ui.py  —  Visual layer for the Market Regime Classifier
All CSS injection, chart layout defaults, and HTML component builders live here.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


# Design tokens — colours, typography, and chart defaults

PALETTE = {
    "bg":          "#02040a",
    "bg1":         "#060b14",
    "bg2":         "#080f1c",
    "bg3":         "#0c1525",
    "border":      "#0f2040",
    "border_hi":   "#1a3560",
    "text":        "#8aafd4",
    "text_hi":     "#c5deff",
    "text_dim":    "#3a5578",
    "accent":      "#0affef",
    "accent2":     "#005eff",
    "accent3":     "#ff2d6b",
    "grid":        "#0a1828",
    "grid_hi":     "#0d2035",
}

REGIME_LABELS = [
    ("Bull Trend",      "#00ffaa"),
    ("Bear Trend",      "#ff2d6b"),
    ("High Volatility", "#ffb800"),
    ("Sideways Range",  "#4d9fff"),
    ("Breakout",        "#0affef"),
]

# Base layout dict applied to every Plotly figure
CHART_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(
        color=PALETTE["text"],
        family="'IBM Plex Mono', monospace",
        size=10,
    ),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickcolor=PALETTE["text_dim"],
    ),
    yaxis=dict(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickcolor=PALETTE["text_dim"],
    ),
)


# Global CSS — injected once on app startup

def inject_css() -> None:
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;600;700;900&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Rajdhani', sans-serif;
    background: #02040a;
    color: #8aafd4;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(0,94,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(10,255,239,0.03) 0%, transparent 50%),
        #02040a;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(10,255,239,0.006) 2px,
        rgba(10,255,239,0.006) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #030810 0%, #020509 100%);
    border-right: 1px solid #0f2040;
    backdrop-filter: blur(20px);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

h1 {
    font-family: 'Orbitron', monospace !important;
    color: #c5deff !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    background: linear-gradient(135deg, #c5deff 0%, #0affef 50%, #4d9fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-family: 'Orbitron', monospace !important;
    color: #4d9fff !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    margin: 1.5rem 0 0.75rem !important;
}

h3, h4 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #8aafd4 !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #060b14 0%, #080f1c 100%);
    border: 1px solid #0f2040;
    border-top: 1px solid #1a3560;
    border-radius: 0;
    padding: 14px 16px;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
    transition: border-color 0.2s;
}

[data-testid="metric-container"]:hover {
    border-color: #1a3560;
}

[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 10px; height: 10px;
    background: #0f2040;
    clip-path: polygon(0 0, 100% 100%, 100% 0);
}

[data-testid="metric-container"] label {
    color: #3a5578 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #c5deff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
}

.stButton > button {
    background: transparent !important;
    color: #0affef !important;
    border: 1px solid #0affef !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    transition: all 0.15s ease !important;
    position: relative !important;
}

.stButton > button:hover {
    background: rgba(10,255,239,0.08) !important;
    box-shadow: 0 0 20px rgba(10,255,239,0.2), inset 0 0 20px rgba(10,255,239,0.04) !important;
    transform: none !important;
}

.stButton > button:active {
    background: rgba(10,255,239,0.15) !important;
}

.stTextInput input, .stSelectbox select,
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background: #06101e !important;
    border: 1px solid #0f2040 !important;
    border-radius: 0 !important;
    color: #8aafd4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #0affef !important;
    box-shadow: 0 0 0 1px rgba(10,255,239,0.3) !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: #0affef !important;
}

div[data-testid="stSlider"] > div > div > div[role="slider"] {
    background: #0affef !important;
    border: 2px solid #02040a !important;
    box-shadow: 0 0 8px rgba(10,255,239,0.5) !important;
}

div[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #8aafd4 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #0f2040 !important;
    gap: 0 !important;
    padding: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #3a5578 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #8aafd4 !important;
    background: rgba(10,255,239,0.03) !important;
}

.stTabs [aria-selected="true"] {
    color: #0affef !important;
    border-bottom: 2px solid #0affef !important;
    background: rgba(10,255,239,0.05) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

.stDataFrame {
    border: 1px solid #0f2040 !important;
}

.stDataFrame table {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
}

.stAlert {
    border-radius: 0 !important;
    border-left: 2px solid #0f2040 !important;
    background: #060b14 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #0affef, #005eff) !important;
    box-shadow: 0 0 8px rgba(10,255,239,0.4) !important;
}

hr { border-color: #0f2040 !important; margin: 1rem 0 !important; }

.stCaption {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem !important;
    color: #3a5578 !important;
    letter-spacing: 0.08em !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #02040a; }
::-webkit-scrollbar-thumb { background: #0f2040; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #1a3560; }

.sys-panel {
    background: linear-gradient(135deg, #060b14 0%, #040810 100%);
    border: 1px solid #0f2040;
    border-top: 1px solid #1a3560;
    padding: 16px 18px;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
}

.sys-panel::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 12px; height: 12px;
    background: #0f2040;
    clip-path: polygon(0 0, 100% 100%, 100% 0);
}

.sys-panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a5578;
    margin-bottom: 6px;
}

.sys-panel-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: #c5deff;
    line-height: 1.2;
}

.regime-chip {
    display: inline-block;
    padding: 2px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);
}

.section-rule {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #3a5578;
}

.section-rule::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #0f2040, transparent);
}

.info-sys {
    background: #040d1a;
    border-left: 2px solid #005eff;
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #8aafd4;
    line-height: 1.6;
}

.warn-sys {
    background: #100a04;
    border-left: 2px solid #ffb800;
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #8aafd4;
    line-height: 1.6;
}

.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a5578;
    margin: 16px 0 6px;
}

.header-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid #0f2040;
    margin-bottom: 20px;
}

.header-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #3a5578;
    border: 1px solid #0f2040;
    padding: 2px 8px;
}

.ticker-badge {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border: 1px solid;
    display: inline-block;
    margin: 2px;
    clip-path: polygon(4px 0%, 100% 0%, calc(100% - 4px) 100%, 0% 100%);
}

.wf-stat {
    background: #040810;
    border: 1px solid #0f2040;
    padding: 14px;
    text-align: center;
}

.wf-stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
}

.wf-stat-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3a5578;
    margin-top: 4px;
}

.wf-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3a5578;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# HTML component helpers — return strings for st.markdown(unsafe_allow_html=True)

def sys_panel(label: str, value: str, color: str = "#c5deff") -> str:
    return f"""
    <div class="sys-panel">
        <div class="sys-panel-label">{label}</div>
        <div class="sys-panel-value" style="color:{color}">{value}</div>
    </div>"""


def regime_chip(name: str, color: str, extra: str = "") -> str:
    return f"""
    <span class="regime-chip" style="background:{color}18;color:{color};border:1px solid {color}40">
        {name}{(" &nbsp;·&nbsp; " + extra) if extra else ""}
    </span>"""


def section_rule(title: str) -> str:
    return f'<div class="section-rule">{title}</div>'


def info_banner(msg: str) -> str:
    return f'<div class="info-sys">{msg}</div>'


def warn_banner(msg: str) -> str:
    return f'<div class="warn-sys">{msg}</div>'


def sidebar_label(txt: str) -> str:
    return f'<div class="sidebar-label">{txt}</div>'


def ticker_badge(ticker: str, color: str = "#4d9fff") -> str:
    return f'<span class="ticker-badge" style="color:{color};border-color:{color}60">{ticker}</span>'


def header(tickers: list[str]) -> None:
    badges = "".join(ticker_badge(t) for t in tickers)
    st.markdown(f"""
    <div class="header-bar">
        <div style="flex:1">
            <h1 style="margin:0;padding:0">REGIME CLASSIFIER</h1>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                        color:#3a5578;letter-spacing:0.15em;margin-top:4px">
                ML-DRIVEN MARKET STATE DETECTION SYSTEM
            </div>
        </div>
        <div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                        color:#3a5578;letter-spacing:0.1em;margin-bottom:4px">ACTIVE INSTRUMENTS</div>
            {badges}
        </div>
        <div class="header-tag">SYS v2.0</div>
    </div>
    """, unsafe_allow_html=True)


def wf_stat_card(label: str, value: str, sub: str = "", color: str = "#c5deff") -> str:
    return f"""
    <div class="wf-stat">
        <div class="wf-stat-val" style="color:{color}">{value}</div>
        <div class="wf-stat-lbl">{label}</div>
        {f'<div class="wf-sub">{sub}</div>' if sub else ''}
    </div>"""


# Chart builders — each returns a Plotly Figure ready for st.plotly_chart

def price_chart(df: pd.DataFrame, meta: dict) -> go.Figure:
    raw = meta["raw"].reindex(df.index)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.15, 0.15],
        vertical_spacing=0.01,
    )

    # Regime background bands
    for i in range(len(df) - 1):
        fig.add_vrect(
            x0=df.index[i], x1=df.index[i + 1],
            fillcolor=df["color"].iloc[i], opacity=0.06,
            line_width=0, row=1, col=1,
        )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=raw["Open"].squeeze(), high=raw["High"].squeeze(),
        low=raw["Low"].squeeze(),   close=raw["Close"].squeeze(),
        increasing_line_color="#00ffaa",   decreasing_line_color="#ff2d6b",
        increasing_fillcolor="rgba(0,255,170,0.12)",
        decreasing_fillcolor="rgba(255,45,107,0.12)",
        line_width=1,
        name="OHLC",
    ), row=1, col=1)

    # 20 and 50-day moving averages
    c = raw["Close"].squeeze()
    fig.add_trace(go.Scatter(
        x=df.index, y=c.rolling(20).mean(),
        mode="lines", line=dict(color="rgba(10,255,239,0.5)", width=1, dash="dot"),
        name="MA20",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=c.rolling(50).mean(),
        mode="lines", line=dict(color="rgba(77,159,255,0.4)", width=1, dash="dot"),
        name="MA50",
    ), row=1, col=1)

    # Regime colour strip
    fig.add_trace(go.Bar(
        x=df.index, y=[1] * len(df),
        marker_color=df["color"],
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}: " + df["regime"] + "<extra></extra>",
    ), row=2, col=1)

    # RSI with overbought/oversold reference lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"] * 100,
        mode="lines", line=dict(color="#ffb800", width=1),
        name="RSI", fill="tozeroy",
        fillcolor="rgba(255,184,0,0.05)",
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,45,107,0.4)",
                  line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,170,0.4)",
                  line_width=1, row=3, col=1)

    fig.update_layout(
        height=680, xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
            font=dict(family="IBM Plex Mono", size=9, color="#8aafd4"),
            x=0.01, y=0.99,
        ),
        **CHART_LAYOUT,
    )
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    return fig


def latent_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for regime in df["regime"].unique():
        sub   = df[df["regime"] == regime]
        color = sub["color"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            marker=dict(
                size=5, color=color, opacity=0.65,
                line=dict(width=0),
            ),
            name=regime,
            hovertemplate=f"<b>{regime}</b><br>dim1: %{{x:.3f}}<br>dim2: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        height=480, title="LATENT SPACE PROJECTION",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8aafd4")),
        **CHART_LAYOUT,
    )
    return fig


def loss_chart(losses: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=losses, mode="lines",
        line=dict(color="#0affef", width=1.5),
        fill="tozeroy", fillcolor="rgba(10,255,239,0.05)",
        hovertemplate="Epoch %{x}: loss=%{y:.5f}<extra></extra>",
        name="Recon Loss",
    ))
    fig.update_layout(
        height=360, title="AUTOENCODER TRAINING LOSS",
        xaxis_title="EPOCH", yaxis_title="MSE LOSS",
        **CHART_LAYOUT,
    )
    return fig


def transition_heatmap(tmat: pd.DataFrame) -> go.Figure:
    labels = list(tmat.columns)
    z      = tmat.values.tolist()
    text   = [[f"{v:.0%}" for v in row] for row in tmat.values]
    fig    = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text,
        texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono", size=10, color="#c5deff"),
        colorscale=[
            [0.0, "#02040a"],
            [0.3, "#060d1e"],
            [0.6, "#0a2045"],
            [1.0, "#0affef"],
        ],
        showscale=True,
        colorbar=dict(
            tickfont=dict(family="IBM Plex Mono", size=9, color="#8aafd4"),
            outlinecolor="#0f2040", outlinewidth=1,
        ),
        hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{text}<extra></extra>",
    ))
    fig.update_layout(
        height=460, title="REGIME TRANSITION PROBABILITY MATRIX",
        xaxis_title="NEXT STATE", yaxis_title="CURRENT STATE",
        **CHART_LAYOUT,
    )
    return fig


def wf_chart(wf: dict, df: pd.DataFrame) -> go.Figure:
    cutoff   = wf["cutoff_idx"]
    test_lbl = wf["test_labels"]
    lmap     = wf["label_map"]

    test_df  = df.iloc[cutoff:cutoff + len(test_lbl)].copy()
    test_df["oos_regime"] = [lmap.get(l, "?") for l in test_lbl]
    color_lk = dict(zip(df["regime"], df["color"]))
    test_df["oos_color"]  = test_df["oos_regime"].map(
        lambda r: color_lk.get(r, "#888888")
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.02,
    )

    # Training window price
    train_df = df.iloc[:cutoff]
    fig.add_trace(go.Scatter(
        x=train_df.index, y=train_df["close"],
        mode="lines", line=dict(color="#3a5578", width=1.2),
        name="TRAIN", fill="tozeroy",
        fillcolor="rgba(58,85,120,0.05)",
    ), row=1, col=1)

    # Out-of-sample test window price
    fig.add_trace(go.Scatter(
        x=test_df.index, y=test_df["close"],
        mode="lines", line=dict(color="#0affef", width=1.5),
        name="TEST (OOS)", fill="tozeroy",
        fillcolor="rgba(10,255,239,0.04)",
    ), row=1, col=1)

    # Vertical line marking the train/test split
    split_date = df.index[cutoff]
    fig.add_shape(
        type="line",
        x0=split_date, x1=split_date,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#ffb800", dash="dash", width=1.2),
    )
    fig.add_annotation(
        x=split_date, y=1, yref="paper",
        text="TRAIN | OOS",
        showarrow=False, xanchor="left",
        font=dict(family="IBM Plex Mono", size=9, color="#ffb800"),
    )

    # OOS regime colour strip
    fig.add_trace(go.Bar(
        x=test_df.index, y=[1] * len(test_df),
        marker_color=test_df["oos_color"], showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}: " + test_df["oos_regime"] + "<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=520, title="WALK-FORWARD OUT-OF-SAMPLE REGIME DETECTION",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8aafd4")),
        **CHART_LAYOUT,
    )
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    return fig


def hmm_prob_chart(df: pd.DataFrame, state_probs: np.ndarray,
                   label_map: dict, regime_colors: dict) -> go.Figure:
    fig = go.Figure()
    n_states = state_probs.shape[1]

    for cid in range(n_states):
        name  = label_map.get(cid, f"State {cid}")
        color = regime_colors.get(name, "#888888")
        probs = state_probs[:len(df), cid]

        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_c  = f"rgba({r},{g},{b},0.5)"

        fig.add_trace(go.Scatter(
            x=df.index, y=probs, mode="lines",
            name=name, stackgroup="one",
            line=dict(width=0.6, color=color),
            fillcolor=fill_c,
        ))

    fig.update_layout(
        height=380, title="HMM STATE POSTERIOR PROBABILITIES",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8aafd4")),
        **CHART_LAYOUT,
    )
    return fig


def multi_price_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    palette = ["#0affef", "#00ffaa", "#ffb800", "#ff2d6b",
               "#a78bfa", "#4d9fff", "#f472b6"]
    for i, (ticker, (df, _)) in enumerate(results.items()):
        if df is None:
            continue
        norm = df["close"] / df["close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm, mode="lines",
            name=ticker, line=dict(width=1.6, color=palette[i % len(palette)]),
            hovertemplate=f"<b>{ticker}</b> %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.1)", line_width=1)
    fig.update_layout(
        height=440, title="NORMALISED PRICE PERFORMANCE  (BASE = 100)",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8aafd4")),
        **CHART_LAYOUT,
    )
    return fig


def regime_bar_chart(results: dict) -> go.Figure:
    regimes = [r for r, _ in REGIME_LABELS]
    palette = ["#0affef", "#00ffaa", "#ffb800", "#ff2d6b",
               "#a78bfa", "#4d9fff", "#f472b6"]
    fig = go.Figure()
    for i, (ticker, (df, _)) in enumerate(results.items()):
        if df is None:
            continue
        pcts = [
            df["regime"].value_counts(normalize=True).get(r, 0) * 100
            for r in regimes
        ]
        fig.add_trace(go.Bar(
            name=ticker, x=regimes, y=pcts,
            marker_color=palette[i % len(palette)],
            marker_line_width=0,
        ))
    fig.update_layout(
        barmode="group", height=420,
        title="REGIME ALLOCATION BY INSTRUMENT  (%)",
        yaxis_title="% DAYS",
        bargap=0.15, bargroupgap=0.04,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8aafd4")),
        **CHART_LAYOUT,
    )
    return fig


def feature_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of the technical feature matrix."""
    cols = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20",
            "trend", "ma_gap", "macd", "range_pos", "vol_spike",
            "rsi", "atr", "bb_width", "obv_roc"]
    sub  = df[[c for c in cols if c in df.columns]].dropna()
    corr = sub.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono", size=8),
        colorscale=[
            [0.0, "#ff2d6b"],
            [0.5, "#02040a"],
            [1.0, "#0affef"],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(
            tickfont=dict(family="IBM Plex Mono", size=9, color="#8aafd4"),
            outlinecolor="#0f2040", outlinewidth=1,
        ),
    ))
    fig.update_layout(
        height=480, title="FEATURE CORRELATION MATRIX",
        **CHART_LAYOUT,
    )
    return fig


def volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Annualised rolling volatility coloured by regime."""
    fig = go.Figure()
    ann_vol = df["vol_20"] * (252 ** 0.5) * 100

    # Base vol line in muted colour
    fig.add_trace(go.Scatter(
        x=df.index, y=ann_vol,
        mode="lines", line=dict(color="#3a5578", width=1),
        fill="tozeroy", fillcolor="rgba(58,85,120,0.08)",
        name="Ann. Vol",
    ))

    # Overlay each regime's portion in its own colour
    for regime in df["regime"].unique():
        sub   = df[df["regime"] == regime]
        color = sub["color"].iloc[0]
        rvol  = ann_vol.reindex(sub.index)
        fig.add_trace(go.Scatter(
            x=sub.index, y=rvol,
            mode="lines", line=dict(color=color, width=1.5),
            name=regime,
        ))

    fig.update_layout(
        height=360, title="ANNUALISED VOLATILITY BY REGIME  (%)",
        yaxis_title="VOL %",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0f2040", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8aafd4")),
        **CHART_LAYOUT,
    )
    return fig
