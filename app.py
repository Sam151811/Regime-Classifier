"""
app.py  —  Market Regime Classifier

Run with:  streamlit run app.py

Algorithms
    • KMeans + Autoencoder (PyTorch) / PCA fallback
    • Hidden Markov Model (hmmlearn)

Features
    • Yield curve slope proxy  : IEF / SHY spread (10Y-2Y)
    • Credit spread proxy      : LQD / HYG ratio  (IG vs HY)
    • Inflation hedge signal   : GLD momentum
    • Risk-off / rates signal  : TLT momentum
    • Equity risk premium proxy: SPY earnings yield vs 10Y (via price ratio)
    • Cross-asset momentum     : DXY (UUP), commodities (DJP)
    • Volatility regime        : VIX level + VIX momentum (^VIX via yfinance)
    • FRED (optional)          : 10Y-2Y spread, CPI YoY, Unemployment rate
    • RSI, ATR, Bollinger width, OBV momentum

Other features
    • Multi-ticker comparison
    • Walk-forward out-of-sample backtesting
    • Regime transition probability matrix
    • Per-regime Sharpe / drawdown / duration analytics
    • Feature correlation heatmap
    • Regime-coloured volatility chart
    • HMM state posterior time series
    • CSV + PDF export
"""

import warnings
warnings.filterwarnings("ignore")

import io
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from hmmlearn import hmm

import ui
from ui import REGIME_LABELS, PALETTE

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

MAX_REGIMES  = 5
RISK_FREE    = 0.05
TRAIN_FRAC   = 0.70

# Macro proxy tickers sourced via yfinance — no API key required
MACRO_TICKERS = {
    "vix":   "^VIX",     # Volatility index
    "tlt":   "TLT",      # 20Y Treasury  (rates / risk-off)
    "ief":   "IEF",      # 7-10Y Treasury
    "shy":   "SHY",      # 1-3Y Treasury  (IEF/SHY approximates yield curve slope)
    "hyg":   "HYG",      # High-yield credit
    "lqd":   "LQD",      # Investment-grade credit (LQD/HYG approximates credit stress)
    "gld":   "GLD",      # Gold  (inflation hedge / risk-off)
    "uup":   "UUP",      # US Dollar index
    "djp":   "DJP",      # Bloomberg Commodity index ETN
    "spy":   "SPY",      # S&P 500  (for cross-asset signals vs subject ticker)
}


st.set_page_config(
    page_title="Regime Classifier",
    page_icon=None,
    layout="wide",
)
ui.inject_css()


# Autoencoder — encodes the full feature set down to 2 latent dimensions

if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.LeakyReLU(0.1),
                nn.Linear(64, 32),        nn.LeakyReLU(0.1),
                nn.Linear(32, 16),        nn.LeakyReLU(0.1),
                nn.Linear(16, 2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(2, 16),          nn.LeakyReLU(0.1),
                nn.Linear(16, 32),         nn.LeakyReLU(0.1),
                nn.Linear(32, 64),         nn.LeakyReLU(0.1),
                nn.Linear(64, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z


# Technical indicators derived from OHLCV data

def _rsi(close: pd.Series, p: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()

    f = pd.DataFrame(index=df.index)

    f["ret_1"]  = c.pct_change()
    f["ret_5"]  = c.pct_change(5)
    f["ret_20"] = c.pct_change(20)

    f["vol_5"]  = f["ret_1"].rolling(5).std()
    f["vol_20"] = f["ret_1"].rolling(20).std()

    ma20        = c.rolling(20).mean()
    ma50        = c.rolling(50).mean()
    f["trend"]  = (c - ma20) / ma20
    f["ma_gap"] = (ma20 - ma50) / ma50

    ema12       = c.ewm(span=12).mean()
    ema26       = c.ewm(span=26).mean()
    f["macd"]   = (ema12 - ema26) / c

    h20            = h.rolling(20).max()
    l20            = l.rolling(20).min()
    f["range_pos"] = (c - l20) / (h20 - l20 + 1e-9)

    avg_v          = v.rolling(20).mean()
    f["vol_spike"] = (v / avg_v.replace(0, np.nan)).clip(upper=10)

    f["rsi"]       = _rsi(c) / 100.0
    f["atr"]       = _atr(h, l, c) / c
    f["bb_width"]  = (4 * c.rolling(20).std()) / ma20
    f["obv_roc"]   = _obv(c, v).pct_change(10).clip(-5, 5)

    return f.dropna()


# Macro feature engineering — cross-asset signals from ETF proxies

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_macro_data(period: str) -> pd.DataFrame:
    """
    Download macro proxy ETFs from yfinance and engineer cross-asset features.
    Returns a DataFrame indexed by date with all macro features.
    Falls back gracefully if individual tickers fail.
    """
    closes = {}
    for key, ticker in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if not raw.empty:
                closes[key] = raw["Close"].squeeze()
        except Exception:
            pass

    if not closes:
        return pd.DataFrame()

    prices = pd.DataFrame(closes).ffill().bfill()
    m = pd.DataFrame(index=prices.index)

    # Volatility regime: VIX level, z-score, and short-term momentum
    if "vix" in prices:
        vix = prices["vix"]
        m["vix_level"]    = vix / 100.0
        m["vix_zscore"]   = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-9)
        m["vix_momentum"] = vix.pct_change(5).clip(-1, 1)

    # Yield curve slope: IEF (7-10Y) vs SHY (1-3Y)
    # IEF outperforming SHY signals steepening (risk-on); reverse signals inversion (risk-off)
    if "ief" in prices and "shy" in prices:
        curve = (prices["ief"] / prices["shy"]).apply(np.log)
        m["yield_curve"]  = curve.diff(20)
        m["curve_zscore"] = (curve - curve.rolling(60).mean()) / (curve.rolling(60).std() + 1e-9)

    # Credit stress: LQD (IG) vs HYG (HY)
    # LQD outperforming HYG signals rising credit stress (risk-off)
    if "lqd" in prices and "hyg" in prices:
        credit = (prices["lqd"] / prices["hyg"]).apply(np.log)
        m["credit_spread"]  = credit.diff(20)
        m["credit_zscore"]  = (credit - credit.rolling(60).mean()) / (credit.rolling(60).std() + 1e-9)

    # Rates and risk-off signal via long-duration Treasuries (TLT)
    if "tlt" in prices:
        tlt = prices["tlt"]
        m["tlt_momentum"] = tlt.pct_change(20).clip(-0.3, 0.3)
        m["tlt_zscore"]   = (tlt - tlt.rolling(60).mean()) / (tlt.rolling(60).std() + 1e-9)

    # Inflation hedge signal via gold momentum
    if "gld" in prices:
        gld = prices["gld"]
        m["gold_momentum"] = gld.pct_change(20).clip(-0.3, 0.3)
        m["gold_zscore"]   = (gld - gld.rolling(60).mean()) / (gld.rolling(60).std() + 1e-9)

    # US Dollar momentum
    if "uup" in prices:
        uup = prices["uup"]
        m["dollar_momentum"] = uup.pct_change(20).clip(-0.3, 0.3)
        m["dollar_zscore"]   = (uup - uup.rolling(60).mean()) / (uup.rolling(60).std() + 1e-9)

    # Broad commodity momentum
    if "djp" in prices:
        djp = prices["djp"]
        m["commodity_momentum"] = djp.pct_change(20).clip(-0.3, 0.3)
        m["commodity_zscore"]   = (djp - djp.rolling(60).mean()) / (djp.rolling(60).std() + 1e-9)

    # Equity risk appetite: SPY momentum and breadth as a cross-asset signal
    if "spy" in prices:
        spy = prices["spy"]
        m["equity_momentum"] = spy.pct_change(20).clip(-0.3, 0.3)
        m["equity_breadth"]  = (spy.pct_change(5) > 0).rolling(20).mean()

    # Composite risk appetite score: combines credit, rates, and vol into one signal
    # Each component is inverted where high values indicate stress
    risk_components = []
    if "vix_zscore" in m:
        risk_components.append(-m["vix_zscore"])
    if "credit_zscore" in m:
        risk_components.append(-m["credit_zscore"])
    if "tlt_zscore" in m:
        risk_components.append(-m["tlt_zscore"])
    if risk_components:
        m["risk_appetite"] = pd.concat(risk_components, axis=1).mean(axis=1)

    return m.dropna()


def merge_macro_features(
    tech_feats: pd.DataFrame,
    macro_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align macro features to the technical feature index and merge.
    Macro features are forward-filled to handle minor date mismatches.
    """
    if macro_feats.empty:
        return tech_feats

    macro_aligned = macro_feats.reindex(tech_feats.index, method="ffill")
    merged = pd.concat([tech_feats, macro_aligned], axis=1)

    # Drop rows where more than 30% of macro columns are missing
    macro_cols = macro_feats.columns.tolist()
    threshold  = int(len(macro_cols) * 0.7)
    merged     = merged.dropna(subset=macro_cols, thresh=threshold)
    return merged.fillna(method="ffill").dropna()


# Human-readable macro context for display and reports

MACRO_REGIME_DESCRIPTIONS = {
    "Risk-On Growth":    "Low volatility, steepening curve, tight credit spreads, strong equity momentum",
    "Risk-Off / Flight": "VIX elevated, curve flattening/inverted, credit stress, TLT outperforming",
    "Inflationary":      "Gold & commodity momentum positive, dollar weakening, rates pressured",
    "Stagflation":       "Commodities up, equities weak, credit stress, high volatility",
    "Neutral / Transit": "Mixed signals across macro indicators, regime transition likely",
}

def describe_macro_regime(row: pd.Series, macro_cols: list) -> str:
    """
    Produce a human-readable macro context string from the latest feature values.
    Used in the UI and PDF export.
    """
    signals = []

    if "vix_level" in row and not pd.isna(row["vix_level"]):
        v = row["vix_level"] * 100
        signals.append(f"VIX {v:.1f} ({'elevated' if v > 20 else 'low'})")

    if "yield_curve" in row and not pd.isna(row["yield_curve"]):
        signals.append(f"Curve {'steepening' if row['yield_curve'] > 0 else 'flattening'}")

    if "credit_spread" in row and not pd.isna(row["credit_spread"]):
        signals.append(f"Credit {'stress' if row['credit_spread'] > 0 else 'calm'}")

    if "gold_momentum" in row and not pd.isna(row["gold_momentum"]):
        signals.append(f"Gold {row['gold_momentum']*100:+.1f}% (20d)")

    if "risk_appetite" in row and not pd.isna(row["risk_appetite"]):
        ra = row["risk_appetite"]
        signals.append(f"Risk appetite: {'positive' if ra > 0.3 else 'negative' if ra < -0.3 else 'neutral'}")

    return "  |  ".join(signals) if signals else "Macro data unavailable"


# Dimensionality reduction — autoencoder preferred, PCA as fallback

def compress(X: np.ndarray, epochs: int) -> tuple[np.ndarray, list[float], str]:
    if TORCH_AVAILABLE:
        t   = torch.tensor(X, dtype=torch.float32)
        mdl = Autoencoder(X.shape[1])
        opt = torch.optim.AdamW(mdl.parameters(), lr=0.001, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        fn  = nn.MSELoss()
        losses = []
        for _ in range(epochs):
            opt.zero_grad()
            rec, _ = mdl(t)
            loss   = fn(rec, t)
            loss.backward()
            opt.step()
            sch.step()
            losses.append(loss.item())
        with torch.no_grad():
            _, lat = mdl(t)
        return lat.numpy(), losses, "Autoencoder (PyTorch · AdamW + CosineAnnealing)"

    pca = PCA(n_components=2)
    return pca.fit_transform(X), [], "PCA (install torch for neural autoencoder)"


# Hidden Markov Model path — sequential probabilistic clustering

def run_hmm(X: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=42,
    )
    model.fit(X)
    labels      = model.predict(X)
    state_probs = model.predict_proba(X)
    return labels, state_probs


# Walk-forward out-of-sample backtest

def walk_forward(
    X: np.ndarray,
    returns: np.ndarray,
    latent: np.ndarray,
    n_clusters: int,
) -> dict:
    n       = len(latent)
    cutoff  = int(n * TRAIN_FRAC)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(latent[:cutoff])
    test_labels = km.predict(latent[cutoff:])

    rank = (
        pd.Series(returns[:cutoff])
        .groupby(km.predict(latent[:cutoff]))
        .median()
        .sort_values(ascending=False)
    )
    label_map = {cid: REGIME_LABELS[i][0] for i, cid in enumerate(rank.index)}

    ret_test   = returns[cutoff:]
    regime_fwd = {}
    for cid in range(n_clusters):
        mask = test_labels == cid
        if mask.sum() < 5:
            continue
        name = label_map.get(cid, f"Cluster {cid}")
        regime_fwd[name] = {
            "mean_fwd_ret": float(ret_test[mask].mean() * 252),
            "sharpe":       _sharpe(ret_test[mask]),
            "n_days":       int(mask.sum()),
            "hit_rate":     float((ret_test[mask] > 0).mean()),
        }

    return {
        "cutoff_idx":  cutoff,
        "regime_fwd":  regime_fwd,
        "test_labels": test_labels,
        "label_map":   label_map,
    }


# Performance analytics helpers

def _sharpe(r: np.ndarray, rfr: float = RISK_FREE) -> float:
    if len(r) < 2:
        return 0.0
    excess = r - rfr / 252
    std    = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _max_drawdown(r: np.ndarray) -> float:
    cum = (1 + r).cumprod()
    mx  = np.maximum.accumulate(cum)
    return float(((cum - mx) / mx).min())


def _avg_duration(df: pd.DataFrame, regime: str) -> float:
    mask   = (df["regime"] == regime).astype(int)
    starts = (mask.diff().fillna(0) == 1).sum() + (mask.iloc[0] == 1)
    return mask.sum() / max(starts, 1)


def regime_analytics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in df["regime"].unique():
        sub  = df[df["regime"] == regime]
        r    = sub["ret_1"].dropna().values
        rows.append({
            "Regime":       regime,
            "Color":        sub["color"].iloc[0],
            "Days":         len(sub),
            "Pct Time":     f"{len(sub)/len(df)*100:.1f}%",
            "Ann. Return":  f"{r.mean()*252*100:.1f}%",
            "Ann. Vol":     f"{r.std()*np.sqrt(252)*100:.1f}%",
            "Sharpe":       f"{_sharpe(r):.2f}",
            "Max Drawdown": f"{_max_drawdown(r)*100:.1f}%",
            "Avg Duration": f"{_avg_duration(df, regime):.0f}d",
        })
    return pd.DataFrame(rows)


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    regimes = df["regime"].values
    labels  = sorted(df["regime"].unique())
    counts  = pd.DataFrame(0, index=labels, columns=labels)
    for i in range(len(regimes) - 1):
        counts.loc[regimes[i], regimes[i + 1]] += 1
    rs = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(rs, axis=0).fillna(0)


# Average macro indicator values per regime — reveals macro fingerprint of each state

def macro_regime_analytics(df: pd.DataFrame, macro_cols: list) -> pd.DataFrame:
    available = [c for c in macro_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    rows = []
    for regime in df["regime"].unique():
        sub  = df[df["regime"] == regime]
        row  = {"Regime": regime, "Days": len(sub)}
        for col in available:
            row[col] = sub[col].mean()
        rows.append(row)

    return pd.DataFrame(rows).set_index("Regime")


# Main analysis pipeline

@st.cache_data(show_spinner=False)
def run_pipeline(
    ticker:     str,
    period:     str,
    n_clusters: int,
    epochs:     int,
    model_type: str,
    use_macro:  bool,
) -> tuple:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        return None, f"No data returned for '{ticker}'. Verify the symbol."

    tech_feats = build_features(raw)
    if len(tech_feats) < 60:
        return None, "Insufficient history. Select a longer period."

    macro_feats  = pd.DataFrame()
    macro_cols   = []
    macro_loaded = False

    if use_macro:
        macro_feats = fetch_macro_data(period)
        if not macro_feats.empty:
            combined   = merge_macro_features(tech_feats, macro_feats)
            macro_cols = [c for c in macro_feats.columns if c in combined.columns]
            if len(combined) >= 60:
                feats        = combined
                macro_loaded = True
            else:
                feats = tech_feats   # not enough date overlap — fall back to technical only
        else:
            feats = tech_feats
    else:
        feats = tech_feats

    scaler = StandardScaler()
    X      = scaler.fit_transform(feats)

    losses: list[float] = []
    state_probs         = None

    if model_type == "HMM":
        labels, state_probs = run_hmm(X, n_clusters)
        latent = PCA(n_components=2).fit_transform(X)
        method = "Hidden Markov Model  (GaussianHMM · full covariance)"
    else:
        latent, losses, method = compress(X, epochs)
        km     = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        labels = km.fit_predict(latent)

    if macro_loaded:
        method += "  +  Macro Features"

    feats = feats.copy()
    feats["cluster"] = labels
    feats["close"]   = raw["Close"].squeeze().reindex(feats.index)

    rank = (
        feats.groupby("cluster")["ret_20"]
        .median()
        .sort_values(ascending=False)
    )
    label_map       = {cid: REGIME_LABELS[r] for r, cid in enumerate(rank.index)}
    feats["regime"] = feats["cluster"].map(lambda c: label_map[c][0])
    feats["color"]  = feats["cluster"].map(lambda c: label_map[c][1])
    feats["x"]      = latent[:, 0]
    feats["y"]      = latent[:, 1]

    returns   = feats["ret_1"].fillna(0).values
    wf_result = walk_forward(X, returns, latent, n_clusters)
    score     = silhouette_score(latent, labels)

    meta = {
        "raw":          raw,
        "ticker":       ticker,
        "method":       method,
        "losses":       losses,
        "score":        score,
        "wf":           wf_result,
        "state_probs":  state_probs,
        "macro_loaded": macro_loaded,
        "macro_cols":   macro_cols,
        "macro_feats":  macro_feats,
    }
    return feats, meta


# PDF report builder

def build_pdf(
    ticker: str, df: pd.DataFrame, meta: dict,
    analytics: pd.DataFrame, tmat: pd.DataFrame,
) -> bytes:
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)
    sty  = getSampleStyleSheet()

    DARK  = rl_colors.HexColor("#030810")
    BLUE  = rl_colors.HexColor("#0affef")
    BLUE2 = rl_colors.HexColor("#4d9fff")
    LIGHT = rl_colors.HexColor("#c5deff")
    MID   = rl_colors.HexColor("#8aafd4")
    DIM   = rl_colors.HexColor("#3a5578")

    title_sty = ParagraphStyle("T", parent=sty["Title"],
        fontSize=18, textColor=BLUE,  spaceAfter=4)
    sub_sty   = ParagraphStyle("S", parent=sty["Normal"],
        fontSize=8,  textColor=DIM,   spaceAfter=14, fontName="Helvetica")
    h2_sty    = ParagraphStyle("H", parent=sty["Heading2"],
        fontSize=11, textColor=BLUE2, spaceBefore=16, spaceAfter=6)
    body_sty  = ParagraphStyle("B", parent=sty["Normal"],
        fontSize=8,  textColor=MID,   leading=13)

    def tbl_sty():
        return TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  rl_colors.HexColor("#060b14")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  BLUE2),
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#040810"), rl_colors.HexColor("#030710")]),
            ("TEXTCOLOR",      (0, 1), (-1, -1), LIGHT),
            ("GRID",           (0, 0), (-1, -1), 0.25, rl_colors.HexColor("#0f2040")),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 7),
            ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ])

    story = []

    story.append(Paragraph(f"MARKET REGIME REPORT  —  {ticker.upper()}", title_sty))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Model: {meta['method']}  |  "
        f"Silhouette: {meta['score']:.3f}  |  "
        f"{len(df):,} trading days  |  "
        f"Macro: {'YES' if meta['macro_loaded'] else 'NO'}",
        sub_sty,
    ))
    story.append(HRFlowable(width="100%", thickness=0.4,
                             color=rl_colors.HexColor("#0f2040")))
    story.append(Spacer(1, 10))

    cur = df.iloc[-1]
    story.append(Paragraph("Current Signal", h2_sty))
    story.append(Paragraph(
        f"<b>{cur['regime']}</b>  |  "
        f"Last close: ${cur['close']:.2f}  |  "
        f"1d return: {cur['ret_1']*100:+.2f}%  |  "
        f"RSI: {cur['rsi']*100:.1f}",
        body_sty,
    ))

    if meta["macro_loaded"] and meta["macro_cols"]:
        macro_ctx = describe_macro_regime(cur, meta["macro_cols"])
        story.append(Paragraph(f"Macro context: {macro_ctx}", body_sty))

    story.append(Spacer(1, 12))

    story.append(Paragraph("Per-Regime Analytics", h2_sty))
    cols = ["Regime", "Days", "Pct Time", "Ann. Return",
            "Ann. Vol", "Sharpe", "Max Drawdown", "Avg Duration"]
    rows = [cols] + [[str(row[c]) for c in cols] for _, row in analytics.iterrows()]
    t = Table(rows, repeatRows=1)
    t.setStyle(tbl_sty())
    story.append(t)
    story.append(Spacer(1, 12))

    if meta["macro_loaded"] and meta["macro_cols"]:
        story.append(Paragraph("Macro Fingerprint by Regime", h2_sty))
        story.append(Paragraph(
            "Average macro indicator value per regime. "
            "Reveals the macro environment characteristic of each state.",
            body_sty,
        ))
        story.append(Spacer(1, 5))
        key_macro = [c for c in [
            "vix_level", "yield_curve", "credit_spread",
            "gold_momentum", "tlt_momentum", "risk_appetite",
        ] if c in df.columns]
        if key_macro:
            mfp = macro_regime_analytics(df, key_macro)
            mfp_rows = [["Regime"] + key_macro] + [
                [idx] + [f"{mfp.loc[idx, c]:.3f}" if c in mfp.columns else "—"
                         for c in key_macro]
                for idx in mfp.index
            ]
            tm2 = Table(mfp_rows, repeatRows=1)
            tm2.setStyle(tbl_sty())
            story.append(tm2)
            story.append(Spacer(1, 12))

    story.append(Paragraph("Regime Transition Matrix", h2_sty))
    story.append(Paragraph("Row = current regime  |  Column = next regime  |  Values = probability", body_sty))
    story.append(Spacer(1, 5))
    tl    = list(tmat.columns)
    trows = [["→"] + tl] + [
        [r] + [f"{v:.0%}" for v in tmat.loc[r]] for r in tl
    ]
    tt = Table(trows, repeatRows=1)
    tt.setStyle(tbl_sty())
    story.append(tt)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Walk-Forward Out-of-Sample Results", h2_sty))
    story.append(Paragraph(
        f"Trained on first {int(TRAIN_FRAC*100)}% ({meta['wf']['cutoff_idx']} days); "
        f"evaluated on remaining {int((1-TRAIN_FRAC)*100)}%.",
        body_sty,
    ))
    story.append(Spacer(1, 5))
    wf_cols = ["Regime", "Days", "Ann. Return", "Sharpe", "Hit Rate"]
    wf_rows = [wf_cols] + [
        [n, str(s["n_days"]),
         f"{s['mean_fwd_ret']*100:.1f}%",
         f"{s['sharpe']:.2f}",
         f"{s['hit_rate']:.0%}"]
        for n, s in meta["wf"]["regime_fwd"].items()
    ]
    tw = Table(wf_rows, repeatRows=1)
    tw.setStyle(tbl_sty())
    story.append(tw)

    doc.build(story)
    return buf.getvalue()


def build_csv(df: pd.DataFrame, macro_cols: list = None) -> str:
    base_cols = ["close", "regime", "ret_1", "ret_5", "ret_20",
                 "vol_20", "rsi", "atr", "bb_width", "obv_roc"]
    extra = [c for c in (macro_cols or []) if c in df.columns]
    export = df[[c for c in base_cols + extra if c in df.columns]].copy()
    export.index.name = "date"
    return export.to_csv()


# Macro dashboard: 4-panel chart showing key indicators coloured by regime

def macro_dashboard_chart(df: pd.DataFrame, macro_feats: pd.DataFrame) -> go.Figure:
    panel_map = {
        "VIX Level":      "vix_level",
        "Yield Curve":    "yield_curve",
        "Credit Spread":  "credit_spread",
        "Risk Appetite":  "risk_appetite",
    }
    available = {k: v for k, v in panel_map.items() if v in df.columns}
    if not available:
        return go.Figure()

    n     = len(available)
    fig   = make_subplots(rows=n, cols=1, shared_xaxes=True,
                          subplot_titles=list(available.keys()),
                          vertical_spacing=0.06)

    for i, (label, col) in enumerate(available.items(), 1):
        series = df[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.values,
                mode="lines",
                line=dict(color="#4d9fff", width=1.2),
                name=label,
                showlegend=False,
            ),
            row=i, col=1,
        )
        # Overlay regime colour on each panel
        for regime in df["regime"].unique():
            mask   = df["regime"] == regime
            color  = df.loc[mask, "color"].iloc[0]
            regime_series = series.reindex(df.index[mask])
            fig.add_trace(
                go.Scatter(
                    x=regime_series.index,
                    y=regime_series.values,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    name=regime,
                    showlegend=(i == 1),
                    legendgroup=regime,
                ),
                row=i, col=1,
            )

    fig.update_layout(
        height=160 * n + 60,
        paper_bgcolor="#030810",
        plot_bgcolor="#040c18",
        font=dict(family="IBM Plex Mono", color="#8aafd4", size=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        gridcolor="#0a1a2e", zerolinecolor="#0a1a2e",
        tickfont=dict(size=8),
    )
    fig.update_yaxes(
        gridcolor="#0a1a2e", zerolinecolor="#0a1a2e",
        tickfont=dict(size=8),
    )
    return fig


# Grouped bar chart showing the average macro z-score fingerprint per regime

def macro_fingerprint_chart(df: pd.DataFrame, macro_cols: list) -> go.Figure:
    key_cols = [c for c in [
        "vix_zscore", "curve_zscore", "credit_zscore",
        "gold_momentum", "tlt_momentum", "dollar_momentum", "risk_appetite",
    ] if c in df.columns]

    if not key_cols:
        return go.Figure()

    mfp = macro_regime_analytics(df, key_cols)
    if mfp.empty:
        return go.Figure()

    fig = go.Figure()
    for regime in mfp.index:
        row   = mfp.loc[regime]
        color = df.loc[df["regime"] == regime, "color"].iloc[0]
        fig.add_trace(go.Bar(
            name=regime,
            x=key_cols,
            y=row[key_cols].values,
            marker_color=color,
            opacity=0.85,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="MACRO FINGERPRINT BY REGIME",
            font=dict(family="Orbitron, monospace", size=12, color="#0affef"),
        ),
        paper_bgcolor="#030810",
        plot_bgcolor="#040c18",
        font=dict(family="IBM Plex Mono", color="#8aafd4", size=10),
        xaxis=dict(gridcolor="#0a1a2e", tickangle=-30),
        yaxis=dict(gridcolor="#0a1a2e", title="Z-score / Normalised value"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=380,
        margin=dict(l=50, r=20, t=50, b=80),
    )
    return fig


# Sidebar configuration

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Orbitron\',monospace;font-size:0.75rem;'
        'font-weight:700;color:#0affef;letter-spacing:0.2em;'
        'text-transform:uppercase;margin-bottom:8px">SYSTEM CONFIG</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(ui.sidebar_label("Instruments"), unsafe_allow_html=True)
    raw_tickers = st.text_input(
        "Tickers",
        "SPY, QQQ",
        label_visibility="collapsed",
        help="Yahoo Finance symbols, comma-separated. e.g. SPY, QQQ, BTC-USD, GLD, AAPL",
    )
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    st.markdown(ui.sidebar_label("History Window"), unsafe_allow_html=True)
    period = st.selectbox(
        "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
        index=4, label_visibility="collapsed",
    )

    st.divider()

    st.markdown(ui.sidebar_label("Detection Algorithm"), unsafe_allow_html=True)
    model_type = st.radio(
        "Algorithm",
        ["KMeans + AE/PCA", "HMM"],
        label_visibility="collapsed",
        horizontal=True,
        help="KMeans clusters the compressed latent space. HMM is a sequential probabilistic model.",
    )

    st.markdown(ui.sidebar_label("Regime Count"), unsafe_allow_html=True)
    n_clusters = st.slider("Regimes", 2, MAX_REGIMES, 4, label_visibility="collapsed")

    st.markdown(ui.sidebar_label("AE Training Epochs"), unsafe_allow_html=True)
    epochs = st.slider(
        "Epochs", 50, 600, 200, step=50,
        label_visibility="collapsed",
        disabled=(model_type == "HMM"),
    )

    st.divider()

    st.markdown(ui.sidebar_label("Macro Features"), unsafe_allow_html=True)
    use_macro = st.toggle(
        "Include macro features",
        value=True,
        help=(
            "Adds cross-asset macro signals to the feature set: "
            "VIX, yield curve slope (IEF/SHY), credit spread (LQD/HYG), "
            "gold, TLT, dollar, commodities, and a risk appetite composite. "
            "All sourced from yfinance — no API key required."
        ),
    )
    if use_macro:
        st.markdown(
            ui.info_banner(
                "Macro proxies: VIX · IEF/SHY (curve) · LQD/HYG (credit) · "
                "GLD · TLT · UUP · DJP · SPY<br>"
                "All via yfinance — no API key needed."
            ),
            unsafe_allow_html=True,
        )

    if not TORCH_AVAILABLE:
        st.markdown(
            ui.info_banner(
                "PyTorch unavailable — using PCA.<br>"
                "Install <code>torch</code> for the neural autoencoder."
            ),
            unsafe_allow_html=True,
        )

    st.divider()
    run = st.button("RUN ANALYSIS")


# Landing state — shown before the user kicks off a run

if not run:
    ui.header(tickers)
    st.markdown(
        ui.info_banner(
            "Configure instruments and model parameters in the sidebar, "
            "then press <strong>RUN ANALYSIS</strong> to begin classification."
        ),
        unsafe_allow_html=True,
    )
    st.stop()


# Run the pipeline for each ticker

results: dict[str, tuple] = {}
prog = st.progress(0, text="Initialising…")

for idx, ticker in enumerate(tickers):
    prog.progress(idx / len(tickers), text=f"Processing {ticker}…")
    results[ticker] = run_pipeline(ticker, period, n_clusters, epochs, model_type, use_macro)

prog.progress(1.0, text="Classification complete.")
prog.empty()

valid  = {t: v for t, v in results.items() if v[0] is not None}
failed = {t: v[1] for t, v in results.items() if v[0] is None}

for t, err in failed.items():
    st.error(f"{t}: {err}")

if not valid:
    st.stop()


# Header and top-level metrics

primary_ticker      = list(valid.keys())[0]
primary_df, p_meta  = valid[primary_ticker]
cur                 = primary_df.iloc[-1]

ui.header(list(valid.keys()))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Instrument",       primary_ticker)
c2.metric("Detected Regime",  cur["regime"])
c3.metric("Last Close",       f"${cur['close']:.2f}")
c4.metric("1D Return",        f"{cur['ret_1']*100:+.2f}%")
c5.metric("Silhouette Score", f"{p_meta['score']:.3f}",
          help="Cluster quality: 1 = perfect separation, -1 = poor")
c6.metric("RSI",              f"{cur['rsi']*100:.1f}")

macro_tag = "  |  MACRO: ON" if p_meta["macro_loaded"] else "  |  MACRO: OFF"
st.caption(
    f"MODEL: {p_meta['method'].upper()}   |   "
    f"PERIOD: {period}   |   "
    f"DAYS: {len(primary_df):,}   |   "
    f"REGIMES: {n_clusters}"
    f"{macro_tag}"
)

if p_meta["macro_loaded"] and p_meta["macro_cols"]:
    macro_ctx = describe_macro_regime(cur, p_meta["macro_cols"])
    st.markdown(
        ui.info_banner(f"<strong>MACRO CONTEXT:</strong>  {macro_ctx}"),
        unsafe_allow_html=True,
    )


# Tabs

(
    tab_price, tab_macro, tab_multi, tab_trans,
    tab_latent, tab_wf, tab_analytics,
    tab_features, tab_loss, tab_export,
) = st.tabs([
    "PRICE + REGIMES",
    "MACRO DASHBOARD",
    "MULTI-TICKER",
    "TRANSITIONS",
    "LATENT SPACE",
    "WALK-FORWARD",
    "ANALYTICS",
    "FEATURE MATRIX",
    "TRAINING",
    "EXPORT",
])


with tab_price:
    sel = (
        st.selectbox("Instrument", list(valid.keys()), key="p_sel")
        if len(valid) > 1 else primary_ticker
    )
    sel_df, sel_meta = valid[sel]

    st.plotly_chart(ui.price_chart(sel_df, sel_meta), use_container_width=True)
    st.plotly_chart(ui.volatility_chart(sel_df), use_container_width=True)

    st.markdown(ui.section_rule("REGIME ALLOCATION"), unsafe_allow_html=True)
    counts = sel_df["regime"].value_counts(normalize=True).mul(100).round(1)
    rcols  = st.columns(len(counts))
    for col, (regime, pct) in zip(rcols, counts.items()):
        color = sel_df.loc[sel_df["regime"] == regime, "color"].iloc[0]
        col.markdown(
            ui.sys_panel(regime, f"{pct}%", color),
            unsafe_allow_html=True,
        )


with tab_macro:
    sel_m = (
        st.selectbox("Instrument", list(valid.keys()), key="mc_sel")
        if len(valid) > 1 else primary_ticker
    )
    m_df, m_meta = valid[sel_m]

    if not m_meta["macro_loaded"]:
        st.markdown(
            ui.info_banner(
                "Macro features are disabled or unavailable. "
                "Enable the <strong>Macro Features</strong> toggle in the sidebar and re-run."
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ui.section_rule("CURRENT MACRO SNAPSHOT"), unsafe_allow_html=True)

        snap_cols = {
            "VIX":            ("vix_level",    lambda v: f"{v*100:.1f}",  "↑ = stress"),
            "Yield Curve":    ("yield_curve",  lambda v: f"{v:+.4f}",    "↑ = steepen"),
            "Credit Spread":  ("credit_spread",lambda v: f"{v:+.4f}",    "↑ = stress"),
            "Gold 20d Mom":   ("gold_momentum",lambda v: f"{v*100:+.1f}%","↑ = inflationary"),
            "TLT 20d Mom":    ("tlt_momentum", lambda v: f"{v*100:+.1f}%","↑ = risk-off"),
            "Risk Appetite":  ("risk_appetite",lambda v: f"{v:+.3f}",    "↑ = risk-on"),
        }
        snap_available = {k: v for k, v in snap_cols.items() if v[0] in m_df.columns}
        if snap_available:
            snap_c = st.columns(len(snap_available))
            for col, (label, (field, fmt, hint)) in zip(snap_c, snap_available.items()):
                val   = m_df.iloc[-1][field]
                color = "#00ffaa" if val > 0 else "#ff2d6b"
                col.markdown(
                    ui.sys_panel(label, fmt(val), color) +
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:0.55rem;color:#3a5578;margin-top:4px">{hint}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(ui.section_rule("MACRO INDICATORS × REGIMES"), unsafe_allow_html=True)
        st.markdown(
            ui.info_banner(
                "Lines are coloured by the detected market regime on each day. "
                "This reveals the macro environment characteristic of each state."
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            macro_dashboard_chart(m_df, m_meta["macro_feats"]),
            use_container_width=True,
        )

        st.markdown(ui.section_rule("MACRO FINGERPRINT BY REGIME"), unsafe_allow_html=True)
        st.markdown(
            ui.info_banner(
                "Average z-score of each macro indicator per regime. "
                "Positive = above its 60-day mean. Reveals what macro "
                "environment each regime tends to occur in."
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            macro_fingerprint_chart(m_df, m_meta["macro_cols"]),
            use_container_width=True,
        )

        st.markdown(ui.section_rule("MACRO ANALYTICS TABLE"), unsafe_allow_html=True)
        key_mc = [c for c in [
            "vix_level", "vix_zscore", "yield_curve", "credit_spread",
            "gold_momentum", "tlt_momentum", "dollar_momentum", "risk_appetite",
        ] if c in m_df.columns]

        if key_mc:
            mfp_df = macro_regime_analytics(m_df, key_mc).reset_index()
            mfp_df.insert(1, "Days", [
                len(m_df[m_df["regime"] == r]) for r in mfp_df["Regime"]
            ])
            fmt_dict = {c: "{:.4f}" for c in key_mc}
            st.dataframe(
                mfp_df.style.format(fmt_dict, subset=key_mc),
                use_container_width=True,
            )


with tab_multi:
    if len(valid) == 1:
        st.markdown(
            ui.info_banner("Add more tickers in the sidebar to enable multi-instrument comparison."),
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(ui.multi_price_chart(valid), use_container_width=True)
        st.plotly_chart(ui.regime_bar_chart(valid),  use_container_width=True)

        st.markdown(ui.section_rule("CURRENT REGIME BY INSTRUMENT"), unsafe_allow_html=True)
        ccols = st.columns(len(valid))
        for col, (t, (df, meta)) in zip(ccols, valid.items()):
            r  = df.iloc[-1]["regime"]
            c  = df.iloc[-1]["color"]
            sc = meta["score"]
            col.markdown(
                ui.sys_panel(t, r, c) +
                f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:0.6rem;color:#3a5578;margin-top:4px">'
                f'SILHOUETTE {sc:.3f}</div>',
                unsafe_allow_html=True,
            )


with tab_trans:
    sel_t = (
        st.selectbox("Instrument", list(valid.keys()), key="tr_sel")
        if len(valid) > 1 else primary_ticker
    )
    t_df = valid[sel_t][0]
    tmat = transition_matrix(t_df)

    st.plotly_chart(ui.transition_heatmap(tmat), use_container_width=True)

    st.markdown(ui.section_rule("MOST PROBABLE NEXT STATE"), unsafe_allow_html=True)
    ml_cols = st.columns(len(tmat.columns))
    for col, from_r in zip(ml_cols, tmat.index):
        row    = tmat.loc[from_r]
        next_r = row.idxmax()
        prob   = row.max()
        color  = (
            t_df.loc[t_df["regime"] == next_r, "color"].iloc[0]
            if next_r in t_df["regime"].values else "#888888"
        )
        col.markdown(
            ui.sys_panel(f"{from_r} →", f"{next_r}", color) +
            f'<div style="font-family:\'IBM Plex Mono\',monospace;'
            f'font-size:0.6rem;color:#3a5578;margin-top:4px">'
            f'PROB {prob:.0%}</div>',
            unsafe_allow_html=True,
        )


with tab_latent:
    sel_l = (
        st.selectbox("Instrument", list(valid.keys()), key="la_sel")
        if len(valid) > 1 else primary_ticker
    )
    l_df, l_meta = valid[sel_l]

    st.markdown(
        ui.info_banner(
            "Each point represents one trading day projected into the model's "
            "2-dimensional latent space. Clustering reveals structurally similar "
            "market conditions regardless of calendar time."
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(ui.latent_chart(l_df), use_container_width=True)

    if model_type == "HMM" and l_meta.get("state_probs") is not None:
        rank_l = (
            l_df.groupby("cluster")["ret_20"]
            .median().sort_values(ascending=False)
        )
        lmap_l = {int(cid): REGIME_LABELS[r][0]
                  for r, cid in enumerate(rank_l.index)}
        color_l = {REGIME_LABELS[r][0]: REGIME_LABELS[r][1]
                   for r in range(len(REGIME_LABELS))}
        st.plotly_chart(
            ui.hmm_prob_chart(l_df, l_meta["state_probs"], lmap_l, color_l),
            use_container_width=True,
        )


with tab_wf:
    sel_wf = (
        st.selectbox("Instrument", list(valid.keys()), key="wf_sel")
        if len(valid) > 1 else primary_ticker
    )
    wf_df, wf_meta = valid[sel_wf]
    wf             = wf_meta["wf"]

    st.plotly_chart(ui.wf_chart(wf, wf_df), use_container_width=True)

    if wf["regime_fwd"]:
        st.markdown(ui.section_rule("OUT-OF-SAMPLE REGIME PERFORMANCE"), unsafe_allow_html=True)
        wf_stat_cols = st.columns(len(wf["regime_fwd"]))
        for col, (rname, stats) in zip(wf_stat_cols, wf["regime_fwd"].items()):
            ret   = stats["mean_fwd_ret"] * 100
            ret_c = "#00ffaa" if ret > 0 else "#ff2d6b"
            col.markdown(
                ui.wf_stat_card(
                    rname,
                    f"{ret:+.1f}%",
                    f"Sharpe {stats['sharpe']:.2f}  ·  Hit {stats['hit_rate']:.0%}  ·  N={stats['n_days']}",
                    ret_c,
                ),
                unsafe_allow_html=True,
            )


with tab_analytics:
    sel_a = (
        st.selectbox("Instrument", list(valid.keys()), key="an_sel")
        if len(valid) > 1 else primary_ticker
    )
    a_df  = valid[sel_a][0]
    an_df = regime_analytics(a_df)

    for _, row in an_df.iterrows():
        color = row["Color"]
        cols  = st.columns(8)
        pairs = [
            ("REGIME",      row["Regime"]),
            ("DAYS",        str(row["Days"])),
            ("PCT TIME",    row["Pct Time"]),
            ("ANN. RETURN", row["Ann. Return"]),
            ("ANN. VOL",    row["Ann. Vol"]),
            ("SHARPE",      row["Sharpe"]),
            ("MAX DD",      row["Max Drawdown"]),
            ("AVG DUR.",    row["Avg Duration"]),
        ]
        for col, (label, val) in zip(cols, pairs):
            col.markdown(
                ui.sys_panel(label, val, color),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


with tab_features:
    sel_f = (
        st.selectbox("Instrument", list(valid.keys()), key="fe_sel")
        if len(valid) > 1 else primary_ticker
    )
    f_df = valid[sel_f][0]

    st.plotly_chart(ui.feature_heatmap(f_df), use_container_width=True)

    st.markdown(ui.section_rule("RECENT FEATURE VALUES"), unsafe_allow_html=True)
    feat_cols = ["ret_1", "ret_5", "ret_20", "vol_20", "rsi",
                 "atr", "bb_width", "obv_roc", "macd", "range_pos"]
    preview_f = f_df[[c for c in feat_cols if c in f_df.columns]].tail(30)
    preview_f.index = preview_f.index.strftime("%Y-%m-%d")
    st.dataframe(
        preview_f.style
            .background_gradient(subset=["rsi"], cmap="RdYlGn")
            .format({c: "{:.4f}" for c in preview_f.columns}),
        use_container_width=True,
    )


with tab_loss:
    sel_lo = (
        st.selectbox("Instrument", list(valid.keys()), key="lo_sel")
        if len(valid) > 1 else primary_ticker
    )
    lo_meta = valid[sel_lo][1]

    if lo_meta["losses"]:
        final = lo_meta["losses"][-1]
        min_l = min(lo_meta["losses"])
        st.markdown(
            ui.info_banner(
                f"Final reconstruction loss: <strong>{final:.5f}</strong>  |  "
                f"Minimum: <strong>{min_l:.5f}</strong>  |  "
                f"Epochs: <strong>{len(lo_meta['losses'])}</strong><br>"
                "Loss still falling at the right edge? Increase epoch count in settings."
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(ui.loss_chart(lo_meta["losses"]), use_container_width=True)
    else:
        st.markdown(
            ui.info_banner(
                "No training curve available in PCA or HMM mode. "
                "Install <code>torch</code> and select KMeans + AE/PCA to enable."
            ),
            unsafe_allow_html=True,
        )


with tab_export:
    sel_ex = (
        st.selectbox("Instrument", list(valid.keys()), key="ex_sel")
        if len(valid) > 1 else primary_ticker
    )
    ex_df, ex_meta = valid[sel_ex]
    ex_an  = regime_analytics(ex_df)
    ex_mat = transition_matrix(ex_df)

    st.markdown(ui.section_rule("DATA EXPORT"), unsafe_allow_html=True)
    col_csv, col_pdf = st.columns(2)

    with col_csv:
        st.download_button(
            label="DOWNLOAD CSV",
            data=build_csv(ex_df, ex_meta.get("macro_cols")),
            file_name=f"{sel_ex}_regimes_{period}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Daily OHLCV + regime labels + all indicators + macro features")

    with col_pdf:
        if REPORTLAB_AVAILABLE:
            pdf = build_pdf(sel_ex, ex_df, ex_meta, ex_an, ex_mat)
            st.download_button(
                label="DOWNLOAD PDF REPORT",
                data=pdf,
                file_name=f"{sel_ex}_regime_report_{period}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("Full report: analytics, macro fingerprint, transition matrix, walk-forward")
        else:
            st.markdown(
                ui.warn_banner("Install <code>reportlab</code> to enable PDF export."),
                unsafe_allow_html=True,
            )

    st.markdown(ui.section_rule("DATA PREVIEW  (LAST 20 ROWS)"), unsafe_allow_html=True)
    preview = ex_df[["close", "regime", "ret_1", "vol_20", "rsi",
                      "atr", "bb_width", "obv_roc"]].tail(20)
    preview.index = preview.index.strftime("%Y-%m-%d")
    st.dataframe(
        preview.style
            .background_gradient(subset=["rsi"], cmap="RdYlGn")
            .format({
                "close":    "${:.2f}",
                "ret_1":    "{:+.3%}",
                "vol_20":   "{:.4f}",
                "rsi":      "{:.3f}",
                "atr":      "{:.5f}",
                "bb_width": "{:.4f}",
                "obv_roc":  "{:+.4f}",
            }),
        use_container_width=True,
    )
