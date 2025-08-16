import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Stock Volatility")

# ---------- Helper for last-point labels ----------
def add_last_point_annotation(fig, series: pd.Series, fmt_func, *, secondary_y: bool = False):
    """Add a label at the most recent non-NaN point of a pandas Series."""
    if series is None or series.dropna().empty:
        return
    s = series.dropna()
    x_last = s.index[-1]
    y_last = s.iloc[-1]
    fig.add_annotation(
        x=x_last,
        y=y_last,
        xref="x",
        yref="y2" if secondary_y else "y",
        text=fmt_func(y_last),
        showarrow=True,
        arrowhead=2,
        arrowcolor="black",
        ax=0,
        ay=-30,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1,
        font=dict(size=12,color="black"),
        align="center"
    )

symbol = st.text_input("Stock Ticker", "BA").upper()

# --- Date Range Selection ---
today = pd.Timestamp.today().normalize()
default_start = pd.to_datetime("2023-01-01")
range_opts = ["Custom", "Last 30 days", "Last 90 days", "Last year", "Full range"]
sel_range = st.selectbox("Select data range", range_opts)
if sel_range == "Custom":
    start_date, end_date = st.date_input(
        "Date Range",
        value=[default_start.date(), today.date()],
        min_value=default_start.date(),
        max_value=today.date()
    )
else:
    end_date = today.date()
    if sel_range == "Last 30 days":
        start_date = (today - pd.Timedelta(days=30)).date()
    elif sel_range == "Last 90 days":
        start_date = (today - pd.Timedelta(days=90)).date()
    elif sel_range == "Last year":
        start_date = (today - pd.DateOffset(years=1)).date()
    else:
        start_date = default_start.date()

# --- Core Inputs ---
periodicity = st.selectbox("Return Periodicity", ["Daily", "Weekly", "Monthly"])

# --- Fetch price data ---
if not symbol or pd.to_datetime(start_date) >= pd.to_datetime(end_date):
    st.error("Please select a valid ticker and date range.")
    st.stop()

interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
data = yf.download(symbol, start=start_date, end=end_date, interval=interval_map[periodicity])
data.columns = data.columns.get_level_values(0)
if data.empty:
    st.error(f"No data for {symbol}.")
    st.stop()

data['Return'] = data['Close'].pct_change()
data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
returns = data['Return'].dropna()
log_returns = data['Log_Return'].dropna()
mu, sigma = returns.mean(), returns.std()

# --- Figure 0: Price Time Series ---
st.subheader("Price Time Series")

# Presets for the price chart
presets0 = ["Full range", "Last 30 days", "Last 90 days", "Last year", "Custom"]
sel0 = st.selectbox("Select chart period (price)", presets0, index = 3)

end_idx0 = data.index.max()
if sel0 == "Full range":
    start_idx0 = data.index.min()
elif sel0 == "Last 30 days":
    start_idx0 = end_idx0 - pd.Timedelta(days=30)
elif sel0 == "Last 90 days":
    start_idx0 = end_idx0 - pd.Timedelta(days=90)
elif sel0 == "Last year":
    start_idx0 = end_idx0 - pd.DateOffset(years=1)
else:
    cs0, ce0 = st.date_input(
        "Custom period (price chart)",
        [data.index.min().date(), data.index.max().date()],
        key="price_custom_range"
    )
    start_idx0, end_idx0 = pd.to_datetime(cs0), pd.to_datetime(ce0)

# Clip to available data and guard against empty slice
start_idx0 = max(start_idx0, data.index.min())
end_idx0 = min(end_idx0, data.index.max())
pv = data.loc[start_idx0:end_idx0]
if pv.empty:
    st.warning("No data in selected range for the price chart. Showing full range.")
    pv = data.copy()

# Build chart on the filtered range
fig_price = make_subplots(specs=[[{"secondary_y": True}]])
fig_price.add_trace(
    go.Bar(x=pv.index, y=pv['Volume'], name='Volume', opacity=0.25),
    secondary_y=True
)
fig_price.add_trace(
    go.Scatter(
        x=pv.index, y=pv['Close'], mode='lines',
        name='Close Price', line=dict(color='white', width=2)
    ),
    secondary_y=False
)

fig_price.update_xaxes(title_text='Date', type='date')
fig_price.update_yaxes(title_text='Price', tickprefix='$', tickformat=',.2f', secondary_y=False)
fig_price.update_yaxes(title_text='Volume', secondary_y=True)
fig_price.update_layout(
    title=f"{symbol} Close Price & Volume",
    legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
)

# Last-point label (uses filtered series)
add_last_point_annotation(
    fig_price,
    pv['Close'],
    fmt_func=lambda v: f"${v:,.2f}",
    secondary_y=False
)

# Ensure annotation is visible in dark mode (if you didn't already update the helper)
fig_price.update_annotations(
    font=dict(color="black"), bgcolor="white", bordercolor="black", arrowcolor="black"
)

st.plotly_chart(fig_price, use_container_width=True)

# --- Optimal symmetric interval ---
st.subheader("Most Probable Return Range")
min_prob = st.slider("Min Probability Coverage", 0.0, 1.0, 0.5, 0.01)
z_vals = np.linspace(0.01, 5, 1000)
probs = norm.cdf(z_vals) - norm.cdf(-z_vals)
widths = 2 * z_vals * sigma
valid = probs >= min_prob
if not valid.any():
    st.warning(f"No interval up to 5σ achieves {min_prob:.0%} coverage.")
    st.stop()
z_opt = z_vals[valid][np.argmin(widths[valid])]
lower_opt = mu - z_opt * sigma
upper_opt = mu + z_opt * sigma
prob_opt = norm.cdf(z_opt) - norm.cdf(-z_opt)

x = np.linspace(returns.min(), returns.max(), 1000)
pdf = norm.pdf(x, mu, sigma)

# --- Figure 1: Optimal Interval ---
fig1 = go.Figure()
fig1.add_trace(go.Histogram(
    x=returns, histnorm='probability density', nbinsx=50,
    marker=dict(color='lightgrey', line=dict(color='black', width=1)), showlegend=False))
fig1.add_trace(go.Scatter(x=x, y=pdf, mode='lines', line=dict(color='white', width=2), showlegend=False))
mask_opt = (x >= lower_opt) & (x <= upper_opt)
fig1.add_trace(go.Scatter(
    x=x[mask_opt], y=pdf[mask_opt], fill='tozeroy',
    name=f"Range [{lower_opt:.2%}, {upper_opt:.2%}] ({prob_opt:.2%})",
    fillcolor='rgba(128,0,128,0.3)', line=dict(color='rgba(128,0,128,0)')))
fig1.update_layout(
    title=f"{symbol} Returns: Probable Range",
    xaxis_title='Return', yaxis_title='Density',
    legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5))
fig1.update_xaxes(tickformat='.2%')
st.plotly_chart(fig1, use_container_width=True)

# --- Figure 2: Rolling Volatility & Price ---
st.subheader("Rolling Volatility & Close Price")
window = st.slider("Rolling window size", 2, len(log_returns), min(20, len(log_returns)))
ann_factor = np.sqrt({"Daily":252, "Weekly":52, "Monthly":12}[periodicity])
data['RollingVol'] = data['Log_Return'].rolling(window).std() * ann_factor

presets2 = ["Full range", "Last 30 days", "Last 90 days", "Last year", "Custom"]
sel2 = st.selectbox("Select chart period", presets2, index = 3)
end_idx = data.index.max()
if sel2 == "Full range":
    start_idx = data.index.min()
elif sel2 == "Last 30 days":
    start_idx = end_idx - pd.Timedelta(days=30)
elif sel2 == "Last 90 days":
    start_idx = end_idx - pd.Timedelta(days=90)
elif sel2 == "Last year":
    start_idx = end_idx - pd.DateOffset(years=1)
else:
    cs, ce = st.date_input("Custom period", [data.index.min().date(), data.index.max().date()])
    start_idx, end_idx = pd.to_datetime(cs), pd.to_datetime(ce)

rv = data.loc[start_idx:end_idx]

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=rv.index, y=rv['RollingVol'], mode='lines', name=f"Volatility ({window})"),
               secondary_y=False)
fig2.add_trace(go.Scatter(x=rv.index, y=rv['Close'], mode='lines', name='Close Price',
                          line=dict(color='white', width=1.5)),
               secondary_y=True)
fig2.update_xaxes(title_text='Date')
fig2.update_yaxes(title_text='Annualized Volatility', tickformat='.2%', secondary_y=False)
fig2.update_yaxes(title_text='Close Price', secondary_y=True)
fig2.update_layout(
    title=f"{symbol} Rolling Volatility & Price",
    legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5))

# NEW: latest labels for both series in Figure 2
add_last_point_annotation(
    fig2,
    rv['RollingVol'],
    fmt_func=lambda v: f"{v:.2%}",
    secondary_y=False
)
add_last_point_annotation(
    fig2,
    rv['Close'],
    fmt_func=lambda v: f"${v:,.2f}",
    secondary_y=True
)

st.plotly_chart(fig2, use_container_width=True)

# --- Options IV & Expected Move ---
st.subheader("Options: Implied Vol & Expected Move")
ticker = yf.Ticker(symbol)
expiries = ticker.options
exp_move = None

def _sum_oi(df):
    if 'openInterest' not in df.columns:
        return np.nan
    return int(pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).sum())

def _pcr_text(puts, calls):
    if pd.isna(puts) or pd.isna(calls):
        return "—"
    if calls == 0:
        return "∞" if puts > 0 else "—"
    return f"{puts / calls:.2f}"

# Robust strike matcher (handles float/str and tiny rounding)
def _match_row(df, strike_value):
    if 'strike' not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df['strike'], errors='coerce')
    mask = np.isfinite(s) & np.isclose(s, float(strike_value), atol=1e-6)
    return df.loc[mask]

# Prefer trading-day DTE for options math; fallback to calendar if needed
def _dte_trading_years(expiry_str):
    # business days (Mon-Fri) inclusive of today→expiry
    today_date = pd.Timestamp.today().date()
    expiry_date = pd.to_datetime(expiry_str).date()
    # +1 to include expiry day; clamp at >=1 so same-day doesn't become 0
    bd = max(np.busday_count(today_date, expiry_date) + 1, 1)
    return bd / 252.0, bd  # years, raw business days

if expiries:
    sel_expiry = st.selectbox("Select expiry", expiries)

    chains = ticker.option_chain(sel_expiry)
    calls, puts = chains.calls.copy(), chains.puts.copy()

    # Expiry-wide OI + PCR
    call_oi_all = _sum_oi(calls)
    put_oi_all  = _sum_oi(puts)
    pcr_all_text = _pcr_text(put_oi_all, call_oi_all)

    st.caption(
        f"Expiry OI (all strikes) — Calls: {('—' if pd.isna(call_oi_all) else f'{call_oi_all:,}')} · "
        f"Puts: {('—' if pd.isna(put_oi_all) else f'{put_oi_all:,}')} · "
        f"Put/Call OI: {pcr_all_text}"
    )

    # strikes from union of calls/puts (handles asymmetry)
    strikes_arr = np.unique(
        np.concatenate([
            pd.to_numeric(calls.get('strike', pd.Series([])), errors='coerce').dropna().values,
            pd.to_numeric(puts.get('strike', pd.Series([])), errors='coerce').dropna().values
        ])
    )
    strikes_arr.sort()

    if strikes_arr.size == 0:
        st.write("No strikes available for this expiry.")
        sel_strike = None
    else:
        spot = float(data['Close'].iloc[-1])
        default_idx = int(np.abs(strikes_arr - spot).argmin())
        sel_strike = st.selectbox(
            "Select strike",
            strikes_arr.tolist(),
            index=default_idx,
            format_func=lambda x: f"{x:,.2f}"
        )
        st.caption(f"Nearest to last close (${spot:,.2f}) → default strike {strikes_arr[default_idx]:,.2f}")

    if sel_strike is not None:
        # Pull IVs robustly
        call_row = _match_row(calls, sel_strike)
        put_row  = _match_row(puts,  sel_strike)

        call_iv = float(call_row['impliedVolatility'].iloc[0]) if not call_row.empty and pd.notna(call_row['impliedVolatility'].iloc[0]) else np.nan
        put_iv  = float(put_row['impliedVolatility'].iloc[0])  if not put_row.empty  and pd.notna(put_row['impliedVolatility'].iloc[0])  else np.nan
        iv_strike = np.nanmean([call_iv, put_iv])  # average if both present

        # OI + PCR at this strike
        def _safe_oi(row):
            if row.empty or 'openInterest' not in row.columns or pd.isna(row['openInterest'].iloc[0]):
                return np.nan
            return int(row['openInterest'].iloc[0])

        call_oi = _safe_oi(call_row)
        put_oi  = _safe_oi(put_row)
        pcr_strike_text = _pcr_text(put_oi, call_oi)

        # Show IVs (surface NaNs honestly)
        st.write(
            f"Strike {sel_strike:,.2f}: "
            f"Call IV {('—' if np.isnan(call_iv) else f'{call_iv:.2%}')}, "
            f"Put IV {('—' if np.isnan(put_iv)  else f'{put_iv:.2%}')}, "
            f"Avg {('—' if np.isnan(iv_strike) else f'{iv_strike:.2%}')}"
        )
        st.caption(
            f"Open interest — Calls: {('—' if pd.isna(call_oi) else f'{call_oi:,}')} · "
            f"Puts: {('—' if pd.isna(put_oi) else f'{put_oi:,}')} · "
            f"Put/Call OI: {pcr_strike_text}"
        )

        # Time to expiry (trading days)
        t_years_252, dte_bd = _dte_trading_years(sel_expiry)

        # Expected move as a % of spot (1-sigma)
        exp_move      = (iv_strike * np.sqrt(t_years_252)) if not np.isnan(iv_strike) else np.nan
        exp_move_call = (call_iv  * np.sqrt(t_years_252)) if not np.isnan(call_iv)  else np.nan
        exp_move_put  = (put_iv   * np.sqrt(t_years_252)) if not np.isnan(put_iv)   else np.nan

        st.write(
            f"Expected move over {dte_bd} trading day(s): "
            f"±{('—' if np.isnan(exp_move) else f'{exp_move:.2%}')}"
        )
        st.write(
            f"Expected move over {dte_bd} trading day(s) - "
            f"**Call**: ±{('—' if np.isnan(exp_move_call) else f'{exp_move_call:.2%}')}, "
            f"**Put**: ±{('—' if np.isnan(exp_move_put) else f'{exp_move_put:.2%}')}"
        )
else:
    st.write("No option data.")


# --- Figure 3: Threshold Tail (Post-Options) ---
st.subheader("Threshold Tail Chart")
use_em = st.checkbox("Use expected move as threshold", value=True)
if use_em and exp_move is not None:
    call_put = st.selectbox("Select Call or Put", ['Calls','Puts'],0)
    if call_put == 'Calls':
        lower_thr = exp_move_call; upper_thr = max(returns)
    else:
        lower_thr = min(returns); upper_thr = -exp_move_put
else:
    cond = st.selectbox("Threshold condition", ["≥","≤","Between"], format_func=lambda x: f"Return {x}")
    if cond == "≥":
        val = st.number_input("Return ≥ (%)", -100.0, 100.0, 0.0, 0.1)/100
        lower_thr, upper_thr = val, None
    elif cond == "≤":
        val = st.number_input("Return ≤ (%)", -100.0, 100.0, 0.0, 0.1)/100
        lower_thr, upper_thr = None, val
    else:
        low = st.number_input("Lower (%)", -100.0, 100.0, -1.0, 0.1)/100
        high= st.number_input("Upper (%)", -100.0, 100.0, 1.0, 0.1)/100
        lower_thr, upper_thr = low, high

# compute mask and prob
if upper_thr is None:
    mask_thr = x >= lower_thr
    prob_thr = 1 - norm.cdf((lower_thr - mu)/sigma)
    label = f"Return ≥ {lower_thr:.2%}"
elif lower_thr is None:
    mask_thr = x <= upper_thr
    prob_thr = norm.cdf((upper_thr - mu)/sigma)
    label = f"Return ≤ {upper_thr:.2%}"
else:
    mask_thr = (x >= lower_thr) & (x <= upper_thr)
    prob_thr = norm.cdf((upper_thr - mu)/sigma) - norm.cdf((lower_thr - mu)/sigma)
    label = f"Return between {lower_thr:.2%} and {upper_thr:.2%}"

# plot
fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=returns, histnorm='probability density', nbinsx=50,
    marker=dict(color='lightgrey', line=dict(color='black', width=1)), showlegend=False))
fig3.add_trace(go.Scatter(x=x, y=pdf, mode='lines', line=dict(color='white', width=2), showlegend=False))
fig3.add_trace(go.Scatter(
    x=x[mask_thr], y=pdf[mask_thr], fill='tozeroy', name=f"{label} ({prob_thr:.2%})",
    fillcolor='rgba(255,0,0,0.3)', line=dict(color='rgba(255,0,0,0)')))
fig3.update_layout(
    title=f"{symbol} Returns: Threshold Tail",
    xaxis_title='Return', yaxis_title='Density',
    legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5))
fig3.update_xaxes(tickformat='.2%')
st.plotly_chart(fig3, use_container_width=True)

# --- Summary Statistics ---
freq_map = {"Daily":252, "Weekly":52, "Monthly":12}
mu_ann = mu * freq_map[periodicity]
vol_ann = sigma * np.sqrt(freq_map[periodicity])
sharpe = mu_ann / vol_ann if vol_ann else np.nan
st.subheader("Summary Statistics")
st.write(f"Mean: **{mu:.2%}**")
st.write(f"Std Dev: **{sigma:.2%}**")
st.write(f"Annualized Return: **{mu_ann:.2%}**")
st.write(f"Annualized Vol: **{vol_ann:.2%}**")
st.write(f"Sharpe Ratio: **{sharpe:.2f}**")
st.write(f"Probable Return Range: **[{lower_opt:.2%}, {upper_opt:.2%}] ({prob_opt:.2%})**")
st.write(f"Probability of expected move: **{prob_thr:.2%}**")
