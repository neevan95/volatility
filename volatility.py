import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Stock Volatility")

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
symbol = st.text_input("Stock Ticker", "AAPL").upper()
periodicity = st.selectbox("Return Periodicity", ["Daily", "Weekly", "Monthly"])
min_prob = st.slider("Min Probability Coverage", 0.0, 1.0, 0.5, 0.01)

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

# --- Optimal symmetric interval ---
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

# prepare pdf
x = np.linspace(returns.min(), returns.max(), 1000)
pdf = norm.pdf(x, mu, sigma)

# --- Figure 0: Price Time Series ---
st.subheader("Price Time Series")
# create subplot with secondary y-axis for price and volume
fig_price = make_subplots(specs=[[{"secondary_y": True}]])
# add volume as column chart on secondary y-axis
fig_price.add_trace(
    go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        #marker_color='blue',
        opacity=0.25,
        yaxis='y2'
    ),
    secondary_y=True
)
# add close price line on primary y-axis
fig_price.add_trace(
    go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='white', width=2)
    ),
    secondary_y=False
)
# layout tweaks for axes
fig_price.update_xaxes(
    title_text='Date',
    type='date'
)
fig_price.update_yaxes(
    title_text='Price',
    tickprefix='$',
    tickformat=',.2f',
    secondary_y=False
)
fig_price.update_yaxes(
    title_text='Volume',
    secondary_y=True
)
fig_price.update_layout(
    title=f"{symbol} Close Price & Volume",
    legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
)
st.plotly_chart(fig_price, use_container_width=True)

# --- Figure 1: Optimal Interval ---
st.subheader("Optimal Interval")
fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=returns, histnorm='probability density', nbinsx=50,
    marker=dict(color='lightgrey', line=dict(color='black', width=1)), showlegend=False))
fig1.add_trace(go.Scatter(x=x, y=pdf, mode='lines', line=dict(color='white', width=2), showlegend=False))
mask_opt = (x >= lower_opt) & (x <= upper_opt)
fig1.add_trace(go.Scatter(x=x[mask_opt], y=pdf[mask_opt], fill='tozeroy',
    name=f"Optimal [{lower_opt:.2%}, {upper_opt:.2%}] ({prob_opt:.2%})",
    fillcolor='rgba(128,0,128,0.3)', line=dict(color='rgba(128,0,128,0)')))
fig1.update_layout(title=f"{symbol} Returns: Optimal Interval",
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
sel2 = st.selectbox("Select chart period", presets2)
end_idx = data.index.max()
if sel2 == "Full range": start_idx = data.index.min()
elif sel2 == "Last 30 days": start_idx = end_idx - pd.Timedelta(days=30)
elif sel2 == "Last 90 days": start_idx = end_idx - pd.Timedelta(days=90)
elif sel2 == "Last year":   start_idx = end_idx - pd.DateOffset(years=1)
else:
    cs, ce = st.date_input("Custom period", [data.index.min().date(), data.index.max().date()])
    start_idx, end_idx = pd.to_datetime(cs), pd.to_datetime(ce)
rv = data.loc[start_idx:end_idx]
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=rv.index, y=rv['RollingVol'], mode='lines', name=f"Volatility ({window})"), secondary_y=False)
fig2.add_trace(go.Scatter(x=rv.index, y=rv['Close'], mode='lines', name='Close Price',line=dict(color='white', width=1.5)), secondary_y=True)
fig2.update_xaxes(title_text='Date')
fig2.update_yaxes(title_text='Annualized Volatility', tickformat='.2%', secondary_y=False)
fig2.update_yaxes(title_text='Close Price', secondary_y=True)
fig2.update_layout(title=f"{symbol} Rolling Volatility & Price",
    legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5))
st.plotly_chart(fig2, use_container_width=True)

# --- Options IV & Expected Move ---
st.subheader("Options: Implied Vol & Expected Move")
ticker = yf.Ticker(symbol)
expiries = ticker.options
exp_move = None
if expiries:
    sel_expiry = st.selectbox("Select expiry", expiries)
    chains = ticker.option_chain(sel_expiry)
    calls, puts = chains.calls, chains.puts
    strikes = sorted(calls['strike'].unique())
    sel_strike = st.selectbox("Select strike", strikes)
    call_iv = calls.loc[calls.strike==sel_strike, 'impliedVolatility'].iloc[0]
    put_iv  = puts.loc[puts.strike==sel_strike,   'impliedVolatility'].iloc[0]
    iv_strike = np.nanmean([call_iv, put_iv])
    st.write(f"Strike {sel_strike}: Call {call_iv:.2%}, Put {put_iv:.2%}, Avg {iv_strike:.2%}")
    days_to = max((pd.to_datetime(sel_expiry) - today).days, 0)
    t_years = days_to / 365
    exp_move = iv_strike * np.sqrt(t_years) if t_years > 0 else 0
    exp_move_call = call_iv * np.sqrt(t_years) if t_years > 0 else 0
    exp_move_put = put_iv * np.sqrt(t_years) if t_years > 0 else 0
    st.write(f"Expected move over {days_to} days: ±{exp_move:.2%}")
    st.write(f"Expected move over {days_to} days - **Call**: ±{exp_move_call:.2%}")
    st.write(f"Expected move over {days_to} days - **Put**: ±{exp_move_put:.2%}")
else:
    st.write("No option data.")

# --- Figure 3: Threshold Tail (Post-Options) ---
st.subheader("Threshold Tail Chart")
use_em = st.checkbox("Use expected move as threshold", value=True)
if use_em and exp_move is not None:
    call_put = st.selectbox("Select Call or Put", ['Calls','Puts'],0)
    if call_put == 'Calls':
        lower_thr = -exp_move_call; upper_thr = exp_move_call
    else:
        lower_thr = -exp_move_put; upper_thr = exp_move_put
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
fig3.add_trace(go.Histogram(x=returns, histnorm='probability density', nbinsx=50,
    marker=dict(color='lightgrey', line=dict(color='black', width=1)), showlegend=False))
fig3.add_trace(go.Scatter(x=x, y=pdf, mode='lines', line=dict(color='white', width=2), showlegend=False))
fig3.add_trace(go.Scatter(x=x[mask_thr], y=pdf[mask_thr], fill='tozeroy', name=f"{label} ({prob_thr:.2%})",
    fillcolor='rgba(255,0,0,0.3)', line=dict(color='rgba(255,0,0,0)')))
fig3.update_layout(title=f"{symbol} Returns: Threshold Tail",
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
st.write(f"Optimal Interval: **[{lower_opt:.2%}, {upper_opt:.2%}] ({prob_opt:.2%})**")
st.write(f"Probability of expected move: **{prob_thr:.2%}**")