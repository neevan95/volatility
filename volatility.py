import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go

st.title("Stock Return Analysis")

# User inputs
symbol = st.text_input("Stock Ticker", value="AAPL").upper()
start_date, end_date = st.date_input(
    "Date Range",
    value=[pd.to_datetime("2023-01-01"), pd.to_datetime("2025-07-30")]
)
periodicity = st.selectbox(
    "Return Periodicity",
    ["Daily", "Weekly", "Monthly"]
)
min_prob = st.slider(
    "Minimum Probability Coverage",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)
operator = st.selectbox(
    "Threshold Condition",
    ["≥", "≤", "Between"],
    format_func=lambda x: f"Return {x}"
)

# Threshold inputs based on operator
if operator == "Between":
    lower_thr = st.number_input(
        "Lower Return Threshold (%)",
        min_value=-100.0,
        max_value=100.0,
        value=-1.0,
        step=0.1
    )
    upper_thr = st.number_input(
        "Upper Return Threshold (%)",
        min_value=-100.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )
else:
    threshold = st.number_input(
        "Return Threshold (%)",
        min_value=-100.0,
        max_value=100.0,
        value=0.0,
        step=0.1
    )

# Fetch and process data
if symbol and start_date < end_date:
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval_map[periodicity])
    if data.empty:
        st.error(f"No data found for {symbol} in the given range.")
    else:
        data['Return'] = data['Close'].pct_change()
        returns = data['Return'].dropna()
        mu, sigma = returns.mean(), returns.std()

        # Find smallest symmetric interval achieving >= min_prob
        z_vals = np.linspace(0.01, 5, 1000)
        probs = norm.cdf(z_vals) - norm.cdf(-z_vals)
        widths = 2 * z_vals * sigma
        valid = probs >= min_prob
        if not valid.any():
            st.warning(f"No symmetric interval up to 5σ achieves {min_prob:.0%} coverage.")
            st.stop()
        idx = np.argmin(widths[valid])
        z_opt = z_vals[valid][idx]
        lower_opt, upper_opt = mu - z_opt * sigma, mu + z_opt * sigma
        prob_opt = norm.cdf(z_opt) - norm.cdf(-z_opt)
        width_opt = upper_opt - lower_opt

        # Prepare PDF
        x = np.linspace(returns.min(), returns.max(), 1000)
        pdf = norm.pdf(x, loc=mu, scale=sigma)

        # Determine threshold mask and probability
        if operator == "≥":
            thr = threshold / 100.0
            z_thr = (thr - mu) / sigma
            prob_thr = 1 - norm.cdf(z_thr)
            mask_thr = x >= thr
            thr_label = f"Return ≥ {threshold:.2f}%"
        elif operator == "≤":
            thr = threshold / 100.0
            z_thr = (thr - mu) / sigma
            prob_thr = norm.cdf(z_thr)
            mask_thr = x <= thr
            thr_label = f"Return ≤ {threshold:.2f}%"
        else:
            lower = lower_thr / 100.0
            upper = upper_thr / 100.0
            z_lower = (lower - mu) / sigma
            z_upper = (upper - mu) / sigma
            prob_thr = norm.cdf(z_upper) - norm.cdf(z_lower)
            mask_thr = (x >= lower) & (x <= upper)
            thr_label = f"Return between {lower_thr:.2f}% and {upper_thr:.2f}%"

        # Legend labels
        interval_label = f"Optimal interval [{lower_opt:.2%}, {upper_opt:.2%}] ({prob_opt:.2%})"
        threshold_label = f"{thr_label} ({prob_thr:.2%})"

        # Figure 1: Optimal Interval only in legend, legend at bottom
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=returns,
            histnorm='probability density',
            nbinsx=50,
            showlegend=False,
            marker=dict(color='lightgrey', line=dict(color='black', width=1))
        ))
        fig1.add_trace(go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            showlegend=False,
            line=dict(color='black', width=2)
        ))
        mask_opt = (x >= lower_opt) & (x <= upper_opt)
        fig1.add_trace(go.Scatter(
            x=x[mask_opt],
            y=pdf[mask_opt],
            fill='tozeroy',
            name=interval_label,
            fillcolor='rgba(128,0,128,0.3)',
            line=dict(color='rgba(128,0,128,0)')
        ))
        fig1.update_layout(
            title=f"{symbol} Returns: Optimal Interval",
            xaxis_title="Return",
            yaxis_title="Density",
            bargap=0.1,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.2,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='black'
            )
        )
        fig1.update_xaxes(tickformat='.2%')
        st.plotly_chart(fig1, use_container_width=True)

        # Figure 2: Threshold only in legend, legend at bottom
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=returns,
            histnorm='probability density',
            nbinsx=50,
            showlegend=False,
            marker=dict(color='lightgrey', line=dict(color='black', width=1))
        ))
        fig2.add_trace(go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            showlegend=False,
            line=dict(color='black', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=x[mask_thr],
            y=pdf[mask_thr],
            fill='tozeroy',
            name=threshold_label,
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='rgba(255,0,0,0)')
        ))
        fig2.update_layout(
            title=f"{symbol} Returns: Threshold Tail",
            xaxis_title="Return",
            yaxis_title="Density",
            bargap=0.1,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.2,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='black'
            )
        )
        fig2.update_xaxes(tickformat='.2%')
        st.plotly_chart(fig2, use_container_width=True)

        # Summary Statistics
        vol_ann = sigma * np.sqrt(252)
        st.subheader("Summary Statistics")
        st.write(f"**Mean return:** {mu:.2%}")
        st.write(f"**Std dev return:** {sigma:.2%}")
        st.write(f"**Annualized volatility:** {vol_ann:.2%}")
        st.write("---")
        st.write(f"**Optimal symmetric interval achieving ≥ {min_prob:.0%} coverage:**")
        st.write(f"Range: [{lower_opt:.2%}, {upper_opt:.2%}]" )
        st.write(f"Width: {width_opt:.2%}")
        st.write(f"Probability covered: {prob_opt:.2%}")
        st.write("---")
        st.write(f"**Threshold event:** {thr_label}")
        st.write(f"Probability of event: {prob_thr:.2%}")
