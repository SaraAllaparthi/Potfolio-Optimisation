import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt

# --- Page Config & Custom CSS ---
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional: Custom CSS to improve layout readability
st.markdown(
    """
    <style>
    .main { 
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    .big-font {
        font-size:20px !important;
    }
    .header {
        font-size:26px;
        font-weight: bold;
    }
    .subheader {
        font-size:20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar for Inputs ---
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN")
investment_value = st.sidebar.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

# --- Main Title ---
st.title("Portfolio Optimization Dashboard")
st.markdown("#### Optimize your portfolio based on investment value using live data from yfinance.")

if tickers_input:
    # Clean and prepare ticker list
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    st.markdown(f"### Fetching Historical Data for: **{', '.join(tickers)}**")
    
    # Download historical data for 1 year
    data = yf.download(tickers, period="1y")
    
    # --- Debug Section (hidden by default) ---
    with st.expander("Debug Info: Data Structure", expanded=False):
        st.write("Data Columns:", data.columns)
    
    data_extracted = False

    # Handle MultiIndex: for multiple tickers, yfinance returns a MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        st.info("MultiIndex detected in downloaded data.")
        st.write("**First level (fields):**", list(data.columns.get_level_values(0).unique()))
        st.write("**Second level (tickers):**", list(data.columns.get_level_values(1).unique()))
        # Try to extract "Adj Close" first; if not available, fallback to "Close"
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
            data_extracted = True
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
            data_extracted = True
        else:
            st.error("The data does not contain 'Adj Close' or 'Close' in the first level.")
    else:
        # For a single ticker, ensure it is a DataFrame with one column.
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
            data_extracted = True
        elif "Close" in data.columns:
            data = data[["Close"]]
            data_extracted = True
        else:
            st.error("The data does not contain an 'Adj Close' or 'Close' column.")

    if not data_extracted:
        st.stop()

    if data.empty:
        st.error("No data was fetched. Please check the ticker symbols or try again later.")
    else:
        data.dropna(inplace=True)
        st.success("Historical data successfully fetched!")
        
        # --- Portfolio Optimization ---
        st.markdown("## Portfolio Optimization")
        st.markdown("Calculating expected returns and optimizing for maximum Sharpe ratio...")
        
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Optimize portfolio (maximizing Sharpe ratio)
        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
        except Exception as e:
            st.error(f"Error during optimization: {e}")
            st.stop()
        
        # Display Optimal Weights in a clean format
        st.markdown("### Optimal Portfolio Weights")
        for ticker, weight in cleaned_weights.items():
            st.write(f"**{ticker}:** {weight:.4f}")
        
        # Display expected performance
        try:
            exp_return, exp_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
        except Exception as e:
            st.error(f"Error calculating portfolio performance: {e}")
            st.stop()
        
        st.markdown("### Portfolio Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{exp_return*100:.2f}%")
        col2.metric("Annual Volatility", f"{exp_volatility*100:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # --- Investment Allocation ---
        st.markdown("### Investment Allocation (in $)")
        allocation = {ticker: weight * investment_value for ticker, weight in cleaned_weights.items()}
        alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=["$ Allocation"])
        st.dataframe(alloc_df.style.format({"$ Allocation": "${:,.2f}"}))
        
        # --- Hypothetical Portfolio Value Over Time ---
        st.markdown("### Hypothetical Portfolio Value Over Time")
        daily_returns = data.pct_change().dropna()
        weights_series = pd.Series(cleaned_weights)
        
        if len(weights_series) == 1:
            portfolio_daily_returns = daily_returns * weights_series.iloc[0]
        else:
            portfolio_daily_returns = (daily_returns * weights_series).sum(axis=1)
        
        portfolio_value = (1 + portfolio_daily_returns).cumprod() * investment_value
        st.line_chart(portfolio_value)
        
        # --- Efficient Frontier Plot ---
        st.markdown("### Efficient Frontier (Optional)")
        try:
            from pypfopt import plotting
            # Create a new instance for plotting
            ef_plot = EfficientFrontier(mu, S)
            fig, ax = plt.subplots(figsize=(6, 4))
            plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
            st.pyplot(fig)
        except Exception as e:
            st.warning("Efficient Frontier plot could not be generated.")
            st.write(e)
        
        # --- Disclaimer ---
        st.markdown("### Disclaimer")
        st.info("This dashboard is for informational and educational purposes only and does not constitute financial advice.")
