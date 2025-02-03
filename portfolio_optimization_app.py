import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import plotly.express as px

# --- Page Config & Custom CSS ---
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
st.sidebar.markdown("Enter **2 to 5** stock tickers (comma-separated) below:")
tickers_input = st.sidebar.text_input("Stock Tickers")
investment_value = st.sidebar.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

# --- Main Title ---
st.title("Portfolio Optimization Dashboard")
st.markdown("#### Optimize your portfolio based on investment value using live data from yfinance.")

# Validate ticker input: require 2 to 5 tickers
if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    if len(tickers) < 2:
        st.error("Please enter at least **2** stock tickers.")
    elif len(tickers) > 5:
        st.error("Please enter no more than **5** stock tickers.")
    else:
        st.markdown(f"### Fetching Historical Data for: **{', '.join(tickers)}**")
        
        # Download historical data for 1 year
        data = yf.download(tickers, period="1y")
        
        # Remove debug info for a clean dashboard
        data_extracted = False

        if isinstance(data.columns, pd.MultiIndex):
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
            
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            ef = EfficientFrontier(mu, S)
            try:
                weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                st.stop()
            
            st.markdown("### Optimal Portfolio Weights")
            for ticker, weight in cleaned_weights.items():
                st.write(f"**{ticker}:** {weight:.4f}")
            
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
            
            st.markdown("### Investment Allocation (in $)")
            allocation = {ticker: weight * investment_value for ticker, weight in cleaned_weights.items()}
            alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=["$ Allocation"])
            st.dataframe(alloc_df.style.format({"$ Allocation": "${:,.2f}"}))
            
            st.markdown("### Hypothetical Portfolio Value Over Time")
            daily_returns = data.pct_change().dropna()
            weights_series = pd.Series(cleaned_weights)
            if len(weights_series) == 1:
                portfolio_daily_returns = daily_returns * weights_series.iloc[0]
            else:
                portfolio_daily_returns = (daily_returns * weights_series).sum(axis=1)
            
            portfolio_value = (1 + portfolio_daily_returns).cumprod() * investment_value
            st.line_chart(portfolio_value)
            
            # --- Efficient Frontier Plot (Interactive) ---
            st.markdown("### Efficient Frontier")
            st.markdown(
                """
                **Efficient Frontier Explained:**  
                This curve shows you the best balance between risk and return you can achieve with your chosen stocks.  
                Portfolios on this curve are considered optimal—they offer the highest expected return for a given level of risk.
                """
            )
            
            # Calculate a range of target returns along the frontier
            points = 50
            target_returns = np.linspace(mu.min(), mu.max(), points)
            frontier_vols = []
            frontier_rets = []
            
            for target in target_returns:
                ef_temp = EfficientFrontier(mu, S)
                try:
                    ef_temp.efficient_return(target)
                    ret, vol, _ = ef_temp.portfolio_performance(verbose=False)
                    frontier_vols.append(vol)
                    frontier_rets.append(ret)
                except Exception:
                    # If optimization fails for a target, skip it.
                    pass
            
            df_frontier = pd.DataFrame({
                "Volatility (Risk)": frontier_vols,
                "Expected Return": frontier_rets
            })
            fig = px.line(
                df_frontier,
                x="Volatility (Risk)",
                y="Expected Return",
                title="Efficient Frontier: Optimal Risk-Return Trade-off",
                labels={"Volatility (Risk)": "Risk (Volatility)", "Expected Return": "Expected Return"}
            )
            fig.update_traces(mode="markers+lines", marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Disclaimer")
            st.info("This dashboard is for informational and educational purposes only and does not constitute financial advice.")
