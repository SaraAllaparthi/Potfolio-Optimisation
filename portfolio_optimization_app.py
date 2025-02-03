# portfolio_optimization_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title("Portfolio Optimization Dashboard")
st.caption("Optimize your portfolio based on investment value using live data from yfinance.")

st.write("### Portfolio Input")
st.write("Enter a list of stock tickers (comma-separated) and your total investment amount.")

# User inputs
tickers_input = st.text_input("Stock Tickers", value="AAPL, MSFT, GOOGL, AMZN")
investment_value = st.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    st.write(f"Fetching historical data for: {', '.join(tickers)}")
    
    # Fetch 1 year of historical adjusted closing price data
    data = yf.download(tickers, period="1y")["Adj Close"]
    
    if data.empty:
        st.error("No data was fetched. Please check your ticker symbols.")
    else:
        # Drop rows with any missing data
        data.dropna(inplace=True)
        st.write("Historical data successfully fetched!")
        
        st.write("### Portfolio Optimization")
        st.write("Calculating expected returns and optimizing for maximum Sharpe ratio...")
        
        # Calculate expected returns and the sample covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Optimize the portfolio for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        st.write("#### Optimal Weights")
        st.write(cleaned_weights)
        
        # Calculate expected portfolio performance
        exp_return, exp_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
        st.write(f"**Expected Annual Return:** {exp_return*100:.2f}%")
        st.write(f"**Expected Annual Volatility:** {exp_volatility*100:.2f}%")
        st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
        
        # Display dollar allocation
        st.write("#### Investment Allocation (in $)")
        allocation = {ticker: weight * investment_value for ticker, weight in cleaned_weights.items()}
        st.write(allocation)
        
        # Plot a hypothetical portfolio value over time
        st.write("### Hypothetical Portfolio Value Over Time")
        # Compute daily returns weighted by the portfolio allocation
        daily_returns = data.pct_change().dropna()
        # Convert weights into a Series aligned with the data columns
        weights_series = pd.Series(cleaned_weights)
        portfolio_daily_returns = (daily_returns * weights_series).sum(axis=1)
        portfolio_value = (1 + portfolio_daily_returns).cumprod() * investment_value
        
        st.line_chart(portfolio_value)
        
        # Additional plot: Efficient Frontier (optional)
        st.write("### Efficient Frontier (Optional)")
        try:
            from pypfopt import plotting
            fig, ax = plt.subplots(figsize=(6, 4))
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            st.pyplot(fig)
        except Exception as e:
            st.write("Could not generate efficient frontier plot.", e)

        st.write("### Disclaimer")
        st.write("This dashboard is for informational and educational purposes only and does not constitute financial advice.")
