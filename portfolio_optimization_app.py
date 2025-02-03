import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt

st.title("Portfolio Optimization Dashboard")
st.caption("Optimize your portfolio based on investment value using live data from yfinance.")

st.write("### Portfolio Input")
st.write("Enter a list of stock tickers (comma-separated) and your total investment amount.")

# User inputs
tickers_input = st.text_input("Stock Tickers", value="AAPL, MSFT, GOOGL, AMZN")
investment_value = st.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

if tickers_input:
    # Clean and prepare ticker list
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    st.write(f"Fetching historical data for: {', '.join(tickers)}")
    
    # Fetch 1 year of historical data
    data = yf.download(tickers, period="1y")
    
    # Debug: Show the columns and structure of the DataFrame
    st.write("### Debug Info: Data Columns")
    st.write(data.columns)
    
    # Initialize a flag for successful extraction
    data_extracted = False
    
    # Check if the DataFrame has a MultiIndex (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        st.write("MultiIndex detected. First level:", list(data.columns.get_level_values(0).unique()))
        # Try the common options for adjusted close
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
            data_extracted = True
        elif "Adj_Close" in data.columns.get_level_values(0):
            data = data["Adj_Close"]
            data_extracted = True
        else:
            st.error("The MultiIndex data does not contain 'Adj Close' or 'Adj_Close' in the first level.")
    else:
        # For a single ticker, try to extract a DataFrame with one column
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
            data_extracted = True
        elif "Adj_Close" in data.columns:
            data = data[["Adj_Close"]]
            data_extracted = True
        else:
            st.error("The data does not contain an 'Adj Close' or 'Adj_Close' column.")
    
    if not data_extracted:
        st.stop()
    
    if data.empty:
        st.error("No data was fetched. Please check your ticker symbols or the data source.")
    else:
        # Drop rows with missing data
        data.dropna(inplace=True)
        st.write("Historical data successfully fetched!")
        
        st.write("### Portfolio Optimization")
        st.write("Calculating expected returns and optimizing for maximum Sharpe ratio...")
        
        # Calculate expected returns and the sample covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Optimize the portfolio for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
        except Exception as e:
            st.error(f"Error during portfolio optimization: {e}")
            st.stop()
        
        st.write("#### Optimal Weights")
        st.write(cleaned_weights)
        
        # Calculate expected portfolio performance
        try:
            exp_return, exp_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
        except Exception as e:
            st.error(f"Error calculating portfolio performance: {e}")
            st.stop()
        
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
        weights_series = pd.Series(cleaned_weights)
        
        if len(weights_series) == 1:
            portfolio_daily_returns = daily_returns * weights_series.iloc[0]
        else:
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
