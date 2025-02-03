import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ------------------------------
# Define the Portfolio Optimization Agent
# ------------------------------
class PortfolioOptimizationAgent:
    def __init__(self, tickers, investment_value=10000.0, period="1y"):
        """
        Initialize the agent with a list of tickers, total investment value, and data period.
        """
        self.tickers = [ticker.upper() for ticker in tickers]
        self.investment_value = investment_value
        self.period = period
        self.data = None
        self.optimal_weights = None
        self.performance = None  # Tuple: (expected return, volatility, sharpe ratio)

    def fetch_data(self):
        """
        Fetch historical data from yfinance and extract the price column.
        Handles multiple tickers by checking if the data is returned as a MultiIndex.
        """
        st.info(f"Fetching data for: {', '.join(self.tickers)}")
        data = yf.download(self.tickers, period=self.period)

        # Handle multiple tickers: yfinance returns a DataFrame with a MultiIndex for columns.
        if isinstance(data.columns, pd.MultiIndex):
            # Try "Adj Close" first; if not, fall back to "Close"
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                data = data["Close"]
            else:
                st.error("Data does not contain 'Adj Close' or 'Close'.")
                return
        else:
            # Single ticker: ensure data is in DataFrame format
            if "Adj Close" in data.columns:
                data = data[["Adj Close"]]
            elif "Close" in data.columns:
                data = data[["Close"]]
            else:
                st.error("Data does not contain an 'Adj Close' or 'Close' column.")
                return

        data.dropna(inplace=True)
        self.data = data

    def optimize_portfolio(self):
        """
        Compute the optimal portfolio weights by:
          - Calculating expected returns and the sample covariance matrix.
          - Optimizing for maximum Sharpe ratio.
        """
        if self.data is None:
            st.error("No data available. Please check your ticker symbols.")
            return

        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)

        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
            self.optimal_weights = ef.clean_weights()  # Clean the weights (rounding, etc.)
            self.performance = ef.portfolio_performance(verbose=False)
        except Exception as e:
            st.error(f"Error during optimization: {e}")

    def get_portfolio_recommendation(self):
        """
        Returns a dictionary with:
          - Optimal weights
          - Expected annual return
          - Annual volatility
          - Sharpe ratio
          - Investment allocation (in $)
        """
        if self.optimal_weights is None or self.performance is None:
            st.error("Portfolio optimization did not complete.")
            return None

        exp_return, volatility, sharpe_ratio = self.performance
        recommendation = {
            "optimal_weights": self.optimal_weights,
            "expected_annual_return": exp_return,
            "annual_volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "investment_allocation": {ticker: weight * self.investment_value
                                      for ticker, weight in self.optimal_weights.items()}
        }
        return recommendation

# ------------------------------
# Streamlit User Interface
# ------------------------------

st.set_page_config(page_title="Portfolio Optimization Agent", layout="wide")

st.title("Portfolio Optimization Agent Demo")
st.markdown("Enter stock tickers (comma-separated) and your total investment value, then click the button below to optimize your portfolio.")

# User inputs
tickers_input = st.text_input("Stock Tickers", value="AAPL, MSFT, GOOGL")
investment_value = st.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

if st.button("Optimize Portfolio"):
    # Clean input tickers
    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
    if len(tickers) < 2:
        st.error("Please enter at least 2 stock tickers.")
    else:
        # Create an instance of the agent and run the optimization
        agent = PortfolioOptimizationAgent(tickers, investment_value)
        agent.fetch_data()
        if agent.data is not None:
            agent.optimize_portfolio()
            recommendation = agent.get_portfolio_recommendation()

            if recommendation:
                st.markdown("## Optimal Portfolio Weights")
                for ticker, weight in recommendation["optimal_weights"].items():
                    st.write(f"**{ticker}:** {weight:.4f}")

                st.markdown("## Portfolio Performance")
                exp_return = recommendation["expected_annual_return"]
                volatility = recommendation["annual_volatility"]
                sharpe = recommendation["sharpe_ratio"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Annual Return", f"{exp_return*100:.2f}%")
                col2.metric("Annual Volatility", f"{volatility*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

                st.markdown("## Investment Allocation (in $)")
                allocation = recommendation["investment_allocation"]
                alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=["$ Allocation"])
                st.dataframe(alloc_df.style.format({"$ Allocation": "${:,.2f}"}))
