import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.express as px
import matplotlib.pyplot as plt

# ------------------------------
# Portfolio Optimization Agent Class
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
        Fetch historical data from yfinance and extract the appropriate price column.
        """
        st.info(f"Fetching data for: {', '.join(self.tickers)}")
        data = yf.download(self.tickers, period=self.period)

        # Handle multiple tickers (MultiIndex) or a single ticker.
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                data = data["Close"]
            else:
                st.error("Data does not contain 'Adj Close' or 'Close'.")
                return
        else:
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
        Calculate expected returns, covariance matrix, and optimize the portfolio for maximum Sharpe ratio.
        """
        if self.data is None:
            st.error("No data available. Please check your ticker symbols.")
            return

        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)

        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
            self.optimal_weights = ef.clean_weights()
            self.performance = ef.portfolio_performance(verbose=False)
        except Exception as e:
            st.error(f"Error during optimization: {e}")

    def get_portfolio_recommendation(self):
        """
        Returns a dictionary containing:
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
st.markdown("Enter stock tickers (comma-separated) and your total investment value, then click **Optimize Portfolio** to see your results.")

# Sidebar Inputs
st.sidebar.header("Portfolio Inputs")
st.sidebar.markdown("Enter **2 to 5** stock tickers (comma-separated):")
tickers_input = st.sidebar.text_input("Stock Tickers")
investment_value = st.sidebar.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

if st.button("Optimize Portfolio"):
    # Clean input tickers
    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
    if len(tickers) < 2:
        st.error("Please enter at least 2 stock tickers.")
    elif len(tickers) > 5:
        st.error("Please enter no more than 5 stock tickers.")
    else:
        # Instantiate and run the agent
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

                # Calculate portfolio daily returns and cumulative value over time.
                daily_returns = agent.data.pct_change().dropna()
                weights_series = pd.Series(agent.optimal_weights)
                if len(weights_series) == 1:
                    portfolio_daily_returns = daily_returns * weights_series.iloc[0]
                else:
                    portfolio_daily_returns = (daily_returns * weights_series).sum(axis=1)
                portfolio_value = (1 + portfolio_daily_returns).cumprod() * investment_value

                st.markdown("## Cumulative Portfolio Value Over Time")
                st.line_chart(portfolio_value)

                # Additional Chart: Monthly Returns
                st.markdown("## Monthly Returns")
                # Compute monthly returns from daily returns
                monthly_returns = portfolio_daily_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_df = monthly_returns.reset_index()
                monthly_returns_df["Month"] = monthly_returns_df["Date"].dt.strftime("%b %Y")
                fig = px.bar(monthly_returns_df, x="Month", y=portfolio_daily_returns.name or "Return",
                             labels={"y": "Monthly Return"},
                             title="Monthly Portfolio Returns")
                st.plotly_chart(fig, use_container_width=True)
