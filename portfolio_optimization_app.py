import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

class PortfolioOptimizationAgent:
    def __init__(self, tickers, investment_value=10000.0, period="1y"):
        """
        Initialize the agent with a list of tickers, total investment value, and data period.
        :param tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
        :param investment_value: Total dollar amount for investment.
        :param period: Historical period for data fetching (default "1y").
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
        The method handles multiple tickers and attempts to extract "Adj Close"
        or falls back to "Close" if necessary.
        """
        print(f"Fetching data for: {', '.join(self.tickers)}")
        data = yf.download(self.tickers, period=self.period)
        # If multiple tickers, yfinance returns a MultiIndex for columns.
        if isinstance(data.columns, pd.MultiIndex):
            # Try to extract "Adj Close" if available; otherwise, fallback to "Close".
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                data = data["Close"]
            else:
                raise ValueError("Data does not contain 'Adj Close' or 'Close'.")
        else:
            # Single ticker: ensure we work with a DataFrame
            if "Adj Close" in data.columns:
                data = data[["Adj Close"]]
            elif "Close" in data.columns:
                data = data[["Close"]]
            else:
                raise ValueError("Data does not contain an 'Adj Close' or 'Close' column.")
        
        data.dropna(inplace=True)
        self.data = data
        print("Data fetching complete.")

    def optimize_portfolio(self):
        """
        Compute the optimal portfolio weights by:
          - Calculating expected returns and the sample covariance matrix.
          - Optimizing for maximum Sharpe ratio.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)
        
        # Optimize portfolio for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S)
        self.optimal_weights = ef.max_sharpe()
        self.optimal_weights = ef.clean_weights()  # Clean and round the weights
        
        self.performance = ef.portfolio_performance(verbose=False)
        print("Portfolio optimization complete.")

    def get_portfolio_recommendation(self):
        """
        Returns a dictionary containing the optimal weights, expected annual return,
        annual volatility, and Sharpe ratio.
        """
        if self.optimal_weights is None or self.performance is None:
            raise ValueError("Portfolio not optimized. Call optimize_portfolio() first.")
        
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

# --- Example usage ---
if __name__ == "__main__":
    # Example tickers (user can modify these or input via a UI)
    tickers = ["AAPL", "MSFT", "GOOGL"]
    investment_value = 10000.0

    # Create an instance of the agent
    agent = PortfolioOptimizationAgent(tickers=tickers, investment_value=investment_value)

    # Fetch data, optimize portfolio, and print recommendations
    agent.fetch_data()
    agent.optimize_portfolio()
    recommendation = agent.get_portfolio_recommendation()

    # Print out the results
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in recommendation["optimal_weights"].items():
        print(f"  {ticker}: {weight:.4f}")
    
    print("\nPerformance Metrics:")
    print(f"  Expected Annual Return: {recommendation['expected_annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {recommendation['annual_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {recommendation['sharpe_ratio']:.2f}")
    
    print("\nInvestment Allocation (in $):")
    for ticker, alloc in recommendation["investment_allocation"].items():
        print(f"  {ticker}: ${alloc:.2f}")
