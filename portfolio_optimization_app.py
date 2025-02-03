import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier
import plotly.express as px

# -------------------------------
# Page Configuration & Custom CSS
# -------------------------------
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* General layout */
    body {
        background-color: #f0f2f6;
        color: #333333;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Main container styling */
    .css-18e3th9 {
        padding: 2rem;
    }
    /* Title styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2, h3, h4 {
        color: #34495e;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #dfe6e9;
    }
    /* Button styling */
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    /* Metric styling */
    .css-1um0q6e {
        background-color: #ecf0f1;
        border: 1px solid #bdc3c7;
        border-radius: 5px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Portfolio Inputs")
st.sidebar.markdown("Enter **2 to 5** stock tickers (comma-separated):")
tickers_input = st.sidebar.text_input("Stock Tickers")
investment_value = st.sidebar.number_input("Total Investment Value ($)", value=10000.0, step=100.0)

# -------------------------------
# Main Title
# -------------------------------
st.title("AI Portfolio Optimization Agent")
st.markdown("#### This agent automatically optimizes your portfolio using historical data and efficient frontier logic.")

# -------------------------------
# Helper Functions
# -------------------------------
def fetch_price_data(tickers, period="1y"):
    """
    Fetch historical price data from yfinance and extract the adjusted close (or close) prices.
    """
    data = yf.download(tickers, period=period)
    data_extracted = False
    if isinstance(data.columns, pd.MultiIndex):
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
        return None
    data.dropna(inplace=True)
    return data

def optimal_portfolio_from_frontier(mu, S, target_returns):
    """
    Loop over a range of target returns using the efficient frontier and return the portfolio with the highest Sharpe ratio.
    """
    best_sharpe = -np.inf
    best_weights = None
    best_perf = None  # (expected return, volatility, sharpe ratio)
    
    for target in target_returns:
        ef_temp = EfficientFrontier(mu, S)
        try:
            ef_temp.efficient_return(target)
            ret, vol, sr = ef_temp.portfolio_performance(verbose=False)
            if sr > best_sharpe:
                best_sharpe = sr
                best_weights = ef_temp.clean_weights()
                best_perf = (ret, vol, sr)
        except Exception:
            continue
    return best_weights, best_perf

# -------------------------------
# Main Logic
# -------------------------------
if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    if len(tickers) < 2:
        st.error("Please enter at least **2** stock tickers.")
    elif len(tickers) > 5:
        st.error("Please enter no more than **5** stock tickers.")
    else:
        st.markdown(f"### Fetching Historical Data for: **{', '.join(tickers)}**")
        data = fetch_price_data(tickers, period="1y")
        if data is None or data.empty:
            st.error("No data was fetched. Please check the ticker symbols or try again later.")
        else:
            st.success("Historical data successfully fetched!")
            
            # -------------------------------
            # Portfolio Optimization Using Historical Returns
            # -------------------------------
            st.markdown("## Portfolio Optimization")
            st.markdown("Calculating expected returns and constructing the efficient frontier...")
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            # Create a range of target returns – use a slightly wider range for flexibility.
            target_returns = np.linspace(mu.min() * 0.9, mu.max() * 1.1, 50)
            opt_weights, opt_perf = optimal_portfolio_from_frontier(mu, S, target_returns)
            
            # Fallback to max_sharpe() if frontier search fails
            if opt_weights is None:
                st.warning("Efficient frontier search failed; falling back to max_sharpe() optimization.")
                try:
                    ef = EfficientFrontier(mu, S)
                    opt_weights = ef.max_sharpe()
                    opt_weights = ef.clean_weights()
                    opt_perf = ef.portfolio_performance(verbose=False)
                except Exception as e:
                    st.error(f"Error during fallback optimization: {e}")
                    st.stop()
            
            st.markdown("### Optimal Portfolio Weights")
            for ticker, weight in opt_weights.items():
                st.write(f"**{ticker}:** {weight:.4f}")
            
            exp_return, exp_volatility, sharpe_ratio = opt_perf
            st.markdown("### Portfolio Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Annual Return", f"{exp_return*100:.2f}%")
            col2.metric("Annual Volatility", f"{exp_volatility*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            st.markdown("### Investment Allocation (in $)")
            allocation = {ticker: weight * investment_value for ticker, weight in opt_weights.items()}
            alloc_df = pd.DataFrame.from_dict(allocation, orient="index", columns=["$ Allocation"])
            st.dataframe(alloc_df.style.format({"$ Allocation": "${:,.2f}"}))
            
            # -------------------------------
            # Cumulative Portfolio Value Over Time
            # -------------------------------
            st.markdown("### Cumulative Portfolio Value Over Time")
            daily_returns = data.pct_change().dropna()
            weights_series = pd.Series(opt_weights)
            if len(weights_series) == 1:
                portfolio_daily_returns = daily_returns * weights_series.iloc[0]
            else:
                portfolio_daily_returns = (daily_returns * weights_series).sum(axis=1)
            portfolio_value = (1 + portfolio_daily_returns).cumprod() * investment_value
            st.line_chart(portfolio_value)
            
            # -------------------------------
            # Efficient Frontier Plot (Interactive)
            # -------------------------------
            st.markdown("### Efficient Frontier")
            st.markdown(
                """
                **What is the Efficient Frontier?**  
                It represents the best balance between risk and return – the "sweet spot" portfolios that offer the highest expected return for a given level of risk.
                """
            )
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
                    continue
            df_frontier = pd.DataFrame({
                "Risk (Volatility)": frontier_vols,
                "Expected Return": frontier_rets
            })
            fig = px.line(
                df_frontier,
                x="Risk (Volatility)",
                y="Expected Return",
                title="Efficient Frontier: Optimal Risk-Return Trade-off",
                labels={"Risk (Volatility)": "Risk (Volatility)", "Expected Return": "Expected Return"}
            )
            fig.update_traces(mode="markers+lines", marker=dict(size=8, color="#2c3e50"))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Disclaimer")
            st.info("This dashboard is for informational and educational purposes only and does not constitute financial advice.")
