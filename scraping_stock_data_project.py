
# Import Required Modules


import yfinance as yf                 # For fetching financial data
import pandas as pd                  # For data manipulation
import numpy as np                   # For numerical operations
import matplotlib.pyplot as plt      # For plotting
import seaborn as sns                # For enhanced visualization
import sqlite3                       # For in-memory SQL-based operations
from statsmodels.tsa.arima.model import ARIMA  # For time-series forecasting


# Stock and Crypto Fetching


# Download historical stock data from Yahoo Finance
def fetch_stock_data(tickers, period="6mo"):
    data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True)
    return data

# Display detailed information for a specific stock
def fetch_additional_info(ticker):
    stock = yf.Ticker(ticker)
    print(f"\nFetching additional data for {ticker}...")
    print("Market Cap:", stock.info.get("marketCap", "N/A"))
    print("P/E Ratio:", stock.info.get("trailingPE", "N/A"))
    print("Dividend Yield:", stock.info.get("dividendYield", "N/A"))
    print("Recent Dividends:\n", stock.dividends.tail())
    print("Stock Splits:\n", stock.splits.tail())
    print("Financials:\n", stock.financials.head())
    print("Balance Sheet:\n", stock.balance_sheet.head())
    print("Cash Flow Statement:\n", stock.cashflow.head())
    print("Analyst Recommendations:\n", stock.recommendations.tail())
    if stock.options:
        print("Available Option Expiration Dates:", stock.options)

# Fetch recent data for a cryptocurrency
def fetch_crypto_data(crypto_ticker):
    crypto = yf.Ticker(crypto_ticker)
    print(f"\nFetching cryptocurrency data for {crypto_ticker}...")
    print(crypto.history(period="1mo"))

# Save any DataFrame to a CSV file
def save_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data saved to {filename}")


# Dashboard Visualization


# Build and display plots for multiple aspects of stock behavior
def visualization_dashboard(stock_history, tickers):
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), constrained_layout=True)
    axes = axes.flatten()

    # Closing price trends
    for ticker in tickers:
        stock_history[ticker]['Close'].plot(ax=axes[0], label=f'{ticker} Closing Price')
    axes[0].set_title('Closing Price Trends')
    axes[0].legend()
    axes[0].grid()

    # Correlation matrix of closing prices
    close_prices = pd.DataFrame({ticker: stock_history[ticker]['Close'] for ticker in tickers})
    sns.heatmap(close_prices.corr(), annot=True, cmap='coolwarm', ax=axes[1])
    axes[1].set_title('Correlation Matrix')

    # Volume trends
    volume = pd.DataFrame({ticker: stock_history[ticker]['Volume'] for ticker in tickers})
    volume.plot(ax=axes[2])
    axes[2].set_title('Trading Volume Trends')

    # Daily returns
    daily_returns = close_prices.pct_change()
    daily_returns.plot(ax=axes[3])
    axes[3].set_title('Daily Returns')

    # Daily returns distribution (histograms with KDE)
    for i, ticker in enumerate(tickers[:2]):
        sns.histplot(daily_returns[ticker].dropna(), kde=True, ax=axes[4 + i], color='orange')
        axes[4 + i].set_title(f'{ticker} Daily Returns Distribution')

    plt.show()


# Advanced SQL + ARIMA Analysis


def advanced_analysis():
    # Load previously saved CSVs
    apple_df = pd.read_csv('AAPL_stock_data.csv')
    stocks_df = pd.read_csv('stocks_data.csv', skiprows=2)

    # Rename columns for clarity and consistency
    stocks_df.columns = [
        'Date', 'AAPL_Close', 'GOOGL_Close', 'TSLA_Close',
        'AAPL_High', 'GOOGL_High', 'TSLA_High',
        'AAPL_Low', 'GOOGL_Low', 'TSLA_Low',
        'AAPL_Open', 'GOOGL_Open', 'TSLA_Open',
        'AAPL_Volume', 'GOOGL_Volume', 'TSLA_Volume'
    ]

    # Convert date columns to datetime format
    apple_df['Date'] = pd.to_datetime(apple_df['Date'], utc=True).dt.tz_convert(None)
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

    # Create in-memory SQLite database and load data
    conn = sqlite3.connect(':memory:')
    apple_df.to_sql('apple_stock', conn, index=False)
    stocks_df.to_sql('stocks', conn, index=False)

    # Join Apple and other stock data on the same date
    query = """
    SELECT a.Date, a.Open AS AAPL_Open_1, a.High AS AAPL_High_1, a.Low AS AAPL_Low_1, a.Close AS AAPL_Close_1, a.Volume AS AAPL_Volume_1,
           s.AAPL_Open AS AAPL_Open_2, s.AAPL_High AS AAPL_High_2, s.AAPL_Low AS AAPL_Low_2, s.AAPL_Close AS AAPL_Close_2, s.AAPL_Volume AS AAPL_Volume_2,
           s.GOOGL_Open, s.GOOGL_High, s.GOOGL_Low, s.GOOGL_Close, s.GOOGL_Volume,
           s.TSLA_Open, s.TSLA_High, s.TSLA_Low, s.TSLA_Close, s.TSLA_Volume
    FROM apple_stock a
    JOIN stocks s ON date(a.Date) = date(s.Date)
    """
    combined_df = pd.read_sql(query, conn, parse_dates=['Date'])
    combined_df.set_index('Date', inplace=True)

    # Calculate daily percentage returns
    combined_df['AAPL_Return'] = combined_df['AAPL_Close_2'].pct_change()
    combined_df['GOOGL_Return'] = combined_df['GOOGL_Close'].pct_change()
    combined_df['TSLA_Return'] = combined_df['TSLA_Close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    volatility = combined_df[['AAPL_Return', 'GOOGL_Return', 'TSLA_Return']].std()

    # Apply ARIMA model to forecast AAPL prices
    model = ARIMA(combined_df['AAPL_Close_2'].dropna(), order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=15)

    # Value at Risk (VaR) and Conditional VaR (CVaR) calculation
    var_cvar = {}
    confidence = 0.95
    for stock in ['AAPL_Return', 'GOOGL_Return', 'TSLA_Return']:
        var = np.percentile(combined_df[stock].dropna(), (1 - confidence) * 100)
        cvar = combined_df[stock][combined_df[stock] <= var].mean()
        var_cvar[stock] = {'VaR': var, 'CVaR': cvar}

    # Correlation matrix for prices and volume
    corr_matrix = combined_df[['AAPL_Close_2', 'GOOGL_Close', 'TSLA_Close', 'AAPL_Volume_2', 'GOOGL_Volume', 'TSLA_Volume']].corr()

    
    # Dashboard Visualization
    

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Heatmap of correlations
    sns.heatmap(corr_matrix, annot=True, ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Matrix')

    # ARIMA Forecast Plot
    forecast.plot(ax=axes[0, 1], title='AAPL Forecast')
    axes[0, 1].set_xlabel('Days')
    axes[0, 1].set_ylabel('Price')

    # Daily Returns Plot
    daily_returns = combined_df[['AAPL_Return', 'GOOGL_Return', 'TSLA_Return']]
    daily_returns.plot(ax=axes[1, 0], title='Daily Returns')

    # Volatility as bar plot
    volatility.plot.bar(ax=axes[1, 1], title='Volatility')

    # Cumulative returns over time
    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns.plot(ax=axes[2, 0], title='Cumulative Returns')

    # Difference between two sources of AAPL closing price
    combined_df['AAPL_Close_Diff'] = combined_df['AAPL_Close_1'] - combined_df['AAPL_Close_2']
    combined_df['AAPL_Close_Diff'].plot(ax=axes[2, 1], marker='o', linestyle='', title='AAPL Closing Price Difference')
    axes[2, 1].axhline(0, color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()
    conn.close()


# Main Execution Block


if __name__ == "__main__":
    # Get stock tickers from user
    tickers = input("Enter stock ticker symbols separated by commas (e.g., AAPL, TSLA, GOOGL): ").split(",")
    tickers = [t.strip().upper() for t in tickers]

    # Fetch and save stock data
    hist = fetch_stock_data(tickers)
    save_to_csv(hist, "stocks_data.csv")

    # Visual analysis dashboard
    visualization_dashboard(hist, tickers)

    # Optional: fetch additional info for each ticker
    for ticker in tickers:
        fetch_additional_info(ticker)

    # Optional: Crypto data
    crypto_ticker = input("Enter cryptocurrency ticker (e.g., BTC-USD) or press Enter to skip: ")
    if crypto_ticker:
        fetch_crypto_data(crypto_ticker)

    # Optional: Run advanced SQLite + ARIMA dashboard
    run_advanced = input("\nDo you want to run the advanced SQLite/ARIMA analysis (requires 'AAPL_stock_data.csv')? [y/n]: ").strip().lower()
    if run_advanced == 'y':
        advanced_analysis()
