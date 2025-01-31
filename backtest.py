import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def annualized_metrics(returns_series, periods_per_year=252):
    """Calculate annualized return, vol, and Sharpe for a daily returns series."""
    mean_ret_daily = returns_series.mean()
    std_daily = returns_series.std()
    
    ann_return = (1 + mean_ret_daily)**periods_per_year - 1
    ann_vol = std_daily * np.sqrt(periods_per_year)
    ann_sharpe = (mean_ret_daily / std_daily) * np.sqrt(periods_per_year) if std_daily != 0 else np.nan
    
    return ann_return, ann_vol, ann_sharpe
# ETFs or proxies for the indices:
sp600_ticker = "^SP600"  # SPDR S&P 600 Small Cap ETF (as a rough proxy for S&P600)
sp500_ticker = "^GSPC"  # SPDR S&P 500 ETF
magnificent_7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

#1 Year
start_date = "2023-01-01"
end_date = "2024-01-01"

# Download data from yfinance (adjusted close for total return approximation)
all_tickers = [sp600_ticker, sp500_ticker] + magnificent_7
df_prices = yf.download(all_tickers, start=start_date, end=end_date, progress=False)["Adj Close"]

df_returns = df_prices.pct_change().dropna()

ret_sp600 = df_returns[sp600_ticker]
ret_sp500 = df_returns[sp500_ticker]

# Standard Strategy daily returns
standard_strategy_returns = ret_sp600 - ret_sp500

# We'll do a naive approach: equally weight the M7 in the offset.
m7_returns = df_returns[magnificent_7]

# If we have 7 stocks, each gets weight = 1/7
m7_equal_weight_return = m7_returns.mean(axis=1)

m7_equal_weight_return = df_returns[magnificent_7].mean(axis=1)

# "Short ex-M7" daily return:
# short_sp500_return = - ret_sp500
# long_m7basket_return = + m7_equal_weight_return
# net_short_ex_m7_return = short_sp500_return + long_m7basket_return
net_short_ex_m7_return = -ret_sp500 + m7_equal_weight_return

# Then the total strategy (long S&P600, short ex-M7) daily return:
modified_strategy_returns = ret_sp600 - net_short_ex_m7_return
# Simplifies to: ret_sp600 + ret_sp500 - m7_equal_weight_return


df_strat = pd.DataFrame({
    "Standard_L600_S500": standard_strategy_returns,
    "Modified_L600_ExM7": modified_strategy_returns
})


df_cum = (1 + df_strat).cumprod()  # naive cumulative product

plt.figure(figsize=(10, 6))
for col in df_cum.columns:
    plt.plot(df_cum.index, df_cum[col], label=col)

plt.title("Cumulative Returns: Standard vs. Modified Strategy")
plt.legend()
plt.ylabel("Growth of 1 unit")
plt.show()

for strat_name in df_strat.columns:
    ann_ret, ann_vol, ann_sr = annualized_metrics(df_strat[strat_name])
    print(f"{strat_name}:")
    print(f"  Annualized Return: {ann_ret*100:.2f}%")
    print(f"  Annualized Vol   : {ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio     : {ann_sr:.2f}\n")
