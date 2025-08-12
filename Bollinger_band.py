import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === CONFIGURATION ===
FILE = "test_data.csv"
INITIAL_BALANCE = 1000
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40
PIP_VALUE = 0.0001
SPREAD = 0.0005

# === ALWAYS CREATE SAMPLE DATA ===
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=300, freq="H")
price = 1.1000 + np.cumsum(np.random.randn(len(dates)) * 0.0005)
df_sample = pd.DataFrame({
    "Gmt time": dates,
    "Open": price,
    "High": price + np.random.rand(len(dates)) * 0.0010,
    "Low": price - np.random.rand(len(dates)) * 0.0010,
    "Close": price + (np.random.rand(len(dates)) - 0.5) * 0.0010
})
df_sample.to_csv(FILE, index=False)

# === LOAD DATA ===
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df['Gmt time'] = pd.to_datetime(df['Gmt time'])
df.set_index('Gmt time', inplace=True)
df.sort_index(inplace=True)

for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

# === INDICATORS ===
df['MA'] = df['Close'].rolling(window=20).mean()
df['STD'] = df['Close'].rolling(window=20).std()
df['Upper'] = df['MA'] + 2 * df['STD']
df['Lower'] = df['MA'] - 2 * df['STD']

# === SIGNALS ===
df['Signal'] = None
position = False

for i in range(20, len(df)):
    close = df['Close'].iloc[i]
    upper = df['Upper'].iloc[i]
    lower = df['Lower'].iloc[i]

    if close < lower and not position:
        df.iloc[i, df.columns.get_loc('Signal')] = 'Buy'
        position = True
    elif close > upper and position:
        df.iloc[i, df.columns.get_loc('Signal')] = 'Sell'
        position = False

# === BACKTEST ===
balance = INITIAL_BALANCE
equity = [balance]
positions = []
open_trade = None

for i in range(len(df)):
    row = df.iloc[i]
    price = row['Close']
    signal = row['Signal']

    if signal == 'Buy' and not open_trade:
        entry = price + SPREAD
        sl = entry - STOP_LOSS_PIPS * PIP_VALUE
        tp = entry + TAKE_PROFIT_PIPS * PIP_VALUE
        open_trade = {'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': row.name}

    elif open_trade:
        high = row['High']
        low = row['Low']
        exit_price = None
        result = None

        if low <= open_trade['sl']:
            exit_price = open_trade['sl']
            result = 'Loss'
        elif high >= open_trade['tp']:
            exit_price = open_trade['tp']
            result = 'Win'

        if exit_price:
            pips = (exit_price - open_trade['entry']) / PIP_VALUE
            balance += pips
            positions.append({
                'entry_time': open_trade['entry_time'],
                'exit_time': row.name,
                'entry': open_trade['entry'],
                'exit': exit_price,
                'result': result,
                'pips': round(pips, 2),
                'balance': round(balance, 2)
            })
            open_trade = None

    equity.append(balance)

df['Equity'] = equity[:len(df)]
trades_df = pd.DataFrame(positions)

# === METRICS ===
if not trades_df.empty:
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'Win'])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100)
    max_drawdown = round(np.max([INITIAL_BALANCE - e for e in df['Equity']]), 2)
    net_profit = balance - INITIAL_BALANCE

    metrics_names = ['Win Rate (%)', 'Net Profit ($)', 'Final Balance ($)', 'Max Drawdown ($)']
    metrics_values = [f"{win_rate:.2f}%", f"{net_profit:.2f}", f"{balance:.2f}", f"{max_drawdown:.2f}"]
else:
    metrics_names = ['Note']
    metrics_values = ['No trades executed']

# === PLOTTING ===
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.6, 0.25, 0.15],
                    specs=[[{"type": "candlestick"}],
                           [{"type": "scatter"}],
                           [{"type": "table"}]])

# Candlestick + Bollinger Bands
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='Candles',
    increasing_line_color='limegreen', decreasing_line_color='red',
    line_width=2), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['Upper'],
                         line=dict(color='orange', width=1), name='Upper Band'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Lower'],
                         line=dict(color='orange', width=1), name='Lower Band'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MA'],
                         line=dict(color='blue', width=1), name='MA'), row=1, col=1)

# Buy/Sell Signals
buy_signals = df[df['Signal'] == 'Buy']
sell_signals = df[df['Signal'] == 'Sell']

fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'],
                         mode='markers', marker=dict(color='green', size=10, symbol='arrow-up'),
                         name='Buy Signal'), row=1, col=1)

fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'],
                         mode='markers', marker=dict(color='red', size=10, symbol='arrow-down'),
                         name='Sell Signal'), row=1, col=1)

# Equity Curve
fig.add_trace(go.Scatter(x=df.index, y=df['Equity'],
                         line=dict(color='magenta', width=2),
                         name='Equity'), row=2, col=1)

# Metrics Table
fig.add_trace(go.Table(
    header=dict(values=['Metric', 'Value'],
                fill_color='darkslategray',
                font=dict(color='white', size=14),
                align='left'),
    cells=dict(values=[metrics_names, metrics_values],
               fill_color='lightgray',
               align='left',
               font=dict(color='black', size=12)) ),
    row=3, col=1)

# Layout Settings
fig.update_layout(
    title="📈 Bollinger Band Strategy with Generated Market Data and Performance Metrics",
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    height=900,
    showlegend=True)

fig.update_yaxes(range=[df['Low'].min() - 0.001, df['High'].max() + 0.001], row=1, col=1)

# Show Plot
fig.show()
