# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import yfinance as yf
apple = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
SMA12 = 0
SMA26 = 0
SMA9 = 0
apple["EMA12"] = 0
apple["EMA26"] = 0
for i in range(0,13):
    SMA12 += apple["Adj Close"][i]
SMA12 = SMA12/12
for i in range (0,26):
    SMA26 += apple["Adj Close"][i]
SMA26= SMA26/26
apple.loc[:12, "EMA12"] = SMA12
apple.loc[:26, "EMA26"] = SMA26

for i in range(12,len(apple)):
    apple.loc[i:, "EMA12"] = (apple["Adj Close"][i] - apple["EMA12"][i-1])*(2/13) + apple["EMA12"][i-1]

for i in range(26,len(apple)):
    apple.loc[i:, "EMA26"] = (apple["Adj Close"][i] - apple["EMA26"][i-1])*(2/27) + apple["EMA12"][i-1]
    
apple["MACD"] = apple["EMA12"]-apple["EMA26"]
for i in range (0,10):
    SMA9 += apple["MACD"][i]
SMA9 = SMA9/9
apple.loc[:9, "Signal"] = SMA9

for i in range(9,len(apple)):
    apple.loc[i:, "Signal"] = (apple["MACD"][i] - apple["Signal"][i-1])*(2/10) + apple["Signal"][i-1]

plt.figure(figsize=(12,8))
plt.plot(apple.index, apple['Adj Close'], label='Adj Close', color ='blue')
plt.plot(apple.index, apple['EMA12'], label='EMA12', color = 'yellow')
plt.plot(apple.index, apple['EMA26'], label='EMA26', color = 'pink' )
plt.plot(apple.index, apple['MACD'], label='MACD', color='green')
plt.plot(apple.index, apple['Signal'], label='Signal', color='red')
plt.legend()
plt.show()

buy_signals = []
sell_signals = []

for i in range(1, len(apple)):
    if apple["MACD"][i-1] < apple["Signal"][i-1] and apple["MACD"][i] > apple["Signal"][i]:
        # Signal d'achat (MACD croise la ligne de signal vers le haut)
        buy_signals.append(apple.index[i])
    elif apple["MACD"][i-1] > apple["Signal"][i-1] and apple["MACD"][i] < apple["Signal"][i]:
        # Signal de vente (MACD croise la ligne de signal vers le bas)
        sell_signals.append(apple.index[i])

fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(apple.index, apple['Adj Close'], label='Adj Close', color='blue')
ax1.plot(apple.index, apple['EMA12'], label='EMA12', color='yellow')
ax1.plot(apple.index, apple['EMA26'], label='EMA26', color='green')
ax1.set_ylabel('Prix de clôture ajusté')

ax1.scatter(buy_signals, apple.loc[buy_signals, 'Adj Close'], label='Buy Signal', marker='^', color='green', s=100)
ax1.scatter(sell_signals, apple.loc[sell_signals, 'Adj Close'], label='Sell Signal', marker='v', color='red', s=100)

ax2 = ax1.twinx()
ax2.plot(apple.index, apple['MACD'], label='MACD', color='orange')
ax2.plot(apple.index, apple['Signal'], label='Signal', color='red')
ax2.set_ylabel('MACD / Signal')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Adj Close, EMA, MACD, Signal avec Signaux d\'achat et vente')
plt.show()

