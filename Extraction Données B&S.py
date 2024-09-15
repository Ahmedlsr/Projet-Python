# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:57:35 2024

@author: Ahmed a optimiser psk la on bloque une date et j'entre pas mal de truc manuellement"""

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import scipy.stats as stats
from scipy.stats import norm #Loi normale

AAPL = yf.Ticker("AAPL")
expiration = AAPL.options
apple = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
plt.figure(figsize=(10,6))
plt.plot(apple['Adj Close'])
plt.show()
données = AAPL.option_chain(date = "2024-08-16")
calls = données.calls
puts = données.puts

# 1 Fonction Black-Scholes pour le calcul du prix d'une option
def BandS(S, K, r, v, T, option):
    d1 = (np.log(S/K) + (r + (v**2)/2) * T) / (v * np.sqrt(T))
    d2 = d1 - (v * np.sqrt(T))
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    if option == "put":
        return put
    else:
        return call
    
BandS(100, 50, 0.03, 0.02, 1, "put")

# Fonction pour calculer Vega
def vega(S, K, r, v, T):
    d1 = (np.log(S/K) + (r + (v**2)/2) * T) / (v * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# Fonction pour calculer la volatilité implicite
def Volimplicite(S, K, r, T, market_price, option_type='call', tol=1e-6, max_iter=100):
    vol = 0.3 
    for i in range(max_iter):
        if option_type == 'call':
            price = BandS(S, K, r, vol, T, option="call")
        elif option_type == 'put':
            price = BandS(S, K, r, vol, T, option="put")
        diff = price - market_price
        if abs(diff) < tol:
            return vol
        vega_value = vega(S, K, r, vol, T)
        if vega_value < 1e-10:
            return np.nan
        vol = vol - diff / vega_value  
    return vol


Volimplicite(100, 110, 0.02, 1, market_price=10, option_type="call")

#3 Calcul de la vol implicite sur les options extraites

def calculate_volatility(row, option_type):
    S = apple['Adj Close'][-1]
    K = row['strike']
    T = (pd.to_datetime(expiration[0]) - pd.Timestamp.today()).days / 365.25
    market_price = row['lastPrice']
    r = 0.05 #rf
    
    vol = Volimplicite(S, K, r, T, market_price, option_type=option_type)
    return vol

calls = calls[calls['volume'] >= 1000]
puts = puts[puts['volume'] >= 1000]
calls['VolCalcule'] = calls.apply(lambda row:  calculate_volatility(row, 'call'), axis=1)
puts['VolCalcule'] = puts.apply(lambda row: calculate_volatility(row, 'put'), axis=1)
#On garde que les lignes ou y'a pas de NaN
calls = calls.dropna(subset=['VolCalcule'])
puts = puts.dropna(subset=['VolCalcule'])

calls['Comparaison'] = abs(calls["impliedVolatility"]-calls["VolCalcule"])
puts['Comparaison'] = abs(puts["impliedVolatility"]-puts["VolCalcule"])
Spot = AAPL.info['previousClose']
T = (datetime(2024, 8, 16) - datetime.today()).days / 365.25

r = 0.05
calls["prix theorique"]=calls.apply(lambda row: BandS(Spot, row["strike"], r, row["VolCalcule"], T, option='call'), axis=1)
puts["prix theorique"]=puts.apply(lambda row: BandS(Spot, row["strike"], r, row["VolCalcule"], T, option='put'), axis=1)

calls.to_csv('calls.csv', index=False)
puts.to_csv('puts.csv', index=False)