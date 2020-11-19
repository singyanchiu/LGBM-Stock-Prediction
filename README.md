# LGBM-Stock-Prediction
A machine learning pipeline that ingest and process a 20-year historical stock price dataset and try to predict future prices using LightGBM.

# Data collection
The python script `01_Data_and_features.py` sends API requests to Alpha Vantage and downloads daily stock price data of the 50 constituent stocks of the Hang Seng Index. It then does basic data cleaning like converting zeros to nulls, etc. Basic features engineering are also done, eg. technical indicators like RSI, momentum, Bollinger Bands, simple and exponential moving averages, etc.
