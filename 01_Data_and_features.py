from utils import *
import pandas as pd
from pandas import DataFrame
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
import time
warnings.filterwarnings("ignore")
import logging

path_log = 'C:/Users/admin/Desktop/'
filename = 'stock_programs.log'

logging.basicConfig(filename=path_log + filename, format='%(asctime)s %(message)s', level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Momentum score function
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    #regress = stats.linregress(x, log_ts)
    mask = ~np.isnan(x) & ~np.isnan(log_ts)
    regress = stats.linregress(x[mask], log_ts[mask])
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)

# def momentum(closes):
#     returns = np.log(closes)
#     x = np.arange(len(returns))
#     mask = ~np.isnan(x) & ~np.isnan(returns)
#     slope, _, rvalue, _, _ = stats.linregress(x[mask], returns[mask])
#     return ((1 + slope) ** 252) * (rvalue ** 2)  # annualize slope and multiply by R^2
#
# def mmi(closes):
#     m = closes.median()
#     nh=0
#     nl=0
#     for i in range(1, len(closes)):
#         if closes[i] > m and closes[i] > closes[i-1]:
#             nl+=1
#         elif closes[i] < m and closes[i] < closes[i-1]:
#             nh+=1
#     return 100*(nl+nh)/(len(closes)-1)

def slope(ts):
    x = np.arange(len(ts))
    mask = ~np.isnan(x) & ~np.isnan(ts)
    regress = stats.linregress(x[mask], ts[mask])
    return regress[0]

def RSI(df, symbol, n=14):
    df_symbol = df.loc[df.index.get_level_values('symbol') == symbol]
    deltas = df_symbol['close'].diff()
    deltas = deltas.replace({np.nan:0.0})
    #-----------
    dUp, dDown = deltas.copy(), deltas.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp=dUp.rolling(14, min_periods=1).mean()
    RolDown=dDown.rolling(14, min_periods=1).mean().abs()
    rsi = deltas.copy()
    rsi[:] = 0
    for i in range(n+2, len(deltas)):
        RolUp[i] = (RolUp[i-1]*13+dUp[i])/14
        RolDown[i] = (RolDown[i-1]*13-dDown[i])/14
        rsi[i] = 100 - 100/(1+RolUp[i]/RolDown[i])
    return rsi

today = datetime.today()
logging.info("Today's date:" + today.strftime("%Y-%m-%d"))

key = '1MCOWQ6JUJ9ETPQ3'
path = '/Users/sing/Desktop/AI Plan/Finance with AI/Notebooks/'
path_pc = 'C:/Users/admin/Desktop/AI Plan/Finance with AI/Notebooks/'
symbols_all50 = ['0001.HK','0002.HK','0003.HK','0005.HK','0006.HK','0011.HK','0012.HK', '0016.HK','0017.HK','0019.HK','0027.HK','0066.HK',
 '0083.HK', '0101.HK','0151.HK','0175.HK','0267.HK','0288.HK','0386.HK','0388.HK','0669.HK','0688.HK','0700.HK','0762.HK','0823.HK',
 '0857.HK','0883.HK','0939.HK','0941.HK','1038.HK','1044.HK','1088.HK','1093.HK','1109.HK','1113.HK','1177.HK','1299.HK','1398.HK',
 '1928.HK','1997.HK','2007.HK','2018.HK','2313.HK','2318.HK','2319.HK','2382.HK','2388.HK','2628.HK','3328.HK','3988.HK']

prices_new = load_csv(path_pc, 'prices_new.csv')

#Get latest stock prices from alpha_vantage
for i in range(0,46,5):
    symbols_new_chunk = [symbols_all50[i], symbols_all50[i+1], symbols_all50[i+2], symbols_all50[i+3], symbols_all50[i+4]]
    logging.info("Trying to retrieve: ")
    logging.info(symbols_new_chunk)
    logging.info('-------------------------------------')
    prices_new_chunk = get_symbols(symbols_new_chunk, key)
    logging.info("Adding to dataset...")
    logging.info('-------------------------------------')
    prices_new = prices_new.combine_first(prices_new_chunk)
    if i != 45:
        logging.info("wait for a minute")
        logging.info('-------------------------------------')
        time.sleep(55)
    else:
        logging.info("Done!")

prices_new.replace(0, np.nan, inplace=True)

logging.info("Saving prices_new.csv...")
save_csv(prices_new, path_pc, 'prices_new.csv')
logging.info("Saved.")

ma_50 = lambda x: x.rolling(50, min_periods=1).mean()
ema_50 = lambda x: x.ewm(span=50, min_periods=1).mean()
macd_signal = lambda x: x.ewm(span=9, min_periods=1).mean() #which is ema 9
macd = lambda x: x.ewm(span=12, min_periods=1).mean() - x.ewm(span=26, min_periods=1).mean()

logging.info("Calculating moving averages...")
prices_new['ema50']=prices_new.close.groupby(level='symbol').apply(ema_50)
prices_new['macd']=prices_new.close.groupby(level='symbol').apply(macd)
prices_new['macd_signal']=prices_new.macd.groupby(level='symbol').apply(macd_signal)
prices_new['macd_signal_pct_diff']= (prices_new['macd']-prices_new['macd_signal'])/prices_new['macd_signal']
prices_new['sma50']=prices_new.close.groupby(level='symbol').apply(ma_50)

logging.info("Calculating Bollinger Bands...")
bbands50_upper = lambda x: x.rolling(50, min_periods=1).mean() + 2*x.rolling(50, min_periods=1).std(ddof=0)
bbands50_lower = lambda x: x.rolling(50, min_periods=1).mean() - 2*x.rolling(50, min_periods=1).std(ddof=0)
prices_new['bbands50_upper']=prices_new.close.groupby(level='symbol').apply(bbands50_upper)
prices_new['bbands50_lower']=prices_new.close.groupby(level='symbol').apply(bbands50_lower)

logging.info("RSIs...")
for symbol in symbols_all50:
     logging.info("Calculating for " + symbol)
     rsi = RSI(prices_new, symbol)
     prices_new = prices_new.combine_first(rsi.to_frame().rename(columns={"close": "rsi14"}))

logging.info("Calculating log vol, intraday change, etc...")
pct_chg_fxn_1 = lambda x: x.pct_change(1)
prices_new['volume_pct_change_1_day'] = prices_new.groupby(level='symbol').volume.apply(pct_chg_fxn_1)
pct_chg_fxn_5 = lambda x: x.pct_change(5)
prices_new['close_pct_change_5_day'] = prices_new.groupby(level='symbol').close.apply(pct_chg_fxn_5)
prices_new['intraday_chg'] = (prices_new.close - prices_new.open)/prices_new.open
prices_new['log volume'] = prices_new.volume.apply(np.log)
std_50 = lambda x: x.rolling(window=50, min_periods=20).std()
prices_new['volatility50'] = prices_new.groupby(level='symbol').close.apply(std_50)
prices_new['volatility50_ratio'] = prices_new.volatility50/prices_new.close

logging.info("Scaling open, high, low, close, volume...")
zscore_50 = lambda x: (x - x.rolling(window=50, min_periods=1).mean())/x.rolling(window=50, min_periods=1).std()
cols = ['open','high','low','close','volume']
for col in cols:
    prices_new[col+'_scaled50'] =prices_new.groupby(level='symbol')[col].apply(zscore_50)

logging.info("Scaling ema50, sma50. Calculating (close/ema)-1, upper/close-1, 1-lower/close...")
t_cols = ['ema50', 'sma50']
for t_col in t_cols:
    prices_new[t_col+'_scaled50'] =prices_new.groupby(level='symbol')[t_col].apply(zscore_50)
prices_new['(close/ema)-1'] = (prices_new.close/prices_new.ema50)-1
prices_new['(close/sma)-1'] = (prices_new.close/prices_new.sma50)-1
prices_new['upper/close-1'] = (prices_new.bbands50_upper/prices_new.close)-1
prices_new['1-lower/close'] = 1-(prices_new.bbands50_lower/prices_new.close)

logging.info("Calculating slopes...")
for symbol in symbols_all50:
    logging.info("Calculating for " + symbol)
    for col in ['bbands50_lower','bbands50_upper','close','ema50','high','low','open','sma50','volume','rsi14']:
        slope_df = prices_new.copy()
        logging.info("-----" + col + '_slope')
        slope_df[col+'_slope']  = slope_df.loc[slope_df.index.get_level_values('symbol') == symbol][col].rolling(14, min_periods = 1).apply(slope)
        prices_new = prices_new.combine_first(slope_df)

momentum_window = 50
minimum_momentum = 1

logging.info("Calculating momentums...")

for symbol in symbols_all50:
    logging.info("Calculating for " + symbol)
    df_mom = prices_new.copy()
    df_mom['momentum'] = df_mom.loc[df_mom.index.get_level_values('symbol') == symbol].close.rolling(momentum_window, min_periods = minimum_momentum).apply(momentum_score)
    prices_new = prices_new.combine_first(df_mom)

logging.info("Saving to file...")
save_csv(prices_new, path_pc, 'prices_new0116.csv')

#Calculate return
logging.info("Calculating return...")
outcomes_new = prices_new.copy()
return_1 = lambda x: (x.shift(-1)-x)/x
return_2 = lambda x: (x.shift(-2)-x)/x
return_3 = lambda x: (x.shift(-3)-x)/x
return_4 = lambda x: (x.shift(-4)-x)/x
return_5 = lambda x: (x.shift(-5)-x)/x
return_10 = lambda x: (x.shift(-10)-x)/x
log_return_1 = lambda x: np.log(x.shift(-1)/x)
log_return_2 = lambda x: np.log(x.shift(-2)/x)
log_return_3 = lambda x: np.log(x.shift(-3)/x)
log_return_4 = lambda x: np.log(x.shift(-4)/x)
log_return_5 = lambda x: np.log(x.shift(-5)/x)
log_return_10 = lambda x: np.log(x.shift(-10)/x)
logging.info('log_return_1')
outcomes_new['log_return_1'] = outcomes_new.groupby(level='symbol')['close'].apply(log_return_1)
logging.info('log_return_2')
outcomes_new['log_return_2'] = outcomes_new.groupby(level='symbol')['close'].apply(log_return_2)
logging.info('log_return_3')
outcomes_new['log_return_3'] = outcomes_new.groupby(level='symbol')['close'].apply(log_return_3)
logging.info('log_return_4')
outcomes_new['log_return_4'] = outcomes_new.groupby(level='symbol')['close'].apply(log_return_4)
logging.info('log_return_5')
outcomes_new['log_return_5'] = outcomes_new.groupby(level='symbol')['close'].apply(log_return_5)
logging.info('log_return_10')
outcomes_new['log_return_10'] = outcomes_new.groupby(level='symbol')['close'].apply(log_return_10)
logging.info('return_1')
outcomes_new['return_1'] = outcomes_new.groupby(level='symbol')['close'].apply(return_1)
logging.info('return_2')
outcomes_new['return_2'] = outcomes_new.groupby(level='symbol')['close'].apply(return_2)
logging.info('return_3')
outcomes_new['return_3'] = outcomes_new.groupby(level='symbol')['close'].apply(return_3)
logging.info('return_4')
outcomes_new['return_4'] = outcomes_new.groupby(level='symbol')['close'].apply(return_4)
logging.info('return_5')
outcomes_new['return_5'] = outcomes_new.groupby(level='symbol')['close'].apply(return_5)
logging.info('return_10')
outcomes_new['return_10'] = outcomes_new.groupby(level='symbol')['close'].apply(return_10)

#new features
past_return_1 = lambda x: (x-x.shift(1))/x.shift(1)
past_return_2 = lambda x: (x-x.shift(2))/x.shift(2)
past_return_3 = lambda x: (x-x.shift(3))/x.shift(3)
past_return_4 = lambda x: (x-x.shift(4))/x.shift(4)
past_return_5 = lambda x: (x-x.shift(5))/x.shift(5)
past_return_10 = lambda x: (x-x.shift(10))/x.shift(10)
past_log_return_1 = lambda x: np.log(x/x.shift(1))
past_log_return_2 = lambda x: np.log(x/x.shift(2))
past_log_return_3 = lambda x: np.log(x/x.shift(3))
past_log_return_4 = lambda x: np.log(x/x.shift(4))
past_log_return_5 = lambda x: np.log(x/x.shift(5))
past_log_return_10 = lambda x: np.log(x/x.shift(10))
logging.info('past_log_return_1')
outcomes_new['past_log_return_1'] = outcomes_new.groupby(level='symbol')['close'].apply(past_log_return_1)
logging.info('past_log_return_2')
outcomes_new['past_log_return_2'] = outcomes_new.groupby(level='symbol')['close'].apply(past_log_return_2)
logging.info('past_log_return_3')
outcomes_new['past_log_return_3'] = outcomes_new.groupby(level='symbol')['close'].apply(past_log_return_3)
logging.info('past_log_return_4')
outcomes_new['past_log_return_4'] = outcomes_new.groupby(level='symbol')['close'].apply(past_log_return_4)
logging.info('past_log_return_5')
outcomes_new['past_log_return_5'] = outcomes_new.groupby(level='symbol')['close'].apply(past_log_return_5)
logging.info('past_log_return_10')
outcomes_new['past_log_return_10'] = outcomes_new.groupby(level='symbol')['close'].apply(past_log_return_10)
logging.info('past_return_1')
outcomes_new['past_return_1'] = outcomes_new.groupby(level='symbol')['close'].apply(past_return_1)
logging.info('past_return_2')
outcomes_new['past_return_2'] = outcomes_new.groupby(level='symbol')['close'].apply(past_return_2)
logging.info('past_return_3')
outcomes_new['past_return_3'] = outcomes_new.groupby(level='symbol')['close'].apply(past_return_3)
logging.info('past_return_4')
outcomes_new['past_return_4'] = outcomes_new.groupby(level='symbol')['close'].apply(past_return_4)
logging.info('past_return_5')
outcomes_new['past_return_5'] = outcomes_new.groupby(level='symbol')['close'].apply(past_return_5)
logging.info('past_return_10')
outcomes_new['past_return_10'] = outcomes_new.groupby(level='symbol')['close'].apply(past_return_10)

#Get last date of dataset. Since today may not be a trading day
last_date = outcomes_new.index.get_level_values('date')[-1].to_pydatetime()

filename = 'outcomes_' + last_date.strftime("%Y-%m-%d") + '.csv'
logging.info("Saving to file: " + filename)
save_csv(outcomes_new, path_pc, filename)

logging.info("Done")
