import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy import stats
import lightgbm as lgb
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
import os.path
import time

def halflife(yp):
    yp_diff = yp-yp.shift()
    delta_y = yp_diff[1:]
    y_lag = yp[:-1]
    regress = stats.linregress(y_lag, delta_y)
    return -np.log(2)/regress[0]

def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
        if t == 0:
            mins, secs = divmod(t, 60)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            print(timeformat, end='\r', flush = True)
    return

def atr(df, symbol, n=14):
    df_symbol = df.loc[df.index.get_level_values('symbol') == symbol]
    high = df_symbol['high']
    low = df_symbol['low']
    close = df_symbol['close']
    df_symbol['tr0'] = abs(high - low)
    df_symbol['tr1'] = abs(high - close.shift(1))
    df_symbol['tr2'] = abs(low - close.shift(1))
    tr = df_symbol[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr

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

def mmi(closes):
    m = np.median(closes)
    nh=0
    nl=0
    for i in range(1, len(closes)):
        if closes[i] > m and closes[i] > closes[i-1]:

            nl+=1
        elif closes[i] < m and closes[i] < closes[i-1]:
            nh+=1
    return (nl+nh)/(len(closes)-1)

def save_csv(df, path, filename):
    df.to_csv(path+filename, date_format='%Y-%m-%d %H:%M:%S')

def load_csv(path, filename):
    df = pd.read_csv(path+filename, parse_dates = True)
    df['date'] = df['date'].astype('datetime64[ns]')
    df = df.set_index(['date','symbol'])
    #df = df.loc[df['close'] != 0]
    return df

def load_FX_csv(path, filename):
    df = pd.read_csv(path+filename, parse_dates = True)
    df['date'] = df['date'].astype('datetime64[ns]')
    df = df.set_index('date')
    #df = df.loc[df['close'] != 0]
    return df

def load_portfo_csv(path, filename):
    portfolio = pd.read_csv(path+filename, parse_dates = True)
    # cols = ['date', 'date_close','symbol','position','cost','target_lower','target_upper']
    # portfolio['symbol'] = portfolio['symbol'].astype('str')
    # portfolio['date'] = portfolio['date'].astype('datetime64[ns]')
    # portfolio['date_close'] = portfolio['date_close'].astype('datetime64[ns]')
    # portfolio = portfolio[cols]
    #portfolio = pd.read_csv(path_pc+'portfolio.csv', parse_dates = True)
    portfolio['symbol'] = portfolio['symbol'].astype('str')
    portfolio['date'] = portfolio['date'].astype('datetime64[ns]')
    portfolio['date_close'] = portfolio['date_close'].astype('datetime64[ns]')
    portfolio.drop(columns=['Unnamed: 0'], inplace=True)
    return portfolio

def load_log_csv(path, filename):
    log = pd.read_csv(path+filename, parse_dates = True)
    cols = ['01) date_buy', '02) date_sell','03) symbol','04) position','05) price','06) amount','07) BOT/SLD']
    log['01) date_buy'] = log['01) date_buy'].astype('datetime64[ns]')
    log['02) date_sell'] = log['02) date_sell'].astype('datetime64[ns]')
    log = log[cols]
    return log

def get_symbols(symbols, key, outputsize='compact', adjusted=False, skipped_symbols=[]):
    ts = TimeSeries(key, output_format='pandas')
    out = pd.DataFrame()
    if adjusted == True:
        func = ts.get_daily_adjusted
        cols = ['open','high','low','close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
    else:
        func = ts.get_daily
        cols = ['open','high','low','close','volume']
    for symbol in symbols:
        if symbol in skipped_symbols:
            print ('Skipping {} as instructed.'.format(symbol))
            continue
        else:
            print('Trying to download ', symbol)
            while True:
                try:
                    df, meta = func(symbol=symbol, outputsize=outputsize)
                except ValueError as e:
                    print('*')
                    print('* Valueerror from Alpha Vantage: ', e)
                    if 'Invalid API call' in str(e):
                        print('Symbol {} not available on Alpha Vantage. Skippping it.'.format(symbol))
                        break
                    elif 'Thank' in str(e):
                        print('API call frequency exceeded as advised by Alpha Vantage. Wait for a minute and try again.')
                        countdown(60)
                    print()
                else:
                    df.columns = cols
                    df['symbol'] = symbol # add a new column which contains the symbol so we can keep multiple symbols in the same dataframe
                    df.reset_index(level=0, inplace=True)
                    df = df.set_index(['date','symbol'])
                    out = pd.concat([out,df],axis=0) #stacks on top of previously collected data
                    break
    return out.sort_index()

def get_symbols_intraday(symbols, key, outputsize='full'):
    ts = TimeSeries(key, output_format='pandas')
    out = pd.DataFrame()
    for symbol in symbols:
        df, meta = ts.get_intraday(symbol=symbol, interval='1min', outputsize=outputsize)
        df.columns = ['open','high','low','close','volume'] #my convention: always lowercase
        df['symbol'] = symbol # add a new column which contains the symbol so we can keep multiple symbols in the same dataframe
        #df = df.set_index(['symbol'])
        df.reset_index(level=0, inplace=True)
        df = df.set_index(['date','symbol'])
        out = pd.concat([out,df],axis=0) #stacks on top of previously collected data
    return out.sort_index()

def get_FX_symbols_intraday(symbols, key, outputsize='full'):
    fe = ForeignExchange(key, output_format='pandas')
    out = pd.DataFrame()
    for symbol in symbols:
        print('Trying to download ', symbol)
        while True:
            try:
                df, meta = fe.get_currency_exchange_intraday(from_symbol=symbol[0:3], to_symbol=symbol[4:], interval='1min', outputsize=outputsize)
            except ValueError as e:
                print('*')
                print('* Valueerror from Alpha Vantage: ', e)
                if 'Invalid API call' in str(e):
                    print('Symbol {} not available on Alpha Vantage. Skippping it.'.format(symbol))
                    break
                elif 'Thank' in str(e):
                    print('API call frequency exceeded as advised by Alpha Vantage. Wait for a minute and try again.')
                    countdown(60)
                print()
            else:
                df.columns = ['open','high','low','close']
                df['symbol'] = symbol[0:3]+'.'+ symbol[4:]
                df.reset_index(level=0, inplace=True)
                df = df.set_index(['date','symbol'])
                out = pd.concat([out,df],axis=0) #stacks on top of previously collected data
                break
    return out.sort_index()

def get_FX_symbols_daily(symbols, key, outputsize='full'):
    fe = ForeignExchange(key, output_format='pandas')
    out = pd.DataFrame()
    for symbol in symbols:
        print('Trying to download ', symbol)
        while True:
            try:
                df, meta = fe.get_currency_exchange_daily(from_symbol=symbol[0:3], to_symbol=symbol[4:], outputsize=outputsize)
            except ValueError as e:
                print('*')
                print('* Valueerror from Alpha Vantage: ', e)
                if 'Invalid API call' in str(e):
                    print('Symbol {} not available on Alpha Vantage. Skippping it.'.format(symbol))
                    break
                elif 'Thank' in str(e):
                    print('API call frequency exceeded as advised by Alpha Vantage. Wait for a minute and try again.')
                    countdown(60)
                print()
            else:
                df.columns = ['open','high','low','close']
                df['symbol'] = symbol[0:3]+'.'+ symbol[4:]
                df.reset_index(level=0, inplace=True)
                df = df.set_index(['date','symbol'])
                out = pd.concat([out,df],axis=0) #stacks on top of previously collected data
                break
    return out.sort_index()

def date_query (df, begin, end):
    return df[(df.index.get_level_values('date')>= begin) &
             (df.index.get_level_values('date')<= end)]

#binary switch - log return > 90% quantile will be 1, otherwise 0
def switch_upper(ts, upper_threshold):
    result = ts.copy()
    for i in range(len(ts)):
        if ts[i] >= upper_threshold:
            result[i] = 1
        else: result[i] = 0
    return result

#binary switch - log return < 10% quantile will be 1, otherwise 0
def switch_lower(ts, lower_threshold):
    result = ts.copy()
    for i in range(len(ts)):
        if ts[i] <= lower_threshold:
            result[i] = 1
        else: result[i] = 0
    return result

from sklearn.metrics import f1_score, precision_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def lgb_precision_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'precision_score', precision_score(y_true, y_hat), True

def to_days(days):
    return pd.Timedelta('{} days'.format(str(days)))

def class_switch_binary(y_valid, y_pred, prob_threshold):
    result = []
    for prob in y_pred:
        if prob > float(prob_threshold):
            result.append(1)
        else: result.append(0)
    result_df = y_valid.copy()
    result_df = result_df.to_frame()
    #result_df.reset_index(level=0, inplace=True)
    result_df['pred'] = result
    return result_df['pred']

# def train_valid_test_split(df, start_date, start_date_valid, start_date_test, end_date_test):
#     X_y_train = df[start_date : start_date_valid - pd.Timedelta('1 day')]
#     X_y_valid = df[start_date_valid: start_date_test - pd.Timedelta('1 day')]
#     X_y_test = df[start_date_test: end_date_test]
#     return X_y_train, X_y_valid, X_y_test

def train_valid_test_split(df, start_date, start_date_valid, end_date_valid, start_date_test, end_date_test):
    X_y_train = df[start_date : start_date_valid]
    X_y_valid = df[start_date_valid +  pd.Timedelta('1 day'): end_date_valid]
    X_y_test = df[start_date_test +  pd.Timedelta('1 day'): end_date_test]
    return X_y_train, X_y_valid, X_y_test

def train_valid_split(df, start_date, start_date_valid, end_date_valid):
    X_y_train = df[start_date : start_date_valid]
    X_y_valid = df[start_date_valid +  pd.Timedelta('1 day'): end_date_valid]
    return X_y_train, X_y_valid

def add_target_upper(X_y_train, X_y_valid, X_y_test, q_upper, target_col, return_col):
    upper_threshold = X_y_train[return_col].quantile(q=q_upper)
    X_y_train[target_col] = switch_upper(X_y_train[return_col], upper_threshold)
    X_y_valid[target_col] = switch_upper(X_y_valid[return_col], upper_threshold)
    X_y_test[target_col] = switch_upper(X_y_test[return_col], upper_threshold)
    return X_y_train, X_y_valid, X_y_test

def add_target_upper_notest(X_y_train, X_y_valid, q_upper, target_col, return_col):
    upper_threshold = X_y_train[return_col].quantile(q=q_upper)
    print("upper_threshold: ", upper_threshold)
    X_y_train[target_col] = switch_upper(X_y_train[return_col], upper_threshold)
    X_y_valid[target_col] = switch_upper(X_y_valid[return_col], upper_threshold)
    return X_y_train, X_y_valid

def add_target_lower_notest(X_y_train, X_y_valid, q_lower, target_col, return_col):
    lower_threshold = X_y_train[return_col].quantile(q=q_lower)
    print("lower_threshold: ", lower_threshold)
    X_y_train[target_col] = switch_lower(X_y_train[return_col], lower_threshold)
    X_y_valid[target_col] = switch_lower(X_y_valid[return_col], lower_threshold)
    return X_y_train, X_y_valid

def add_target_lower(X_y_train, X_y_valid, X_y_test, q_lower, target_col, return_col):
    lower_threshold = X_y_train[return_col].quantile(q=q_lower)
    X_y_train[target_col] = switch_lower(X_y_train[return_col], lower_threshold)
    X_y_valid[target_col] = switch_lower(X_y_valid[return_col], lower_threshold)
    X_y_test[target_col] = switch_lower(X_y_test[return_col], lower_threshold)
    return X_y_train, X_y_valid, X_y_test

def downsample(X_y_train, target_col, test_ratio, random_seed):
    df_positive = X_y_train.loc[X_y_train[target_col]==1]
    df_negative = X_y_train.loc[X_y_train[target_col]==0]
    df_negative_bigger, df_negative_downsampled = train_test_split(df_negative,
                                                               test_size=test_ratio, random_state=random_seed)
    X_y_train_resampled = pd.concat([df_positive, df_negative_downsampled])
    X_y_train_resampled = X_y_train_resampled.sort_index()
    return X_y_train_resampled

def downsample_3class(X_y_train, target_col, random_seed):
    class_list = [1,0,-1]
    tuple = (len(X_y_train.loc[X_y_train[target_col] == 1]),len(X_y_train.loc[X_y_train[target_col] == 0]),
             len(X_y_train.loc[X_y_train[target_col] == -1]))

    lowest_n_class = class_list[tuple.index(min(tuple))]
    class_list.pop(tuple.index(min(tuple)))

    df_keep = X_y_train.loc[X_y_train[target_col] == lowest_n_class]

    X_y_train_resampled = df_keep.copy()

    for class_label in class_list:
        df_to_downsample = X_y_train.loc[X_y_train[target_col] == class_label]
        test_ratio = len(df_keep)/len(df_to_downsample)
        df_to_downsample_bigger, df_downsampled = train_test_split(df_to_downsample,
                                                                   test_size=test_ratio, random_state=random_seed)
        X_y_train_resampled = pd.concat([X_y_train_resampled, df_downsampled])
        X_y_train_resampled = X_y_train_resampled.sort_index()
    return X_y_train_resampled

def downsample_positive(X_y_train, target_col, test_ratio, random_seed):
    df_positive = X_y_train.loc[X_y_train[target_col]==1]
    df_negative = X_y_train.loc[X_y_train[target_col]==0]
    df_positive_bigger, df_positive_downsampled = train_test_split(df_positive,
                                                               test_size=test_ratio, random_state=random_seed)
    X_y_train_resampled = pd.concat([df_negative, df_positive_downsampled])
    X_y_train_resampled = X_y_train_resampled.sort_index()
    return X_y_train_resampled

def feature_target_split(df, features_cols, target_col):
    X_train = df[features_cols]
    y_train = df[target_col]
    return X_train, y_train

def knn_train(X_train, y_train, X_valid, y_valid, X_valid_close, p_range, leaf_size_range, n_neighbors_range, return_col_actual, prob_threshold = 0.7, sign = 1):
    max_total_gain = float("-inf")
    max_auc = float("-inf")
    #max_precision_total_gain = float("-inf")
    for p in p_range:
        for leaf_size in leaf_size_range:
            for n_neighbors in n_neighbors_range:
                knn = KNeighborsClassifier(p = p, leaf_size = leaf_size, n_neighbors = n_neighbors)
                model = knn.fit(X_train, y_train)
                y_pred = model.predict(X_valid)
                #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
                y_class_pred = class_switch_binary(y_valid, y_pred, prob_threshold)
                #precision = precision_score(y_valid, y_class_pred)
                #print('-'*80)
                #print('p = ', p, ' leaf_size = ', leaf_size, ' n_neighbors = ', n_neighbors)
                #print(classification_report(y_valid, y_class_pred_var_threshold))
                auc = roc_auc_score(y_valid, y_class_pred)
                #print(auc)
                X_valid_close_pred = pd.merge(X_valid_close, y_class_pred, left_index=True, right_index=True)
                X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
                total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
                if auc > max_auc:
                    max_auc = auc
                    best_auc_model = model
                    p_at_max_auc = p
                    leaf_size_at_max_auc = leaf_size
                    n_neighbors_at_max_auc = n_neighbors
                if total_gain > max_total_gain:
                    max_total_gain = total_gain
                    best_model = model
                    p_at_max_tt = p
                    leaf_size_at_max_tt = leaf_size
                    n_neighbors_at_max_tt = n_neighbors
    #print("----------------------")
    return (best_model, best_auc_model, max_auc, p_at_max_auc, leaf_size_at_max_auc, n_neighbors_at_max_auc, p_at_max_tt, leaf_size_at_max_tt, n_neighbors_at_max_tt, y_class_pred)
    # optimal_depth, optimal_num_leaves, max_precision, optimal_precision_depth, optimal_precision_num_leaves, max_precision_total_gain)

def lgb_train_v2(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, prob_threshold_range,
              return_col_actual, min_data = 11, metric = 'auc', sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    max_precision = float("-inf")
    max_auc = float("-inf")
    max_f1 = float("-inf")
    max_precision_total_gain = float("-inf")
    for prob_threshold in prob_threshold_range:
        for max_depth in max_depth_range:
            for num_leaves in num_leaves_range:
                parameters = {
                    'application': 'binary',
                    'metric': metric,
                    'is_unbalance': 'false',
                    #'scale_pos_weight': 9,
                    'boosting': 'gbdt',
                    'num_leaves': num_leaves,
                    'feature_fraction': 0.95,
                    'bagging_fraction': 0.2,
                    'bagging_freq': 20,
                    'learning_rate': 0.1,
                    'verbose': -1,
                    'min_data_in_leaf': min_data,
                    'max_depth': max_depth
                }
                #print("Using ", metric)
                model = lgb.train(parameters,
                                       train_data,
                                       valid_sets=valid_data,
                                       num_boost_round=5000,
                                        verbose_eval=False,
                                       #feval=lgb_f1_score,
                                       early_stopping_rounds=100)
                y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
                #print("model.eval_valid:")
                #print(model.eval_valid())
                #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
                y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold)
                precision = precision_score(y_valid, y_class_pred_var_threshold)
                auc = roc_auc_score(y_valid, y_pred)
                f1 = f1_score(y_valid, y_class_pred_var_threshold)
                X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
                X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
                total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
                if precision > max_precision:
                    max_precision = precision
                    best_pres_model = model
                    optimal_precision_depth = max_depth
                    optimal_precision_num_leaves = num_leaves
                    max_precision_total_gain = total_gain
                    opt_precision_thres = prob_threshold
                if auc > max_auc:
                    max_auc = auc
                    best_auc_model = model
                    optimal_auc_depth = max_depth
                    optimal_auc_num_leaves = num_leaves
                    max_auc_total_gain = total_gain
                    opt_auc_thres = prob_threshold
                if total_gain > max_total_gain:
                    max_total_gain = total_gain
                    best_model = model
                    optimal_depth = max_depth
                    optimal_num_leaves= num_leaves
                    opt_tt_thres = prob_threshold
                if f1 > max_f1:
                    max_f1 = f1
                    best_f1_model = model
                    optimal_f1_depth = max_depth
                    optimal_f1_num_leaves = num_leaves
                    max_f1_total_gain = total_gain
                    opt_f1_thres = prob_threshold
        #print("max auc = ", max_auc, " at depth = ", optimal_auc_depth, " and num_leaves = ", optimal_auc_num_leaves,
        #      ' with total gain = ', max_auc_total_gain)
    return (best_model, best_auc_model, best_f1_model, max_total_gain, max_auc_total_gain, max_f1_total_gain,
    optimal_depth, optimal_num_leaves, max_precision, optimal_precision_depth, optimal_precision_num_leaves, max_precision_total_gain,
    opt_precision_thres, opt_auc_thres, opt_tt_thres, opt_f1_thres)

def lgbv2_train_multi(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, prob_threshold = 0.7, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_precision = float("-inf")
    max_auc = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'multiclass',
                'num_class': 3,
                'is_unbalance': 'false',
                'metric': 'multi_logloss',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            #print("Using ", metric)
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_f1_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold)
            precision = precision_score(y_valid, y_class_pred_var_threshold)
            auc = roc_auc_score(y_valid, y_class_pred_var_threshold)
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            if precision > max_precision:
                max_precision = precision
                best_pres_model = model
                optimal_precision_depth = max_depth
                optimal_precision_num_leaves = num_leaves
                max_precision_total_gain = total_gain
            if auc > max_auc:
                max_auc = auc
                best_auc_model = model
                optimal_auc_depth = max_depth
                optimal_auc_num_leaves = num_leaves
                max_auc_total_gain = total_gain
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
                optimal_depth = max_depth
                optimal_num_leaves= num_leaves
    #print("max auc = ", max_auc, " at depth = ", optimal_auc_depth, " and num_leaves = ", optimal_auc_num_leaves,
    #      ' with total gain = ', max_auc_total_gain)
    return (best_model, best_pres_model, max_total_gain,
    optimal_depth, optimal_num_leaves, max_precision, optimal_precision_depth, optimal_precision_num_leaves, max_precision_total_gain)

def lgb_train(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, metric = 'auc', prob_threshold = 0.7, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    max_precision = float("-inf")
    max_auc = float("-inf")
    max_precision_total_gain = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'binary',
                'metric': metric,
                'is_unbalance': 'false',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            #print("Using ", metric)
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_f1_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold)
            precision = precision_score(y_valid, y_class_pred_var_threshold)
            auc = roc_auc_score(y_valid, y_class_pred_var_threshold)
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            if precision > max_precision:
                max_precision = precision
                best_pres_model = model
                optimal_precision_depth = max_depth
                optimal_precision_num_leaves = num_leaves
                max_precision_total_gain = total_gain
            if auc > max_auc:
                max_auc = auc
                best_auc_model = model
                optimal_auc_depth = max_depth
                optimal_auc_num_leaves = num_leaves
                max_auc_total_gain = total_gain
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
                optimal_depth = max_depth
                optimal_num_leaves= num_leaves
    #print("max auc = ", max_auc, " at depth = ", optimal_auc_depth, " and num_leaves = ", optimal_auc_num_leaves,
    #      ' with total gain = ', max_auc_total_gain)
    return (best_model, best_pres_model, max_total_gain,
    optimal_depth, optimal_num_leaves, max_precision, optimal_precision_depth, optimal_precision_num_leaves, max_precision_total_gain)

def lgb_train_auc(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, metric = 'auc', prob_threshold = 0.7, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    max_precision = float("-inf")
    max_auc = float("-inf")
    max_precision_total_gain = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'binary',
                'metric': metric,
                'is_unbalance': 'false',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            #print("Using ", metric)
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_f1_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold)
            #precision = precision_score(y_valid, y_class_pred_var_threshold)
            auc = roc_auc_score(y_valid, y_class_pred_var_threshold)
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            # if precision > max_precision:
            #     max_precision = precision
            #     best_pres_model = model
            #     optimal_precision_depth = max_depth
            #     optimal_precision_num_leaves = num_leaves
            #     max_precision_total_gain = total_gain
            if auc > max_auc:
                max_auc = auc
                best_auc_model = model
                optimal_auc_depth = max_depth
                optimal_auc_num_leaves = num_leaves
                max_auc_total_gain = total_gain
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
                optimal_depth = max_depth
                optimal_num_leaves= num_leaves
    #print("max auc = ", max_auc, " at depth = ", optimal_auc_depth, " and num_leaves = ", optimal_auc_num_leaves,
    #      ' with total gain = ', max_auc_total_gain)
    return (best_model, best_auc_model, max_total_gain,
    optimal_depth, optimal_num_leaves, max_auc, optimal_auc_depth, optimal_auc_num_leaves, max_auc_total_gain)

def lgb_train_feature_importance(X_train, y_train, X_valid, y_valid, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, metric = 'auc', prob_quantile = 0.85, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'binary',
                'metric': metric,
                'is_unbalance': 'false',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_precision_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold[0])
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
    print("----------------------")
    return best_model

def total_gain(model, X_test, X_test_close, y_test, prob_quantile, return_col_actual, sign=1):
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    prob_threshold = pd.DataFrame(y_test_pred).quantile(q=prob_quantile)
    y_class_pred = class_switch_binary(y_test, y_test_pred, prob_threshold[0])
    X_test_close_pred = pd.merge(X_test_close, y_class_pred, left_index=True, right_index=True)
    X_test_close_pred['gain'] = X_test_close_pred[return_col_actual] * X_test_close_pred.pred
    test_total_gain = X_test_close_pred.groupby(level='symbol').gain.sum().sum() * sign
    return test_total_gain, y_class_pred

def total_actual_gain_knn(model, X_test, X_test_close, y_test, prob_threshold, return_col_actual, sign=1):
    y_test_pred = model.predict(X_test)
    y_class_pred = class_switch_binary(y_test, y_test_pred, prob_threshold)
    X_test_close_pred = pd.merge(X_test_close, y_class_pred, left_index=True, right_index=True)
    X_test_close_pred['amount_spent'] = X_test_close_pred.pred * X_test_close_pred.next_day_open * X_test_close_pred.num_shares
    X_test_close_pred['actual_gain'] = (1+X_test_close_pred[return_col_actual]) * X_test_close_pred.pred * X_test_close_pred.close \
    * X_test_close_pred.num_shares - X_test_close_pred.next_day_open * X_test_close_pred.pred * X_test_close_pred.num_shares
    test_total_gain = X_test_close_pred.groupby(level='symbol').actual_gain.sum().sum() * sign
    total_amount_spent = X_test_close_pred.amount_spent.sum()
    return test_total_gain, total_amount_spent, y_class_pred

def total_actual_gain(model, X_test, X_test_close, y_test, prob_threshold, return_col_actual, sign=1):
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    #prob_threshold = pd.DataFrame(y_test_pred).quantile(q=prob_quantile)
    #prob_threshold = 0.6
    #print("Prob Threshold = 0.7")
    y_class_pred = class_switch_binary(y_test, y_test_pred, prob_threshold)
    X_test_close_pred = pd.merge(X_test_close, y_class_pred, left_index=True, right_index=True)
    X_test_close_pred['amount_spent'] = X_test_close_pred.pred * X_test_close_pred.next_day_open * X_test_close_pred.num_shares
    X_test_close_pred['actual_gain'] = (1+X_test_close_pred[return_col_actual]) * X_test_close_pred.pred * X_test_close_pred.close \
    * X_test_close_pred.num_shares - X_test_close_pred.next_day_open * X_test_close_pred.pred * X_test_close_pred.num_shares
    test_total_gain = X_test_close_pred.groupby(level='symbol').actual_gain.sum().sum() * sign
    total_amount_spent = X_test_close_pred.amount_spent.sum()
    return test_total_gain, total_amount_spent, y_class_pred

def ensemble_total_gain(y_class_pred, X_test_close, return_col_actual, sign=1):
    X_test_close_pred = pd.merge(X_test_close, y_class_pred, left_index=True, right_index=True)
    X_test_close_pred['gain'] = X_test_close_pred[return_col_actual] * X_test_close_pred.pred
    test_total_gain = X_test_close_pred.groupby(level='symbol').gain.sum().sum() * sign
    return test_total_gain

def ensemble_actual_gain(y_class_pred, X_test_close, return_col_actual, sign=1):
    X_test_close_pred = pd.merge(X_test_close, y_class_pred, left_index=True, right_index=True)
    X_test_close_pred['actual_gain'] = (1+X_test_close_pred[return_col_actual]) * X_test_close_pred.pred * X_test_close_pred.close \
    * X_test_close_pred.num_shares - X_test_close_pred.next_day_open * X_test_close_pred.pred * X_test_close_pred.num_shares
    X_test_close_pred['amount_spent'] = X_test_close_pred.pred * X_test_close_pred.next_day_open * X_test_close_pred.num_shares
    test_total_gain = X_test_close_pred.groupby(level='symbol').actual_gain.sum().sum() * sign
    total_amount_spent = X_test_close_pred.amount_spent.sum()
    return test_total_gain, total_amount_spent

def multi_lgb_predict(models, X, y):
    df = X.copy()
    cols = []
    for j, model in enumerate(models, 1):
        if type(model).__name__ == 'KNeighborsClassifier':
            df['y_pred_{}'.format(str(j))] = [item[1] for item in list(model.predict_proba(X))]
            cols.append('y_pred_{}'.format(str(j)))
        else:
            df['y_pred_{}'.format(str(j))] = model.predict(X, num_iteration=model.best_iteration)
            cols.append('y_pred_{}'.format(str(j)))
    df = df[cols]
    df['target'] = y
    return df, cols

def multi_lgb_predict_no_y(models, X):
    df = X.copy()
    cols = []
    for j, model in enumerate(models, 1):
        df['y_pred_{}'.format(str(j))] = model.predict(X, num_iteration=model.best_iteration)
        cols.append('y_pred_{}'.format(str(j)))
    df = df[cols]
    return df, cols

def compress(data, selectors, threshold):
    return (d for d, s in zip(data, selectors) if s > threshold)

def load_latest(today, prefix, path):
    count = 0
    while True:
        filename = prefix + today.strftime("%Y-%m-%d") + '.csv'
        print("Trying ", filename)
        if os.path.isfile(path + filename):
            print("Loading file: ", filename)
            if prefix == 'outcomes_' or prefix == 'FX_all_intraday_' or prefix == 'outcomes_new_features_':
                content = load_csv(path, filename)
                break
            elif 'master_scoreboard_' in prefix or 'predict_' in prefix:
                content = pd.read_csv(path + filename)
                break
            elif 'price_intraday_' in prefix:
                content = load_csv(path, filename)
                break
            elif 'predictions_' in prefix:
                content = load_portfo_csv(path, filename)
                break
            elif 'FX_EUR_USD_intraday_' in prefix:
                content = load_FX_csv(path, filename)
                break
            else:
                content = load_csv(path, filename)
                # content = lgb.Booster(model_file=path + filename)
                break
        else:
            today = today - to_days(1)
            count += 1
            if count > 5:
                print("No valid dataframe file.")
                break
    return content

def get_last_date(df):
    y = df.index.get_level_values('date')[-1].year
    m = df.index.get_level_values('date')[-1].month
    d = df.index.get_level_values('date')[-1].day
    #print("will return ", datetime(y,m,d))
    return datetime(y,m,d)

def get_last_date_dropna(df):
    df_dropna = df.dropna()
    y = df_dropna.index.get_level_values('date')[-1].year
    m = df_dropna.index.get_level_values('date')[-1].month
    d = df_dropna.index.get_level_values('date')[-1].day
    return datetime(y,m,d)

def symbol_to_str(self, symbol):
    return '0'*(4-len(str(symbol))) + str(symbol)+'.HK'

def gain_vs_loss(ts):
    dUp, dDown = ts.copy(), ts.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    return dUp.sum()/(-dDown.sum())

def get_dataset_dates(df):
    """
    Input: dataset with date as one of the MultiIndex
    Output: a list of all dates in this dataset as datetime objects
    """
    #get all trading dates from dataset of this symbol
    data_dates = sorted(list(set(df.index.get_level_values(0))))
    data_converted_dates = []
    for ts in data_dates:
        data_converted_dates.append(ts.to_pydatetime())
    return data_converted_dates

def to_tick_price_FX(price):
    return round(round(price/0.00005)*0.00005, 5)

def to_tick_price(price):
    """
    Input: price
    Output: price that is rounded to the tick size allowed by HKSE, 3 decimal place
    """
    tick_dict={5000:5,
          2000:2,
          1000:1,
          500:0.5,
          200:0.2,
          100:0.1,
          20:0.05,
          10:0.02,
          0.5:0.01,
          0.25:0.005,
          0.01:0.001
          }
    for tier in tick_dict.keys():
        if price >= tier:
            tick = tick_dict[tier]
            break
    return round(round(price/tick)*tick,3)

def symbol_converted(symbol):
    """
    Input: symbol in the form '0005.HK' (from Alpha Vantange data)
    Output: symbol in the form '5' (for TWS)
    """
    slice_index = 0
    for i in range(0,4):
        if symbol[i]=='0':
            slice_index = i+1
        else:
            break
    return symbol[slice_index:-3]
