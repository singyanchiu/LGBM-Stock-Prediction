import info
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
from utils import *

class Context():
    def __init__(self):
        self.capital_base = 1000

class Order():
    order_id = int(0)
    def __init__(self, order_type, symbol, num_shares, date_open, traded_price, date_limit, target_up, target_down):
        Order.order_id += int(1)
        self.order_type = order_type
        self.symbol = symbol
        self.num_shares = num_shares
        self.date_open = date_open
        self.traded_price = traded_price
        self.date_limit = date_limit
        self.target_up = target_up
        self.target_down = target_down

def initialize(context):
    context.capital_base = 500000
    context.capital_base_short = 500000
    context.year_length = 252
    context.cash = context.capital_base
    context.symbol = '1928.HK'
    context.positions = pd.DataFrame()
    context.log = pd.DataFrame()
    context.current_margin = 0
    context.margin_ratio_long = 0.5#1/3.18
    context.margin_ratio_short = 0.5#1/2.65
    context.positions_value = 0
    #context.factor = 0.4
    context.average_holding_period = 3.0
    context.lot_threshold_fraction = 0.9
    context.typical_amount = 40000
    context.max_positions = int((context.positions_value+context.cash)/(context.typical_amount*(context.margin_ratio_long+context.margin_ratio_short)/2))
    context.max_daily_orders = int(context.max_positions/context.average_holding_period)
    context.max_daily_margin_limit = (context.positions_value+context.cash)/context.average_holding_period

def inverse_volatility_weights(short_list, today, outcomes_new):
    weight_dict={}
    for symbol in short_list:
        weight_dict[symbol] = 1/outcomes_new.loc[(today, symbol)]['past_return_5_std50']
    factor=1/sum(weight_dict.values())
    for k in weight_dict:
        weight_dict[k] = weight_dict[k]*factor
    return weight_dict

def ann_ret(ts):
    return np.power((ts[-1]/ts[0]),(context.year_length/len(ts)))-1

def ann_ret(ts):
    return np.power((ts[-1]/ts[0]),(context.year_length/len(ts)))-1

def dd(ts):
    return np.min(ts/np.maximum.accumulate(ts))-1

def performance(result, window):
    perf = result.copy()
    perf['daily_returns'] = perf['porfolio_value'].diff()/perf['porfolio_value']-1
    rolling_window = result.porfolio_value.rolling(window)
    perf['annualized_return'] = rolling_window.apply(ann_ret)
    perf['annualized_volatility'] = result.daily_returns.rolling(window).std()*np.sqrt(252/window)
    perf['sharpe_ratio'] = perf['annualized_return']/perf['annualized_volatility']
    perf['drawdown'] = rolling_window.apply(dd)
    perf['exposure'] = perf['positions_value']/context.capital_base
    return perf

def get_price_list(outcomes_new, date):
    outcomes_close = outcomes_new['close']

    #Fill NaN close price for symbols that stop trading on certain days.
    #Replace with preceding value, group by symbols. (By using the unstack method)
    outcomes_close_fillna = outcomes_close.unstack(0).fillna(method='ffill', axis=1).unstack()

    #There are still NaN due to non-existence of certain stock during some days.
    #Just drop them as it is immpossible for it to appear on porfolio on that date
    outcomes_close_fillna = outcomes_close_fillna.dropna()

    #check if date is in dataset
    data_dates = get_dataset_dates(outcomes_close_fillna)
    if date not in data_dates:
        print("Date not in data set! It's probably not a trading date.")
        raise SystemExit('Error ocurred! Exiting.')

    return outcomes_close_fillna.loc[slice(date,date)]

def get_price_list_open(outcomes_new, date):
    outcomes_close = outcomes_new['open']

    #Fill NaN close price for symbols that stop trading on certain days.
    #Replace with preceding value, group by symbols. (By using the unstack method)
    outcomes_close_fillna = outcomes_close.unstack(0).fillna(method='bfill', axis=1).unstack()

    #There are still NaN due to non-existence of certain stock during some days.
    #Just drop them as it is immpossible for it to appear on porfolio on that date
    outcomes_close_fillna = outcomes_close_fillna.dropna()

    #check if date is in dataset
    data_dates = get_dataset_dates(outcomes_close_fillna)
    if date not in data_dates:
        print("Date not in data set! It's probably not a trading date.")
        raise SystemExit('Error ocurred! Exiting.')

    return outcomes_close_fillna.loc[slice(date,date)]

def order_target_num_shares(order,  num_shares, order_id, context):
    print (order.order_type)
    if order.order_type == 'BUY':

        #open LONG position
        print(num_shares)
        num_lots = num_shares // info.board_lots[order.symbol]
        if num_lots <= 0:
            print ('Number of shares is smaller than 1 lot of {}.'.format(order.symbol))
            print ('Order cancelled')
            return
        num_shares = num_lots * info.board_lots[order.symbol]
        amount_needed = num_shares*order.traded_price
        print(num_lots)
        print(num_shares)
        print('Amount needed = ', num_shares*order.traded_price)
        print('Current cash = ', context.cash)

        #Check if enough money left
        if amount_needed > context.cash:
            print ('Not enough cash to buy {} shares of {}'.format(num_shares, order.symbol))
            print ('Order cancelled!')
            return

        context.cash -= amount_needed
        print('Cash balance = ', context.cash)

        position = {'order_id' : Order.order_id,
                   'symbol' : order.symbol,
                   'num_shares' : num_shares,
                   'date_open' : order.date_open,
                   'traded_price' : order.traded_price,
                   'date_limit' : order.date_limit,
                   'target_up' : order.target_up,
                   'target_down' : order.target_down
                    }
        context.positions = context.positions.append(position, ignore_index=True)
    elif order.order_type == 'SHORT':

        #open SHORT position
        print(num_shares)
        num_lots = num_shares // info.board_lots[order.symbol]
        if num_lots <= 0:
            print ('Number of shares is smaller than 1 lot of {}.'.format(order.symbol))
            print ('Order cancelled')
            print ()
            return
        num_shares = num_lots * info.board_lots[order.symbol]
        print('Number of shares to SHORT: {}'.format(num_shares))

        amount_received = num_shares * order.traded_price
        print('Amount received = ', amount_received)

        context.cash += amount_received
        print('Cash balance = ', context.cash)

        position = {'order_id' : Order.order_id,
                    'symbol' : order.symbol,
                   'num_shares' : -num_shares,
                   'date_open' : order.date_open,
                   'traded_price' : order.traded_price,
                   'date_limit' : order.date_limit,
                   'target_up' : order.target_up,
                   'target_down' : order.target_down
                    }
        context.positions = context.positions.append(position, ignore_index=True)

    elif order.order_type == 'CLOSE':

        #Close LONG position
        #check if the order_id is in positions, if so, close that position (no partial close at this moment)
        print('order_id = ', order_id)
        index_to_drop = context.positions[context.positions['order_id'] == order_id].index[0]
        print('indext to drop = ', index_to_drop)
        cash_temp = context.cash
        cash_temp += context.positions.loc[index_to_drop]['num_shares'] * order.traded_price
        if cash_temp < 0:
            print()
            print("#######WARNING############")
            print("Unable to close SHORT position as there's not enough cash to buy stock!")
            print("CLOSE order cancelled")
            print("##########################")
            return
        context.cash = cash_temp

        print('Cash gain: ', context.positions.loc[index_to_drop]['num_shares'] * order.traded_price)
        print('Balance: ', context.cash)
        context.positions.drop(index_to_drop, inplace=True)
    return

def order_target_percent(order, percentage, order_id, context, remaining_margin_limit):
    if order.order_type != 'CLOSE':
        if order.order_type == 'BUY':
            margin_ratio = context.margin_ratio_long
            sign = 1
        elif order.order_type == 'SHORT':
            margin_ratio = context.margin_ratio_short
            sign = -1

        #open position
        #num_shares = (percentage * context.capital_base * context.factor)/order.traded_price
        num_shares = (percentage * (remaining_margin_limit/margin_ratio))/order.traded_price
        num_lots = num_shares // info.board_lots[order.symbol]
        if num_lots <= 0:
            print ('Percentage of captital is not enough for 1 lot of {}.'.format(order.symbol))
            #If num shares still bigger than a certain percentage of the board log
            #then we still make an order. Just make it 1 boart lot
            if num_shares >= context.lot_threshold_fraction*info.board_lots[order.symbol]:
                print('##########################################################')
                print('#  Re-adjusting amount a bit to meet board lot. ')
                print('##########################################################')
                num_lots = 1
            elif num_shares < context.lot_threshold_fraction*info.board_lots[order.symbol]:
                print('ORDER CANCELLED.')
                return 'min_charge_not_hitting', 0

        num_shares = num_lots * info.board_lots[order.symbol]
        amount = num_shares * order.traded_price
        print('Amount needed = ', num_shares*order.traded_price)
        print('Current cash = ', context.cash)

        #Margin check:
        margin_change = amount * margin_ratio
        print('Margin change = ', margin_change)
        if context.positions_value+context.cash < context.current_margin + margin_change:
            print('\n'+'#'*80)
            print('WARNING: NOT meeting margin requirement.')
            print ('{} Order cancelled'.format(order.order_type))
            print('\n'+'#'*80)
            return 'margin_overshoot', 0

        #Update MARGIN, CASH and positions
        context.current_margin += margin_change
        context.cash -= amount*sign
        context.positions_value += amount*sign
        print('Cash balance = ', context.cash)
        print('Current margin: ', context.current_margin)

        position = {'order_id' : Order.order_id,
                   'symbol' : order.symbol,
                   'num_shares' : num_shares*sign,
                   'date_open' : order.date_open,
                   'traded_price' : order.traded_price,
                   'date_limit' : order.date_limit,
                   'target_up' : order.target_up,
                   'target_down' : order.target_down
                    }
        log =      {'action' : order.order_type,
                   'symbol' : order.symbol,
                   'num_shares' : num_shares*sign,
                   'date_open' : order.date_open,
                   'traded_price' : order.traded_price,
                   'date_limit' : order.date_limit,
                   'target_up' : order.target_up,
                   'target_down' : order.target_down,
                   'amount' : amount,
                   'current_margin' : context.current_margin
                    }
        context.log = context.log.append(log, ignore_index=True)
        context.positions = context.positions.append(position, ignore_index=True)
        return 'successful', margin_change

    elif order.order_type == 'CLOSE':

        #Close position
        #check if the order_id is in positions, if so, close that position (no partial close at this moment)
        print('order_id = ', order_id)
        index_to_drop = context.positions[context.positions['order_id'] == order_id].index[0]
        print('indext to drop = ', index_to_drop)

        cash_temp = context.cash
        cash_temp += context.positions.loc[index_to_drop]['num_shares'] * order.traded_price
        context.cash = cash_temp

        print('Cash gain: ', context.positions.loc[index_to_drop]['num_shares'] * order.traded_price)
        print('Balance: ', context.cash)
        context.positions.drop(index_to_drop, inplace=True)
    return 'successful', 0

def positions_value(positions, price_list):
    value = 0
    for i, row in positions.iterrows():
        date = price_list.index.get_level_values('date')[0].strftime("%Y-%m-%d")
        price = price_list[:,row['symbol']].values[0]
#         print('Order_id: ', row['order_id'])
#         print('Symbol = ', row['symbol'])
#         print('Price on {} = {}'.format(date, price))
#         print('Num shares = ', row['num_shares'])
#         print('Amount = ', price * row['num_shares'])
#         print()
        value += price * row['num_shares']
    return value

def positions_value_absolute(positions, price_list):
    value = 0
    for i, row in positions.iterrows():
        price = price_list[:,row['symbol']].values[0]
        if row['num_shares']>=0:
            value += price * row['num_shares']
        elif row['num_shares']<0:
            value -= price * row['num_shares']
    return value

def backtest_lgb(outcomes_new, symbols):
    trade_date_list = sorted(list(set(outcomes_new.index.get_level_values('date'))))
    outcomes_new = outcomes_new.drop(['log_return_10', 'return_10'], axis=1)
    outcomes_new_dropna = outcomes_new.dropna()

    next_day_open = lambda x: x.shift(-1)
    outcomes_new_dropna['next_day_open'] = outcomes_new_dropna.groupby(level='symbol').apply(next_day_open)['open']

    features_selected = info.features_selected

    prob_threshold = 0.7
    test_period = 20
    test_size = 1
    valid_size = 400
    training_size = 2000
    valid_test_gap = 4
    q_upper = 0.9
    q_lower = 0.1
    return_col = 'log_return_5'
    return_col_actual = 'return_5'
    max_depth_range = np.arange(2,5,1)
    num_leaves_range = np.arange(9,15,1)

    #make a list of test dates according to test period, by back counting from the latest date with target
    test_dates = sorted(list(set(outcomes_new_dropna.index.get_level_values('date'))))[-test_period:]

    test_dates_converted = []
    for ts in test_dates:
        test_dates_converted.append(ts.to_pydatetime())

    results = pd.DataFrame()

    for test_date in test_dates_converted:

        print()

        price_list = get_price_list(outcomes_new, test_date)

        price_list_open = get_price_list_open(outcomes_new, test_date)

        if 'buy_list' not in globals() and 'buy_list' not in locals():
            buy_list = []

        #Iterate throught the buy_list (which is created by the previous day's ML model run)
        #Make BUY order for each symbol in buy_list, at open price
        #Imagine the time now is before market open. Trying to buy stock at open price
        if len(buy_list)>0:
            for symbol_to_buy in buy_list:
                order_type = 'BUY'
                #order_id = ''
                num_lots = 1
                num_shares = num_lots * info.board_lots[symbol_to_buy]
                date_open = test_date
                traded_price = price_list_open[:,symbol_to_buy].values[0]
                date_limit = trade_date_list[trade_date_list.index(date_open)+4]
                target_up = 33.7
                target_down = 27.5
                percentage = 0.2

                print("Making buy order.", date_open.strftime("%Y-%m-%d"), traded_price, date_limit.strftime("%Y-%m-%d"))
                order = Order(order_type, symbol_to_buy, num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage)

        #Iterate through positions and see if need to sell any of them on this date
        #Imagine the time now is right before market close. Trying to sell stock at close price.
        for i, row in context.positions.iterrows():
            if test_date == row['date_limit']:

                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = price_list[:,row['symbol']].values[0]
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id=order_id)

        #After all the buy and sell (imagine the time now is after market close)
        #Update porfolio value
        value = positions_value(context.positions, price_list)

        if 'symbol' in context.positions.columns:
            positions_list = list(context.positions['symbol'])
        else:
            positions_list = []

        result = {'positions': positions_list, #context.positions.to_dict('records'),
                'date': test_date,
                 'positions_value': value,
                 'cash_value': context.cash,
                 'porfolio_value': value+context.cash}
        results = results.append(result, ignore_index=True)

        #HERE's the PREDICTION PART
        #Loop throught list of symbols
        #For each symbol, train LGB with available data to to this date (minus test valid gap)
        #Then make predictions on test date

        buy_list=[]

        for symbol in symbols:

            print('Predicting for {} on {}'.format(symbol, test_date))

            outcomes_new_selected_symbol = outcomes_new_dropna.loc[outcomes_new_dropna.index.get_level_values('symbol')==symbol]

            #get all trading dates from dataset of this symbol
            data_dates = sorted(list(set(outcomes_new_selected_symbol.index.get_level_values(0))))
            data_converted_dates = []
            for ts in data_dates:
                data_converted_dates.append(ts.to_pydatetime())

            #Check if test data is a trading date for this stock
            if test_date not in data_converted_dates:
                print('Stock data for {} not available on {}. Skipping this symbol for this date'.format(symbol, test_date))
                continue

            #Calculate training start date and valid start date by back counting
            start_date = data_converted_dates[data_converted_dates.index(test_date)-(valid_size + training_size + valid_test_gap + test_size)]
            start_date_valid = data_converted_dates[data_converted_dates.index(test_date)-(valid_size + valid_test_gap + test_size)]
            end_date_valid = data_converted_dates[data_converted_dates.index(test_date)-valid_test_gap-test_size]
            start_date_test = data_converted_dates[data_converted_dates.index(test_date)-test_size]
            end_date_test = test_date
            #print('start_date = ', start_date)
#             print('start_date_valid = ', start_date_valid)
#             print('end_date_valid = ', end_date_valid)
            #print('test_date = ', test_date)

            X_y_train, X_y_valid, X_y_test = train_valid_test_split(outcomes_new_selected_symbol, start_date,
                                                                            start_date_valid, end_date_valid, start_date_test, end_date_test)

            #calculate upper threshold in training set, then create targets for both valid and test
            X_y_train, X_y_valid, X_y_test = add_target_upper(X_y_train, X_y_valid,
                                                              X_y_test, q_upper, 'target', return_col)

            X_y_test['symbol'] = X_y_test.index.get_level_values('symbol')

            #downsample the training set's negative data points
            X_y_train_resampled = downsample(X_y_train, 'target', test_ratio=0.11, random_seed=11)

            #create 2 extra sets for calculating gain
            X_valid_close = X_y_valid[['close',return_col_actual]]
            X_test_close = X_y_test[['close',return_col_actual, 'symbol','next_day_open']]

            num_shares = []

            for i, row in X_test_close.iterrows():
                num_shares.append(info.board_lots[row[2]] * info.multiplier[row[2]])

            X_test_close['num_shares'] = num_shares

            #split into features and target sets
            X_train, y_train = feature_target_split(X_y_train_resampled, features_selected, 'target')
            X_valid, y_valid = feature_target_split(X_y_valid, features_selected, 'target')
            X_test, y_test = feature_target_split(X_y_test, features_selected, 'target')

            (best_model, best_pres_model, max_total_gain, optimal_depth, optimal_num_leaves, max_precision,
            optimal_precision_depth, optimal_precision_num_leaves,
            max_precision_total_gain) = lgb_train(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, return_col_actual,
                                                 min_data = 11, metric = 'auc', prob_threshold = prob_threshold)

            y_test_pred = best_model.predict(X_test, num_iteration=best_model.best_iteration)

            y_class_pred = class_switch_binary(y_test, y_test_pred, prob_threshold)

            if y_class_pred[0] == 1:
                print('Predicted POSITIVE for {}'.format(symbol))
                buy_list.append(symbol)

    return results

def backtest_random(outcomes_new, context):

    outcomes_new_dropna = outcomes_new.dropna()

    test_period = 200

    trade_date_list = sorted(list(set(outcomes_new.index.get_level_values('date'))))

    test_dates = sorted(list(set(outcomes_new_dropna.index.get_level_values('date'))))[-test_period:]

    test_dates_converted = []
    for ts in test_dates:
        test_dates_converted.append(ts.to_pydatetime())

    results = pd.DataFrame()

    for test_date in test_dates_converted:

        print()

        price_list = get_price_list(outcomes_new, test_date)

        price_list_open = get_price_list_open(outcomes_new, test_date)

        if 'buy_list' not in globals() and 'buy_list' not in locals():
            buy_list = []

        if 'short_list' not in globals() and 'short_list' not in locals():
            short_list = []

        #Iterate throught the buy_list (which is created by the previous day's ML model run)
        #Make BUY order for each symbol in buy_list, at open price
        #Imagine the time now is before market open. Trying to buy stock at open price
        if len(buy_list)>0:
            weights = inverse_volatility_weights(buy_list, test_date, outcomes_new)
            for symbol_to_buy in buy_list:
                order_type = 'BUY'
                order_id = ''
                num_lots = 1
                num_shares = num_lots * info.board_lots[symbol_to_buy]
                date_open = test_date
                traded_price = price_list_open[:,symbol_to_buy].values[0]
                date_limit = trade_date_list[trade_date_list.index(date_open)+4]
                target_up = outcomes_new.loc[(date_open,symbol_to_buy)]['target_upper_v2']
                target_down = outcomes_new.loc[(date_open,symbol_to_buy)]['target_lower_v2']
                percentage = weights[symbol_to_buy]

                print("Making buy order. Date open: {}, Date limit: {}, Traded price: {}.".format(date_open.strftime("%Y-%m-%d"),
                                                                                                    date_limit.strftime("%Y-%m-%d"), traded_price))
                order = Order(order_type, symbol_to_buy, num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)

        #Iterate throught the short_list (which is created by the previous day's ML model run)
        #Make SHORT order for each symbol in buy_list, at open price
        #Allocate weights to each symbol by inverse volatility of past_return_5
        if len(short_list)>0:
            weights = inverse_volatility_weights(short_list, test_date, outcomes_new)
            for symbol_to_short in short_list:
                order_type = 'SHORT'
                order_id = ''
                num_lots = 1
                num_shares = 1
                date_open = test_date
                traded_price = price_list_open[:,symbol_to_short].values[0]
                date_limit = trade_date_list[trade_date_list.index(date_open)+4]
                target_up = outcomes_new.loc[(date_open,symbol_to_short)]['target_upper_v2']
                target_down = outcomes_new.loc[(date_open,symbol_to_short)]['target_lower_v2']
                percentage = weights[symbol_to_short]

                print("Making short order. Date open: {}, Date limit: {}, Traded price: {}.".format(date_open.strftime("%Y-%m-%d"),
                                                                                                    date_limit.strftime("%Y-%m-%d"), traded_price))
                order = Order(order_type, symbol_to_short, num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)

        #listen to market price of all stocks in positions
        #If hitting target price, then sell
        for i, row in context.positions.iterrows():
            if outcomes_new.loc[(test_date,row['symbol'])]['high']>row['target_up']:
                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = row['target_up']
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)
                continue
            if outcomes_new.loc[(test_date,row['symbol'])]['low']<row['target_down']:
                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = row['target_down']
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)
                continue

        #Near market close - check what stock in position, with date limit = today. Sell it at close price
        for i, row in context.positions.iterrows():
            #print('type of row[date_limit]', type(row['date_limit']))
            if test_date >= row['date_limit']:
                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = price_list[:,row['symbol']].values[0]
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)

        #After all the buy and sell (imagine the time now is after market close)
        #Update porfolio value
        value = positions_value(context.positions, price_list)

        if 'symbol' in context.positions.columns:
            positions_list = list(context.positions['symbol'])
        else:
            positions_list = []

        result = {'positions': positions_list, #context.positions.to_dict('records'),
                'date': test_date,
                 'positions_value': value,
                 'cash_value': context.cash,
                 'porfolio_value': value+context.cash}
        results = results.append(result, ignore_index=True)

        #HERE's the PREDICTION PART

        buy_list=[]
        short_list=[]

        print('Test date: ', test_date)

        X_test = outcomes_new.loc[test_date]

        buy_df = pd.DataFrame()
        short_df = pd.DataFrame()

        symbol_universe = list(X_test.index.get_level_values('symbol'))
        X_test_samples = X_test.loc[X_test.index.get_level_values('symbol').isin(random.sample(symbol_universe,10))].sample(frac=1)
        buy_df=X_test_samples[:5]
        short_df=X_test_samples[5:]

        buy_list = list(buy_df.index.get_level_values('symbol'))
        short_list = list(short_df.index.get_level_values('symbol'))

        print("--------------------------------")
        print('Buy list: ', buy_list)
        print('Short list: ', short_list)
        print("--------------------------------")
    return results

def backtest_lgb_v2(outcomes_new, context):

    outcomes_new_dropna = outcomes_new.dropna()

    trade_date_list = sorted(list(set(outcomes_new.index.get_level_values('date'))))

    features_selected = info.features_lgb_v2

    max_impact_pct = 0.001
    max_impact_volatility = 'price_chg_1_ema_std50'
    total_budget = context.capital_base * 1.2

    test_period = 250
    test_size = 1
    valid_size = 1
    training_size = 2000
    valid_test_gap = 4
    label = 'label_v2'

    #make a list of test dates according to test period, by back counting from the latest date with target
    test_dates = sorted(list(set(outcomes_new_dropna.index.get_level_values('date'))))[-test_period:]

    test_dates_converted = []
    for ts in test_dates:
        test_dates_converted.append(ts.to_pydatetime())

    results = pd.DataFrame()

    context.predictions = pd.DataFrame()

    for test_date in test_dates_converted:

        print()

        price_list = get_price_list(outcomes_new, test_date)

        price_list_open = get_price_list_open(outcomes_new, test_date)

        if 'buy_list' not in globals() and 'buy_list' not in locals():
            buy_list = []

        if 'short_list' not in globals() and 'short_list' not in locals():
            short_list = []

        #Iterate throught the buy_list (which is created by the previous day's ML model run)
        #Make BUY order for each symbol in buy_list, at open price
        #Imagine the time now is before market open. Trying to buy stock at open price

        max_impact = (context.positions_value+context.cash) * max_impact_pct

        #####################################################################################################
        #Need to change algorithm so that it loops through alternate buy and short on the list, so each side
        #gets equal chance to be ordered, due to the margin limit
        #########################################################################

        len_buy = len(buy_list)
        len_short = len(short_list)

        #calculate weights for both lists according to inverse volatility (std of past_return_5 )
        if len_buy>0:
            weights_buy = inverse_volatility_weights(buy_list, test_date, outcomes_new)

        if len_short>0:
            weights_short = inverse_volatility_weights(short_list, test_date, outcomes_new)

        #Alternate buy and short lists to make order. Since we check margin for each order, so we want to give buy and short
        #roughly equal chance to get ordered
        for symbol_to_buy, symbol_to_short in zip(buy_list, short_list):

            #open long position
            order_type = 'BUY'
            order_id = ''
            #max_impact/outcomes_new.loc[(test_date, symbol_to_buy)][max_impact_volatility]
            num_shares = 1
            date_open = test_date
            traded_price = price_list_open[:,symbol_to_buy].values[0]
            date_limit = trade_date_list[trade_date_list.index(date_open)+4]
            target_up = outcomes_new.loc[(date_open,symbol_to_buy)]['target_upper_v2']
            target_down = outcomes_new.loc[(date_open,symbol_to_buy)]['target_lower_v2']
            percentage = weights_buy[symbol_to_buy]
            print()
            print("[BUY] Symbol: {} Date open: {}, Date limit: {}, Traded price: {}.".format(symbol_to_buy,
                                                                                             date_open.strftime("%Y-%m-%d"),
                                                                                             date_limit.strftime("%Y-%m-%d"),
                                                                                             traded_price))
            order = Order(order_type, symbol_to_buy, num_shares, date_open, traded_price, date_limit, target_up, target_down)
            order_target_percent(order,  percentage, order_id, context)

            #open short position
            order_type = 'SHORT'
            order_id = ''
            #num_shares = max_impact/outcomes_new.loc[(test_date, symbol_to_buy)][max_impact_volatility]
            num_shares = 1
            date_open = test_date
            traded_price = price_list_open[:,symbol_to_short].values[0]
            date_limit = trade_date_list[trade_date_list.index(date_open)+4]
            target_up = outcomes_new.loc[(date_open,symbol_to_short)]['target_upper_v2']
            target_down = outcomes_new.loc[(date_open,symbol_to_short)]['target_lower_v2']
            percentage = weights_short[symbol_to_short]
            print()
            print("[SHORT] Symbol: {} Date open: {}, Date limit: {}, Traded price: {}.".format(symbol_to_short,
                                                                                               date_open.strftime("%Y-%m-%d"),
                                                                                               date_limit.strftime("%Y-%m-%d"),
                                                                                               traded_price))
            order = Order(order_type, symbol_to_short, num_shares, date_open, traded_price, date_limit, target_up, target_down)
            order_target_percent(order,  percentage, order_id, context)

        if len_buy>len_short:
            for k in range(len_short, len_buy):
                symbol_to_buy = buy_list[k]
                order_type = 'BUY'
                order_id = ''
                #max_impact/outcomes_new.loc[(test_date, symbol_to_buy)][max_impact_volatility]
                num_shares = 1
                date_open = test_date
                traded_price = price_list_open[:,symbol_to_buy].values[0]
                date_limit = trade_date_list[trade_date_list.index(date_open)+4]
                target_up = outcomes_new.loc[(date_open,symbol_to_buy)]['target_upper_v2']
                target_down = outcomes_new.loc[(date_open,symbol_to_buy)]['target_lower_v2']
                percentage = weights_buy[symbol_to_buy]
                print()
                print("[BUY] Symbol: {} Date open: {}, Date limit: {}, Traded price: {}.".format(symbol_to_buy,
                                                                                                 date_open.strftime("%Y-%m-%d"),
                                                                                                 date_limit.strftime("%Y-%m-%d"),
                                                                                                 traded_price))
                order = Order(order_type, symbol_to_buy, num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)

        elif len_short>len_buy:
            for k in range(len_buy, len_short):
                symbol_to_short = short_list[k]
                order_type = 'SHORT'
                order_id = ''
                #num_shares = max_impact/outcomes_new.loc[(test_date, symbol_to_buy)][max_impact_volatility]
                num_shares = 1
                date_open = test_date
                traded_price = price_list_open[:,symbol_to_short].values[0]
                date_limit = trade_date_list[trade_date_list.index(date_open)+4]
                target_up = outcomes_new.loc[(date_open,symbol_to_short)]['target_upper_v2']
                target_down = outcomes_new.loc[(date_open,symbol_to_short)]['target_lower_v2']
                percentage = weights_short[symbol_to_short]
                print()
                print("[SHORT] Symbol: {} Date open: {}, Date limit: {}, Traded price: {}.".format(symbol_to_short,
                                                                                                   date_open.strftime("%Y-%m-%d"),
                                                                                                   date_limit.strftime("%Y-%m-%d"),
                                                                                                   traded_price))
                order = Order(order_type, symbol_to_short, num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)

        #listen to market price of all stocks in positions
        #If hitting target price, then sell
        for i, row in context.positions.iterrows():
            if outcomes_new.loc[(test_date,row['symbol'])]['high']>row['target_up']:
                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = row['target_up']
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print()
                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)
                continue

            if outcomes_new.loc[(test_date,row['symbol'])]['low']<row['target_down']:
                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = row['target_down']
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print()
                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)
                continue

        #Near market close - check what stock in position, with date limit = today. Sell it at close price
        for i, row in context.positions.iterrows():
            #print('type of row[date_limit]', type(row['date_limit']))
            if test_date >= row['date_limit']:
                order_type = 'CLOSE'
                order_id = row['order_id']
                num_lots = 1
                num_shares = float('Nan')
                date_open = float('Nan')
                traded_price = price_list[:,row['symbol']].values[0]
                date_limit = float('Nan')
                target_up = float('Nan')
                target_down = float('Nan')
                percentage = 0

                print()
                print("Making CLOSE order.", test_date.strftime("%Y-%m-%d"), traded_price)
                order = Order(order_type, row['symbol'], num_shares, date_open, traded_price, date_limit, target_up, target_down)
                order_target_percent(order,  percentage, order_id, context)


        #After all the buy and sell (imagine the time now is after market close)
        #Update porfolio value
        context.positions_value = positions_value(context.positions, price_list)

        if 'symbol' in context.positions.columns:
            positions_list = list(context.positions['symbol'])
        else:
            positions_list = []

        result = {'positions': positions_list, #context.positions.to_dict('records'),
                'date': test_date,
                 'positions_value': context.positions_value,
                 'cash_value': context.cash,
                 'porfolio_value': context.positions_value+context.cash}

        results = results.append(result, ignore_index=True)

        #Update margin value due to the closing positions and STOCK VALUE CHANGE. USE CLOSE VALUE FOR THE DAY
        context.current_margin = positions_value_absolute(context.positions, price_list) * context.margin_ratio_long

        #CHECK MAINTENANCE MARGIN
        if context.positions_value+context.cash <= context.current_margin*0.8:
            print ('ELV lower than!!! MAINTENANCE MARGIN. GAME OVER LIAO.')
            return results

        #End of Day SMA calculation: ELV-Reg T Margin >= 0, cash>=-0.5*stock_value
        if context.positions_value+context.cash < context.positions_value * 0.5:
            print ('SMA smaller than zero! GAME OVER.')
            return results


        #HERE's the PREDICTION PART
        buy_list=[]
        short_list=[]

        print('Test date: ', test_date)
        #print('type of test_date: ', type(test_date))

        #get all trading dates from dataset of this symbol
        data_dates = sorted(list(set(outcomes_new_dropna.index.get_level_values(0))))
        data_converted_dates = []
        for ts in data_dates:
            data_converted_dates.append(ts.to_pydatetime())

        #Calculate training start date and valid start date by back counting
        start_date = data_converted_dates[data_converted_dates.index(test_date)-(valid_size + training_size + valid_test_gap + test_size)]
        start_date_valid = data_converted_dates[data_converted_dates.index(test_date)-(valid_size + valid_test_gap + test_size)]
        end_date_valid = data_converted_dates[data_converted_dates.index(test_date)-valid_test_gap-test_size]
        start_date_test = data_converted_dates[data_converted_dates.index(test_date)-test_size]
        end_date_test = test_date

#         print('start_date = ', start_date)
#         print('start_date_valid = ', start_date_valid)
#         print('end_date_valid = ', end_date_valid)
#         print('start_date_test = ', start_date_test)
        #print('test_date = ', test_date)

        X_y_train, X_y_valid, X_y_test = train_valid_test_split(outcomes_new_dropna, start_date,
                                                                    start_date_valid, end_date_valid, start_date_test, end_date_test)

        #downsample the training set so each class has same number of samples
        X_y_train_resampled = downsample_3class(X_y_train, label, 42)
        #X_y_valid_resampled = downsample_3class(X_y_valid, label, 42)
        #X_y_test_resampled = downsample_3class(X_y_test, label, 42)

        #rename all -1 labels to 2 as LGB only takes 0,1,2...
        X_y_train_resampled.loc[X_y_train_resampled[label] == -1, label] = 2
        X_y_valid.loc[X_y_valid[label] == -1, label] = 2
        X_y_test.loc[X_y_test[label] == -1, label] = 2

        #split into features and target sets
        X_train, y_train = feature_target_split(X_y_train_resampled, features_selected, label)
        X_valid, y_valid = feature_target_split(X_y_valid, features_selected, label)
        X_test, y_test = feature_target_split(X_y_test, features_selected, label)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        max_acc = float("-inf")
        boosting = 'gbdt'

        learning_rate = 0.006985117638031729
        min_data = 16
        num_leaves = 47
        max_depth = 7

#         Hyperparameters with April 9th data:
#         learning_rate = 0.0010430175663616878
#         max_depth = 10
#         num_leaves = 52
#         min_data = 19
#
#             Training with Max Depth = 8 Num Leaves = 77 Min data = 18 Learning Rate = 0.0029137891706607867
#
#             Training accuracy =  0.6237882177479492
#             Validation accuracy =  0.47768670309653916
#             Test accuracy =  0.5101626016260162

        parameters = {
            'application': 'multiclass',
            'num_class': 3,
            #'is_unbalance': 'false',
            #'metric': 'auc',
            #'scale_pos_weight': 9,
            'boosting': boosting,
            'num_leaves': num_leaves,
            'feature_fraction': 0.95,
            'bagging_fraction': 0.2,
            'bagging_freq': 10,
            'learning_rate': learning_rate,
            'verbose': 0,
            'min_data_in_leaf': min_data,
            'max_depth': max_depth
        }
        model = lgb.train(parameters,
                               train_data,
                               valid_sets=valid_data,
                               num_boost_round=5000,
                                verbose_eval=False,
                               #feval=lgb_f1_score,
                               early_stopping_rounds=100)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_class_pred = [pred.argmax() for pred in y_pred]

        X_test['pred'] = y_class_pred

        buy_df = pd.DataFrame()
        short_df = pd.DataFrame()

        #search criteria for coming up with buy/short list
        search_criteria = 'momentum_25'

        for i, row in X_test.iterrows():
            if row['pred'] == 1: #and row['momentum_50']>10:
#                                 and row['close_scaled50']>1.5\
#                                 and row['volume_scaled50']>0.5:
                #print('Predicted POSITIVE for {}'.format(i[1]))
                entry = {'symbol': i[1],
                        'volatility': row['past_return_5_std50'],
                        search_criteria: row[search_criteria]}
                buy_df=buy_df.append(entry, ignore_index=True)
            elif row['pred'] == 2: #and row['momentum_5']<-60 \
#                                   and row['close_scaled50']<1.5\
#                                   and row['volume_scaled50']>1\
#                                   and row['bull_ratio']<0:
                #print('Predicted NEGATIVE for {}'.format(i[1]))
                entry = {'symbol': i[1],
                        'volatility': row['past_return_5_std50'],
                       search_criteria: row[search_criteria]}
                short_df=short_df.append(entry, ignore_index=True)

        if len(buy_df)!=0 or len(short_df)!=0:
            positive_pct = len(buy_df)/(len(buy_df)+len(short_df))
            negative_pct = len(short_df)/(len(buy_df)+len(short_df))
        else:
            positive_pct = 0
            negative_pct = 0

        print('Postive percentage: ', positive_pct)
        print('Negative percentage: ', negative_pct)
        print()

        if len(buy_df)>0:
            positive_list = list(buy_df['symbol'])
        else:
            positive_list = []

        if len(short_df)>0:
            negative_list = list(short_df['symbol'])
        else:
            negative_list = []

        predictions_entry = {'date':test_date,
                            'num_predictions_long':len(buy_df),
                            'num_predictions_short':len(short_df),
                            'predictions_long':positive_list,
                            'predictions_short':negative_list,
                            'percentage_long':positive_pct,
                            'percentage_short':negative_pct}

        context.predictions=context.predictions.append(predictions_entry, ignore_index=True)

        buy_drop_indexes = []
        short_drop_indexes = []

        for i, row in buy_df.iterrows():
            if len(context.positions)>0:
                if row['symbol'] in list(context.positions['symbol']):
#                     print('####################################')
#                     print('{} already in porfolio. It will not be included in buy list.'.format(row['symbol']))
#                     print('####################################')
                    buy_drop_indexes.append(i)

        for i, row in short_df.iterrows():
            if len(context.positions)>0:
                if row['symbol'] in list(context.positions['symbol']):
#                     print('####################################')
#                     print('{} already in porfolio. It will not be included in short list.'.format(row['symbol']))
#                     print('####################################')
                    short_drop_indexes.append(i)

        buy_df=buy_df.drop(buy_drop_indexes)
        short_df=short_df.drop(short_drop_indexes)

        #if too many predictions, shortlist them by inverse volatility

        if positive_pct == 1:
            max_num_long = 6
            max_num_short = 0
        elif positive_pct >= 0.83:
            max_num_long = 5
            max_num_short = 1
        elif positive_pct >= 0.67:
            max_num_long = 4
            max_num_short = 1
        elif positive_pct >= 0.5:
            max_num_long = 3
            max_num_short = 2
        elif positive_pct >= 0.33:
            max_num_long = 2
            max_num_short = 3
        elif positive_pct >= 0.16:
            max_num_long = 1
            max_num_short = 4
        else:
            max_num_long = 0
            max_num_short = 5

        #####################################
        # SET TO 5 SYMBOLS MAX for each
        ###########################
        max_num_long = 5
        max_num_short = 5

        #Search criteria
        search_criteria = 'volatility'

        #Sort it so that smallest volatility comes first
        if len(buy_df)>0:
            buy_df = buy_df.sort_values(by=[search_criteria], ascending=True)

        if len(buy_df)>max_num_long:
            buy_list = list(buy_df['symbol'][:max_num_long])
        elif len(buy_df)>0:
            buy_list = list(buy_df['symbol'])
        else:
            buy_list = []

        #########################################################
        # SET BUY LIST = NULL SO IT WON"T OPEN ANY LONG POSITION#s
        #########################################################
        #buy_list = []

        #Sort it so that smallest volatility comes first
        if len(short_df)>0:
            short_df = short_df.sort_values(by=[search_criteria], ascending=True)

        if len(short_df)>max_num_short:
            short_list = list(short_df['symbol'][:max_num_short])
        elif len(short_df)>0:
            short_list = list(short_df['symbol'])
        else:
            short_list = []

        #short_list = []

        print("--------------------------------")
        print('Buy list: ', buy_list)
        print('Short list: ', short_list)
        print("--------------------------------")

    return results
