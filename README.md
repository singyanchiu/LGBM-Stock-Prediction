# LGBM-Stock-Prediction
A machine learning pipeline that ingest and process a 20-year historical stock price dataset and try to predict future prices using LightGBM.

# Data collection
The python script `01_Data_and_features.py` sends API requests to Alpha Vantage and downloads daily stock price data of the 50 constituent stocks of the Hang Seng Index. It then does basic data cleaning like converting zeros to nulls, etc. Basic features engineering are also done, eg. technical indicators like RSI, momentum, Bollinger Bands, simple and exponential moving averages, etc.

# Feature engineering
The jupyter notebook `Stock Data Feature Engineering.ipynb` we generate even more features from the dataset, eg. ATR (active trading range) and volatilities with various lookback windows. Feature importance will be studied in `Stock Prediction with LightGBM.ipynb.` 

# Labeling
In `Stock Data Feature Engineering.ipynb`, we employed the Triple-Barrier Method proposed by Marcos Lopez de Prado in his book [Advances in Financial Machine Learning.](https://www.amazon.sg/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=asc_df_1119482089/?tag=googleshoppin-22&linkCode=df0&hvadid=389050130770&hvpos=&hvnetw=g&hvrand=17837895927692401192&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9062548&hvtargid=pla-422557754574&psc=1) First, for every data point, we set a profit-taking (stop-loss) limit as it’s closing price plus(minus) twice its rolling 50-period standard deviation. Within a time limit of 5 days, if the profit-taking limit is touched first, we label that data point as +1. If the stop-loss limit is touched first, we label it as -1. If the time limit is touched first, we label it as 0.

# Model training and tuning
## Resampling
In `Stock Prediction with LightGBM.ipynb`, the dataset was splitted into training, validation and test set. Some samples in the training set were randomly dropped to ensure that the numbers of samples for each label are roughly the same.
## Hyperparameters tuning
We then use our data to train a LightGBM model and tune its hyperparameters using a random search approach. The hyperparameters being tuned are **max_depth, num_leaves, min_data, learning_rate** with **multi-class accuracy score** as our metric.
## Feature Importance and Result Analysis
With the optimized hyperparameters, we run our LightGBM model on the test set and analyse the results with a confusion matrix and study its feature importance. 

# Backtest
In `Backtest with LightGBM.ipynb`, we deploy our LightGBM model with optimized hyperparameters, in a backtest to test whether the strategy is profitable.

## Walk Forward Approach
We train, validate and test our model with a Walk Forward approach as described in https://alphascientist.com/walk_forward_model_building.html For each sample, the backtest algorithm predicts its label by training a LightGBM model (with optimized hyperparameters found above) with the preceding 2000 (the lookback period can be adjusted) data points.

## Making Stock Orders
The algorithm will then make long/short stock orders on each day according to the predicted labels. Also, on each day, it checks if there’s any open position that hits profit-taking or stop loss limits and closes the position if so. Any position that doesn’t hit these limits within 5 day will also be closed. Commissions and transaction costs are ignored in this backtest engine. The daily cash balance, portfolio value are recorded in a result dataframe. We did backtests with various parameters like sorting criteria in prioritized list of stock to order.

## Backtest Result Analysis
The performance of various strategies with different parameters are plotted and compared in `Results_Analysis.ipynb`. The Hang Seng Index was used as a baseline. All plots are rebased so that they can be compared on the same graph.
