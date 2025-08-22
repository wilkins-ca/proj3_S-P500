import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from warnings import filterwarnings
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LinearRegression as linreg
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor as gradboostreg
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import ta

filterwarnings('ignore')

# list of files we want to examine
pathstring = 'sp500_5yr_data/'
companylist = [pathstring + 'AAPL_data.csv',
               pathstring + 'AMZN_data.csv',
               pathstring + 'GOOG_data.csv',
               pathstring + 'MSFT_data.csv']

# put all the data together in a df
companydata = []
for file in companylist:
    current = pd.read_csv(file)
    companydata.append(current)

df0 = pd.concat(companydata, ignore_index = True)


# have a look at the data
print(df0.describe()) # length of 4752
print(df0.isnull().sum()) # no nulls
print(df0.columns) # date, open, high, low, close, volume, Name
print(df0.head())
print(df0.dtypes) # date is str, open, high, low, close are float, volume is int, name is str


# convert date col to datetime type
df0['date'] = pd.to_datetime(df0['date'])
print(df0['date'].dtypes)

# strlist of company names
companynames = df0['Name'].unique()

# plot date vs close for each company
plt.figure(figsize = (9,7))
for idx, company in enumerate(companynames, 1):
    plt.subplot(2,2,idx)
    filter1 = df0['Name'] == company
    df_temp = df0[filter1]
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    plt.plot(df_temp['date'], df_temp['close'])
    plt.title(company)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

# calc rolling avg w window of 7 days and plot
plt.figure(figsize = (9, 7))
for idx, company in enumerate(companynames, 1):
    plt.subplot(2,2,idx)
    filter1 = df0['Name'] == company
    df_temp = df0[filter1]
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    plt.plot(df_temp['date'], df_temp['high'].rolling(window = 7).mean())
    plt.title(company)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()

# look at different window sizes for rolling avg
df1 = (df0.assign(
    high_14 = df0['high'].rolling(window = 14).mean(),
    high_21 = df0['high'].rolling(window = 21).mean(),
    high_28 = df0['high'].rolling(window = 28).mean()
))

plt.figure(figsize = (9, 7))
for idx, company in enumerate(companynames, 1):
    plt.subplot(2,2,idx)
    filter1 = df1['Name'] == company
    df = df1[filter1]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace = True)
    df[['high_14','high_21','high_28']].plot(ax = plt.gca())
    plt.title(company)

plt.tight_layout()
plt.show()

# we know each line in the df represents a day, so call pct_change() on the close column
# to see the returns for the day; we will focus on Microsoft
msft_df = df1[df1['Name'] == 'MSFT']
msft_df['Daily return in %'] = msft_df['close'].pct_change() * 100
# plot daily return percentage
fig, ax = plt.subplots()
ax.plot(pd.to_datetime(msft_df['date']), msft_df['Daily return in %'])
ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.xlabel('Date')
plt.xticks(rotation = 45)
plt.ylabel('Percent Return')
plt.title('Microsoft Daily Return Percentage')
plt.tight_layout()
plt.show()


# look at quarterly avg return by company (time resampling)
pd.to_datetime(df1['date'])
df1.set_index('date', inplace = True)
df1['pct_return'] = df1.groupby('Name')['close'].pct_change() * 100
q_avg = (
    df1.groupby('Name')['pct_return']
    .resample('Q')
    .mean()
    .reset_index()
)
print(q_avg.head())
print(q_avg.shape)
print(q_avg.columns)

# plot quarterly avg
plt.figure(figsize = (9, 7))
for idx, company in enumerate(companynames, 1):
    plt.subplot(2,2,idx)
    filter1 = q_avg['Name'] == company
    df = q_avg[filter1]
    df['date'] = pd.to_datetime(df['date'])
    plt.plot(df['date'], df['pct_return'])
    plt.title(company)
    plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()

# correlation between closing prices?
apple = df1[df1['Name'] == 'AAPL']
microsoft = df1[df1['Name'] == 'MSFT']
amazon = df1[df1['Name'] == 'AMZN']
google = df1[df1['Name'] == 'GOOG']

closing = pd.DataFrame()
closing['apple'] = apple['close']
closing['microsoft'] = microsoft['close']
closing['amazon'] = amazon['close']
closing['google'] = google['close']

corr_mat = closing.corr()
sns.heatmap(corr_mat, annot = True)
plt.show() 
# shows that google and apple prices are not well correlated, 
# and amazon and google are the most correlated at 0.98

# check if daily change in stock price is correlated to daily returns in stock price
for col in closing.columns:
    closing[col + '_pctChange'] = (closing[col] - closing[col].shift(1)) / closing[col].shift(1) * 100

closing2 = closing[['apple_pctChange', 'microsoft_pctChange','amazon_pctChange','google_pctChange']]

grid = sns.PairGrid(data = closing2)
grid.map_diag(sns.histplot)
grid.map_lower(sns.scatterplot)
grid.map_upper(sns.kdeplot)
plt.show()

closingCorrMat = closing2.corr()
sns.heatmap(closingCorrMat, annot = True)
plt.show()

# ==================================================================

# price movement classification (will it go up or down)
# price range/volatility prediction
# calculating risk metric with Sharpe ratio
# cluster similar stocks

df1.reset_index(inplace = True)

# up or down price movement
# make target variable (1 if price goes up, 0 if price stays same or goes down)
df1['target'] = (df1['close'].shift(-1) > df1['close']).astype(int)

# calc features to feed into model
# daily return
df2 = (df1.assign(
    dailyReturn = (df1['close'] - df1['open']) / df1['open'], # shows momentum
    prevDayReturn = df1['close'].shift(1), # patterns could repeat
    dailyRange = df1['high'] - df1['low'], # shows volatility
    mvAvg7days = df1['close'].rolling(window = 7).mean(), # signal trends possibly
    mvAvg14days = df1['close'].rolling(window = 14).mean(), # signal trends possibly
    mvAvg30days = df1['close'].rolling(window = 30).mean(), # added after first pass, trying to refine features
    volChange = (df1['volume'] - df1['volume'].shift(1)) / df1['volume'].shift(1) # stronger conviction on the trade
))

# looking at trends over multiple days
for i in range(1, 4):
    df2[f'returnLag{i}'] = df2['dailyReturn'].shift(i)

# additional features added post first pass, weekly returns (time resampled), RSI (relative strength index):
# where rsi > 70, stock might be due for a drop, and rsi < 70, due for upswing
# MACD (moving avg convergence divergence): diff btw 2 avgs, indicates momentum,
# if macd crosses above signal line, trending upwards; crosses below, trending down
df2.set_index('date', inplace = True)
df2['weeklyReturns'] = df2['close'].resample('W').last().pct_change().reindex(df2.index, method = 'ffill')
df2['rsi'] = ta.momentum.RSIIndicator(df2['close'], window = 14).rsi()

macd = ta.trend.MACD(df2['close'])
df2['macd'] = macd.macd()
df2['macd_signal'] = macd.macd_signal()
df2['macd_diff'] = macd.macd_diff()


# inspect and remove null values
print(df2.isnull().sum())
df2.dropna(inplace = True)

# split btw test and train
trainSize = int(len(df2) * 0.75)
train = df2.iloc[:trainSize]
test = df2.iloc[trainSize:]

features = ['weeklyReturns', 'prevDayReturn', 'dailyRange', 'mvAvg7days',
                'mvAvg14days','mvAvg30days','volChange', 'returnLag1','returnLag2',
                'returnLag3', 'rsi', 'macd', 'macd_signal','macd_diff']

xtrain = train[features]
print(xtrain.head())
ytrain = train['target']
print(ytrain.head())

xtest = test[features]
ytest = test['target']

# create and apply model
model = lr()
model.fit(xtrain, ytrain)

# make prediction
yPred = model.predict(xtest)

# check accuracy
print(f"linear regression accuracy score = {accuracy_score(ytest, yPred)}")
print(confusion_matrix(ytest, yPred))
# initial run of model gave accuracy score of 0.516 and conf mat of [[1 572], [0 608]]
# conf mat essentially means that the model is always predicting the stock goes up (hence the roughly 50% accuracy)
# add more/refine features, try random forest

# random forest
# on second pass of rf, using shallow trees, bc they perform better for noisy data like stocks (capturing broad, robust patterns)
rf = rfc(n_estimators = 200, max_depth = 5, min_samples_split = 50, random_state = 42)
rf.fit(xtrain, ytrain)
# check for which features are actually being used by the model
importances = pd.Series(rf.feature_importances_, index = xtrain.columns)
print(importances.sort_values(ascending = False).head(10))
rfPred = rf.predict(xtest)
# check accuracy
print(f"random forest and improved features accuracy score = {accuracy_score(ytest, rfPred)}")
print(confusion_matrix(ytest, rfPred))


# price range/volatility prediction; will use regression for first pass, and measure accuracy by 
# MSE/RSE (lower is better)
df3 = (df2.assign(
    ATR14 = np.maximum(
        df2['high'] - df2['low'], 
        np.maximum(abs(df2['high'] - df2['close'].shift(1)), 
                   abs(df2['low'] - df2['close'].shift(1))))
        .rolling(14).mean()
)) # ATR14 is rolling mean of true range

print(df3.isnull().sum())

# drop nulls
df3.dropna(inplace = True)

target = 'ATR14'

train = df3.iloc[:trainSize]
test = df3.iloc[trainSize:]

xtrain = train[features]
ytrain = train[target]

print(xtrain)
print(ytrain)

xtest = test[features]
ytest = test[target]

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)

# fit model
linReg = linreg()
linReg.fit(xtrain, ytrain)

# make pred
yPred_lin = linReg.predict(xtest)

# evaluate performance of lin reg model
mse = mean_squared_error(ytest, yPred_lin)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, yPred_lin)

print(f"Linear Reg results: mean squared error = {mse}; root mse = {rmse}; r2 = {r2}")
# on first pass, mse = 2.77, rmse = 1.66 and r2 = -36.1 (poor performance, however, given
# how noisy stock volatility is, poor performance is not surprising)

# now we try gradient boosting regression for the price volatility prediction
gbr = gradboostreg(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 4,
    random_state = 42
)

gbr.fit(xtrain, ytrain)
ypredgbr = gbr.predict(xtest)

mse_gbr = mean_squared_error(ytest, ypredgbr)
rmse_gbr = np.sqrt(mse_gbr)
r2_gbr = r2_score(ytest, ypredgbr)

print(f"grad boost mse = {mse_gbr}; grad boost rmse = {rmse_gbr}; grad boost r2 = {r2_gbr}")
# with gradient boosting, mse = 0.11, rmse = 0.33, and r2 = -0.48. Much better than plain 
# linear regression, but still doing worse than predicting mean of target vals

# risk metric with Sharpe ratio: measure of how much excess return per unit risk
riskFreeRate = 0.0

meanReturn = df3['dailyReturn'].mean()
std_return = df3['dailyReturn'].std()

sharpe = (meanReturn - riskFreeRate) / std_return
print(f"Sharpe Ratio (risk metric) = {sharpe}")

# annualized sharpe
sharpe_ann = (meanReturn * 252) / (std_return * np.sqrt(252))
print(f"annualized sharpe ratio = {sharpe_ann}")

# checking stability over time of sharpe ratio
window = 90 # days
avg90days = df3['dailyReturn'].rolling(window).mean()
std90days = df3['dailyReturn'].rolling(window).std()

# rolling sharpe not annualized
rollingsharpe = (avg90days - riskFreeRate) / std90days
# rolling sharpe annualized
rollSharpeAnn = (avg90days * 252) / (std90days * np.sqrt(252))

plt.figure(figsize = (9, 6))
plt.plot(df3.index, rollSharpeAnn, label = "rolling annualized sharpe ratio (90 day window)")
plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
plt.title("Rolling Annualized sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.tight_layout()
plt.show()

# at first pass, significant drops noticed in early 2014 and late 2017, with a spike in early 2017;
# still very noisy, so trying longer window of 180 days
# checking stability over time of sharpe ratio
window = 180 # days
avgrolling = df3['dailyReturn'].rolling(window).mean()
stdrolling = df3['dailyReturn'].rolling(window).std()

# rolling sharpe not annualized
rollingsharpe2 = (avgrolling - riskFreeRate) / stdrolling
# rolling sharpe annualized
rollSharpeAnn2 = (avgrolling * 252) / (stdrolling * np.sqrt(252))

plt.figure(figsize = (9, 6))
plt.plot(df3.index, rollSharpeAnn2, label = "rolling annualized sharpe ratio (180 day window)")
plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
plt.axhline(1, color = 'red', linestyle = '--', linewidth = 1)
plt.axhline(2, color = 'red', linestyle = '--', linewidth = 1)
plt.title("Rolling Annualized sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.tight_layout()
plt.show()
# ^ 180 window shows big spikes in mid 2013 and early-middle 2017; mostly the rest of the time it hovers around 0


# clustering similar stocks

# read in more stocks, collect in a df
files = glob.glob("sp500_5yr_data/*.csv") # changed from 150 of the 500 stocks to all 500 on second pass

collected_list = []

for f in files:
    ticker = f.replace(".csv", "").replace("sp500_5yr_data\\", "")
    dfTemp = pd.read_csv(f, parse_dates = ["date"])
    dfTemp.set_index("date", inplace = True)

    # resample
    monthlyClose = dfTemp["close"].resample("M").last()
    # monthly returns: less noisy than daily returns
    monthlyReturns = monthlyClose.pct_change()
    # store as df
    collected_list.append(monthlyReturns.rename(ticker))

# merge into one df
returns = pd.concat(collected_list, axis = 1)
# handling missing data
returns = returns.dropna(axis = 0, how = 'any')

print(f"shape of monthly returns df = {returns.shape}")
print(returns.head())

# correlation matrices are most often used for stock data
corrmat = returns.corr()

# heat map for above matrix
plt.figure(figsize = (9, 6))
sns.heatmap(corrmat, cmap = 'coolwarm', center = 0)
plt.title('Stock Cluster Correlation Matrix')
plt.tight_layout()
plt.show()

# kmeans clustering
# turning correlation into distance so highly correlated stocks are "close"
distance = 1 - corrmat

# input distance into kmeans model
kmeans = KMeans(n_clusters = 5, random_state = 37)
labels = kmeans.fit_predict(distance)

# add cluster labels to tickers
clusters = pd.DataFrame({
    "Ticker": corrmat.columns,
    "Cluster": labels
})

print(clusters.sort_values("Cluster").head(20))

# plot with pca viz
returns_filled = returns.fillna(0).T
# PCA to 2 components (instead of component per month)
pca = PCA(n_components = 2)
components = pca.fit_transform(returns_filled)
# put into df with cluster labels
pca_df = pd.DataFrame(components, columns = ['PC1', 'PC2'])
pca_df['Ticker'] = returns.columns
pca_df['Cluster'] = labels

# scatter plot for clusters
plt.figure(figsize = (9, 6))
for cluster in sorted(pca_df['Cluster'].unique()):
    subset = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(subset['PC1'], subset['PC2'], label = f"Cluster {cluster}", alpha = 0.7)

plt.title("KMeans Clusters of Stocks (PCA Projection)")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.legend()
plt.tight_layout()
plt.show()

# at first pass, most stocks in cluster 2, which centers mostly on the origin of the pca projection
# not surprising, bc most stocks have returns that behave similarly. Now we try with all 500 stocks

# 2 clusters when run with all 500 stocks, and still most are centered around the origin