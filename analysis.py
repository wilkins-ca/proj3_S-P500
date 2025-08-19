import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from warnings import filterwarnings
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score, confusion_matrix
import ta

filterwarnings('ignore')

# list of files we want to examine
pathstring = 'individual_stocks_5yr/'
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


# look at quarterly avg return by company
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
    volChange = (df1['volume'] - df1['volume'].shift(1)) / df1['volume'].shift(1) # stronger conviction on the trade
))

# looking at trends over multiple days
for i in range(1, 4):
    df2[f'returnLag{i}'] = df2['dailyReturn'].shift(i)

# inspect and remove null values
print(df2.isnull().sum())
df2.dropna(inplace = True)

# split btw test and train
trainSize = int(len(df2) * 0.75)
train = df2.iloc[:trainSize]
test = df2.iloc[trainSize:]

xtrain = train[['dailyReturn', 'prevDayReturn', 'dailyRange', 'mvAvg7days',
                'mvAvg14days','volChange', 'returnLag1','returnLag2','returnLag3']]
ytrain = train['target']

xtest = test[['dailyReturn', 'prevDayReturn', 'dailyRange', 'mvAvg7days',
                'mvAvg14days','volChange', 'returnLag1','returnLag2','returnLag3']]
ytest = test['target']

# create and apply model
model = lr()
model.fit(xtrain, ytrain)

# make prediction
yPred = model.predict(xtest)

# check accuracy
print(accuracy_score(ytest, yPred))
print(confusion_matrix(ytest, yPred))
# initial run of model gave accuracy score of 0.516 and conf mat of [[1 572], [0 608]]
# conf mat essentially means that the model is always predicting the stock goes up (hence the roughly 50% accuracy)
# add more/refine features, try random forest
