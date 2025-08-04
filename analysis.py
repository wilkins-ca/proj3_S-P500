import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from warnings import filterwarnings

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