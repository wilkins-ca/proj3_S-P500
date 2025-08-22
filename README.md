# S&P 500 Data Analysis and Machine Learning

## Introduction

*This project focuses mainly on trends in stock data from 2013 to 2018 for Apple, Microsoft, Amazon, and Google. It began as a guided project from a Udemy course, but I extended it significantly with machine learning. This included feature engineering, model experimentation, and evaluation of risk metrics like the Sharpe ratio.*

*The goal was to apply predictive models to real-world-emulating data and investigage how different technical indicators can inform data such as price movement classification and volatility prediction.*

*In addition to experimenting with models, this proect emphasized an understanding of risk-adjusted performance: how much return an investment may generate relative to the risk taken. Through this work, I strengthened my skills in data preprocessing, feature engineering, evaluation of model performance, and interpreting financial metrics in a practical, applied context.*

## Data Collection and Preprocessing

* dataset came from Udemy course resources, each row representing a day
* checked for where the null values where, how many, and whether they were proportionally significant to the size of the dataset
* resampled time index in order to look at quarterly average return
* calculated data includes: daily return, several rolling averages, correlation matrix for closing prices of above mentioned 4 companies, correlation matrix for daily change in stock price vs. daily returns

## Exploratory Data Analysis (EDA)

* checked for column names and data types
* reindexed and reset several times using date column converted to datetime object
* heat maps for correlation matrices, one pairgrid for stock price change vs. stock return, line charts for time-series data

## Statistical Trends

* *Findings*
  * Google and Apple stock prices are not well correlated, whereas Amazon and Google stock prices are strongly correlated at 0.98
  * all three moving averages calculated for Google and Microsoft (windows of 7 days, 14 days, and 21 days) dropped very sharply in 2013, whereas Apple and Amazon trended upwards
  * sharp increase in quarterly returns for Google, mid 2015

## Price Movement Classification

* target variable defined to be 1 if stock price increases, 0 if stock price remains the same or decreases
* features put into first pass: daily return, previous day return, daily price range, moving average over 7 days, moving average over 14 days, change in volume traded, and daily return lagged by 1, 2, and 3 days
  * first pass was execusted with linear regression model
* on second pass, weekly returns added as feature, as well as RSI (relative strength index) over a window of 14 days and Moving Average Convergence Divergence
  * second pass was executed with random forest model
  * random forest did not perform better than linear regression

## Price Range/Volatility Prediction

* first pass executed with linear regression
* ATR14 as input feature (rolling average of "true range", which is a measure of a stock's daily price volatility: max(high - low, |high - previousClose|, |low - previousClose|))
* evaluated performance using mean squared error, root mean squared error, and coefficient of determination r^2
* on first pass, MSE = 2.77, RMSE = 1.66, and r^2 = -36.1; this is poor performance, but to be expected, considering how noisy stock data is
* second pass executed with gradient boosting regression
* on second pass, MSE = 0.11, RMSE = 0.33, and r^2 = -0.48; better performance, but still not very good

## Risk Metrics: Sharpe Ratio

* annualized (not rolling) = 0.03
* not annualized (not rolling) = 0.47
* when charting stability over time of the sharpe ratio on first pass (window of 90 days), drops noticed in early 2014 and late 2017, and a spike in early 2017
* on second pass of sharpe stability charting (180 day window), spikes noticed in mid 2013 and early-middle 2017, and otherwise hovered around 0

## Clustering Similar Stocks

* clustered by monthly returns
* all 500 of S&P 500 stocks used
* correlation matrix used in calculating distance values, such that highly correlated stocks would be "close" to each other
* KMeans used for clustering
* PCA projection used for generation of a scatter plot, with components = 2
* most of the stocks are clustered around the origin, because most stocks' monthly return behavior is similar

## Challenges and Lessons Learned

* needed to forward fill when calculating weekly returns
* made sure to check for nulls and remove where necessary and reasonable
* predicting stock behavior is known to be a difficult thing, but it seemed that the Gradient Boosted Regression performed the best
* feature engineering making use of feature importances
* learned concept of confusion matrix
