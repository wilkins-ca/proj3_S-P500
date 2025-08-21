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

## Price Range/Volatility Prediction

## Risk Metrics: Sharpe Ratio

## Challenges and Lessons Learned
