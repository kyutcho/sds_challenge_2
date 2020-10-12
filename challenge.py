# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:08:25 2020

@author: david
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from scipy.stats import norm, skew

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

plt.style.use("ggplot")

# file_path = "\challenge_1"
win_path = "D:/Learning/Projects/sds_challenges_2/challenge_2/data/public_cars.csv"
mac_path = "~/Projects/sds_challenge_2/challenge_2/data/public_cars.csv"
data_path = "challenge_2/data/public_cars.csv"

cars = pd.read_csv(data_path)

# columns
cars.columns

# high-level info
cars.info()

cars.head()

# shape
cars.shape

# column with missing values
cars.isnull().sum()[cars.isnull().sum() != 0]

def print_skew_kurt(col):
    print(f'skewnewss: {col.skew()}')
    print(f'kurtosis: {col.kurt()}')

# price var
cars["price_usd"].describe()
sns.distplot(cars["price_usd"])
print_skew_kurt(cars["price_usd"])

# log(price)
sns.distplot(np.log(cars["price_usd"]))
print_skew_kurt(np.log(cars["price_usd"]))             

# sqrt(price)
sns.distplot(np.sqrt(cars["price_usd"]))
print_skew_kurt(np.sqrt(cars["price_usd"]))

def calc_IQR(col):
    return col.quantile(0.75) - col.quantile(0.25)

def num_outlier(col):
    IQR = calc_IQR(cars[col])
    
    return cars[cars[col] >= (cars[col].quantile(0.75) + 1.5*IQR)].sum()+\
           cars[cars[col] <= (cars[col].quantile(0.25) - 1.5*IQR)].sum()

# categorical variable univariate analysis function
def cat_analysis(col_x, col_y = "price_usd", sorted_by = "count"):
    col_count = cars[col_x].value_counts()
    col_desc = cars.groupby(col_x)[col_y].agg(["mean", "median", "std", calc_IQR])
    
    if sorted_by == "count":
        idx = col_count.index
    else:
        idx = col_desc.sort_values(by = sorted_by, ascending = False).index
    
    col_count = col_count.reindex(idx)
    col_desc = col_desc.reindex(idx)
    
    col_combined = pd.concat([col_count, col_desc], axis = 1)
    col_combined.rename(columns = {col_x: "total"}, inplace = True)
    
    plt.figure(figsize = (12, 6))
    
    if len(col_combined) > 20:
        sns.boxplot(data = cars[cars[col_x].isin(list(idx[:20]))], \
            x = col_x, \
            y = col_y,
            order = idx[:20])
        plt.xticks(rotation = 90)
        print(col_combined.head(20))
    else:
        sns.boxplot(data = cars, \
            x = col_x, \
            y = col_y,
            order = idx)
        print(col_combined)

# number of unique values for categorical vars        
cars.select_dtypes("object").apply(pd.Series.nunique, axis = 0)

# manufacturer_name
cat_analysis("manufacturer_name")
cars["manufacturer_name"].value_counts().sort_values().head(10)

# model_name
cat_analysis("model_name")
(cars["model_name"].value_counts() < 5).mean()

# transmission
cat_analysis("transmission")

# color
cat_analysis("color")

# engine_fuel
cat_analysis("engine_fuel")

# engine_has_gas
cat_analysis("engine_has_gas")

# engine_type
cat_analysis("engine_type")

# body_type
cat_analysis("body_type")

# has_warranty
cat_analysis("has_warranty", "median")

# state
cat_analysis("state")

# drivetrain
cat_analysis("drivetrain")

# features
for i in range(10):
    var = "feature_" + str(i)
    cat_analysis(var)
    
# pivot table
pd.pivot_table(cars, values = "price_usd", 
               index = "manufacturer_name", columns = "body_type",
               aggfunc="mean", fill_value=0)

# engine_fuel vs engine_type
pd.crosstab(index = cars["engine_fuel"], columns = cars["engine_type"])

# gas type replaced by gasoline
cars["engine_fuel"] = np.where(cars["engine_fuel"] == "gas", "gasoline", cars["engine_fuel"])

# Confirm changes gas -> gasoline
pd.crosstab(index = cars["engine_fuel"], columns = cars["engine_type"])

# Drop engine_type
cars.drop("engine_type", axis = 1, inplace=True)

# odometer_value
plt.figure(figsize = (10, 5))
sns.distplot(cars["odometer_value"], color = "green")

sns.relplot(data = cars, x = "odometer_value", 
            y = "price_usd", size = 10, legend = False)

# year_produced
cars["year_produced"].value_counts()
round(cars["year_produced"].value_counts(normalize = True) * 100, 2)
sns.distplot(cars["year_produced"])

cars["year_produced"].plot(kind = "hist", x = "year_produced", bins = 20)

# duration_listed
sns.distplot(cars["duration_listed"])
print_skew_kurt(cars["duration_listed"])

# log(duration_listed)
# sns.distplot(np.log(cars["duration_listed"]))
# print_skew_kurt(np.log(cars["duration_listed"]))

# for lam in np.arange(-2, 2.5, 0.5):
#     bc_trans = boxcox1p(cars["duration_listed"], lam)
#     print("skewness:{} when lambda {}".format(bc_trans.skew(), lam))
    
# ax = sns.regplot(x = "duration_listed", y = "price_usd", data = cars)
# ax.set_yscale("log")
# ax.set_xscale("log")

# engine_capacity
cars["engine_capacity"].value_counts()
cars["engine_capacity"].hist(bins = 30)

# sns.distplot(np.log(cars["engine_capacity"]))
# sns.distplot(boxcox1p(cars["engine_capacity"], 0))

# print_skew_kurt(np.log(cars["engine_capacity"]))
# print_skew_kurt(boxcox1p(cars["engine_capacity"], 0))

# correlation
num_vars = ["price_usd", "odometer_value", "year_produced", "engine_capacity", "duration_listed"]
corr = cars[num_vars].corr()

plt.figure(figsize = (10, 5))
sns.heatmap(corr, annot=True, cmap = "YlOrRd")

# Pairplot
sns.pairplot(cars[num_vars])

# Handling missing data
cars["engine_capacity"].describe()
cars.loc[cars["engine_capacity"].isnull(), ["engine_has_gas", "engine_fuel", "engine_capacity"]]

# All missing values are electric cars which does not have engine capacity
cars["engine_capacity"].fillna(0, inplace = True)

# Generate label encoded columns for columns with 2 unique values
lbl_enc = LabelEncoder()

for col in cars.columns.to_list():
    if (cars[col].dtype == "object" or cars[col].dtype == "bool"):
        if len(list(cars[col].unique())) == 2:
            cars[col] = lbl_enc.fit_transform(cars[col])
            print("Column {} is label encoded!".format(col))
            
# Drop model names column
cars.drop("model_name", axis = 1, inplace = True)

# Make dummy variable
cars = pd.get_dummies(cars, drop_first = True)

# Make X, y
X = cars.drop("price_usd", axis = 1)
y = cars["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Linear Regression
linearM = LinearRegression()

# Fit linear regression model
linearM.fit(X_train, y_train)

# Prediction for test
linearM.score(X_test, y_test)

# Predicted price for price_usd using linear model
y_pred = linearM.predict(X_test)

# visualization: actual vs pred
df = pd.DataFrame({"actual":y_test, 
                   "pred":y_pred})

plt.figure(figsize = (10, 5))
sns.regplot(data = df, x = "actual", y = "pred")

plt.title("Car price in USD actual vs predicted")
plt.gca().spines["top"].set_color(None)
plt.gca().spines["right"].set_color(None)
plt.legend(loc = "best", fontsize = 10)

