# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:08:25 2020

@author: david
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew

plt.style.use("ggplot")

# file_path = "\challenge_1"
win_path = "D:/Learning/Projects/sds_challenges_2/challenge_2/data/public_cars.csv"
mac_path = "~/Projects/sds_challenge_2/challenge_1/data/public_flights.csv"

cars = pd.read_csv(win_path)

# columns
cars.columns

# high-level info
cars.info()

cars.head()

# shape
cars.shape

# column with missing values
cars.isnull().sum()[cars.isnull().sum() != 0]

# price var
cars["price_usd"].describe()
sns.distplot(cars["price_usd"])
print(f'skewnewss: {cars["price_usd"].skew()}')
print(f'kurtosis: {cars["price_usd"].kurt()}')

# log(price)
cars["log_price_usd"] = np.log(cars["price_usd"])
sns.distplot(cars["log_price_usd"])
print(f'skewnewss: {cars["log_price_usd"].skew()}')
print(f'kurtosis: {cars["log_price_usd"].kurt()}')

def calc_IQR(col):
    return col.quantile(0.75) - col.quantile(0.25)

def num_outlier(col, df = "cars"):
    IQR = calc_IQR(col)
    
    return df[col >= (col.quantile(0.75) + 1.5*IQR) |\
              col <= (col.quantile(0.25) - 1.5*IQR)]

# categorical variable univariate analysis function
def cat_analysis(col, sortedBy = "count"):
    col_count = cars[col].value_counts()
    col_desc = cars.groupby(col)["price_usd"]\
                   .agg(["mean", "median", "std", calc_IQR])
                                          
    if sortedBy == "count":
        idx = col_count.index
    else:
        idx = col_desc.sort_values(by = sortedBy, ascending = False).index
    
    col_count = col_count.reindex(idx)
    col_desc = col_desc.reindex(idx)
    
    col_combined = pd.concat([col_count, col_desc], axis = 1)
    
    plt.figure(figsize = (12, 6))
    
    if len(col_combined) > 20:
        sns.boxplot(data = cars[cars[col].isin(list(idx[:20]))], \
            x = col, \
            y = "price_usd",
            order = idx[:20])
        plt.xticks(rotation = 90)
        print(col_combined.head(20))
    else:
        sns.boxplot(data = cars, \
            x = col, \
            y = "price_usd",
            order = idx)
        print(col_combined)
        

# number of unique values for categorical vars        
cars.select_dtypes("object").apply(pd.Series.nunique, axis = 0)

# manufacturer_name
cat_analysis("manufacturer_name", "median")

# model_name
cat_analysis("model_name")

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
cat_analysis("body_type", "median")

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

# odometer_value
plt.figure(figsize = (10, 5))
sns.distplot(cars["odometer_value"], color = "green")

sns.relplot(data = cars, x = "odometer_value", 
            y = "price_usd", size = 10, legend = False)

# year_produced
cars["year_produced"].value_counts()
round(cars["year_produced"].value_counts(normalize = True) * 100, 2)

cars["year_produced"].plot(kind = "hist", x = "year_produced", bins = 20)

# duration_listed
sns.distplot(cars["duration_listed"])

ax = sns.regplot(x = "duration_listed", y = "price_usd", data = cars)
ax.set_yscale("log")
ax.set_xscale("log")

# engine_capacity
cars["engine_capacity"].value_counts()
cars["engine_capacity"].hist(bins = 30)

# correlation
num_vars = ["price_usd", "odometer_value", "year_produced", "engine_capacity", "duration_listed"]
corr = cars[num_vars].corr()

plt.figure(figsize = (10, 5))
sns.heatmap(corr, annot=True, cmap = "YlOrRd")
