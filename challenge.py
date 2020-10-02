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
cars["log_price_usd"] = np.log10(cars["price_usd"])
sns.distplot(cars["log_price_usd"])
print(f'skewnewss: {cars["log_price_usd"].skew()}')
print(f'kurtosis: {cars["log_price_usd"].kurt()}')

# categorical variable univariate analysis function
def cat_analysis(col):
    val_count = cars[col].value_counts()
    val_count = val_count.rename("count")
    val_pct = round(cars[col].value_counts(normalize = True), 2)
    val_pct = val_pct.rename("pct")
    val_combined = pd.concat([val_count, val_pct], axis = 1)
    
    if len(val_combined) > 20:
        print(val_combined.head(20))
    else:
        print(val_combined)

def box_plot_viz(col):
    idx = cars.groupby(col)\
              .agg({"price_usd":"median"})\
              .sort_values("price_usd", ascending = False).index
    plt.figure(figsize = (12, 6))
    sns.boxplot(data = cars, \
            x = col, \
            y = "price_usd",
            order = idx)
    if len(cars[col].unique()) > 20:
        plt.xticks(rotation = 90)

# number of unique values for categorical vars        
cars.select_dtypes("object").apply(pd.Series.nunique, axis = 0)

# manufacturer_name
cat_analysis("manufacturer_name")

box_plot_viz("manufacturer_name")


# model_name
cat_analysis("model_name")


# transmission
cat_analysis("transmission")

box_plot_viz("transmission")


# color
cat_analysis("color")

box_plot_viz("color")


# engine_fuel
cat_analysis("engine_fuel")

box_plot_viz("engine_fuel")


# engine_has_gas
cat_analysis("engine_has_gas")

box_plot_viz("engine_has_gas")


# engine_type
cat_analysis("engine_type")

box_plot_viz("engine_type")

# body_type
cat_analysis("body_type")

box_plot_viz("body_type")


# has_warranty
cat_analysis("has_warranty")

box_plot_viz("has_warranty")


# state
cat_analysis("state")

box_plot_viz("state")

# drivetrain
cat_analysis("drivetrain")

box_plot_viz("drivetrain")


# features
for i in range(10):
    var = "feature_" + str(i)
    cat_analysis(var)
