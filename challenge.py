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

cars.head()

# shape
cars.shape

# column with missing values
cars.isnull().sum()[cars.isnull().sum() != 0]

# categorical variable univariate analysis function
def cat_analysis(col):
    val_count = cars[col].value_counts()
    val_pct = round(cars[col].value_counts(normalize = True), 2)
    val_combined = pd.concat([val_count, val_pct], axis = 1)
    
    if len(val_combined) > 20:
        print(val_combined.head(20))
    else:
        print(val_combined)

# manufacturer_name
cat_analysis("manufacturer_name")

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

# drivetrain
cat_analysis("drivetrain")

# features
for i in range(10):
    var = "feature_" + str(i)
    cat_analysis(var)
