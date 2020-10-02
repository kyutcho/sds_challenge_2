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
win_path = "D:/Learning/Projects/sds_challenges/challenge_1/data/public_flights.csv"
mac_path = "~/Projects/sds_challenge/challenge_1/data/public_flights.csv"
flight = pd.read_csv(mac_path)
