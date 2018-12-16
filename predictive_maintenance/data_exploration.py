"""
Multi-variate timestamp Data from fleet of similar engines.

Each engine starts with different degrees of initial wear and manufacturing
variation which is unknown to the user.
This wear and variation is considered normal, i.e., it is not considered a fault condition.

There are three operational settings that have a substantial effect on engine performance.
The data are contaminated with sensor noise.

The engine is operating normally at the start of each time series, and starts to degrade at some point
during the series.

In the training set,
    the degradation grows in magnitude until a predefined threshold is reached beyond which it is not
    preferable to operate the engine.
In the test set,
    the time series ends some time prior to complete degradation.

The objective of the competition is
    to predict the number of remaining operational cycles before in the test set,
    i.e., the number of operational cycles after the last cycle that the engine will
    continue to operate properly.
"""
# %%
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %%
PROJECT_DIR = os.path.abspath(os.curdir)
BASE_DIR = os.path.join(PROJECT_DIR, "predictive_maintenance")

train_pd = pd.read_csv(os.path.join(BASE_DIR, "train.txt"),
                       sep=' ', header=None)

# %%

train_data = train_pd[list(range(26))].copy()
col_list = ['unit_number', 'time_cycles']
col_dict = {"operational_setting": 3, "sensor_measurement": 21}

for key in col_dict:
    for i in range(1, col_dict[key] + 1):
        col_list.append('{}_{}'.format(key, i))

train_data.columns = col_list
del train_pd

# %%
print("Number of Machines - ", len(train_data.unit_number.unique()))

# %%
time_cycles_available = train_data.groupby(["unit_number"])["time_cycles"].size().sort_values(
    ascending=False)

# %%
machine_number = 76
train_data[train_data.unit_number == machine_number].plot("time_cycles", "sensor_measurement_15")
plt.show()

# %%
# Marking the final value as the one where the damage has occurred
train_data.loc[:, "is_faulty"] = np.zeros(len(train_data)).astype('int')

# %%
max_tc_dict = train_data.groupby("unit_number").agg({"time_cycles": np.max}).to_dict()["time_cycles"]
for machine in max_tc_dict:
    time_cycle = max_tc_dict.get(machine)
    train_data.loc[(train_data["unit_number"] == machine) &
                   (train_data["time_cycles"] == time_cycle), "is_faulty"] = 1

#%%
train_data.isnull().sum()
