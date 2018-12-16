# %%
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
gc.enable()
# %%
BASE_DIR = os.path.abspath(os.path.curdir)
TRAIN = os.path.join(BASE_DIR, "GP_hack", "train")


# %%

def read_data():
    train = pd.read_csv(os.path.join(TRAIN, 'train.csv'),
                        sep=',')
    meal_info = pd.read_csv(os.path.join(TRAIN, 'meal_info.csv'),
                            sep=',')
    fc_info = pd.read_csv(os.path.join(TRAIN, 'fulfilment_center_info.csv'),
                          sep=',')
    test = pd.read_csv(os.path.join(BASE_DIR, "GP_hack", 'test.csv'))
    print("train records - ", train.shape)
    print("test records - ", test.shape)
    print("meal info records - ", meal_info.shape)
    print("fulfilment center records - ", fc_info.shape)
    return train, test, meal_info, fc_info


# %%
# Reading the dataset
train_pd, test_pd, meal_info_pd, fc_info_pd = read_data()


# %%

def create_additional_features(data):
    data.loc[:, "base_minus_checkout"] = data.base_price - data.checkout_price
    center_id_meal_list = np.unique(data["center_id"].astype('str') + "-" + data["meal_id"].astype('str'))
    print("number of unique center-meal pair - ", len(center_id_meal_list))
    # add_cols = ["is_discount", "mean_price_deviation", "std_price_deviation", "max_price_deviation"]
    for ctr_meal in tqdm(center_id_meal_list):
        cond1 = (data["center_id"] == int(ctr_meal.split("-")[0])) & \
                (data["meal_id"] == int(ctr_meal.split("-")[1]))
        t_stats1 = data.loc[cond1, "base_minus_checkout"].describe().to_dict()
        # t_stats1 = t1["base_minus_checkout"].describe().to_dict()
        data.loc[cond1, "is_discount"] = data.loc[:, "base_minus_checkout"] >= t_stats1["mean"]
        data.loc[cond1, "mean_price_deviation"] = [t_stats1["mean"]] * len(data[cond1])
        data.loc[cond1, "std_price_deviation"] = [t_stats1["std"]] * len(data[cond1])
        data.loc[cond1, "max_price_deviation"] = [t_stats1["max"]] * len(data[cond1])
        # data.loc[cond1, add_cols] = t1[add_cols]
        # del t1
        gc.collect()
    data.loc[:, "cuisine_category"] = data["cuisine"] + "-" + data["category"]
    del data["category"]
    del data["cuisine"]
    return data


def correct_dtypes(data, type_dict, is_ohe=False):
    for type_ in type_dict:
        print("Processing Type - ", type_)
        for col in type_dict[type_]:
            print("processing Column - ", col)
            if type_ == "bool":
                data.loc[:, col] = data[col].astype('bool')
            elif type_ == "cat":
                if is_ohe:
                    print("Creating One Hot Encodings...")
                    data = pd.concat([data, pd.get_dummies(data[col])], axis=1)
                    del data[col]
                else:
                    print("Converting to Categorical data types...")
                    data.loc[:, col] = pd.Categorical(data[col])
    return data


def data_transform(data_pd, is_test=False):
    if is_test:
        remove_cols = ["id", 'week', "base_price", "center_id", "meal_id"]
        predictor = None
    else:
        remove_cols = ["id", 'week', "num_orders", "base_price", "center_id",
                       "meal_id"]
        predictor = "num_orders"

    req_cols = []
    for col in data_pd.columns:
        if col not in remove_cols:
            req_cols.append(col)
    print("columns used for prediction - ", req_cols)
    dataset1 = data_pd[req_cols]

    if is_test:
        target1 = None
    else:
        target1 = data_pd[predictor].values
        print("Target shape - ", target1.shape)
    print("Input data shape - ", dataset1.shape)
    return dataset1, target1


def get_deviation_cat(price_dev, desc_dict):
    low1 = desc_dict['mean'] - desc_dict["std"]
    low2 = desc_dict['mean'] + desc_dict["std"]
    if price_dev <= low1:
        return 'min_change'
    elif low1 < price_dev <= low2:
        return "avg_change"
    else:
        return "max_change"


def get_window(window, stop):
    start = 1
    while start + window <= stop:
        yield (start, start + window)
        start = start + window


def get_average(data, window):
    windows_list = get_window(window, len(data))
    total_window_orders = {}
    window_week_map = {}
    for counter, range_ in enumerate(windows_list):
        start, stop = range_
        window_week_map[counter + 1] = list(range(start, stop))
        total_window_orders[counter + 1] = data[data["week"].isin(
            list(range(start, stop)))]["num_orders"].mean()
    return total_window_orders, window_week_map


# %%
joined_pd = pd.merge(train_pd, meal_info_pd, on="meal_id", how="inner")
joined_pd = pd.merge(joined_pd, fc_info_pd, on="center_id", how="inner")
joined_pd.sort_values(["week", "center_id", "meal_id"], ascending=True, inplace=True)
print("All the data merged shape - ", joined_pd.shape)

joined_test = pd.merge(test_pd, meal_info_pd, on="meal_id", how="inner")
joined_test = pd.merge(joined_test, fc_info_pd, on="center_id", how="inner")
joined_test.sort_values(["week", "center_id", "meal_id"], ascending=True, inplace=True)
print("All the TEST data merged shape - ", joined_test.shape)

dtypes_dict = {'bool': ["emailer_for_promotion", "homepage_featured"],
               'cat': ["cuisine_category", "center_type"]}

joined_pd = create_additional_features(joined_pd)
joined_test = create_additional_features(joined_test)

joined_pd = correct_dtypes(joined_pd, dtypes_dict, is_ohe=True)

joined_test = correct_dtypes(joined_test, dtypes_dict, is_ohe=True)

# %%
joined_pd, target = data_transform(joined_pd)

# %%
# Features from train dataset
train_sample = train_pd.copy()
train_sample.loc[:, "base_minus_checkout"] = np.abs(train_sample.base_price - train_sample.checkout_price)

# %%
print("number of unique center-meal pair")
center_meal_list = np.unique(train_sample["center_id"].astype('str') + "-" + train_sample["meal_id"].astype('str'))

# for i in center_meal_list:
cond = (train_pd["center_id"] == int(center_meal_list[504].split("-")[0])) & \
       (train_pd["meal_id"] == int(center_meal_list[504].split("-")[1]))

t = train_sample[cond].copy()

# %%
time_avg_dict = {'month': 4, 'quarter': 13, 'year': 52}

for col in time_avg_dict:
    time_avg, time_week_map = get_average(t, time_avg_dict[col])
    t.loc[:, col] = [""] * len(t)
    t.loc[:, '{}_avg'.format(col)] = [""] * len(t)
    for i in time_week_map:
        t.loc[t["week"].isin(time_week_map[i]), col] = i
        t.loc[t["week"].isin(time_week_map[i]),
              '{}_avg'.format(col)] = time_avg[i]

# %%
# t.loc[:, "base_minus_checkout_shifted"] = t["base_minus_checkout"].shift(1).fillna(0.0)
# t.loc[:, "dev_shift_diff"] = np.abs(t["base_minus_checkout"] - t["base_minus_checkout_shifted"])

t_stats = t["base_minus_checkout"].describe().to_dict()

get_cat = partial(get_deviation_cat, desc_dict=t_stats)
t.loc[:, "is_discount"] = t["base_minus_checkout"] >= t_stats["mean"]
t.loc[:, "mean_price_deviation"] = [t_stats["mean"]] * len(t)
t.loc[:, "std_price_deviation"] = [t_stats["std"]] * len(t)
# t.loc[:, "min_price_deviation"] = [t_stats["min"]] * len(t)
t.loc[:, "max_price_deviation"] = [t_stats["max"]] * len(t)

# %%


# %%
plt.plot(range(len(t)), t.base_minus_checkout)
plt.plot(range(len(t)), [2 * t_stats["std"]] * len(t))
plt.plot(range(len(t)), [t_stats["std"]] * len(t))
plt.show()
# %%
from sklearn.model_selection import ShuffleSplit

dataset, target = data_transform(joined_pd)
shuffle = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2019)
train_index, test_index = list(shuffle.split(dataset))[0]
print("Train - Test split | ", len(train_index), " - ", len(test_index))
X_train = dataset.iloc[train_index].values
X_val = dataset.iloc[test_index].values
y_train = target[train_index]
y_val = target[test_index]

print("TRAIN SHAPE || data - {} | target - {}".format(X_train.shape,
                                                      y_train.shape))
print("VAL SHAPE || data - {} | target - {}".format(X_val.shape,
                                                    y_val.shape))
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def rms_log_error(true, pred):
    return np.sqrt(mean_squared_error(true, pred))


def model_results(model):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    print("Train In-sample metric : ", rms_log_error(y_train,
                                                     y_train_pred))
    print("Validation metric : ", rms_log_error(y_val,
                                                y_val_pred))


# %%
# Train In-sample metric :  126.45464585357654
# Validation metric :  177.32391066863423

# Train In-sample metric :  141.14969283617793
# Validation metric :  174.42153355518724

# Train In-sample metric :  179.2193494423072
# Validation metric :  196.3168023972855

rf_config = {"n_estimators": 1000,
             "max_depth": 8,
             "max_features": 'sqrt',
             "random_state": 111,
             "n_jobs": 4,
             "verbose": 1}

rf = RandomForestRegressor(**rf_config)

rf.fit(X_train, y_train)

model_results(rf)

# %%

print("Number of centers - ", train_pd.center_id.nunique())
print("Number of centers-meal combinations - ",
      len(train_pd.groupby(['center_id', 'meal_id']).size()))

# %%
"""
Meal id plot, with and without rolling averages
"""
meal_id = 1247
# cond = (train_pd["center_id"] == center_id) & \
#        (train_pd["meal_id"] == meal_id)
cond = (train_pd["meal_id"] == meal_id)
cond_pd = train_pd[cond].reset_index()
for i in cond_pd.center_id.unique():
    print('for center - ', i)
    temp = cond_pd[cond_pd["center_id"] == i].copy()

    # without rolling averages
    # plt.plot(list(range(len(temp))),
    #          temp["num_orders"])

    plt.plot(list(range(len(temp))),
             temp["num_orders"].rolling(window=10).mean())

    del temp
# cond_pd["num_orders"].plot()
plt.show()

# %%
"""
Average order for the center - meal combination
"""
joined_pivot = joined_pd.pivot_table(values=["num_orders", "week"],
                                     index="center_id",
                                     columns="meal_id",
                                     aggfunc=np.mean
                                     )

# %%
"""
Observing the changes in orders, when the checkout price is reduced
"""
center_id = 10
meal_id = 1109
cond = (joined_pd["center_id"] == center_id) & \
       (joined_pd["meal_id"] == meal_id)
cond_pd = joined_pd[cond].reset_index()
plt.plot(list(range(len(cond_pd))),
         cond_pd["num_orders"])
plt.plot(list(range(len(cond_pd))),
         cond_pd["base_minus_checkout"])
plt.show()

# %%
# joined_pd.to_csv(os.path.join(TRAIN, "joined_data.csv"), index=False)

# %%

# %%
test_pd = pd.read_csv(os.path.join(BASE_DIR, "GP_hack", 'test.csv'))
test_merge = pd.merge(test_pd, meal_info_pd, on="meal_id", how="inner")
test_merge = pd.merge(test_pd, fc_info_pd, on="center_id", how="inner")

test_dataset, _ = data_transform(test_merge, is_test=True)

print("Test shape : ", test_dataset.shape)

# %%
importance_pd = pd.DataFrame(dict([("column", dataset.columns.tolist()),
                                   ("score", rf.feature_importances_)]))

importance_pd.sort_values("score", ascending=False, inplace=True)

# %%
corr_pd = dataset.select_dtypes(include=["int", "float"]).corr()
