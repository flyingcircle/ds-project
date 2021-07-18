import time
import datetime
import pandas as pd
from matplotlib import pyplot as plt

def to_timestamp(date_str, sec_num):
    gmtime = time.gmtime(sec_num)
    date_list = date_str.split("/")
    timestamp = pd.Timestamp(
        year=2000 + int(date_list[2]),
        month=int(date_list[0]),
        day=int(date_list[1]),
        hour=gmtime.tm_hour,
        minute=gmtime.tm_min,
        second=gmtime.tm_sec,
    )
    # 60 * 60 * 24 = 86400 (some of the days have more than 24 hours of arrival times somehow...)
    if sec_num > 86400:
        timestamp += datetime.timedelta(days=1)
    return timestamp

# should work from what I understand, but doesn't...
# daily[categories] = daily[categories].astype("category")
def make_categories(make_category, df):
    for category in make_category:
        df[category] = pd.Categorical(df[category])

# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
def flatten(t):
    return [item for sublist in t for item in sublist]

def plot(r, l, preds, labels):
    assert len(preds) == len(labels)
    preds = preds[r:l]
    labels = labels[r:l]
    x = list(range(len(preds)))
    plt.scatter(x, labels, label="labels")
    plt.scatter(x, preds, label="predictions")
    plt.legend()
    plt.show()