import os
import pandas as pd


def get_transit_daily() :
    data_dir = os.path.join(os.getcwd(), "..", "data")
    daily_file = os.path.join(data_dir, "transit_daily.csv")
    return pd.read_csv(daily_file, low_memory=False)