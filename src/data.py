import os
import pandas as pd
import numpy as np
import util

INT_TYPES = [
    "trip_id",
    "arrival_time",
    "departure_time",
    "stop_time",
    "door",
    "gtfs_trip_id",
    "gtfs_stop_time_id",
    "stop_id",
    "trip_number",
    "route_number",
    "vehicle_number",
    "deviance",
]

CATEGORIES = [
    "trip_number",
    "gtfs_stop_time_id",
    "gtfs_stop_id",
    "gtfs_trip_id",
    "train",
    "trip_id",
    "data_agency",
    "data_source",
    "direction",
    "schedule_status",
    "service_key",
    "stop_id",
    "vehicle_number",
    "route_number",
    "trip_number",
    "time_cat",
    "door",
    #"stop_id_pairing",
]

USED_COLS = [
    "direction",
    "door",
    "lift",
    "dwell",
    "ons",
    "offs",
    "maximum_speed",
    "service_key",
    "schedule_status",
    "deviance",
    "secs_stopped",
    "time_cat",
    "stop_id",
    "delta_pattern_distance",
    "avg_speed",
    "label",
]

DROP_COLS = [
    "early",
    "on_time",
    "late",
    "location_distance",
    "capacity",
]

MINUTES_PER_TIME_CAT = 5

def get_transit_daily():
    data_dir = os.path.join(os.getcwd(), "..", "data")
    daily_file = os.path.join(data_dir, "transit_daily.csv")
    return pd.read_csv(daily_file, low_memory=False)

def get_mega_stop() :
    data_dir = os.path.join(os.getcwd(), "..", "data")
    daily_file = os.path.join(data_dir, "mega_stop_event.hdf")
    return pd.read_hdf(daily_file, start=0, stop=10000)

def get_data_subset(d, n):
    d = d.head(n)
    d = d.dropna(axis=0, subset=["service_date", "arrival_time", "deviance"])
    d = d.sort_values(["service_date", "arrival_time"])
    return d

def daily_sort(d):
    d["timestamp"] = np.vectorize(util.to_timestamp)(
        d["service_date"], d["arrival_time"]
    )
    d = d.sort_values(["trip_id", "timestamp"])
    d.deviance = d.deviance.apply(lambda x: pd.to_timedelta(x).total_seconds())
    return d

def daily_zerofill_int_fields(d):
    d[INT_TYPES] = d[INT_TYPES].fillna(0).astype("int32")
    return d

def daily_compute_times(d):
    d["secs_stopped"] = d.departure_time - d.arrival_time
    d["time_cat"] = (d.arrival_time % (60 * 60 * 24))  // 60 // MINUTES_PER_TIME_CAT
    return d

"""
sort by trip_id or vechicle number and then timestamp before taking these diffs. 
That's because if you sort just by time your mixing all the busses together.
"""
def daily_compute_deltas(d):
    dist = ["pattern_distance", "train_mileage"]
    delta_dist = ["delta_" + x for x in dist]
    d[delta_dist] = d[dist].diff(1)
    d[delta_dist] = np.where(d[delta_dist] < 0, 0, d[delta_dist])
    return d

def daily_compute_avg_speed(d):
    d["avg_speed"] = d.delta_pattern_distance / (
        d.arrival_time - d.departure_time.shift(1)
    )
    return d

def daily_remove_unused_columns(d):
    d = d.drop(axis=1, columns=DROP_COLS)
    return d

def data_transforms(d):
    d = get_data_subset(d, 100000)
    d = daily_sort(d)
    d = daily_zerofill_int_fields(d)
    d = daily_compute_times(d)
    util.make_categories(CATEGORIES, d)
    d = daily_compute_deltas(d)
    d = daily_compute_avg_speed(d)
    d = daily_remove_unused_columns(d)
    d['label'] = d.groupby('trip_number')['deviance'].shift(-1)
    #d['label'] = d.deviance.shift(-5) # making a prediction from 5 stops previously.
    # https://stackoverflow.com/a/45745154/6293070
    d = d[~d.isin([np.nan, np.inf, -np.inf]).any(1)]
    return d