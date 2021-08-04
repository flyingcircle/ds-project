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

DIR_0_STOP_IDS = [
    9301,
    7636,
    7640,
    7594,
    7602,
    3397,
    13732,
    13772,
    13778,
    13791,
    13792,
    4622,
    4625,
    4627,
    12951,
    4644,
    4647,
    4648,
    4651,
    4656,
    4662,
    4666,
    4667,
    11920,
    4675,
    11921,
    4679,
    4680,
    4682,
    4686,
    11922,
    4691,
    4693,
    4699,
    4704,
    4707,
    4542,
    4562,
    4564,
    4504,
    4567,
    4568,
    4571,
    4573,
    11742,
    4580,
    4582,
    4584,
    4585,
    4588,
    4592,
    4593,
    4596,
    4598,
    4599,
    4600,
    4602,
    4606,
    4608,
    4610,
    4612,
    4534,
    4615,
    8812,
    4618,
    4522,
    4620,
    4507,
    8742,
    4515,
    4513,
    4552,
    4555,
    12866,
    12867,
    4557,
    3715,
    3725,
    3683,
    10858,
]
DIR_1_STOP_IDS = [
    10858,
    7605,
    13033,
    12862,
    9347,
    4558,
    12868,
    12863,
    4556,
    4553,
    12864,
    4516,
    8743,
    4508,
    4621,
    4523,
    4617,
    4554,
    4616,
    4614,
    4613,
    4611,
    4609,
    4607,
    4603,
    11786,
    4506,
    4597,
    4595,
    4591,
    4587,
    4586,
    4583,
    4581,
    4578,
    4576,
    4575,
    4572,
    4570,
    4569,
    4566,
    4503,
    4565,
    4563,
    4561,
    4543,
    4709,
    4706,
    4700,
    4698,
    4695,
    4692,
    11923,
    4684,
    4683,
    4681,
    11924,
    4676,
    4674,
    11925,
    4668,
    4664,
    4663,
    4657,
    4654,
    4653,
    4649,
    4646,
    4643,
    4631,
    4628,
    4626,
    4623,
    4538,
    13825,
    13773,
    13733,
    3398,
    13780,
    7773,
    7756,
    7767,
    7747,
    7751,
    9300,
]

MINUTES_PER_TIME_CAT = 5

def get_transit_daily():
    data_dir = os.path.join(os.getcwd(), "..", "data")
    daily_file = os.path.join(data_dir, "transit_daily.csv")
    return pd.read_csv(daily_file, low_memory=False)

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
    d = get_data_subset(d, 10000)
    d = daily_sort(d)
    d = daily_zerofill_int_fields(d)
    d = daily_compute_times(d)
    util.make_categories(CATEGORIES, d)
    d = daily_compute_deltas(d)
    d = daily_compute_avg_speed(d)
    d = daily_remove_unused_columns(d)
    d['label'] = d.groupby('trip_id')['deviance'].shift(-1)
#     d['label'] = d.deviance.shift(-5) # making a prediction from 5 stops previously.
    # https://stackoverflow.com/a/45745154/6293070
    d = d[~d.isin([np.nan, np.inf, -np.inf]).any(1)]
    return d