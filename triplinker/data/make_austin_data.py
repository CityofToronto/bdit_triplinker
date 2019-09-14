"""Script to generate `austin_data.csv`.

This script transforms open data from the Ride Austin ridehailing non-profit
into a format usable by `triplinker`. The data can be found in two parts at

https://data.world/ride-austin/ride-austin-june-6-april-13
https://data.world/ride-austin/ride-austin-june-6-april-13-part-2

Parameters
----------
rawpath : str
    Path to Rides_Data csv files.
outfile : str
    Output file path and name.

Example
-------
If the input files are in the current working directory, and the desired
output filename is `./out.hdf5`, then:

```
python make_austin_data.py ./ ./out.hdf5
```

"""

import pandas as pd
import pytz


def fixtime_utcminus10_nozone(t):
    if isinstance(t, str):
        # pytz's GMT definitions are inverted - if this issue is ever resolved,
        # this needs to be fixed!
        # https://github.com/pandas-dev/pandas/issues/21509
        return (pd.to_datetime(t[:-6]).tz_localize('Etc/GMT+10')
                .tz_convert('US/Central'))
    return pd.NaT


def read_csvs(rawpath):
    # Read in raw data.
    df_a = pd.read_csv(rawpath + '/Rides_DataA.csv')
    df_b = pd.read_csv(rawpath + '/Rides_DataB.csv',
                       usecols=(0, 6, 10, 11, 12, 13, 14, 15))

    # Merge the two tables.
    df = pd.merge(df_a, df_b, on='RIDE_ID')
    df.reset_index(inplace=True, drop=True)

    # Fix timestamps, then strip time zones.
    for name in ['started_on', 'completed_on', 'driver_reached_on']:
        df[name] = pd.to_datetime(df[name])
    df['dispatched_on'] = (pd.to_datetime(df['dispatched_on'], utc=True)
                           .dt.tz_convert('US/Central'))
    df['driver_accepted_on'] = (df['driver_accepted_on']
                                .apply(fixtime_utcminus10_nozone))

    # Fix column names.
    df = df[['RIDE_ID', 'dispatched_on', 'driver_accepted_on',
             'dispatch_location_lat', 'dispatch_location_long',
             'driver_reached_on', 'started_on', 'start_location_lat',
             'start_location_long', 'completed_on', 'end_location_lat',
             'end_location_long', 'driving_distance_to_rider',
             'distance_travelled', 'active_driver_id']]

    df.columns = ['ID', 'dispatch_time', 'driver_accept_time', 'dispatch_lat',
                  'dispatch_lon', 'driver_reach_start_time', 'start_time',
                  'start_lat', 'start_lon', 'complete_time', 'complete_lat',
                  'complete_lon', 'driving_distance_to_rider', 'trip_distance',
                  'driver_id']

    return df


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("rawpath", type=str,
                        help="Path to Rides_Data csv files.")
    parser.add_argument("outfile", type=str,
                        help="Output file path and name.")
    args = parser.parse_args()

    df = read_csvs(args.rawpath)

    # Output only the week of March 31, 2017 as test data.
    timespan = ((df['dispatch_time'] >= '2017/03/13 04:00:00-05:00') &
                (df['dispatch_time'] < '2017/03/20 04:00:00-05:00'))
    
    #.dt.tz_localize(None)
    #.tz_localize(tz=None)

    dfs = df.loc[timespan, :].copy()
    # Strip time zone stamps before outputting.
    for name in ['dispatch_time', 'driver_accept_time',
                 'driver_reach_start_time', 'start_time', 'complete_time']:
        dfs[name] = dfs[name].dt.tz_localize(None)

    dfs.to_csv(args.outfile, index=False)
