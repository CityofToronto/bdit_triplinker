"""Tests for trip linker.

Also houses importable full paths of sample data, following the format of the
github.com/mhvk/baseband data directory.  This makes .csv files visible to
pytest.
"""

from os import path as _path


def _full_path(name, dirname=_path.dirname(_path.abspath(__file__))):
    return _path.join(dirname, name)


NYC_TAXI_TRIPDATA = _full_path('test_data_nyc.csv')
"""Small subset of taxi dropoff and pickup points in NYC on 2011/01/19, from
7:00 to 11:00 AM.  The first 25 rows are guaranteed to have sensible
feasible connections (see docstring for test_grapher.py).  Original
data downloaded from
https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""

AUSTIN_RIDESHARE_TRIPDATA = _full_path('test_data_austin.csv')
"""Data from the Ride Austin rideshare network for all trips requested between
2017/03/15 04:00:00 to 2017/03/16 04:00:00.  Original data downloaded from
https://data.world/ride-austin/ride-austin-june-6-april-13 and
https://data.world/ride-austin/ride-austin-june-6-april-13-part-2
"""
