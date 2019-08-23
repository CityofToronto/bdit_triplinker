"""Sample data for testing and documentation.

Follows the format of the https://www.github.com/mhvk/baseband data directory.
This makes .csv files visible to pytest.
"""

from os import path as _path


def _full_path(name, dirname=_path.dirname(_path.abspath(__file__))):
    return _path.join(dirname, name)


AUSTIN_DATA = _full_path('austin_data.csv')
"""Data from the Ride Austin rideshare network for all trips requested between
2017/03/15 04:00:00 to 2017/03/16 04:00:00.  Original data downloaded from
https://data.world/ride-austin/ride-austin-june-6-april-13 and
https://data.world/ride-austin/ride-austin-june-6-april-13-part-2
"""

NYC_TAXI_TRIPDATA = _full_path('test_data_nyc.csv')