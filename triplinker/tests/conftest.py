"""Test suite preprocessing.

Main purpose is to load the sample Ride Austin data from `../../data`, and then
passing it on as a fixture to the various tests.  This enables multiple tests
to use the sample data while only loading it once.

References:
https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
https://stackoverflow.com/questions/17801300/how-to-run-a-method-before-all-tests-in-all-classes
https://pythontesting.net/framework/pytest/pytest-session-scoped-fixtures/

"""

import pytest
import pandas as pd
from .. import grapher

from ..data import AUSTIN_DATA


@pytest.fixture(scope="session", autouse=True)
def austin_data():
    """Load Austin data into test environment."""

    df = pd.read_csv(AUSTIN_DATA, parse_dates=[1, 2, 5, 6, 9])
    # Only consider 6 - 7:30 PM to make tests go faster.
    df.drop(labels=['ptctripid', 'request_datetime',
                    'driver_accept_datetime',
                    'request_latitude', 'request_longitude',
                    'driver_reach_datetime',
                    'trip_distance', 'driver_id'], axis=1, inplace=True)
    df = df.loc[
        ((df['pickup_datetime'] >= '2017-03-15 18:00:00') &
         (df['pickup_datetime'] < '2017-03-15 19:30:00')), :]

    gridangle_austin = (-97.746966, 30.267280, -97.741567, 30.281615)
    gphr = grapher.TripGrapherManhattan(
        df, pd.Timedelta('20 minutes'), gridangle=gridangle_austin,
        speed=7.56)
    net = gphr.create_graph(max_connections=30)
    return {'df': df, 'gphr': gphr, 'net': net}
