"""Test suite preprocessing.

Main purpose is to load the sample Ride Austin data from `../../data`, and then
passing it on as a fixture to the various tests.  This allows us to load the
data just once but use it in multiple tests.

References:
https://docs.pytest.org/en/latest/fixture.html#fixture-finalization-executing-teardown-code
https://stackoverflow.com/questions/17801300/how-to-run-a-method-before-all-tests-in-all-classes
https://pythontesting.net/framework/pytest/pytest-session-scoped-fixtures/

"""

import pytest
import sys
import pandas as pd

from .. import grapher

from ..data import AUSTIN_DATA


@pytest.fixture(scope="session", autouse=True)
def austin_data(request):
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
    gphr = grapher.GrapherManhattan(
        df, pd.Timedelta('20 minutes'), gridangle=gridangle_austin,
        speed=7.56)
    net = gphr.create_graph(max_connections=30)

    # Post test-suite check that df is unchanged after all testing.
    # Exception handling from https://stackoverflow.com/a/6062799, Update 2.
    df_copy = df.copy()

    def check_df_unchanged():
        try:
            pd.testing.assert_frame_equal(df, df_copy, check_less_precise=8)
        except AssertionError as exc:
            raise (type(exc)(("OD DataFrame values have changed! This may "
                              "be due to a triplinker process changing its "
                              "input data! See below:\n") +
                             str(exc))
                   .with_traceback(sys.exc_info()[2]))

    request.addfinalizer(check_df_unchanged)

    return {'df': df, 'gphr': gphr, 'net': net, 'gridangle': gridangle_austin}
