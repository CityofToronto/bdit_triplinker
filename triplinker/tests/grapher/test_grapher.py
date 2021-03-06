# """Pytest scripts to test grapher module.

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from ... import grapher
from ...grapher import manhattan
from ... import utils

from ...data import NYC_TAXI_TRIPDATA


class TestGrapherBase:
    """Base class for grapher testers."""

    radius = 6.371e3


class TestGrapherDB(TestGrapherBase):
    """Tests converting dataframe feasible links into graph."""

    @staticmethod
    def _feasiblenet_to_df(df, net):
        """Converts a feasible link network to a DataFrame of connections."""
        from_trip = []
        to_trip = []
        enroute = []

        for edge in net.edges:
            from_trip.append(df.loc[edge[0], 'ptctripid'])
            to_trip.append(df.loc[edge[1], 'ptctripid'])
            # Edge overhead time stored as minutes, but .
            enroute.append(60. * net.edges[edge]['enroute_time'])

        return pd.DataFrame({'from_trip': from_trip,
                             'to_trip': to_trip,
                             'dt': enroute})

    @pytest.fixture()
    def refdf(self, austin_data):
        """Test DataFrame of Austin data that includes dummy trip ID."""
        df = austin_data['df'].copy()
        df['ptctripid'] = ['trip_{0:d}'.format(idx + 10000)
                           for idx in df.index]
        return df

    @pytest.fixture()
    def refnet(self, austin_data):
        """Test network of Austin data with distances removed."""
        # Copying a network deepcopies it.
        # https://stackoverflow.com/questions/39555831/how-do-i-copy-but-not-deepcopy-a-networkx-graph
        net = austin_data['net'].copy()
        for edge in net.edges:
            del net.edges[edge]['r']
        return net

    @pytest.fixture()
    def reflinks(self, refnet, refdf):
        """DataFrame of feasible (Manhattan) links in the Austin data."""
        return self._feasiblenet_to_df(refdf, refnet)

    def test_create_graph(self, refdf, refnet, reflinks):
        """Check if we can translate an entire dataframe to a graph."""
        gphr = grapher.GrapherDB(refdf, reflinks)
        net = gphr.create_graph()
        assert utils.graphs_equivalent(net, refnet)

    def test_create_reduced_graph(self, refdf, refnet, reflinks):
        """Check if we can restrict valid links to below some critical dt."""
        gphr = grapher.GrapherDB(refdf, reflinks)
        t_max = 9.2356
        net_reduced = gphr.create_graph(t_max=t_max)

        # Produce a new links dataframe with only links below t_max.
        link_check = self._feasiblenet_to_df(refdf, net_reduced)

        # Join the original and check dataframes together.
        link_check = pd.merge(reflinks, link_check,
                              on=('from_trip', 'to_trip'),
                              suffixes=('_original', '_check'), how='left')

        ptcid = dict(zip(refdf['ptctripid'].values, refdf.index.values))
        deadheading = []
        for _, row in link_check.iterrows():
            deadheading.append(
                (refdf.loc[ptcid[row['to_trip']], 'pickup_datetime'] -
                 refdf.loc[ptcid[row['from_trip']], 'dropoff_datetime']) /
                 np.timedelta64(1, 'm'))
        link_check['deadheading'] = deadheading

        # Columns that exist in `link_check` should match ones in the original.
        reduced = link_check['dt_check'].notnull()
        assert np.allclose(
            link_check.loc[reduced, 'dt_original'].values,
            link_check.loc[reduced, 'dt_check'].values, rtol=1e-6, atol=1e-8)
        # Columns that don't should all have `dt > t_max`.
        assert np.all(link_check.loc[reduced, 'deadheading'].values <= t_max)
        assert np.all(link_check.loc[~reduced, 'deadheading'].values > t_max)


class TestGrapherConstBase(TestGrapherBase):
    """Base class for grapher testers that use constant boundaries

    When testing TestGrapherHaversine, args should be:
        t_max, r_max, avgspeed
    When testing TestGrapherManhattan, args should be:
        t_max, r_max, gridangle, avgspeed
    """

    def setup(self):
        self.df = pd.read_csv(NYC_TAXI_TRIPDATA,
                              usecols=list(range(1, 7)),
                              parse_dates=[0, 1], infer_datetime_format=True)
        self.df.drop(self.df.index[
            (self.df['dropoff_datetime'] - self.df['pickup_datetime']) <
            1. * pd.Timedelta('1 seconds')].values,
            axis=0, inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df.index.name = 'id'

    def get_triplink_graph(self, subset, max_connections, *args):
        # Create new graph.
        G = nx.DiGraph()
        G.add_nodes_from(self.df.index.values)

        # Create slicer.
        if len(subset):
            subset = slice(*subset)
        else:
            subset = slice(None)

        # Add feasible links as graph edges.
        for i, row in self.df.iloc[subset, :].iterrows():
            feasible_pickups = self.get_feasible_pickups(row, *args)
            if len(feasible_pickups) > 0:
                feasible_pickups.sort_values('distance (km)',
                                             ascending=True, inplace=True)
                new_edges = [
                    (row.name, fr.name,
                     {'r': fr['distance (km)'],
                      'enroute_time': 60. * fr['distance (km)'] / args[-1],
                      'deadhead_time': ((fr['pickup_datetime'] -
                                         row['dropoff_datetime']) /
                                        np.timedelta64(1, 'm'))})
                    for i, fr in
                    feasible_pickups.head(max_connections).iterrows()]
                G.add_edges_from(new_edges)

        return G


class TestGrapherManhattanBase(TestGrapherBase):
    """Tests `GrapherManhattanBase`'s rotation matrix maker."""

    def setup(self):
        self.df = pd.read_csv(NYC_TAXI_TRIPDATA,
                              usecols=list(range(1, 7)),
                              parse_dates=[0, 1], infer_datetime_format=True)
        self.df.drop(self.df.index[
            (self.df['dropoff_datetime'] - self.df['pickup_datetime']) <
            1. * pd.Timedelta('1 seconds')].values,
            axis=0, inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.df.index.name = 'id'

    def get_manhattan_distances(self, gridangle, dropoff_lat, dropoff_lon,
                                pickup_lat, pickup_lon):
        latrad = np.radians(pickup_lat)
        latrad_0 = np.radians(dropoff_lat)
        dlat = latrad - latrad_0
        avglat = 0.5 * (latrad + latrad_0)
        dlon = np.cos(avglat) * np.radians(pickup_lon - dropoff_lon)
        grid = self.radius * np.vstack([dlon, dlat])

        gridangle = np.radians(gridangle)
        rotmtx = np.array([[np.cos(gridangle), -np.sin(gridangle)],
                           [np.sin(gridangle), np.cos(gridangle)]])

        rides_xy = rotmtx.dot(grid)

        return np.sum(np.abs(rides_xy), axis=0)

    # Numbers obtained by hacking longlat_to_gridangle in
    # 2-Distance Metric Testing.ipynb.  First three coordinate sets represent
    # permutations of a segment of NYC's 8th street.  Last coordinates
    # represent two points on Spadina.
    @pytest.mark.parametrize(
        ('ll', 'angle_ref'),
        [((-73.999169, 40.739246, -73.938127, 40.822952), 0.504523325),
         ((-73.938127, 40.822952, -73.999169, 40.739246), 0.504523325),
         ((-73.999169, 40.822952, -73.938127, 40.739246), -0.504523325),
         ((-79.392405, 43.638978, -79.407120, 43.674952), -0.28772656)])
    def test_ll_gridangle(self, ll, angle_ref):
        mb = manhattan.GrapherManhattanBase(None)
        lon1, lat1, lon2, lat2 = ll
        angle = mb.longlat_to_gridangle(lon1, lat1, lon2, lat2)
        assert np.isclose(angle, angle_ref, rtol=1e-6, atol=1e-8)

    def test_get_rotation_matrix(self):
        mb = manhattan.GrapherManhattanBase(None)

        # Check that if we pass `None`, we don't get anything.
        mb.get_rotation_matrix(None)
        assert not hasattr(mb, 'rotation_matrix')

        mb.get_rotation_matrix(0.)
        assert np.allclose(mb.rotation_matrix,
                           np.array([[1., 0.], [0., 1.]]),
                           rtol=1e-6, atol=1e-8)

        mb.get_rotation_matrix(90.)
        assert np.allclose(mb.rotation_matrix,
                           np.array([[0., 1.], [-1., 0.]]),
                           rtol=1e-6, atol=1e-8)

        mb.get_rotation_matrix(-45.)
        assert np.allclose(mb.rotation_matrix,
                           np.array([[2.**-0.5, -2.**-0.5],
                                     [2.**-0.5, 2.**-0.5]]),
                           rtol=1e-6, atol=1e-8)

        # Angle for Manhattan street grid.
        nyc_ll = (-73.999169, 40.739246, -73.938127, 40.822952)
        dlat = nyc_ll[3] - nyc_ll[1]
        dlon = (nyc_ll[2] - nyc_ll[0]) * np.cos(
            np.radians(0.5 * (nyc_ll[1] + nyc_ll[3])))
        gridangle = np.arctan2(dlon, dlat)
        rotmtx = np.array([[np.cos(gridangle), np.sin(gridangle)],
                           [-np.sin(gridangle), np.cos(gridangle)]])

        mb.get_rotation_matrix(nyc_ll)
        assert np.allclose(mb.rotation_matrix, rotmtx,
                           rtol=1e-6, atol=1e-8)

    def test_get_manhattan_distances(self):

        mb = manhattan.GrapherManhattanBase(None)

        nyc_ll = (-73.999169, 40.739246, -73.938127, 40.822952)
        # Dropoff at Times Square, pickup at Penn Station
        dropoff_lat = 40.754337
        dropoff_lon = -73.986888
        pickups_lat = 40.751038
        pickups_lon = -73.994291

        mb.get_rotation_matrix(nyc_ll)
        dist_ref = self.get_manhattan_distances(
            np.degrees(mb.gridangle), dropoff_lat, dropoff_lon, pickups_lat,
            pickups_lon)
        dist = mb.get_manhattan_distances(
            dropoff_lat, dropoff_lon, pickups_lat, pickups_lon)

        assert np.allclose(dist_ref, dist, rtol=1e-6, atol=1e-8)

        toronto_ll = (-79.392405, 43.638978, -79.407120, 43.674952)
        # Dropoff at City Hall, pickup at Dundas Square, Osgoode Hall and CN
        # Tower
        dropoff_lat = 43.654106
        dropoff_lon = -79.383110
        pickups_lat = np.array([43.656316, 43.651852, 43.641787])
        pickups_lon = np.array([-79.380894, -79.386799, -79.386398])

        mb.get_rotation_matrix(toronto_ll)
        dist_ref = self.get_manhattan_distances(
            np.degrees(mb.gridangle), dropoff_lat, dropoff_lon, pickups_lat,
            pickups_lon)
        dist = mb.get_manhattan_distances(
            dropoff_lat, dropoff_lon, pickups_lat, pickups_lon)

        assert np.allclose(dist_ref, dist, rtol=1e-6, atol=1e-8)


class TestGrapherManhattan(TestGrapherConstBase):
    """Tests Manhattan-based grapher."""

    def get_feasible_pickups(self, dropoff_row, t_max, r_max,
                             gridangle, avgspeed):

        gridangle = np.radians(gridangle)
        dropoff_time = dropoff_row['dropoff_datetime']

        # Consider only pickup points that are below the maximum
        # connection distance and time.
        rides_timecut = self.df.loc[
            (self.df['pickup_datetime'] > dropoff_time) &
            (self.df['pickup_datetime'] <= dropoff_time + t_max),
            ['pickup_datetime', 'pickup_latitude', 'pickup_longitude']]
        rides_latlong = rides_timecut[
            ['pickup_latitude', 'pickup_longitude']].values * np.pi / 180.

        # Filter only for feasible rides.
        if len(rides_latlong):
            latrad = rides_timecut['pickup_latitude'].values * np.pi / 180.
            latrad_0 = dropoff_row['dropoff_latitude'] * np.pi / 180.
            dlat = latrad - latrad_0
            avglat = 0.5 * (latrad + latrad_0)
            dlon = np.cos(avglat) * (np.pi / 180.) * (
                (rides_timecut['pickup_longitude'] -
                 dropoff_row['dropoff_longitude']).values)
            grid = self.radius * np.vstack([dlon, dlat])

            rotmtx = np.array([[np.cos(gridangle), -np.sin(gridangle)],
                               [np.sin(gridangle), np.cos(gridangle)]])

            rides_xy = rotmtx.dot(grid)

            manhattan_distances = np.sum(np.abs(rides_xy), axis=0)
            sorted_i = np.argsort(manhattan_distances)
            sorted_distances = manhattan_distances[sorted_i]
            bounded_i = sorted_distances < r_max

            feasible_pickups = rides_timecut.iloc[
                sorted_i[bounded_i], :].copy()
            feasible_pickups['distance (km)'] = sorted_distances[bounded_i]
            cannot_connect = (
                (feasible_pickups['distance (km)'] *
                 pd.Timedelta('1 hours') / avgspeed) >
                (feasible_pickups['pickup_datetime'] -
                 dropoff_row['dropoff_datetime']))
            feasible_pickups.drop(feasible_pickups.index[cannot_connect],
                                  axis=0, inplace=True)
        else:
            feasible_pickups = pd.DataFrame(
                {'pickup_datetime': [], 'pickup_latitude': [],
                 'pickup_longitude': [], 'distance (km)': []})

        return feasible_pickups

    # Number of matches manually determined during test development.
    @pytest.mark.parametrize(
        ('row_num', 't_max', 'r_max', 'gridangle', 'spd', 'n_match'),
        [(14, pd.Timedelta('3 minutes'), 1.5, 28.907056901, 7.56, 3),
         (573, pd.Timedelta('10 minutes'), 2.2, -9.886959713, 10.52, 2),
         (22, pd.Timedelta('6 minutes'), 2.6, -5.78687373, 13.58, 34),
         (22, pd.Timedelta('6 minutes'), 2.6, -69.32789321, 13.58, 42),
         (119, pd.Timedelta('4 minutes'), 0.7, 0.0, 3.2, 0),
         (22, pd.Timedelta('4 minutes'), 0., 19.480565034, 1.7, 0),
         (3, pd.Timedelta('5 minutes'), 2.4, -44.11775023, 0., 0)])
    def test_feasible_pickups(self, row_num, t_max, r_max,
                              gridangle, spd, n_match):
        dropoff_row = self.df.iloc[row_num, :]
        fp_ref = self.get_feasible_pickups(dropoff_row, t_max, r_max,
                                           gridangle, spd)
        # Dummy grapher.
        gphr = grapher.GrapherManhattan(self.df, None, None,
                                            gridangle=gridangle, speed=None)
        fp = gphr.get_feasible_pickups(dropoff_row, t_max, r_max, spd)
        assert np.all(fp_ref['pickup_datetime'].values ==
                      fp['pickup_datetime'].values)
        float_columns = ['pickup_latitude', 'pickup_longitude',
                         'distance (km)']
        assert np.allclose(fp_ref[float_columns].values,
                           fp[float_columns].values)
        assert fp.shape[0] == n_match

    # Number of edges manually determined during test development.
    # Also tests whether auto-calculation of r_max functions.
    @pytest.mark.parametrize(
        ('t_max', 'gridangle', 'spd', 'subset',
         'max_connections', 'n_edges'),
        [(pd.Timedelta('2 minutes'), -69.32789321, 7.56, (), 5, 128),
         (pd.Timedelta('3 minutes'), 5.729577951, 33.8, (0, 2), 1000, 64),
         (pd.Timedelta('0 minutes'), 6.8754935416, 45.2, (15, 19), 20, 0),
         (pd.Timedelta('2 minutes'), 21.199438419, 18.5, (5, 15, 4), 10, 8),
         (pd.Timedelta('4 minutes'), 0., 3.5, (0, 20), 3, 13)])
    def test_graph_generator(self, t_max, gridangle, spd, subset,
                             max_connections, n_edges):
        r_max = spd * t_max / np.timedelta64(1, 'h')
        G_ref = self.get_triplink_graph(subset, max_connections, t_max,
                                        r_max, gridangle, spd)
        gphr = grapher.GrapherManhattan(self.df, t_max,
                                            gridangle=gridangle, speed=spd)
        G = gphr.create_graph(subset=subset, max_connections=max_connections)
        assert utils.graphs_equivalent(G_ref, G)
        assert G.number_of_edges() == n_edges
