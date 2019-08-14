"""Grapher methods that use the rotated Manhattan distance between O/D."""
import numpy as np
import pandas as pd
import networkx as nx

from .base import GrapherBase


class GrapherManhattanBase(GrapherBase):
    """Base class for Manhattan distance-based graphers.

    Includes functions than calculate Manhattan distances on a grid angled from
    due north.

    """

    def get_rotation_matrix(self, gridangle):

        if isinstance(gridangle, tuple):
            self.gridangle = self.longlat_to_gridangle(*gridangle)
        else:
            self.gridangle = (np.radians(gridangle)
                              if gridangle is not None else None)
        if self.gridangle is not None:
            self.rotation_matrix = np.array(
                [[np.cos(self.gridangle), np.sin(self.gridangle)],
                 [-np.sin(self.gridangle), np.cos(self.gridangle)]])

    def get_manhattan_distances(self, dropoff_lat, dropoff_lon,
                                pickups_lat, pickups_lon):
        """Calculate longitudinal and latitudinal Manhattan distances of
        pickup points from dropoff point.  All parameters are in degrees.

        Parameters
        ----------
        dropoff_lat : float
            Dropoff latitude.
        dropoff_lon : float
            Dropoff longitude.
        pickups_lat : float or numpy.ndarray
            Pickup latitude(s).  Can be a float or a 1-D array.
        pickups_lon : float or numpy.ndarray
            Pickup longitude(s).  Can be a float or a 1-D array.

        Returns
        -------
        manhattan_distances : numpy.ndarray
            Array of distances, with the same order as pickups_lon/lat.
        """
        # Convert degrees to radians.
        dropoff_lat = np.radians(dropoff_lat)
        dropoff_lon = np.radians(dropoff_lon)
        pickups_lat = np.radians(pickups_lat)
        pickups_lon = np.radians(pickups_lon)

        dlat = pickups_lat - dropoff_lat
        dlon = (pickups_lon - dropoff_lon) * np.cos(
            0.5 * (pickups_lat + dropoff_lat))

        # Convert to physical distances and rotate.
        xy = self._r_earth * np.c_[dlon, dlat]
        # Transpose of the usual rotation matrix, since we reverse the dot
        # product order.
        xy_rotated = xy.dot(self.rotation_matrix)
        return np.abs(xy_rotated).sum(axis=1)

    @staticmethod
    def longlat_to_gridangle(lon1, lat1, lon2, lat2):
        """Convenience function to determine angle of a street (in physical
        units) from due north.

        Positive angles are clockwise from due north.  Roads that are exactly
        north-south have angle 0, and those exactly east-west have angle
        :math:`\pi/2`.

        Parameters
        ----------
        lon1 : float
            Longitude of first point on the road.
        lat1 : float
            Latitude of first point on the road.
        lon2 : float
            Longitude of second point on the road.
        lat2 : float
            Latitude of second point on the road.
        """
        dlon = np.cos(np.radians(0.5 * (lat1 + lat2))) * (lon2 - lon1)
        dlat = lat2 - lat1
        if dlat < 0:
            dlat *= -1
            dlon *= -1
        return np.arctan2(dlon, dlat)


class GrapherConstLimits(GrapherBase):
    """Base class for graphers with constant r_max, t_max, and speed."""

    # Speed floor, to prevent NaT issues when dividing distance and
    # multiplying by pd.Timedelta.
    _speedfloor = 1e-4

    # 7.56 kph from https://www.latimes.com/nation/la-na-new-york-traffic-manhattan-20180124-story.html
    def __init__(self, df, t_max, r_max=None, speed=7.56):
        self.t_max = t_max
        if r_max is None and t_max is not None:
            r_max = speed * (t_max / np.timedelta64(1, 'h'))
        self.r_max = r_max
        self.speed = speed
        super().__init__(df)

    def get_feasible_pickups(self, dropoff_row, t_max, r_max, speed):
        """Find feasible pickups for dropoff.

        Parameters
        ----------
        dropoff_row : pandas.Series
            One row from a table of dropoff locations and times.  Must include
            'dropoff_datetime', 'dropoff_latitude', 'dropoff_longitude'.
        t_max : pandas.Timedelta
            Maximum time duration between dropoff and pickup.
        r_max : float
            Maximum distance between dropoff and pickup, in km.
        speed : float
            Typical vehicle speed in km/h.

        Returns
        -------
        feasible_pickups : pandas.DataFrame
            All feasible pickups for dropoff.
        """

        dropoff_time = dropoff_row['dropoff_datetime']
        rides_timecut = self.df.loc[
            (self.df['pickup_datetime'] > dropoff_time) &
            (self.df['pickup_datetime'] <= dropoff_time + t_max),
            ['pickup_datetime', 'pickup_latitude', 'pickup_longitude']]

        # If there are rides within the maximum time
        if len(rides_timecut):
            neighbours, distances = self.get_neighbours(
                rides_timecut, dropoff_row['dropoff_latitude'],
                dropoff_row['dropoff_longitude'], r_max)

            # Copying since we're adding new columns.
            feasible_pickups = rides_timecut.iloc[neighbours, :].copy()
            feasible_pickups['distance (km)'] = distances

            # Determine if pickup distance cannot be reached in time travelling
            # at typical taxi speeds.
            cannot_pickup = (
                (feasible_pickups['pickup_datetime'] - dropoff_time) <
                (feasible_pickups['distance (km)'].values *
                 pd.Timedelta('1 hours') / max(speed, self._speedfloor)))
            feasible_pickups.drop(labels=feasible_pickups.index[cannot_pickup],
                                  inplace=True)

        else:
            feasible_pickups = pd.DataFrame(
                {'id': [], 'pickup_datetime': [], 'pickup_latitude': [],
                 'pickup_longitude': [], 'distance (km)': []})

        return feasible_pickups

    def create_graph(self, subset=(), max_connections=10):
        """Create a directed graph from raw pickup and dropoff data.

        Parameters
        ----------
        subset : tuple, optional
            Arguments to generate a `slice` object, for selecting
            only a subset of `nyc_tlc` to process.
        max_connections : int, optional
            Maximum number of connections allowed per dropoff point.

        Returns
        -------
        G : networkx.DiGraph
            Directed graph network of trips.  Distance and travel time between
            connections are stored as values.
        """
        # Generate graph and load all trips into it.
        G = nx.DiGraph()
        G.add_nodes_from(self.df.index.values)

        # Keep max_connections zero of positive.
        max_connections = max(max_connections, 0)

        # Handle subsetting.
        if len(subset):
            subset = slice(*subset)
        else:
            subset = slice(None)

        # For each ride dropoff, determine possible links and encode
        # `max_connection` number of nearest links into the graph.
        for i, row in self.df.iloc[subset, :].iterrows():
            feasible_rides = self.get_feasible_pickups(
                row, self.t_max, self.r_max, self.speed)
            if len(feasible_rides) > 0:
                feasible_rides.sort_values('distance (km)',
                                           ascending=True, inplace=True)
                new_edges = [(
                    row.name, fr.name,
                    {'r': fr['distance (km)'],
                     'deadhead_time': (
                        (fr['pickup_datetime'] - row['dropoff_datetime']) /
                        np.timedelta64(1, 'm')),
                     'overhead_time': (
                        60. * fr['distance (km)'] / self.speed)})
                    for (fr_i, fr) in
                    feasible_rides.head(max_connections).iterrows()]
                G.add_edges_from(new_edges)

        return G


class GrapherManhattan(GrapherConstLimits, GrapherManhattanBase):
    """Transforms trip data into directed graph.  Uses Manhattan distance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns: 'dropoff_datetime', 'dropoff_latitude',
        'dropoff_longitude', 'pickup_datetime', 'pickup_latitude',
        'pickup_longitude'.
    t_max : pandas.Timedelta
        Maximum time duration between dropoff and pickup.
    r_max : float or None, optional
        Maximum Manhattan distance between dropoff and pickup.  If `None`
        (default), will be inferred from `speed` and `t_max`.
    gridangle : float or tuple, optional
        Angle - in degrees - of north-south streets on the city grid from from
        due north.  Alternatively, can pass a tuple of (lon_1, lat_1,
        lon_2, lat_2) of a north-south street in the grid, and the
        angle will be auto-calculated from `longlat_to_gridangle`.  Default: 0.
    speed : float, optional
        Typical vehicle speed, in km/h.  Default: 7.56, the typical NYC
        downtown traffic speed during rush hour.
    """
    # Note: MRO is TripGrapherConstLimits, then RotatedManhattanDistance.
    # Since the latter just copies GrapherBase.__init__, which doesn't call
    # super().__init__, the code in GrapherBase.__init__ is only called
    # once.

    def __init__(self, df, t_max, r_max=None, gridangle=0., speed=7.56):
        super().__init__(df, t_max, r_max=r_max, speed=speed)
        self.get_rotation_matrix(gridangle)

    def get_neighbours(self, rides, lat, lon, r_max):
        """Find closest neighbours of (lat, lon) within distance r_max.

        Uses Manhattan distance.

        Parameters
        ----------
        rides : pandas.DataFrame
            DataFrame of pickup locations and times.  Must include
            'pickup_datetime', 'pickup_latitude', 'pickup_longitude'.  Takes
            the index as ID.
        lat : float
            Dropoff latitude.
        lon : float
            Dropoff longitude.
        gridangle : float, optional
            Angle - in degrees - of north-south streets on the city grid from
            from due north.
        r_max : float
            Maximum Manhattan distance between dropoff and pickup, in km.

        Returns
        -------
        neighbours : numpy.ndarray
            Array of rides indices (can used for rides.iloc).
        distances : numpy.ndarray
            Corresponding distances in km.
        """
        rides_latlon = rides[['pickup_latitude', 'pickup_longitude']].values

        manhattan_distances = self.get_manhattan_distances(
            lat, lon, rides_latlon[:, 0], rides_latlon[:, 1])

        sorted_indices = np.argsort(manhattan_distances)
        sorted_distances = manhattan_distances[sorted_indices]
        bounds = sorted_distances < r_max

        return sorted_indices[bounds], sorted_distances[bounds]
