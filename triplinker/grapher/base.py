"""Base methods for creating feasibile link networks."""
import numpy as np
import pandas as pd
import networkx as nx


class GrapherBase:
    """Base class for transforming trips into directed graphs."""

    # Mean radius of the Earth, from https://en.wikipedia.org/wiki/Earth
    _r_earth = 6.371e3

    def __init__(self, df):
        self.df = df


class GrapherDB(GrapherBase):
    """Translates OD data and a set of feasible links into a graph.

    Parameters
    ----------
    df_data : pandas.DataFrame
        DataFrame of pickup locations and times.  Must include 'ptctripid',
        'pickup_datetime', 'pickup_latitude', 'pickup_longitude'.  Takes the
        index as node ID.
    df_links : pandas.DataFrame
        DataFrame with a column 'from_trip' of drop-off 'ptctripid', a
        corresponding column 'to_trip' of pick-up 'ptctripid', and a column
        'dt' representing en-route time in seconds.
    time_units : str, optional
        If 'sec', switches weight units to seconds.  Default: 'min'.

    """

    def __init__(self, df_data, df_links, time_units='min'):
        super().__init__(df_data)
        self.links = df_links
        self.time_units = time_units

    def create_graph(self, t_max=np.infty):
        """Create a directed graph from raw data and links.

        Parameters
        ----------
        t_max : float, optional
            Maximum deadheading time of feasible link.  Default: `numpy.infty`,
            in which case all links from `self.links` are used.  (Since the
            feasibility graph itself usually is time-limited, the graph
            produced may still have limitations.)

        Returns
        -------
        routed_net : networkx.DiGraph
            Graph of feasible links between trips, with 'deadhead_time' and
            'enroute_time'.
        """
        # Map ptctripids to indices, and merge
        ptcindex = pd.DataFrame({'ptctripid': self.df['ptctripid'],
                                 'ptcindex': self.df.index})
        links_merged = pd.merge(
            pd.merge(self.links, ptcindex,
                     left_on='from_trip', right_on='ptctripid'),
            ptcindex, left_on='to_trip', right_on='ptctripid',
            suffixes=('_from', '_to'))

        # Create routed network.
        routed_net = nx.DiGraph()
        routed_net.add_nodes_from(self.df.index.values)

        # Populate routed network with edges from links.
        for i, row in links_merged.iterrows():
            deadhead_time = (
                (self.df.loc[row['ptcindex_to'], 'pickup_datetime'] -
                 self.df.loc[row['ptcindex_from'], 'dropoff_datetime']) /
                np.timedelta64(1, 'm'))
            if deadhead_time <= t_max:
                if self.time_units == 'sec':
                    routed_net.add_edge(row['ptcindex_from'],
                                        row['ptcindex_to'],
                                        deadhead_time=(60. * deadhead_time),
                                        enroute_time=row['dt'])
                else:
                    routed_net.add_edge(row['ptcindex_from'],
                                        row['ptcindex_to'],
                                        deadhead_time=deadhead_time,
                                        enroute_time=(row['dt'] / 60.))

        return routed_net
