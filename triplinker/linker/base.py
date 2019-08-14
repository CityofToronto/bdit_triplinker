import networkx as nx
import numpy as np
import pandas as pd
import tqdm


class BatchedLinker:
    """Base class for batch-linking trips."""

    def __init__(self, df, gph_feasible, start_time,
                 timespan=pd.Timedelta('24 hour'), progress_bar=False):
        self.df = df
        self.gph_feasible = gph_feasible
        self.start_time = start_time
        self.timespan = timespan
        self._disable_tqdm = True if not progress_bar else False

        # Get list of time-ordered pickup points (`sort_values` is
        # not inplace).
        self.pickups = (self.df[['pickup_datetime']]
                        .sort_values('pickup_datetime'))

    def get_bin_graph(self, soln, cbin_edges):
        """Generate feasible sub-graph."""

        bin_pickup_points = self.pickups[
            (self.pickups['pickup_datetime'] >= cbin_edges[0]) &
            (self.pickups['pickup_datetime'] < cbin_edges[1])].index.values

        bin_graph = nx.DiGraph()
        bin_graph.add_nodes_from(bin_pickup_points)

        # Add edges from feasibility network.  Use `bin_pickup_points` because
        # it's bad form to iterate over a list of nodes that's increasing.
        for node in bin_pickup_points:
            for edge in self.gph_feasible.in_edges(node):
                bin_graph.add_edge(*edge, **self.gph_feasible.edges[edge])

        # Delete any nodes already in use within the solution.  We only have
        # to consider `out_degree` because the nodes in `bin_pickup_points` are
        # guaranteed to have not yet been used; only the new nodes added by the
        # `bin_graph.add_edge` loop might be.
        nodes_to_delete = []
        for node in bin_graph.nodes():
            if soln.out_degree(node):
                nodes_to_delete.append(node)
        bin_graph.remove_nodes_from(nodes_to_delete)

        return bin_graph

    @staticmethod
    def add_edges(soln, bin_graph_soln):
        for edge in bin_graph_soln.edges():
            soln.add_edge(*edge, **bin_graph_soln.edges[edge])

    def link_mtd(self, bin_graph):
        raise NotImplementedError

    def get_batch_linking(self, t_bin):
        """Get batched link solution.

        Parameters
        ----------
        t_bin : pandas.Timedelta
            Period of time of a single bin.

        Returns
        -------
        soln : networkx.DiGraph
            Graph of binned linking solution.
        """
        nbins = int(np.ceil(self.timespan / t_bin))
        bin_edges = [self.start_time + x * t_bin
                     for x in np.arange(nbins + 1)]

        soln = nx.DiGraph()
        soln.add_nodes_from(self.gph_feasible)

        # For each bin, get the feasible sub-graph, link it, and store the
        # solution in `soln`.
        for i in tqdm.tqdm(range(nbins), disable=self._disable_tqdm):
            bin_graph = self.get_bin_graph(soln, bin_edges[i:i + 2])
            if bin_graph.number_of_edges():
                bin_graph_soln = self.link_mtd(bin_graph)
                self.add_edges(soln, bin_graph_soln)

        return soln


class BatchedLinkerVazifeh(BatchedLinker):
    """Link trip in batches to simulate dynamic trip linking.  Uses
    Vazifeh et al. 2018 minimum fleet size linker.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of ptc pickups and dropoffs.
    gph_feasible : networkx.classes.digraph.DiGraph
        Graph of feasible links between trips.  Node IDs should match the
        indices in `df`.
    start_time : pandas.Timestamp
        Start time of day.
    timespan : pandas.Timedelta, optional
        Entire timespan being linked.  Default: 24 hours.
    """

    def link_mtd(self, bin_graph):
        return get_vazifeh_solution(bin_graph)
