import pytest
import numpy as np
import networkx as nx
import pandas as pd

from ...linker import bipartite as lb
from ...linker import BatchedLinker
from ... import utils


class TestLinkerBase:

    @pytest.fixture()
    def df(self, austin_data):
        """Test DataFrame of Austin data."""
        return austin_data['df'].copy()

    @pytest.fixture()
    def gphr(self, austin_data):
        """Austin data GrapherManhattan instance."""
        return austin_data['gphr']

    @pytest.fixture()
    def net(self, austin_data):
        """Test network of Austin data."""
        return austin_data['net'].copy()


class TestBatchedLinkers(TestLinkerBase):
    """Test batched linking classes.

    Since link algorithms are tested separately, we'll just use the minimum
    fleet linker when needed.
    """

    def test_batched_linker_class(self, df, net):
        start_time = pd.Timestamp('2017-03-15 18:00:00')
        timespan = pd.Timedelta('90 minutes')
        linker = lb.MaxCardinalityLinker()
        bl = BatchedLinker(df, net, linker, start_time,
                           timespan=timespan)
        assert bl.df is df
        assert bl.gph_feasible is net
        assert bl.linker == linker
        assert bl.start_time == start_time
        assert bl.timespan == timespan

        assert np.all(
            df[['pickup_datetime']].sort_values('pickup_datetime') ==
            bl.pickups)

    @pytest.mark.parametrize(
        ('start_time', 'stop_time', 'n_prev'),
        [('2017-03-15 18:30:00', '2017-03-15 18:40:00', 3),
         ('2017-03-15 19:15:00', '2017-03-15 19:19:00', 5),
         ('2017-03-15 18:51:00', '2017-03-15 19:00:00', 15)])
    def test_get_bin_graph(self, df, net, start_time, stop_time, n_prev):
        bl = BatchedLinker(
            df, net, lb.MaxCardinalityLinker(),
            pd.Timestamp('2017-03-15 18:00:00'),
            timespan=pd.Timedelta('90 minutes'))

        # Begin with no edges in the solution.
        soln_net = nx.DiGraph()
        soln_net.add_nodes_from(net)

        bin_graph = bl.get_bin_graph(soln_net, [start_time, stop_time])

        # Check that `bin_graph` has all nodes in the bin, and a bunch of extra
        # nodes that are only from previous bins.
        nodes_in_tbin = (
            bl.pickups[
                (bl.pickups['pickup_datetime'] >= start_time) &
                (bl.pickups['pickup_datetime'] < stop_time)]
            .index.values)
        assert len(set(nodes_in_tbin) - set(bin_graph.nodes())) == 0
        prev_bin_nodes = list(set(bin_graph.nodes()) - set(nodes_in_tbin))
        assert len(prev_bin_nodes) >= 0
        assert np.all(bl.pickups.loc[prev_bin_nodes, :] <
                      start_time)

        # For a few nodes from previous bins, add a connection to the solution.
        for node in prev_bin_nodes[:n_prev]:
            soln_net.add_edge(*next(iter(net.out_edges(node))))

        # Check that bin_graph properly removes these.
        bin_graph2 = bl.get_bin_graph(soln_net, [start_time, stop_time])

        expected_nodes = list(set(bin_graph.nodes()) -
                              set(prev_bin_nodes[:n_prev]))
        assert sorted(expected_nodes) == sorted(bin_graph2.nodes())

    def _batch_linking(self, net, bl, t_bin):
        # Define batch solution to output.
        soln_batch = nx.DiGraph()
        soln_batch.add_nodes_from(net)
        linker = lb.MaxCardinalityLinker()

        n_bins = int(bl.timespan / t_bin)
        bin_edges = [bl.start_time + x * t_bin for x in np.arange(n_bins + 1)]

        for i in range(n_bins):
            # Use bl for `get_bin_graph`, which we already tested.
            bin_graph = bl.get_bin_graph(soln_batch, bin_edges[i:i + 2])
            if bin_graph.number_of_nodes():
                bin_graph_soln = linker.get_solution(bin_graph)
                for edge in bin_graph_soln.edges():
                    soln_batch.add_edge(*edge, **bin_graph_soln.edges[edge])
            else:
                nodes_in_tbin = (
                    bl.pickups[
                        (bl.pickups['pickup_datetime'] >= bin_edges[i]) &
                        (bl.pickups['pickup_datetime'] < bin_edges[i + 1])]
                    .index.values)
                assert len(nodes_in_tbin) == 0, "bin_graph is missing nodes!"

        return soln_batch

    @pytest.mark.parametrize(
        ('start_time', 'timespan', 't_bin'),
        [('2017-03-15 18:00:00', '90 minutes', '5 minutes'),
         ('2017-03-15 18:37:00', '82 minutes', '6.2 minutes'),
         ('2017-03-15 19:00:00', '14 minutes', '14 minutes'),
         ('2017-03-15 19:00:00', '14 minutes', '1 minutes')])
    def test_batched_linker_vazifeh(self, df, net, start_time,
                                    timespan, t_bin):
        # Since we can't guarantee the graph's output, we can only test that
        # each bin has
        bl = BatchedLinker(
            df, net, lb.MaxCardinalityLinker(),
            pd.Timestamp(start_time), timespan=pd.Timedelta(timespan))
        bl_soln = bl.get_batch_solution(pd.Timedelta(t_bin))
        ref_soln = self._batch_linking(net, bl, pd.Timedelta(t_bin))

        assert utils.graphs_equivalent(bl_soln, ref_soln)
