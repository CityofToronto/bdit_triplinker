import pytest
import numpy as np
# import networkx as nx

from ...linker import greedy as lg
from .test_base import TestLinkerBase


class TestGreedyLinker(TestLinkerBase):
    """Test greedy linker function."""

    def _from_node_weights(self, net, node, mintype):
        from_node = np.zeros(len(net.in_edges(node)), dtype=int)
        weights = np.zeros(len(net.in_edges(node)))
        for i, edge in enumerate(net.in_edges(node)):
            from_node[i] = edge[0]
            weights[i] = net.edges[edge][mintype]
        return from_node, weights
    
    def _to_node_weights(self, net, node, mintype):
        to_node = np.zeros(len(net.out_edges(node)), dtype=int)
        weights = np.zeros(len(net.out_edges(node)))
        for i, edge in enumerate(net.out_edges(node)):
            to_node[i] = edge[1]
            weights[i] = net.edges[edge][mintype]
        return to_node, weights

    @pytest.mark.parametrize(('whichgreed', 'mintype'),
                             [('driver', 'deadhead_time'),
                              ('passenger', 'deadhead_time'),
                              ('driver', 'enroute_time'),
                              ('passenger', 'enroute_time')])
    def test_greedy_solution(self, df, net, whichgreed, mintype):
        if whichgreed == 'passenger':
            time_ordered_nodes = list(
                df['pickup_datetime'].sort_values().index)
        else:
            time_ordered_nodes = list(
                df['dropoff_datetime'].sort_values().index)
        linker = lg.GreedyLinker(whichgreed=whichgreed, mintype=mintype)

        # Get greedy solution from linker.
        greedy_soln = linker.get_solution(net, time_ordered_nodes)

        # Algorithm to ensure all nodes have optimally greedy links.
        for node in time_ordered_nodes:
            # For each node, start by getting all relevant connecting nodes
            # (origin if passenger, destination if driver).  Also retrieve
            # chosen connecting node from solution.
            if whichgreed == 'passenger':
                connecting_nodes, weights = self._from_node_weights(
                    net, node, mintype)
                soln_link = list(greedy_soln.in_edges(node))
                soln_cn = soln_link[0][0] if len(soln_link) else None
            else:
                connecting_nodes, weights = self._to_node_weights(
                    net, node, mintype)
                soln_link = list(greedy_soln.out_edges(node))
                soln_cn = soln_link[0][1] if len(soln_link) else None
            # Sort the connecting node IDs and distances from nearest to
            # furthest link.
            ordered_args = np.argsort(weights)
            weights = weights[ordered_args]
            connecting_nodes = connecting_nodes[ordered_args]
            # If a solution was picked, find
            if soln_cn is not None:
                link_i = np.where(connecting_nodes == soln_cn)[0][0]
                link_weight = weights[link_i]
            # If link doesn't exist, use dummy values to make next lines work.
            else:
                link_i = len(connecting_nodes)
                link_weight = -999.    # No real weights are negative.
            # If the closest link wasn't chosen (link_i > 0), check that all
            # closer links are already taken.  If the link wasn't chosen,
            # search all of `to_node`.
            for i in range(link_i):
                if whichgreed == 'passenger':
                    greedier_origin = list(
                        greedy_soln.out_edges(connecting_nodes[i]))
                    datetime_col = 'pickup_datetime'
                    s_i = 1
                else:
                    greedier_origin = list(
                        greedy_soln.in_edges(connecting_nodes[i]))
                    datetime_col = 'dropoff_datetime'
                    s_i = 0

                assert len(greedier_origin) in [0, 1], (
                    "node has multiple incoming edges!")

                # Intelligently handle situations where link distances are
                # the same.
                if not np.isclose(weights[i], link_weight,
                                  rtol=1e-7, atol=1e-7):
                    assert (df.loc[greedier_origin[0][s_i], datetime_col] <=
                            df.loc[node, datetime_col]), (
                                "greediest link blocked by future node!")

    def test_greedy_linker_errs(self):
        with pytest.raises(ValueError):
            lg.GreedyLinker(whichgreed='cars', mintype='enroute_time')
