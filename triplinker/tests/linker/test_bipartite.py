"""Pytest scripts to test bipartite linker submodule."""

# import pytest
# import numpy as np
import networkx as nx
# import pandas as pd

from ...linker import bipartite as lb
from ... import utils


class SimpleBipartiteGraphTests:
    """Simple graphs and their expected matchings."""

    def setup(self):
        self.Graphs = []
        self.expected_matchings = []

        # Vazifeh Fig. 1, slightly modified to guarantee one solution.
        self.Graphs.append(nx.DiGraph())
        self.Graphs[-1].add_nodes_from('ABCDEFGHIJKLMNOPQRST')
        self.Graphs[-1].add_edges_from(
            [('A', 'B'), ('B', 'C'), ('C', 'H'), ('C', 'Q'), ('D', 'E'),
             ('E', 'C'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('G', 'I'),
             ('I', 'J'), ('I', 'K'), ('J', 'K'), ('L', 'N'), ('L', 'O'),
             ('M', 'O'), ('N', 'Q'), ('O', 'P'), ('O', 'Q'),
             ('Q', 'R', {'test_float': 489.6853}),
             ('Q', 'S'), ('R', 'S'), ('R', 'T'), ('S', 'T')])
        self.expected_matchings.append(
            {'S_i': 'R_o', 'C_i': 'B_o', 'G_i': 'F_o', 'P_i': 'O_o',
             'F_i': 'E_o', 'Q_i': 'N_o', 'N_i': 'L_o', 'J_i': 'I_o',
             'K_i': 'J_o', 'O_i': 'M_o', 'B_i': 'A_o', 'E_i': 'D_o',
             'R_i': 'Q_o', 'I_i': 'G_o', 'T_i': 'S_o', 'H_i': 'C_o',
             'Q_o': 'R_i', 'B_o': 'C_i', 'I_o': 'J_i', 'D_o': 'E_i',
             'C_o': 'H_i', 'J_o': 'K_i', 'M_o': 'O_i', 'L_o': 'N_i',
             'N_o': 'Q_i', 'G_o': 'I_i', 'R_o': 'S_i', 'S_o': 'T_i',
             'O_o': 'P_i', 'A_o': 'B_i', 'F_o': 'G_i', 'E_o': 'F_i'})

        # Extreme case of single path.
        self.Graphs.append(nx.DiGraph())
        self.Graphs[-1].add_nodes_from('ABCDE')
        self.Graphs[-1].add_edges_from(
            [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
        self.expected_matchings.append(
            {'B_i': 'A_o', 'C_i': 'B_o', 'D_i': 'C_o', 'E_i': 'D_o',
             'A_o': 'B_i', 'B_o': 'C_i', 'C_o': 'D_i', 'D_o': 'E_i'})

        # Extreme case of no connections.
        self.Graphs.append(nx.DiGraph())
        self.Graphs[-1].add_nodes_from('ABCDE')
        self.expected_matchings.append({})

        # Connections staggered in time.
        self.Graphs.append(nx.DiGraph())
        self.Graphs[-1].add_nodes_from('ABCDEF')
        self.Graphs[-1].add_edges_from(
            [('A', 'D'), ('A', 'E'), ('A', 'F'), ('B', 'E'),
             ('B', 'F'), ('C', 'F')])
        self.expected_matchings.append(
            {'D_i': 'A_o', 'E_i': 'B_o', 'F_i': 'C_o',
             'A_o': 'D_i', 'B_o': 'E_i', 'C_o': 'F_i'})

    @staticmethod
    def get_bipartite_graph(G):
        """Convert a digraph to bipartite.

        Based on functions early in triplinker's development.
        """
        Gb = nx.DiGraph()
        nodesb_top = []
        nodesb_bottom = []

        for item in G.nodes:
            nodesb_top.append(str(item) + '_i')
            nodesb_bottom.append(str(item) + '_o')
        Gb.add_nodes_from(nodesb_top, bipartite=0)
        Gb.add_nodes_from(nodesb_bottom, bipartite=1)

        for edge in G.edges:
            Gb.add_edge(str(edge[0]) + '_o', str(edge[1]) + '_i',
                        **G.edges[edge])

        return Gb, nodesb_top

    @staticmethod
    def get_digraph_from_matching(matching, G):
        node_dtype = next(iter(G.nodes)).__class__
        directed_matching = []
        seen_keys = []
        for key in matching.keys():
            if matching[key] not in seen_keys:
                seen_keys.append(key)
                if key[-2:] == "_o":
                    cedge = (node_dtype(key[:-2]),
                             node_dtype(matching[key][:-2]))
                else:
                    cedge = (node_dtype(matching[key][:-2]),
                             node_dtype(key[:-2]))
                directed_matching.append(cedge + (G.edges[cedge].copy(),))

        Gsoln = nx.DiGraph()
        Gsoln.add_nodes_from(G.nodes)
        Gsoln.add_edges_from(directed_matching)

        return Gsoln, directed_matching


class TestBipartiteLinkerBase(SimpleBipartiteGraphTests):
    """Test bipartite linker base module."""

    def test_digraph_to_bipartite(self):
        base = lb.BipartiteLinkerBase()
        for i in range(len(self.Graphs)):
            Gb_ref, topnodes_ref = self.get_bipartite_graph(self.Graphs[i])
            Gb, topnodes = base.digraph_to_bipartite(self.Graphs[i])
            assert utils.graphs_equivalent(Gb_ref, Gb)
            assert topnodes_ref == topnodes

    def test_bipartite_to_digraph(self):
        base = lb.BipartiteLinkerBase()
        for i in range(len(self.Graphs)):
            Gsoln_ref, dmatch_ref = self.get_digraph_from_matching(
                self.expected_matchings[i], self.Graphs[i])
            Gsoln, dmatch = base.matching_to_digraph(
                self.Graphs[i], self.expected_matchings[i],
                return_matching=True)
            assert utils.graphs_equivalent(Gsoln_ref, Gsoln)
            assert dmatch_ref == dmatch


class TestMaxCardinalityLinkerSimple(SimpleBipartiteGraphTests):
    """Test maximum cardinality linker module on simple graphs."""

    def test_maxcardinality_matching(self):
        linker = lb.MaxCardinalityLinker()
        for i in range(len(self.Graphs)):
            Gsoln_ref, _dummy = self.get_digraph_from_matching(
                self.expected_matchings[i], self.Graphs[i])
            assert utils.graphs_equivalent(
                Gsoln_ref, linker.get_solution(self.Graphs[i]))


class TestMaxCardinalityLinkerProduction:
    # TO DO: add production-level tests utilizing Austin data.  These won't be able
    # to test
    pass
