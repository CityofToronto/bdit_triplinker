"""Pytest scripts to test misc utilities."""

import networkx as nx
from .. import utils


class TestUtils:
    """Tests misc utilities."""

    def test_graphs_equivalent(self):

        # Vazifeh Fig. 1, slightly modified to guarantee one solution.
        digraph = nx.DiGraph()
        digraph.add_nodes_from('ABCDEFG')
        digraph.add_edges_from(
            [('A', 'B'), ('B', 'C', {'test_float': 0.23289491891}),
             ('C', 'G'), ('C', 'F'), ('D', 'E')])

        # Copies of digraph with small changes.
        digraph_different1 = digraph.copy()
        digraph_different1.remove_edge('B', 'C')
        digraph_different2 = digraph.copy()
        digraph_different2.remove_node('F')
        digraph_different3 = digraph.copy()
        digraph_different3.edges['B', 'C']['test_float'] = 0.232889

        # Newly made graph, but with exact same nodes and links.
        digraph_eqv = nx.DiGraph()
        digraph_eqv.add_nodes_from('ABCDEFG')
        digraph_eqv.add_edges_from(
            [('A', 'B'), ('B', 'C', {'test_float': 0.23289491997}),
             ('C', 'G'), ('C', 'F'), ('D', 'E')])

        assert utils.graphs_equivalent(digraph, digraph.copy())
        assert utils.graphs_equivalent(digraph, digraph_eqv)
        assert not utils.graphs_equivalent(digraph, digraph_different1)
        assert not utils.graphs_equivalent(digraph, digraph_different2)
        assert not utils.graphs_equivalent(digraph, digraph_different3)

        # Extreme example of empty graph.
        empty_graph = nx.Graph()
        assert utils.graphs_equivalent(empty_graph, empty_graph.copy())

    def test_find_paths(self):

        # Test get_paths on simple paths.  Originally these were linked to
        # TestMaxCardinalityLinkerSimple, but moved here during refactoring.
        Graphs = []
        expected_paths = []

        # Vazifeh Fig. 1, slightly modified to guarantee one solution.
        Graphs.append(nx.DiGraph())
        Graphs[-1].add_nodes_from('ABCDEFGHIJKLMNOPQRST')
        Graphs[-1].add_edges_from(
            [('A', 'B'), ('B', 'C'), ('C', 'H'), ('D', 'E'), ('E', 'F'),
             ('F', 'G'), ('G', 'I'), ('I', 'J'), ('J', 'K'), ('L', 'N'),
             ('M', 'O'), ('N', 'Q'), ('O', 'P'),
             ('Q', 'R', {'test_float': 489.6853}), ('R', 'S'), ('S', 'T')])
        expected_paths.append(
            [['A', 'B', 'C', 'H'],
             ['D', 'E', 'F', 'G', 'I', 'J', 'K'],
             ['L', 'N', 'Q', 'R', 'S', 'T'],
             ['M', 'O', 'P']])

        # Extreme case of single path.
        Graphs.append(nx.DiGraph())
        Graphs[-1].add_nodes_from('ABCDE')
        Graphs[-1].add_edges_from(
            [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
        expected_paths.append(
            [['A', 'B', 'C', 'D', 'E'], ])

        # Extreme case of no connections.
        Graphs.append(nx.DiGraph())
        Graphs[-1].add_nodes_from('ABCDE')
        expected_paths.append(
            [['A', ], ['B', ], ['C', ], ['D', ], ['E', ]])

        # Connections staggered in time.
        Graphs.append(nx.DiGraph())
        Graphs[-1].add_nodes_from('ABCDEF')
        Graphs[-1].add_edges_from(
            [('A', 'D'), ('B', 'E'), ('C', 'F')])
        expected_paths.append(
            [['A', 'D'], ['B', 'E'], ['C', 'F']])

        for i in range(len(Graphs)):
            paths = utils.get_paths(Graphs[i])
            # Path order depends on node iterator order, which appears to be
            # system/OS dependent, so compare sorted paths.
            assert sorted(paths) == sorted(expected_paths[i])
