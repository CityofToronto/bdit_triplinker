"""
Utilities supporting the other modules and test suite.
"""
import numpy as np


def graphs_equivalent(graph1, graph2):
    """Checks if two graphs are equivalent by comparing their nodes, edges and
    values stored along their edges.
    """
    if (sorted(list(graph1.nodes)) != sorted(list(graph2.nodes)) or
            sorted(list(graph1.edges)) != sorted(list(graph2.edges))):
        return False
    edge_list = sorted(graph1.edges)
    graph1_edgevals = []
    graph2_edgevals = []
    # This is 1.2 - 2x faster than creating a list-of-lists or multi-d array.
    for edge in edge_list:
        graph1_cedge = graph1.edges[edge]
        graph2_cedge = graph2.edges[edge]
        graph1_edgevals += [graph1_cedge[key]
                            for key in sorted(graph1_cedge.keys())]
        graph2_edgevals += [graph2_cedge[key]
                            for key in sorted(graph2_cedge.keys())]
    return np.allclose(graph1_edgevals, graph2_edgevals, rtol=1e-6, atol=1e-8)


def graphs_equivalent_withadj(graph1, graph2):
    """Checks if two graphs are equivalent using their adjacency.

    Adjacency returns every node, each of its connections, and any object
    associated with the connection.  For our purposes two graphs are the same
    if these are all the same.  Such a comparison will fail if weights are
    floating point, in which case one should use the slower `graphs_equivalent`.
    """
    # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.adjacency.html
    return list(sorted(graph1.adjacency())) == list(sorted(graph2.adjacency()))
