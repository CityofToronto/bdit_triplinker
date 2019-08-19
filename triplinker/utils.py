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


def get_paths(G):
    """Retrieves the set of all paths in a solution graph.

    Parameters
    ----------
    G : networkx.classes.digraph.DiGraph
        Directed graph representing the trip linking solution, eg. from
        `get_vazifeh_solution`.

    Returns
    -------
    paths : list of lists
        Paths in the path cover.  Each path is a list of nodes, whose order is
        that from the start to the end of the path.
    """
    # Function should be O(N) since it uses a dict to hash node ids.
    remaining_nodes = dict(zip(list(G.nodes),
                               [None for i in range(G.number_of_nodes())]))
    paths = []

    while len(remaining_nodes):
        # (Re-)set current path.
        current_path = []
        # Use "first" node from remaining nodes.
        cnode = next(iter(remaining_nodes))

        # If the node is not at the start of the path, walk backwards until
        # we reach the start, recording nodes in reverse order.
        if len(G.pred[cnode]):
            # Store the node we found - we'll walk forward from it later.
            onode = cnode
            cnode = next(G.predecessors(cnode))
            while True:
                current_path.append(cnode)
                del remaining_nodes[cnode]
                if not len(G.pred[cnode]):
                    break
                # Assumes there's only one predecessor, so can just do next().
                cnode = next(G.predecessors(cnode))
            cnode = onode
            # Make the node order consistent with the path direction.
            current_path = current_path[::-1]
        # Now walk forward until we reach the path's end.
        while True:
            current_path.append(cnode)
            del remaining_nodes[cnode]
            if not len(G.succ[cnode]):
                break
            # Assumes there's only one successor, so can just do next().
            cnode = next(G.successors(cnode))

        paths.append(current_path)

    return paths
