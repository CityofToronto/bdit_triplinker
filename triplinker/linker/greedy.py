"""
Methods for linking ridesourcing trips together.
"""
import networkx as nx
import numpy as np
import pandas as pd
import tqdm


class GreedyLinker:
    """Greedy trip linker.

    Parameters
    ----------
    whichgreed : str, optional
        Either 'passenger' or 'driver'.  Default: 'passenger'.
    mintype : str, optional
        Value to greedily minimize.  Default: 'deadhead_time'.

    Notes
    -----
    Routine is O(N_edges), since it scans through all edges once, though this
    can't feasibly be slower than making the graph in the first place.
    """

    def __init__(self, whichgreed='passenger', mintype='deadhead_time'):
        if whichgreed == 'passenger':
            self._passgreed = True
        elif whichgreed == 'driver':
            self._passgreed = False
        else:
            return ValueError("'whichgreed' must be 'passenger' or 'driver'.")

        self.mintype = mintype

    @property
    def whichgreed(self):
        return "Passenger" if self._passgreed else "Driver"
    
    def _get_greedy_min_edge(self, c_edges, Gw):
        """Determines smallest feasible edge in Gw."""

        min_edge = c_edges[0]
        min_val = Gw.edges[min_edge][self.mintype]
        for edge in c_edges[1:]:
            # "<" handles ties by ignoring them.
            if Gw.edges[edge][self.mintype] < min_val:
                min_edge = edge
                min_val = Gw.edges[edge][self.mintype]

        return min_edge

    def get_solution(self, G, time_ordered_nodes):
        """Get a linking solution.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Digraph where trips are nodes and feasible trip links are edges.
        time_ordered_nodes : listlike
            Nodes ordered by either their drop-off or pick-up time.

        Returns
        -------
        Gsoln : networkx.classes.digraph.DiGraph
            Digraph containing the solution.
        """
        # Copy G, since we'll be destroying it.
        Gw = G.copy()

        # Create the solution and just add the nodes from the working graph.
        Gsoln = nx.DiGraph()
        Gsoln.add_nodes_from(Gw)

        # Using the time-ordered nodes list, scan each node's outgoing edges to
        # determine lowest-cost link.  Record this link, then remove all
        # incident edges to the succeeding node (so no other connections can be
        # made to it).
        for node in time_ordered_nodes:
            if self._passgreed:
                c_edges = list(Gw.in_edges(node))
            else:
                c_edges = list(Gw.out_edges(node))
            if len(c_edges):
                min_edge = get_greedy_min_edge(c_edges, Gw, mintype)
                # Add connection to solution.
                Gsoln.add_edge(*min_edge, **Gw.edges[min_edge])
                # We won't be going back to node, so we can freely delete all links
                # to node given by edge[1].
                if self._passgreed:
                    Gw.remove_edges_from(list(Gw.out_edges(min_edge[0])))
                else:
                    Gw.remove_edges_from(list(Gw.in_edges(min_edge[1])))

        return Gsoln
