"""
Linking methods that require transforming the digraph into a bipartite graph.
"""
import networkx as nx
import numpy as np
import pandas as pd
import tqdm


class BipartiteLinkerBase:
    """Base class for bipartite linking algorithms."""

    @staticmethod
    def digraph_to_bipartite(G, return_top_nodes=True, return_digraph=False):
        """Convert digraph G into a bipartite graph Gb.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Digraph.
        return_top_nodes : bool, optional
            If `True` (default), returns "top" nodes of bipartite graph as well
            as digraph.  These are needed to resolve ambiguities when running
            networkx.bipartite.maximum_matching.
        return_digraph : bool, optional
            If `True`, creates a bipartite digraph.  Default: `False`.

        Returns
        -------
        Gb : networkx.classes.graph.Graph
            (Undirected) bipartite graph.
        nodesb_top : list
            "Top" nodes of bipartite graph.

        Notes
        -----
        By default, graph is undirected for compatibility with networkx's
        Hopcroft-Karp and Eppstein matching algorithms.  See
        https://networkx.github.io/documentation/stable/reference/algorithms/bipartite.html
        """
        if return_digraph:
            Gb = nx.DiGraph()
        else:
            Gb = nx.Graph()
        nodesb_top = []
        nodesb_bottom = []
        for item in list(G.nodes):
            nodesb_top.append(str(item) + '_i')
            nodesb_bottom.append(str(item) + '_o')
        Gb.add_nodes_from(nodesb_top, bipartite=0)
        Gb.add_nodes_from(nodesb_bottom, bipartite=1)

        edgesb = []
        for node, links in G.adjacency():
            if len(links):
                for link in links.keys():
                    edgesb.append((str(node) + '_o', str(link) + '_i',
                                   links[link].copy()))
        Gb.add_edges_from(edgesb)

        if return_top_nodes:
            return Gb, nodesb_top
        return Gb

    @staticmethod
    def matching_to_digraph(G, matching, return_matching=False,
                            check_degree=False):
        """Map matching bipartite solution back to digraph.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Original digraph.
        matching : dict
            Dictionary of {node_from: node_to, ...} indices, as output from
            networkx.bipartite.maximum_matching.
        return_matching : bool, optional
            If `True`, returns directed matching as well as digraph.
            Default: `False`.
        check_degree: bool, optional
            If `True`, checks that nodes in the digraph solution have at most
            degree 2 (one incoming, one outgoing edge).  Default: `False`.

        Returns
        -------
        Gsoln : networkx.classes.digraph.DiGraph
            Digraph containing the solution.
        directed_matching : list
            List of edges in [(node_from, node_to), ...] format.  Only returned
            if `return_matching` is `True`.
        """
        # Convert the node indices to the digraph's native dtype.
        random_node = next(iter(G.nodes))
        # Converting str to str should take an insiginficant amount of time.
        node_type = random_node.__class__

        # Store a set of all unique links in `directed_matching`, stripping the
        # `_i`, # `_o` suffixes in the process.
        directed_matching = []
        seen_keys = []
        for key in matching.keys():
            if matching[key] not in seen_keys:
                seen_keys.append(key)
                if key[-2:] == "_o":
                    cedge = (node_type(key[:-2]), node_type(matching[key][:-2]))
                else:
                    cedge = (node_type(matching[key][:-2]), node_type(key[:-2]))
                directed_matching.append(cedge + (G.edges[cedge].copy(),))

        # Create a new digraph from the nodes of the old, and implement the
        # link solution.
        Gsoln = nx.DiGraph()
        Gsoln.add_nodes_from(G)
        Gsoln.add_edges_from(directed_matching)

        if check_degree:
            for node in Gsoln.nodes():
                assert Gsoln.degree(node) <= 2, (
                    "node {0} has degree {1}".format(node, Gsoln.degree(node)))

        if return_matching:
            return Gsoln, directed_matching
        return Gsoln


class MaxCardinalityLinker(BipartiteLinkerBase):
    """Unweighted maximum cardinality linker.

    This is the algorithm used by Vazifeh et al. 2018.
    """

    def get_solution(self, G, return_matching=False):
        """Get a linking solution.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Digraph where trips are nodes and feasible trip links are edges.
        return_matching : bool
            If `True`, returns directed matching as well as digraph.
            Default: `False`.

        Returns
        -------
        Gsoln : networkx.classes.digraph.DiGraph
            Digraph containing the solution.
        directed_matching : list
            List of edges in [(node_from, node_to), ...] format.  Only
            returned if `return_matching` is `True`.
        """
        Gb, nodesb_top = self.digraph_to_bipartite(G)
        matching = nx.bipartite.maximum_matching(Gb, top_nodes=nodesb_top)
        return self.matching_to_digraph(G, matching,
                                        return_matching=return_matching)
