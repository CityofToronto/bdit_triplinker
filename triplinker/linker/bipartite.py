"""Linking methods that transform the graph into a bipartite one."""
import networkx as nx


class BipartiteLinkerBase:
    """Base class for bipartite linking algorithms."""

    @staticmethod
    def digraph_to_bipartite(G):
        """Convert digraph G into a bipartite graph Gb.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Digraph.

        Returns
        -------
        Gb : networkx.classes.graph.DiGraph
            Bipartite digraph.
        """
        Gb = nx.DiGraph()

        # Need to return nodesb_top for matching purposes.
        nodesb_top = map(lambda node: str(node) + '_i', G.nodes())
        nodesb_bottom = map(lambda node: str(node) + '_o', G.nodes())
        Gb.add_nodes_from(nodesb_top, bipartite=0)
        Gb.add_nodes_from(nodesb_bottom, bipartite=1)

        edgesb = map(lambda edge: (
            str(edge[0]) + '_o', str(edge[1]) + '_i',
            G.edges[edge].copy()), G.edges())
        Gb.add_edges_from(edgesb)

        return Gb
    
    @staticmethod
    def get_top_nodes(Gb):
        """Get "top" nodes (departing links only) of bipartite digraph."""
        return list(
            sorted(filter(lambda node: node.endswith('_i'), Gb.nodes())))

    @staticmethod
    def get_directed_matching(Gsoln):
        return sorted(list(Gsoln.edges()))

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

        if return_matching:
            return Gsoln, directed_matching
        return Gsoln


class MaxCardinalityLinker(BipartiteLinkerBase):
    """Unweighted maximum cardinality linker.

    This is the algorithm used by Vazifeh et al. 2018.

    Notes
    -----
    Unlike `linker.MinWeightMaxCardinalityLinker`, this class simply returns
    *one* maximum cardinality solution, rather than the one with minimum total
    weight.

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
        Gb = self.digraph_to_bipartite(G)
        nodesb_top = self.get_top_nodes(Gb)
        # H-K and Eppstein matching algos require undirected graph, or will
        # fail silently and return an empty set.
        # https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.bipartite.matching.hopcroft_karp_matching.html
        matching = nx.bipartite.maximum_matching(
            Gb.to_undirected(), top_nodes=nodesb_top)
        return self.matching_to_digraph(G, matching,
                                        return_matching=return_matching)
