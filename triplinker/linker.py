"""
Methods for linking ride-hailing trips together.
"""
import networkx as nx
import numpy as np
import pandas as pd
import tqdm


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

    # Create a new digraph from the nodes of the old, and implement the link
    # solution.
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


def get_vazifeh_solution(G, return_matching=False):
    """Trip linker based on Vazifeh et al. 2018.

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
    Gb, nodesb_top = digraph_to_bipartite(G)
    matching = nx.bipartite.maximum_matching(Gb, top_nodes=nodesb_top)
    return matching_to_digraph(G, matching, return_matching=return_matching)


def get_greedy_min_edge(c_edges, Gw, mintype):
    """Determines smallest feasible edge, for loop in greedy solution algos."""

    min_edge = c_edges[0]
    min_val = Gw.edges[min_edge][mintype]
    for edge in c_edges[1:]:
        # "<" handles ties by ignoring them.
        if Gw.edges[edge][mintype] < min_val:
            min_edge = edge
            min_val = Gw.edges[edge][mintype]

    return min_edge


def get_greedy_solution(G, time_ordered_nodes, mintype='deadhead_time'):
    """Trip linker that links dropoff points to the nearest pickup point in
    travel time.

    Algorithm is greedy, and dropoff points earlier in time are linked before
    later ones.  Routine is O(N_edges), since it scans through all edges once,
    though this can't feasibly be slower than making the graph in the first
    place.

    Parameters
    ----------
    G : networkx.classes.digraph.DiGraph
        Digraph where trips are nodes and feasible trip links are edges.
    time_ordered_nodes : listlike
        Nodes ordered by their dropoff time.
    mintype : str, optional
        Value to greedily minimize.  Default: 'deadhead_time'.

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
    # determine lowest-cost link.  Record this link, then remove all incident
    # edges to the succeeding node (so no other connections can be made to it).
    for node in time_ordered_nodes:
        c_edges = list(Gw.out_edges(node))
        if len(c_edges):
            min_edge = get_greedy_min_edge(c_edges, Gw, mintype)
            # Add connection to solution.
            Gsoln.add_edge(*min_edge, **Gw.edges[min_edge])
            # We won't be going back to node, so we can freely delete all links
            # to node given by edge[1].
            Gw.remove_edges_from(list(Gw.in_edges(min_edge[1])))

    return Gsoln


def get_greedy_passenger_solution(G, pickup_time_ordered_nodes,
                                  mintype='deadhead_time'):
    """Trip linker that links dropoff points to the nearest pickup point in
    travel time.

    Algorithm is greedy, and dropoff points earlier in time are linked before
    later ones.  Routine is O(N_edges), since it scans through all edges once,
    though this can't feasibly be slower than making the graph in the first
    place.

    Parameters
    ----------
    G : networkx.classes.digraph.DiGraph
        Digraph where trips are nodes and feasible trip links are edges.
    pickup_time_ordered_nodes : listlike
        Nodes ordered by their pickup time.
    mintype : str, optional
        Value to greedily minimize.  Default: 'deadhead_time'.

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
    # determine lowest-cost link.  Record this link, then remove all incident
    # edges to the succeeding node (so no other connections can be made to it).
    for node in pickup_time_ordered_nodes:
        c_edges = list(Gw.in_edges(node))
        if len(c_edges):
            min_edge = get_greedy_min_edge(c_edges, Gw, mintype)
            # Add connection to solution.
            Gsoln.add_edge(*min_edge, **Gw.edges[min_edge])
            # We won't be going back to node, so we can freely delete all links
            # to node given by edge[1].
            Gw.remove_edges_from(list(Gw.out_edges(min_edge[0])))

    return Gsoln


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


class BatchedLinkerGreedy(BatchedLinker):
    """Link trip in batches to simulate dynamic trip linking.  Uses
    greedy linker.

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
        time_ordered_nodes = list(
            self.df.loc[list(bin_graph.nodes()), ['dropoff_datetime']]
            .sort_values('dropoff_datetime').index)
        return get_greedy_solution(bin_graph, time_ordered_nodes)


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
