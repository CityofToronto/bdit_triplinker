"""
Linking methods that use Google's OR-Tools package
(https://developers.google.com/optimization/).
"""
import numpy as np
from ortools.graph import pywrapgraph as orgraph
from ortools.linear_solver import pywraplp as orls

from .bipartite import BipartiteLinkerBase


class ORLinkerBase(BipartiteLinkerBase):
    """Base class for OR-Tools linkers."""

    _weight_name = None

    def __init__(self, units):
        if units not in ['min', 'sec']:
            raise ValueError('units not recognized!')
        self.units = units
        self._modifier = 60. if self.units == 'min' else 1.

    def get_weighted_graph(self, G):
        raise NotImplementedError

    def get_nodes_and_converters(self, nodesb_top):
        # Top nodes are pickups, bottom nodes dropoffs.
        self._dest_nodes = nodesb_top
        self._or_nodes = ['{0}_o'.format(node[:-2])
                          for node in self._dest_nodes]
        self._all_nodes = self._or_nodes + self._dest_nodes


class MinWeightMaxCardinalityLinker(ORLinkerBase):
    """Minimum weight maximum cardinality linking solution.

    Casts the problem as a minimum cost flow problem following
    https://developers.google.com/optimization/assignment/assignment_min_cost_flow
    Uses OR-Tools's `SimpleMinCostFlow`.
    """
    # This is a class and not a function mainly for hackability and testing
    # purposes.

    min_cost_flow_messages = {
        orgraph.SimpleMinCostFlow.BAD_COST_RANGE: 'BAD_COST_RANGE',
        orgraph.SimpleMinCostFlow.BAD_RESULT: 'BAD_RESULT',
        orgraph.SimpleMinCostFlow.FEASIBLE: 'FEASIBLE',
        orgraph.SimpleMinCostFlow.INFEASIBLE: 'INFEASIBLE',
        orgraph.SimpleMinCostFlow.NOT_SOLVED: 'NOT_SOLVED',
        orgraph.SimpleMinCostFlow.OPTIMAL: 'OPTIMAL',
        orgraph.SimpleMinCostFlow.UNBALANCED: 'UNBALANCED',
    }

    _weight_name = 'minimizer'

    def get_weighted_graph(self, G):

        self.Gw = G.copy()

        for edge in self.Gw.edges():
            self.Gw.edges[edge][self._weight_name] = (
                int(np.round(self._modifier *
                             self.Gw.edges[edge]['overhead_time'])))

    def get_max_cardinality(self):
        self.max_card = lb.get_vazifeh_solution(self.Gw).number_of_edges()

    def get_nodes_and_converters(self, nodesb_top):
        super().get_nodes_and_converters(nodesb_top)

        self.nodes_idx = dict(
            [(node, i + 1) for i, node in enumerate(self._all_nodes)])
        self.idx_nodes = dict(
            [(i + 1, node) for i, node in enumerate(self._all_nodes)])

    def solve_flow(self, Gdb):
        # Deploy min cost flow instance.
        self.min_cost_flow = orgraph.SimpleMinCostFlow()

        # Add arcs in the bipartite graph.
        for edge in Gdb.edges():
            self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                self.nodes_idx[edge[0]], self.nodes_idx[edge[1]], 1,
                Gdb.edges[edge][self._weight_name])

        # Add source arcs.
        for node in self._or_nodes:
            self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                0, self.nodes_idx[node], 1, 0)

        # Add sink arcs.
        sink_idx = Gdb.number_of_nodes() + 1
        for node in self._dest_nodes:
            self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                self.nodes_idx[node], sink_idx, 1, 0)

        # Set supply at source and sink to max cardinality value, zero
        # elsewhere.
        for node in self._all_nodes:
            self.min_cost_flow.SetNodeSupply(self.nodes_idx[node], 0)
        self.min_cost_flow.SetNodeSupply(0, self.max_card)
        self.min_cost_flow.SetNodeSupply(sink_idx, -self.max_card)

        solve_status = self.min_cost_flow.Solve()

        if solve_status != self.min_cost_flow.OPTIMAL:
            raise ValueError('solve_status is {0}'.format(
                self.min_cost_flow_messages[solve_status]))

    def flow_soln_to_matching(self):
        # Timing tests suggest it's slightly cheaper to append to dicts
        # than append to lists and cast to dict later.
        matching = {}
        # `solve_flow` sets the largest node to be the sink.
        sink_idx = self.min_cost_flow.NumNodes() - 1

        for arc in range(self.min_cost_flow.NumArcs()):
            if ((self.min_cost_flow.Tail(arc) != 0) and
                    (self.min_cost_flow.Head(arc) != sink_idx) and
                    (self.min_cost_flow.Flow(arc) > 0)):
                matching[self.idx_nodes[self.min_cost_flow.Tail(arc)]] = (
                    self.idx_nodes[self.min_cost_flow.Head(arc)])

        return matching

    def get_solution(self, G, return_matching=False):
        """Obtains minimum weight maximum cardinality matching solution.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Feasible connections digraph.
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

        # Create a copy of the graph with relevant weights.
        self.get_weighted_graph(G)

        # Get a bipartite graph and obtain the maximum cardinality matching.
        # We'll need this number later.
        self.get_max_cardinality()

        # Get bipartite digraph.
        Gdb, nodesb_top = lb.digraph_to_bipartite(
            self.Gw, return_digraph=True)

        # Converters for node names to numerical indices.
        self.get_nodes_and_converters(nodesb_top)

        # Solve.
        self.solve_flow(Gdb)

        # Get matching.
        matching = self.flow_soln_to_matching()

        return lb.matching_to_digraph(
            self.Gw, matching, return_matching=return_matching)


class MinWeightMaximalLinker(ORLinkerBase):
    """Minimum weight maximal matching linker.

    Casts the problem as a mixed-integer programming one following
    https://developers.google.com/optimization/assignment/assignment_mip .
    Uses OR-Tools's `pywraplp.Solver`.
    """
    # This is a class and not a function mainly for hackability and testing
    # purposes.

    mip_solver_messages = {
        orls.Solver.ABNORMAL: 'ABNORMAL',
        orls.Solver.FEASIBLE: 'FEASIBLE',
        orls.Solver.INFEASIBLE: 'INFEASIBLE',
        orls.Solver.NOT_SOLVED: 'NOT_SOLVED',
        orls.Solver.OPTIMAL: 'OPTIMAL',
        orls.Solver.UNBOUNDED: 'UNBOUNDED'
    }

    _weight_name = 'maximizer'

    def get_weighted_graph(self, G, max_weight=None):
        self.Gw = G.copy()

        # Find maximum weight in entire graph.
        if max_weight is None:
            max_weight = -1.
            for edge in self.Gw.edges():
                if self.Gw.edges[edge]['overhead_time'] > max_weight:
                    max_weight = self.Gw.edges[edge]['overhead_time']
        max_weight = np.round(max_weight * self._modifier)

        # Save floating point maximizer weights (converted to ints later).
        for edge in self.Gw.edges():
            self.Gw.edges[edge][self._weight_name] = (
                max_weight -
                np.round(self._modifier *
                         self.Gw.edges[edge]['overhead_time']))

    def get_gain_matrix(self, Gdb):
        """Generates a (floating point) gain matrix.

        gain[or_node, dest_node] = gain of link.  Invalid links are represented
        with `numpy.nan`.
        """
        self.gain_mtx = np.nan * np.ones(
            [len(self._or_nodes), len(self._dest_nodes)])

        for i, node_o in enumerate(self._or_nodes):
            for j, node_d in enumerate(self._dest_nodes):
                if Gdb.has_edge(node_o, node_d):
                    self.gain_mtx[i, j] = (
                        Gdb.edges[node_o, node_d][self._weight_name])

    def solve_mip(self):

        self.solver = orls.Solver('SolveAssignmentProblemMIP',
                                  orls.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        sum_list = []
        self._x = {}
        for i in range(self.gain_mtx.shape[0]):
            for j in range(self.gain_mtx.shape[1]):
                if not np.isnan(self.gain_mtx[i, j]):
                    self._x[i, j] = self.solver.BoolVar(
                        "x[{0:d},{1:d}]".format(i, j))
                    sum_list.append(self.gain_mtx[i, j] * self._x[i, j])

        # Declare objective.
        self.solver.Maximize(self.solver.Sum(sum_list))

        # Each drop-off is assigned at most 1 pick-up.
        for i in range(self.gain_mtx.shape[0]):
            dropoff_edges = []
            for j in range(self.gain_mtx.shape[1]):
                if not np.isnan(self.gain_mtx[i, j]):
                    dropoff_edges.append(self._x[i, j])
            self.solver.Add(self.solver.Sum(dropoff_edges) <= 1)

        # Each pick-up is assigned at most 1 drop-off.
        for j in range(self.gain_mtx.shape[1]):
            pickup_edges = []
            for i in range(self.gain_mtx.shape[0]):
                if not np.isnan(self.gain_mtx[i, j]):
                    pickup_edges.append(self._x[i, j])
            self.solver.Add(self.solver.Sum(pickup_edges) <= 1)

        solve_status = self.solver.Solve()

        if solve_status != self.solver.OPTIMAL:
            raise ValueError('solve_status is {0}'.format(
                self.mip_solver_messages[solve_status]))

        self.total_cost = self.solver.Objective().Value()

    def mip_soln_to_matching(self):
        # Timing tests suggest it's slightly cheaper to append to dicts
        # than append to lists and cast to dict later.
        matching = {}
        for link in self._x.keys():
            if self._x[link].solution_value() > 0.:
                matching[self._or_nodes[link[0]]] = (
                    self._dest_nodes[link[1]])
        return matching

    def get_solution(self, G, max_weight=None, return_matching=False):
        """Obtains maximum weight matching solution.

        Parameters
        ----------
        G : networkx.classes.digraph.DiGraph
            Feasible connections digraph.
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
        # Create a copy of the graph with relevant weights.
        self.get_weighted_graph(G, max_weight=max_weight)

        # Get bipartite digraph.
        Gdb, nodesb_top = lb.digraph_to_bipartite(
            self.Gw, return_digraph=True)

        # Converters for node names to numerical indices, for generating
        # matching from solution.
        self.get_nodes_and_converters(nodesb_top)

        # Get gain matrix.
        self.get_gain_matrix(Gdb)

        # Solve.
        self.solve_mip()

        # Get matching.
        matching = self.mip_soln_to_matching()

        return lb.matching_to_digraph(
            self.Gw, matching, return_matching=return_matching)
