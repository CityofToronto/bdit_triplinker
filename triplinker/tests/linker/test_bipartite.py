import pytest
# import numpy as np
# import networkx as nx
# import pandas as pd

# from ...data import AUSTIN_DATA

# class TestDelta:
#     def test_gamma_1(self):
#         print('\nIn test_gamma_1()')

#     def test_gamma_2(self, some_resource):
#         print('\nIn test_gamma_2()')

# class TestClass:

#     def test_method1(self, austin_data):
#         assert austin_data['df'].shape[0] > 0



# class TestMinFleetLinker:
#     """Test linker module."""

#     def setup(self):
#         self.Graphs = []
#         self.expected_matchings = []
#         self.expected_paths = []

#         # Vazifeh Fig. 1, slightly modified to guarantee one solution.
#         self.Graphs.append(nx.DiGraph())
#         self.Graphs[-1].add_nodes_from('ABCDEFGHIJKLMNOPQRST')
#         self.Graphs[-1].add_edges_from(
#             [('A', 'B'), ('B', 'C'), ('C', 'H'), ('C', 'Q'), ('D', 'E'),
#              ('E', 'C'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('G', 'I'),
#              ('I', 'J'), ('I', 'K'), ('J', 'K'), ('L', 'N'), ('L', 'O'),
#              ('M', 'O'), ('N', 'Q'), ('O', 'P'), ('O', 'Q'),
#              ('Q', 'R', {'test_float': 489.6853}),
#              ('Q', 'S'), ('R', 'S'), ('R', 'T'), ('S', 'T')])
#         self.expected_matchings.append(
#             {'S_i': 'R_o', 'C_i': 'B_o', 'G_i': 'F_o', 'P_i': 'O_o',
#              'F_i': 'E_o', 'Q_i': 'N_o', 'N_i': 'L_o', 'J_i': 'I_o',
#              'K_i': 'J_o', 'O_i': 'M_o', 'B_i': 'A_o', 'E_i': 'D_o',
#              'R_i': 'Q_o', 'I_i': 'G_o', 'T_i': 'S_o', 'H_i': 'C_o',
#              'Q_o': 'R_i', 'B_o': 'C_i', 'I_o': 'J_i', 'D_o': 'E_i',
#              'C_o': 'H_i', 'J_o': 'K_i', 'M_o': 'O_i', 'L_o': 'N_i',
#              'N_o': 'Q_i', 'G_o': 'I_i', 'R_o': 'S_i', 'S_o': 'T_i',
#              'O_o': 'P_i', 'A_o': 'B_i', 'F_o': 'G_i', 'E_o': 'F_i'})
#         self.expected_paths.append(
#             [['A', 'B', 'C', 'H'],
#              ['D', 'E', 'F', 'G', 'I', 'J', 'K'],
#              ['L', 'N', 'Q', 'R', 'S', 'T'],
#              ['M', 'O', 'P']])

#         # Extreme case of single path.
#         self.Graphs.append(nx.DiGraph())
#         self.Graphs[-1].add_nodes_from('ABCDE')
#         self.Graphs[-1].add_edges_from(
#             [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
#         self.expected_matchings.append(
#             {'B_i': 'A_o', 'C_i': 'B_o', 'D_i': 'C_o', 'E_i': 'D_o',
#              'A_o': 'B_i', 'B_o': 'C_i', 'C_o': 'D_i', 'D_o': 'E_i'})
#         self.expected_paths.append(
#             [['A', 'B', 'C', 'D', 'E'], ])

#         # Extreme case of no connections.
#         self.Graphs.append(nx.DiGraph())
#         self.Graphs[-1].add_nodes_from('ABCDE')
#         self.expected_matchings.append({})
#         self.expected_paths.append(
#             [['A', ], ['B', ], ['C', ], ['D', ], ['E', ]])

#         # Connections staggered in time.
#         self.Graphs.append(nx.DiGraph())
#         self.Graphs[-1].add_nodes_from('ABCDEF')
#         self.Graphs[-1].add_edges_from(
#             [('A', 'D'), ('A', 'E'), ('A', 'F'), ('B', 'E'),
#              ('B', 'F'), ('C', 'F')])
#         self.expected_matchings.append(
#             {'D_i': 'A_o', 'E_i': 'B_o', 'F_i': 'C_o',
#              'A_o': 'D_i', 'B_o': 'E_i', 'C_o': 'F_i'})
#         self.expected_paths.append(
#             [['A', 'D'], ['B', 'E'], ['C', 'F']])

#     @staticmethod
#     def get_bipartite_graph(G):
#         Gb = nx.Graph()
#         nodesb_top = []
#         nodesb_bottom = []

#         for item in G.nodes:
#             nodesb_top.append(str(item) + '_i')
#             nodesb_bottom.append(str(item) + '_o')
#         Gb.add_nodes_from(nodesb_top, bipartite=0)
#         Gb.add_nodes_from(nodesb_bottom, bipartite=1)

#         for edge in G.edges:
#             Gb.add_edge(str(edge[0]) + '_o', str(edge[1]) + '_i',
#                         **G.edges[edge])

#         return Gb, nodesb_top

#     @staticmethod
#     def get_digraph_from_matching(matching, G):
#         node_dtype = next(iter(G.nodes)).__class__
#         directed_matching = []
#         seen_keys = []
#         for key in matching.keys():
#             if matching[key] not in seen_keys:
#                 seen_keys.append(key)
#                 if key[-2:] == "_o":
#                     cedge = (node_dtype(key[:-2]),
#                              node_dtype(matching[key][:-2]))
#                 else:
#                     cedge = (node_dtype(matching[key][:-2]),
#                              node_dtype(key[:-2]))
#                 directed_matching.append(cedge + (G.edges[cedge].copy(),))

#         Gsoln = nx.DiGraph()
#         Gsoln.add_nodes_from(G.nodes)
#         Gsoln.add_edges_from(directed_matching)

#         return Gsoln, directed_matching

#     def test_digraph_to_bipartite(self):
#         for i in range(len(self.Graphs)):
#             Gb_ref, topnodes_ref = self.get_bipartite_graph(self.Graphs[i])
#             Gb, topnodes = linker.digraph_to_bipartite(self.Graphs[i],
#                                                        return_top_nodes=True)
#             assert utils.graphs_equivalent(Gb_ref, Gb)
#             assert topnodes_ref == topnodes

#     def test_bipartite_to_digraph(self):
#         for i in range(len(self.Graphs)):
#             Gsoln_ref, dmatch_ref = self.get_digraph_from_matching(
#                 self.expected_matchings[i], self.Graphs[i])
#             Gsoln, dmatch = linker.matching_to_digraph(
#                 self.Graphs[i], self.expected_matchings[i],
#                 return_matching=True)
#             assert utils.graphs_equivalent(Gsoln_ref, Gsoln)
#             assert dmatch_ref == dmatch

#     def test_matching_and_paths(self):
#         for i in range(len(self.Graphs)):
#             Gsoln_ref, dmatch_ref = self.get_digraph_from_matching(
#                 self.expected_matchings[i], self.Graphs[i])
#             Gsoln = linker.get_vazifeh_solution(self.Graphs[i],
#                                                 return_matching=False)
#             assert utils.graphs_equivalent(Gsoln_ref, Gsoln)

#             paths = linker.get_paths(Gsoln)
#             # Path order depends on node iterator order, which appears to be
#             # system/OS dependent, so compare sorted paths.
#             assert sorted(paths) == sorted(self.expected_paths[i])


# class TestLinkerdfBase:

#     def setup(self):
#         df = pd.read_csv(AUSTIN_RIDESHARE_TRIPDATA,
#                          parse_dates=[1, 2, 5, 6, 9])
#         # Only consider 6 - 7:30 PM to make tests go faster.
#         df.drop(labels=['ptctripid', 'request_datetime',
#                         'driver_accept_datetime',
#                         'request_latitude', 'request_longitude',
#                         'driver_reach_datetime',
#                         'trip_distance', 'driver_id'], axis=1, inplace=True)
#         df = df.loc[
#             ((df['pickup_datetime'] >= '2017-03-15 18:00:00') &
#              (df['pickup_datetime'] < '2017-03-15 19:30:00')), :]

#         gridangle_austin = (-97.746966, 30.267280, -97.741567, 30.281615)
#         gphr = grapher.TripGrapherManhattan(
#             df, pd.Timedelta('20 minutes'), gridangle=gridangle_austin,
#             speed=7.56)
#         net = gphr.create_graph(max_connections=30)

#         self.df = df
#         self.net = net
#         self.gphr = gphr


# class TestGreedyLinker(TestLinkerdfBase):
#     """Test greedy linker function."""

#     @pytest.mark.parametrize(('mintype'), ('deadhead_time', 'overhead_time'))
#     def test_get_greedy_solution(self, mintype):
#         time_ordered_nodes = list(
#             self.df['dropoff_datetime'].sort_values().index)

#         # Get greedy solution from linker.
#         greedy_soln = linker.get_greedy_solution(self.net, time_ordered_nodes,
#                                                  mintype=mintype)

#         # Algorithm to ensure all nodes have optimally greedy links.
#         for node in time_ordered_nodes:
#             # For each node, start by getting all destination nodes.
#             to_node = np.zeros(len(self.net.out_edges(node)), dtype=int)
#             weights = np.zeros(len(self.net.out_edges(node)))
#             for i, edge in enumerate(self.net.out_edges(node)):
#                 to_node[i] = edge[1]
#                 weights[i] = self.net.edges[edge][mintype]
#             # Sort the destination node IDs and distances from nearest to
#             # furthest link.
#             ordered_args = np.argsort(weights)
#             weights = weights[ordered_args]
#             to_node = to_node[ordered_args]
#             # Retrieve what was chosen by the greedy linker.
#             soln_link = list(greedy_soln.out_edges(node))
#             if len(soln_link):
#                 link_i = np.where(to_node == soln_link[0][1])[0][0]
#                 link_weight = weights[link_i]
#             # If link doesn't exist, use dummy values to make next lines work.
#             else:
#                 link_i = len(to_node)
#                 link_weight = -999.    # No real weights are negative.
#             # If the closest link wasn't chosen (link_i > 0), check that all
#             # closer links are already taken.  If the link wasn't chosen,
#             # search all of `to_node`.
#             for i in range(link_i):
#                 greedier_origin = list(greedy_soln.in_edges(to_node[i]))
#                 assert len(greedier_origin) in [0, 1], (
#                     "node has multiple incoming edges!")
#                 # Intelligently handle situations where link distances are
#                 # the same.
#                 if not np.isclose(weights[i], link_weight,
#                                   rtol=1e-7, atol=1e-7):
#                     assert (self.df.loc[greedier_origin[0][0],
#                                         'dropoff_datetime'] <=
#                             self.df.loc[node, 'dropoff_datetime']), (
#                                 "greediest link blocked by future node!")


# class TestGreedyPassengerLinker(TestLinkerdfBase):
#     """Test greedy linker function."""

#     @pytest.mark.parametrize(('mintype'), ('deadhead_time', 'overhead_time'))
#     def test_get_greedy_passenger_solution(self, mintype):
#         time_ordered_nodes = list(
#             self.df['pickup_datetime'].sort_values().index)

#         # Get greedy passenger solution from linker.
#         gp_soln = linker.get_greedy_passenger_solution(
#             self.net, time_ordered_nodes, mintype=mintype)

#         # Algorithm to ensure all nodes have optimally greedy links.
#         for node in time_ordered_nodes:
#             # For each node, start by getting all destination nodes.
#             from_node = np.zeros(len(self.net.in_edges(node)), dtype=int)
#             weights = np.zeros(len(self.net.in_edges(node)))
#             for i, edge in enumerate(self.net.in_edges(node)):
#                 from_node[i] = edge[0]
#                 weights[i] = self.net.edges[edge][mintype]
#             # Sort the destination node IDs and distances from nearest to
#             # furthest link.
#             ordered_args = np.argsort(weights)
#             weights = weights[ordered_args]
#             from_node = from_node[ordered_args]
#             # Retrieve what was chosen by the greedy linker.
#             soln_link = list(gp_soln.in_edges(node))
#             if len(soln_link):
#                 link_i = np.where(from_node == soln_link[0][0])[0][0]
#                 link_weight = weights[link_i]
#             # If link doesn't exist, use dummy values to make next lines work.
#             else:
#                 link_i = len(from_node)
#                 link_weight = -999.    # No real weights are negative.
#             # If the closest link wasn't chosen (link_i > 0), check that all
#             # closer links are already taken.  If the link wasn't chosen,
#             # search all of `to_node`.
#             for i in range(link_i):
#                 greedier_origin = list(gp_soln.out_edges(from_node[i]))
#                 assert len(greedier_origin) in [0, 1], (
#                     "node has multiple outgoing edges!")
#                 # Intelligently handle situations where link distances are
#                 # the same.
#                 if not np.isclose(weights[i], link_weight,
#                                   rtol=1e-7, atol=1e-7):
#                     assert (self.df.loc[greedier_origin[0][1],
#                                         'pickup_datetime'] <=
#                             self.df.loc[node, 'pickup_datetime']), (
#                                 "greediest link blocked by future node!")


# class TestBatchedLinkers(TestLinkerdfBase):
#     """Test batched linking classes.

#     Since link algorithms are tested separately, we'll just use the minimum
#     fleet linker when needed.
#     """

#     def test_batched_linker_class(self):
#         start_time = pd.Timestamp('2017-03-15 18:00:00')
#         timespan = pd.Timedelta('90 minutes')
#         bl = linker.BatchedLinker(self.df, self.net, start_time,
#                                   timespan=timespan)
#         assert bl.df is self.df
#         assert bl.gph_feasible is self.net
#         assert bl.start_time == start_time
#         assert bl.timespan == timespan

#         assert np.all(
#             self.df[['pickup_datetime']].sort_values('pickup_datetime') ==
#             bl.pickups)

#     @pytest.mark.parametrize(
#         ('start_time', 'stop_time', 'n_prev'),
#         [('2017-03-15 18:30:00', '2017-03-15 18:40:00', 3),
#          ('2017-03-15 19:15:00', '2017-03-15 19:19:00', 5),
#          ('2017-03-15 18:51:00', '2017-03-15 19:00:00', 15)])
#     def test_get_bin_graph(self, start_time, stop_time, n_prev):
#         bl = linker.BatchedLinker(
#             self.df, self.net, pd.Timestamp('2017-03-15 18:00:00'),
#             timespan=pd.Timedelta('90 minutes'))

#         # Begin with no edges in the solution.
#         soln_net = nx.DiGraph()
#         soln_net.add_nodes_from(self.net)

#         bin_graph = bl.get_bin_graph(soln_net, [start_time, stop_time])

#         # Check that `bin_graph` has all nodes in the bin, and a bunch of extra
#         # nodes that are only from previous bins.
#         nodes_in_tbin = (
#             bl.pickups[
#                 (bl.pickups['pickup_datetime'] >= start_time) &
#                 (bl.pickups['pickup_datetime'] < stop_time)]
#             .index.values)
#         assert len(set(nodes_in_tbin) - set(bin_graph.nodes())) == 0
#         prev_bin_nodes = list(set(bin_graph.nodes()) - set(nodes_in_tbin))
#         assert len(prev_bin_nodes) >= 0
#         assert np.all(bl.pickups.loc[prev_bin_nodes, :] <
#                       start_time)

#         # For a few nodes from previous bins, add a connection to the solution.
#         for node in prev_bin_nodes[:n_prev]:
#             soln_net.add_edge(*next(iter(self.net.out_edges(node))))

#         # Check that bin_graph properly removes these.
#         bin_graph2 = bl.get_bin_graph(soln_net, [start_time, stop_time])

#         expected_nodes = list(set(bin_graph.nodes()) -
#                               set(prev_bin_nodes[:n_prev]))
#         assert sorted(expected_nodes) == sorted(bin_graph2.nodes())

#     def batch_linking(self, bl, t_bin):
#         # Define batch solution to output.
#         soln_batch = nx.DiGraph()
#         soln_batch.add_nodes_from(self.net)

#         n_bins = int(bl.timespan / t_bin)
#         bin_edges = [bl.start_time + x * t_bin for x in np.arange(n_bins + 1)]

#         for i in range(n_bins):
#             # Use bl for `get_bin_graph`, which we already tested.
#             bin_graph = bl.get_bin_graph(soln_batch, bin_edges[i:i + 2])
#             if bin_graph.number_of_nodes():
#                 bin_graph_soln = linker.get_vazifeh_solution(bin_graph)
#                 for edge in bin_graph_soln.edges():
#                     soln_batch.add_edge(*edge, **bin_graph_soln.edges[edge])
#             else:
#                 nodes_in_tbin = (
#                     bl.pickups[
#                         (bl.pickups['pickup_datetime'] >= bin_edges[i]) &
#                         (bl.pickups['pickup_datetime'] < bin_edges[i + 1])]
#                     .index.values)
#                 assert len(nodes_in_tbin) == 0, "bin_graph is missing nodes!"

#         return soln_batch

#     @pytest.mark.parametrize(
#         ('start_time', 'timespan', 't_bin'),
#         [('2017-03-15 18:00:00', '90 minutes', '5 minutes'),
#          ('2017-03-15 18:37:00', '82 minutes', '6.2 minutes'),
#          ('2017-03-15 19:00:00', '14 minutes', '14 minutes'),
#          ('2017-03-15 19:00:00', '14 minutes', '1 minutes')])
#     def test_batched_linker_vazifeh(self, start_time, timespan, t_bin):
#         # Since we can't guarantee the graph's output, we can only test that
#         # each bin has
#         bl = linker.BatchedLinkerVazifeh(
#             self.df, self.net, pd.Timestamp(start_time),
#             timespan=pd.Timedelta(timespan))
#         bl_soln = bl.get_batch_linking(pd.Timedelta(t_bin))
#         ref_soln = self.batch_linking(bl, pd.Timedelta(t_bin))

#         assert utils.graphs_equivalent(bl_soln, ref_soln)
