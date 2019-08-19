
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