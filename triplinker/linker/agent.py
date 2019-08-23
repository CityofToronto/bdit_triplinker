import numpy as np
import pandas as pd
import networkx as nx
import tqdm
import warnings
import pprint

from . import linker


class DriverShiftBase:

    def __init__(self):
        self.trips = []
        self._start_time = None
        self._latest_time = None

    def __getitem__(self, key):
        return self.trips.__getitem__(key)

    @property
    def n_trips(self):
        return len(self.trips)

    @property
    def last_trip(self):
        return self.trips[-1] if len(self.trips) else None

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = pd.Timestamp(start_time)

    @property
    def latest_time(self):
        return self._latest_time

    @latest_time.setter
    def latest_time(self, latest_time):
        self._latest_time = pd.Timestamp(latest_time)

    def add_trips_to_shift(self, new_trips, start_time, latest_time):
        self.latest_time = latest_time
        if start_time is not None:
            self.start_time = start_time
            self.trips += new_trips
        else:
            self.trips += new_trips[1:]

    def __repr__(self):
        return self.__class__.__name__ + " " + self.trips.__repr__()


class ShiftTrackerBase:

    _driver_class = DriverShiftBase

    def __init__(self, df, **kwargs):
        self.df = df
        if len(kwargs):
            warnings.warn('unused kwargs {0}'.format(kwargs))
        self.shifts = []

    def __getitem__(self, key):
        return self.shifts.__getitem__(key)

    def __len__(self):
        return len(self.shifts)

    def __repr__(self):
        return self.__class__.__name__ + "\n" + pprint.pformat(self.shifts)

    def pruner(self, bin_graph, nodes_to_delete=[], origin_nodes=None):
        bin_graph.remove_nodes_from(nodes_to_delete)

    def get_origin_nodes(self, bin_graph):
        # Origin nodes are nodes with outgoing edges but no incoming edges.
        return [node for node in bin_graph.nodes()
                if (not bin_graph.in_degree(node)) and
                (bin_graph.out_degree(node))]

    def find_connecting_shift(self, first_trip):
        for i in range(len(self.shifts)):
            if self.shifts[i].last_trip == first_trip:
                return i
        return None

    def update_shift(self, idx, new_trips):
        if self.shifts[idx].n_trips == 0:
            start_time = self.df.loc[new_trips[0], 'pickup_datetime']
        else:
            start_time = None
        self.shifts[idx].add_trips_to_shift(
            new_trips, start_time,
            self.df.loc[new_trips[-1], 'dropoff_datetime'])

    def add_solution(self, bin_graph_soln):
        new_paths = linker.get_paths(bin_graph_soln)
        for path in new_paths:
            idx = self.find_connecting_shift(path[0])
            if idx is None:
                self.shifts.append(self._driver_class())
                idx = -1
            self.update_shift(idx, path)

    def refresh(self):
        self.shifts = []

    @property
    def paths(self):
        return [shift.trips for shift in self.shifts]

    @property
    def first_trip(self):
        return [shift[0] for shift in self.shifts]

    @property
    def start_times(self):
        return [shift.start_time for shift in self.shifts]

    @property
    def last_trip(self):
        return [shift[-1] for shift in self.shifts]

    @property
    def latest_times(self):
        return [shift.latest_time for shift in self.shifts]

    @property
    def n_trips(self):
        return [shift.n_trips for shift in self.shifts]

    @property
    def n_shifts(self):
        return len(self.shifts)


class DriverShiftTimeLimit(DriverShiftBase):

    def __init__(self):
        self.exceeded_timelimit = False
        super().__init__()


class ShiftTrackerTimeLimit(ShiftTrackerBase):

    _driver_class = DriverShiftTimeLimit

    def __init__(self, df, time_limit=pd.Timedelta('3 hours'), **kwargs):
        self.time_limit = pd.Timedelta(time_limit)
        super().__init__(df, **kwargs)

    def pruner(self, bin_graph, nodes_to_delete=[], origin_nodes=None):
        if origin_nodes is None:
            origin_nodes = self.get_origin_nodes(bin_graph)
        shift_end_nodes = [shift[-1] for shift in self.shifts
                           if shift.exceeded_timelimit]
        nodes_to_delete = list(
            set(origin_nodes).intersection(set(shift_end_nodes))
            .union(set(nodes_to_delete)))
        super().pruner(bin_graph, nodes_to_delete=nodes_to_delete,
                       origin_nodes=origin_nodes)

    def add_solution(self, bin_graph_soln):
        super().add_solution(bin_graph_soln)
        for shift in self.shifts:
            if shift.latest_time - shift.start_time > self.time_limit:
                shift.exceeded_timelimit = True


class DriverShiftOneRouting(DriverShiftBase):

    def __init__(self):
        self.one_routing = False
        super().__init__()


class ShiftTrackerOneRouting(ShiftTrackerBase):

    _driver_class = DriverShiftOneRouting

    def __init__(self, df, p_onerouting=0.5, **kwargs):
        self.p_onerouting = p_onerouting
        self._is_one_routing = []
        super().__init__(df, **kwargs)

    @property
    def is_one_routing(self):
        return self._is_one_routing

    def pruner(self, bin_graph, nodes_to_delete=[], origin_nodes=None):
        if origin_nodes is None:
            origin_nodes = self.get_origin_nodes(bin_graph)
        shift_end_nodes = [shift[-1] for shift in self.shifts
                           if shift.one_routing]
        nodes_to_delete = list(
            set(origin_nodes).intersection(set(shift_end_nodes))
            .union(set(nodes_to_delete)))
        super().pruner(bin_graph, nodes_to_delete=nodes_to_delete,
                       origin_nodes=origin_nodes)

    def add_solution(self, bin_graph_soln):
        super().add_solution(bin_graph_soln)
        for shift in self.shifts[len(self._is_one_routing):]:
            if np.random.binomial(1, self.p_onerouting, size=1):
                shift.one_routing = True
                self._is_one_routing.append(True)
            else:
                self._is_one_routing.append(False)


class DriverShiftORTL(DriverShiftTimeLimit, DriverShiftOneRouting):
    pass


class ShiftTrackerORTL(ShiftTrackerOneRouting, ShiftTrackerTimeLimit):

    _driver_class = DriverShiftORTL


class BatchedLinkerShiftBase(linker.BatchedLinker):
    """Base class for batch-linker with simple driver agent modelling."""

    _ShiftTracker = ShiftTrackerBase

    def __init__(self, df, gph_feasible, start_time,
                 timespan=pd.Timedelta('24 hour'),
                 progress_bar=False, shift_kwargs={}):
        self.shift_tracker = self._ShiftTracker(df, **shift_kwargs)
        super().__init__(df, gph_feasible, start_time,
                         timespan=timespan, progress_bar=progress_bar)

    def __getitem__(self, key):
        return self.shift_tracker.__getitem__(key)

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

        self.shift_tracker.refresh()

        # For each bin, get the feasible sub-graph, link it, and store the
        # solution in `soln`.
        for i in tqdm.tqdm(range(nbins), disable=self._disable_tqdm):
            bin_graph = self.get_bin_graph(soln, bin_edges[i:i + 2])
            if bin_graph.number_of_edges():
                self.shift_tracker.pruner(bin_graph)
                bin_graph_soln = self.link_mtd(bin_graph)
                self.add_edges(soln, bin_graph_soln)
                self.shift_tracker.add_solution(bin_graph_soln)
            elif bin_graph.number_of_nodes():
                # If `bin_graph` literally has no feasible connections, then
                # just add the nodes in the bin as individual connections.  Any
                # nodes already in `shift_tracker` will simply get re-added
                # without double-counting.
                self.shift_tracker.add_solution(bin_graph)

        return soln


class BatchedLinkerShiftVazifehBase(BatchedLinkerShiftBase,
                                    linker.BatchedLinkerVazifeh):
    pass


class BatchedLinkerShiftTimeLimit(BatchedLinkerShiftVazifehBase):

    _ShiftTracker = ShiftTrackerTimeLimit


class BatchedLinkerShiftOneRouting(BatchedLinkerShiftVazifehBase):

    _ShiftTracker = ShiftTrackerOneRouting


class BatchedLinkerShiftORTL(BatchedLinkerShiftVazifehBase):

    _ShiftTracker = ShiftTrackerORTL
