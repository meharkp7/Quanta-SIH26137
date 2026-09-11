"""Versioned, bounded physical-path cache for snapshot routing costs."""
from collections import OrderedDict
from .path_builder import DirectedPathBuilder

class CostView:
    def __init__(self, edges, *, graph_version, cost_version="free-flow", forecast_version=None,
                 closed_edge_ids=(), travel_time_provider=None, max_entries=4096):
        self.versions = (graph_version, cost_version, forecast_version)
        self.builder = DirectedPathBuilder(edges, closed_edge_ids=closed_edge_ids,
                                          travel_time_provider=travel_time_provider)
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._cache = OrderedDict()

    def shortest_path(self, source, target, *, departure_time_s=0.0):
        # Exact departure is retained within its one-second bucket: no time approximation.
        key = (*self.versions, source, target, int(departure_time_s), departure_time_s)
        if key not in self._cache:
            self._cache[key] = self.builder.shortest_path(source, target, departure_time_s=departure_time_s)
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return self._cache[key]

    def invalidate(self, *, cost_version, forecast_version=None):
        # Clear all entries: an edge decrease can improve paths not previously using it.
        self.versions = (self.versions[0], cost_version, forecast_version)
        self._cache.clear()
