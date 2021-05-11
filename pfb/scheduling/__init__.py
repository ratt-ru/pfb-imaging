'''
Scheduler plugin shamelessly stolen from
https://github.com/JSKenyon/QuartiCal/blob/master/quartical/scheduling/__init__.py
'''

from collections import defaultdict

from distributed.diagnostics.plugin import SchedulerPlugin
from dask.core import reverse_dict
from dask.base import tokenize


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(AutoRestrictor(**kwargs), idempotent=True)


def unravel_deps(hlg_deps, name, unravelled_deps=None):
    """Recursively construct a set of all dependencies for a specific task."""

    if unravelled_deps is None:
        unravelled_deps = set()

    for dep in hlg_deps[name]:
        unravelled_deps |= {dep}
        unravel_deps(hlg_deps, dep, unravelled_deps)

    return unravelled_deps


class AutoRestrictor(SchedulerPlugin):

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None,
                     **kw):
        """Processes dependencies to assign tasks to specific workers."""
        tasks = scheduler.tasks
        dependencies = kw["dependencies"]
        dependents = reverse_dict(dependencies)

        # Terminal nodes have no dependents, root nodes have no dependencies.
        terminal_nodes = {k for (k, v) in dependents.items() if v == set()}
        root_nodes = {k for (k, v) in dependencies.items() if v == set()}

        roots_per_terminal = {}
        terminal_deps = {}

        for tn in terminal_nodes:
            # Get dependencies per terminal node.
            terminal_deps[tn] = unravel_deps(dependencies, tn)
            # Associate terminal nodes with root nodes.
            roots_per_terminal[tn] = root_nodes & terminal_deps[tn]

        # Create a unique token for each set of terminal roots. TODO: This is
        # very strict. What about nodes with very similar roots? Tokenization
        # may be overkill too.
        root_tokens = \
            {tokenize(*sorted(v)): v for v in roots_per_terminal.values()}

        hash_map = defaultdict(set)
        group_offset = 0

        # Associate terminal roots with a specific group if they are not a
        # subset of another larger root set. TODO: This can likely be improved.
        for k, v in root_tokens.items():
            if any([v < vv for vv in root_tokens.values()]):  # Strict subset.
                continue
            else:
                hash_map[k] |= set([group_offset])
                group_offset += 1

        # If roots were a subset, they should share the annotation of their
        # superset/s.
        for k, v in root_tokens.items():
            shared_roots = \
                {kk: None for kk, vv in root_tokens.items() if v < vv}
            if shared_roots:
                hash_map[k] = \
                    set().union(*[hash_map[kk] for kk in shared_roots.keys()])

        workers = list(scheduler.workers.keys())
        n_worker = len(workers)

        for k, deps in terminal_deps.items():

            # TODO: This can likely be improved.
            group = hash_map[tokenize(*sorted(roots_per_terminal[k]))]

            # Set restrictions on a terminal node and its dependencies.
            for tn in [k, *deps]:
                try:
                    task = tasks[tn]
                except KeyError:  # Keys may not have an assosciated task.
                    continue
                if task._worker_restrictions is None:
                    task._worker_restrictions = set()
                task._worker_restrictions |= \
                    {workers[g % n_worker] for g in group}
                task._loose_restrictions = False