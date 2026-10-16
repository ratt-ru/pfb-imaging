"""hci-msv4 `_dispatch` against stub actors: in-process under a fake Ray, and on the session cluster.

The fake reproduces the Ray semantics `_dispatch` depends on -- an actor runs its calls one
at a time in submission order, a call waits for its ref arguments (head-of-line blocking),
an error propagates to every call that consumes the failed ref -- and picks which runnable
actor goes next at random, so several seeds exercise many interleavings in milliseconds.

The real-cluster test is slow (D45): every stub actor is a new worker process, Ray starts at
most `num_cpus` of them at a time, and a start costs seconds. Everything those actors touch is
defined inside `_stubs`, so Ray pickles it by value and a stub process never imports the
scientific stack (unpickling a real `WorkItem` imports `core.hci_msv4`). `_dispatch` only
reads `.block` off an item.
"""

import random
from collections import Counter

import numpy as np
import pytest
import ray

from pfb_imaging.core import hci_msv4
from pfb_imaging.core.hci_msv4 import HciPlan, _dispatch


def _stubs():
    """An item type and stub actor classes, local so Ray pickles them by value."""
    import time  # deferred: the stub actors are pickled by value and must carry their own imports
    from dataclasses import dataclass  # deferred: as above

    @dataclass(frozen=True)
    class Item:
        time_index: int
        band_index: int
        block: int

    class StubWorker:
        def __init__(self, delay=0.0):
            self.delay = delay
            self.reset(None)

        def reset(self, fail_at):
            self.fail_at = fail_at
            self.promised = None
            self.got = []

        def image(self, item, next_item=None):
            if self.promised is not None and item != self.promised:
                raise RuntimeError(f"promised {self.promised} but got {item}")
            self.promised = next_item
            self.got.append(item)
            if item.time_index == self.fail_at:
                raise ZeroDivisionError(f"stub failure on {item.time_index}")
            time.sleep(self.delay)
            mem = {"pid": 0, "rss_gb": 0.0, "peak_gb": 0.0}
            return {"time_index": item.time_index, "band_index": item.band_index}, mem

        def received(self):
            return self.got

    class StubWriter:
        def __init__(self, delay=0.0):
            self.delay = delay

        def reset(self, expected, ipc):
            self.left = dict(expected)
            self.ipc = ipc
            self.open = set()
            self.max_open_blocks = 0

        def add(self, result):
            time.sleep(self.delay)
            key = (result["band_index"], result["time_index"] // self.ipc)
            self.open.add(key[1])
            self.max_open_blocks = max(self.max_open_blocks, len(self.open))
            self.left[key] -= 1
            if self.left[key]:
                return None
            if not any(n for (_, blk), n in self.left.items() if blk == key[1]):
                self.open.discard(key[1])
            return key, {"pid": 0, "rss_gb": 0.0, "peak_gb": 0.0}

        def max_open(self):
            return self.max_open_blocks

    return Item, StubWorker, StubWriter


def _plan(item_cls, ntime=12, nband=2, ipc=2) -> HciPlan:
    items = [item_cls(t, b, t // ipc) for t in range(ntime) for b in range(nband)]
    items.sort(key=lambda it: (it.block, it.band_index, it.time_index))
    empty = np.empty(0)
    return HciPlan(
        items=items,
        out_times=np.arange(ntime, dtype=float),
        out_freqs=np.arange(nband, dtype=float),
        expected=dict(Counter((it.band_index, it.block) for it in items)),
        channel_width=np.zeros(nband),
        field_radecs=np.zeros((1, 2)),
        max_blength=0.0,
        max_freq=0.0,
        all_times=empty,
        all_freqs=empty,
    )


class _Ref:
    def __init__(self):
        self.done, self.value, self.error = False, None, None


class _FakeRay:
    """Just enough of Ray's actor semantics for `_dispatch`, with a seeded scheduler."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.actors = []
        self.workers = []  # actors whose concurrent busyness is tracked
        self.max_busy = 0

    def actor(self, obj):
        a = _FakeActor(obj)
        self.actors.append(a)
        return a

    def _step(self):
        self.max_busy = max(self.max_busy, sum(1 for a in self.workers if a.queue))
        runnable = [a for a in self.actors if a.queue and all(r.done for r in a.queue[0][2] if isinstance(r, _Ref))]
        if not runnable:
            raise RuntimeError("fake Ray deadlock")
        self.rng.choice(runnable).run_one()

    def wait(self, refs, num_returns=1):
        while not any(r.done for r in refs):
            self._step()
        first = next(r for r in refs if r.done)  # Ray returns ready refs in input order
        return [first], [r for r in refs if r is not first]

    def get(self, ref):
        while not ref.done:
            self._step()
        if ref.error is not None:
            raise ref.error
        return ref.value


class _FakeMethod:
    def __init__(self, actor, name, nret=1):
        self.actor, self.name, self.nret = actor, name, nret

    def options(self, num_returns=1):
        return _FakeMethod(self.actor, self.name, num_returns)

    def remote(self, *args):
        refs = [_Ref() for _ in range(self.nret)]
        self.actor.queue.append((self.name, refs, args))
        self.actor.max_queue = max(self.actor.max_queue, len(self.actor.queue))
        return refs[0] if self.nret == 1 else refs


class _FakeActor:
    def __init__(self, obj):
        self.obj, self.queue, self.max_queue = obj, [], 0

    def __getattr__(self, name):
        return _FakeMethod(self, name)

    def run_one(self):
        name, refs, args = self.queue.pop(0)
        failed = next((a.error for a in args if isinstance(a, _Ref) and a.error is not None), None)
        if failed is None:
            try:
                out = getattr(self.obj, name)(*[a.value if isinstance(a, _Ref) else a for a in args])
            except Exception as e:  # noqa: BLE001 -- delivered through the refs, as Ray does
                failed = e
        for k, r in enumerate(refs):
            r.done = True
            if failed is not None:
                r.error = failed
            else:
                r.value = out if len(refs) == 1 else out[k]


@pytest.mark.parametrize("seed", range(4))
def test_dispatch_invariants_in_process(monkeypatch, seed):
    item_cls, stub_worker, stub_writer = _stubs()
    fake = _FakeRay(seed)
    monkeypatch.setattr(hci_msv4, "ray", fake)
    plan = _plan(item_cls, ntime=12, nband=3)
    workers = [stub_worker() for _ in range(3)]
    writer = stub_writer()
    writer.reset(plan.expected, 2)
    writer_handle = fake.actor(writer)
    _dispatch(plan, [fake.actor(w) for w in workers], writer_handle, 2)
    got = [it for w in workers for it in w.got]
    # every item exactly once; a broken promise raises inside the stub worker
    assert sorted(got, key=lambda it: (it.time_index, it.band_index)) == sorted(
        plan.items, key=lambda it: (it.time_index, it.band_index)
    )
    assert 1 <= writer.max_open_blocks <= 2
    assert writer_handle.max_queue <= 2 * len(workers)  # unacked writer adds are capped
    assert not any(a.queue for a in fake.actors)

    # a worker error surfaces with its own type, whichever ref the driver sees first
    fake = _FakeRay(seed)
    monkeypatch.setattr(hci_msv4, "ray", fake)
    for w in workers:
        w.reset(7)
    writer.reset(plan.expected, 2)
    with pytest.raises(ZeroDivisionError, match="stub failure on 7"):
        _dispatch(plan, [fake.actor(w) for w in workers], fake.actor(writer), 2)


@pytest.mark.slow
def test_dispatch_on_ray_with_stub_actors():
    item_cls, stub_worker, stub_writer = _stubs()
    stub_worker = ray.remote(num_cpus=1e-2)(stub_worker)
    stub_writer = ray.remote(num_cpus=1e-2)(stub_writer)
    plan = _plan(item_cls)
    max_open = 2
    # the writer is slower than the workers, so acks lag and the writer cap binds
    workers = [stub_worker.remote(delay=0.002) for _ in range(2)]
    writer = stub_writer.remote(delay=0.004)
    try:
        ray.get(writer.reset.remote(plan.expected, 2))
        _dispatch(plan, workers, writer, max_open)
        got = [it for w in workers for it in ray.get(w.received.remote())]
        assert sorted(got, key=lambda it: (it.time_index, it.band_index)) == sorted(
            plan.items, key=lambda it: (it.time_index, it.band_index)
        )
        assert 1 <= ray.get(writer.max_open.remote()) <= max_open

        ray.get([w.reset.remote(5) for w in workers] + [writer.reset.remote(plan.expected, 2)])
        with pytest.raises(ZeroDivisionError, match="stub failure on 5"):
            _dispatch(plan, workers, writer, max_open)
    finally:
        for a in workers + [writer]:
            ray.kill(a)


@pytest.mark.parametrize("seed", range(4))
def test_narrow_window_keeps_every_worker_busy(monkeypatch, seed):
    """One band, four images per block, two blocks open: eight allowed items for eight workers.

    Promising each dispatched worker its next item as well would hand the eight items to four
    workers and idle the other four.
    """
    item_cls, stub_worker, stub_writer = _stubs()
    fake = _FakeRay(seed)
    monkeypatch.setattr(hci_msv4, "ray", fake)
    plan = _plan(item_cls, ntime=32, nband=1, ipc=4)
    workers = [stub_worker() for _ in range(8)]
    writer = stub_writer()
    writer.reset(plan.expected, 4)
    handles = [fake.actor(w) for w in workers]
    fake.workers = handles
    _dispatch(plan, handles, fake.actor(writer), 2)
    assert fake.max_busy == 8
    assert sorted(it.time_index for w in workers for it in w.got) == list(range(32))
