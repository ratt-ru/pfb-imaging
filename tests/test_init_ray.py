"""Tests for pfb_imaging.init_ray.

These mock ray.init because the behavior under test is which kwargs reach
ray.init: passing cluster properties (num_cpus, object_store_memory) while
connecting to an existing cluster raises inside Ray (see issue #254), and
exercising that for real would require an external cluster.
"""

import ray

from pfb_imaging import init_ray


def _patch_ray(monkeypatch, initialized=False):
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(ray, "init", fake_init)
    monkeypatch.setattr(ray, "is_initialized", lambda: initialized)
    return calls


def test_local_address_forces_new_cluster_with_resources(monkeypatch):
    calls = _patch_ray(monkeypatch)
    runtime_env = {"env_vars": {"FOO": "1"}}

    init_ray(nworkers=4, ray_address="local", runtime_env=runtime_env, object_store_memory=1e9)

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["address"] == "local"
    assert kwargs["num_cpus"] == 4
    assert kwargs["object_store_memory"] == 1e9
    assert kwargs["runtime_env"] is runtime_env


def test_local_is_the_default_address(monkeypatch):
    calls = _patch_ray(monkeypatch)

    init_ray(nworkers=2)

    assert len(calls) == 1
    assert calls[0]["address"] == "local"
    assert calls[0]["num_cpus"] == 2


def test_existing_cluster_address_drops_cluster_properties(monkeypatch):
    calls = _patch_ray(monkeypatch)
    runtime_env = {"env_vars": {"FOO": "1"}}

    init_ray(nworkers=4, ray_address="10.0.0.1:6379", runtime_env=runtime_env, object_store_memory=1e9)

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["address"] == "10.0.0.1:6379"
    assert "num_cpus" not in kwargs
    assert "object_store_memory" not in kwargs
    assert kwargs["runtime_env"] is runtime_env


def test_no_reinit_when_already_initialized(monkeypatch):
    calls = _patch_ray(monkeypatch, initialized=True)

    init_ray(nworkers=4)

    assert calls == []
