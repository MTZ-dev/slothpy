from __future__ import annotations

import socket

import pytest

from slothpy.core.slt_session import (
    SltNodeResources,
    SltResourcePool,
    SltResourceRequest,
    resolve_control_node_name,
)


def test_resolve_control_node_name_explicit() -> None:
    nodes = (
        SltNodeResources("node-a", 8),
        SltNodeResources("node-b", 8),
    )
    assert resolve_control_node_name(nodes, control_node_name="node-b") == "node-b"


def test_packed_allocation_puts_control_node_first() -> None:
    pool = SltResourcePool(
        (
            SltNodeResources("worker-1", 16),
            SltNodeResources("control", 16),
        ),
        control_node_name="control",
        parent_reserved_cores=1,
    )

    allocation = pool.try_acquire(
        SltResourceRequest(num_processes=4, num_threads=2)
    )

    assert allocation is not None
    assert allocation.rank0_node_name == "control"
    assert allocation.nodes[0].node_name == "control"
    assert allocation.nodes[0].ranks >= 1
    assert sum(node.ranks for node in allocation.nodes) == 4


def test_parent_reserved_cores_reduce_control_node_capacity() -> None:
    pool = SltResourcePool(
        (SltNodeResources("localhost", 4),),
        control_node_name="localhost",
        parent_reserved_cores=2,
    )

    # 4 cores - 2 parent = 2 free -> at most 1 rank with 2 threads
    allocation = pool.try_acquire(
        SltResourceRequest(num_processes=2, num_threads=2)
    )

    assert allocation is None

    single = pool.try_acquire(
        SltResourceRequest(num_processes=1, num_threads=2)
    )
    assert single is not None
    assert single.nodes[0].ranks == 1


def test_exact_node_allocation_requires_control_node() -> None:
    pool = SltResourcePool(
        (
            SltNodeResources("node-a", 8),
            SltNodeResources("node-b", 8),
        ),
        control_node_name="node-b",
        parent_reserved_cores=0,
    )

    allocation = pool.try_acquire(
        SltResourceRequest(num_processes=3, num_threads=1, num_nodes=2)
    )

    assert allocation is not None
    assert allocation.nodes[0].node_name == "node-b"
    assert allocation.rank0_node_name == "node-b"


def test_local_pool_uses_localhost_control() -> None:
    pool = SltResourcePool.local(cores=8, parent_reserved_cores=1)
    assert pool.control_node_name == "localhost"

    allocation = pool.try_acquire(
        SltResourceRequest(num_processes=2, num_threads=2)
    )
    assert allocation is not None
    assert allocation.is_localhost_only


def test_hostfile_order_matches_rank0_node() -> None:
    pool = SltResourcePool(
        (
            SltNodeResources("remote", 16),
            SltNodeResources("local", 16),
        ),
        control_node_name="local",
        parent_reserved_cores=0,
    )

    allocation = pool.try_acquire(
        SltResourceRequest(num_processes=3, num_threads=1, num_nodes=2)
    )
    assert allocation is not None

    hostfile = allocation.write_openmpi_hostfile("/tmp/slothpy-test-hostfile")
    lines = hostfile.read_text(encoding="utf-8").strip().splitlines()

    assert lines[0].startswith("local ")
    assert allocation.nodes[0].node_name == "local"


def test_resolve_control_node_matches_hostname() -> None:
    host = socket.gethostname().split(".")[0]
    nodes = (SltNodeResources(host, 4), SltNodeResources("other", 4))
    assert resolve_control_node_name(nodes) == host
