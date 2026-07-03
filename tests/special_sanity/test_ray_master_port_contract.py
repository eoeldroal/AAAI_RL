import socket
from pathlib import Path

import pytest

RAY_BASE = Path("verl/single_controller/ray/base.py")


def _allocate_ray_master_port_for_test():
    pytest.importorskip("ray")
    from verl.single_controller.ray.base import _allocate_ray_master_port

    return _allocate_ray_master_port


def _find_free_port_range(size: int) -> list[int]:
    for start in range(63000, 65536 - size):
        sockets = []
        try:
            for port in range(start, start + size):
                sock = socket.socket()
                sock.bind(("", port))
                sockets.append(sock)
            return [start, start + size - 1]
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("Could not find free test port range")


def test_ray_worker_group_master_port_uses_non_ephemeral_default_range():
    source = RAY_BASE.read_text()

    assert 'VERL_RAY_MASTER_PORT_START", "62000"' in source
    assert 'VERL_RAY_MASTER_PORT_END", "65535"' in source
    assert "_allocate_ray_master_port(master_port_range)" in source
    assert 'raise RuntimeError(f"Could not find a free port in range {master_port_range}")' in source


def test_ray_worker_group_keeps_explicit_master_port_range_override():
    source = RAY_BASE.read_text()

    assert "if master_port_range is None:" in source
    assert "port = _allocate_ray_master_port(master_port_range)" in source


def test_ray_master_port_allocator_does_not_reuse_released_ports(tmp_path, monkeypatch):
    _allocate_ray_master_port = _allocate_ray_master_port_for_test()
    monkeypatch.setenv("VERL_RAY_MASTER_PORT_ALLOC_FILE", str(tmp_path / "ports.txt"))
    start, end = _find_free_port_range(3)

    first = _allocate_ray_master_port([start, end])
    second = _allocate_ray_master_port([start, end])
    third = _allocate_ray_master_port([start, end])

    assert [first, second, third] == [start, start + 1, start + 2]


def test_ray_master_port_allocator_skips_ports_used_by_other_processes(tmp_path, monkeypatch):
    _allocate_ray_master_port = _allocate_ray_master_port_for_test()
    monkeypatch.setenv("VERL_RAY_MASTER_PORT_ALLOC_FILE", str(tmp_path / "ports.txt"))
    start, end = _find_free_port_range(2)
    sock = socket.socket()
    sock.bind(("", start))
    try:
        assert _allocate_ray_master_port([start, end]) == end
    finally:
        sock.close()
