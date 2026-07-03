from pathlib import Path

SERVER = Path("verl/workers/rollout/sglang_rollout/async_sglang_server.py")


def _server_text() -> str:
    return SERVER.read_text()


def test_sglang_http_server_reserves_single_node_nccl_port_for_ray_colocation():
    server = _server_text()

    reservation = 'sock.bind(("127.0.0.1", port))'
    assign_nccl_port = 'args["nccl_port"] = self._nccl_port'
    close_socket = "self._nccl_sock.close()"
    launch_subprocesses = "Engine._launch_subprocesses("

    assert "Ray can" in server
    assert 'VERL_SGLANG_NCCL_PORT_START", "61000"' in server
    assert 'VERL_SGLANG_NCCL_PORT_END", "61999"' in server
    assert "32 * self.replica_rank + self.node_rank" in server
    assert reservation in server
    assert assign_nccl_port in server
    assert close_socket in server
    assert launch_subprocesses in server

    assert server.index(reservation) < server.index(assign_nccl_port)
    assert server.index(assign_nccl_port) < server.index(close_socket)
    assert server.index(close_socket) < server.index(launch_subprocesses)


def test_sglang_http_server_keeps_multi_node_dist_init_addr_separate():
    server = _server_text()

    assert "if self.nnodes > 1:" in server
    assert 'args["dist_init_addr"] = dist_init_addr' in server
    assert "elif self.nnodes == 1:" in server
    assert server.index("elif self.nnodes == 1:") < server.index('args["nccl_port"] = self._nccl_port')
