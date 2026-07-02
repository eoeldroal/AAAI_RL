import importlib

import pytest


CUDA_STACK_ENV = {
    "CUDA_HOME": "/opt/test/cuda",
    "CUDA_PATH": "/opt/test/cuda",
    "CUDAToolkit_ROOT": "/opt/test/cuda",
    "CUDACXX": "/opt/test/cuda/bin/nvcc",
    "CUDNN_HOME": "/opt/test/cudnn",
    "CUDNN_PATH": "/opt/test/cudnn",
    "NCCL_HOME": "/opt/test/nccl",
    "LD_LIBRARY_PATH": "/opt/test/cuda/lib:/opt/test/cudnn/lib:/opt/test/nccl/lib",
    "LIBRARY_PATH": "/opt/test/cuda/lib",
    "CPATH": "/opt/test/cuda/include",
    "CMAKE_PREFIX_PATH": "/opt/test/cuda",
    "CMAKE_INCLUDE_PATH": "/opt/test/cuda/include",
    "CMAKE_LIBRARY_PATH": "/opt/test/cuda/lib",
    "SGLANG_NUMA_BIND_V2": "0",
}


def test_ppo_ray_runtime_env_forwards_cuda_stack(monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("torch")

    for key, value in CUDA_STACK_ENV.items():
        monkeypatch.setenv(key, value)

    constants_ppo = importlib.import_module("verl.trainer.constants_ppo")
    runtime_env = constants_ppo.get_ppo_ray_runtime_env()
    env_vars = runtime_env["env_vars"]

    for key, value in CUDA_STACK_ENV.items():
        assert env_vars[key] == value


def test_cuda_rollout_env_forwards_cuda_stack(monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("torch")

    for key, value in CUDA_STACK_ENV.items():
        monkeypatch.setenv(key, value)

    platform_cuda = importlib.import_module("verl.plugin.platform.platform_cuda")
    env_vars = platform_cuda.PlatformCUDA().rollout_env_vars()

    assert env_vars["NCCL_CUMEM_ENABLE"] == "0"
    for key, value in CUDA_STACK_ENV.items():
        assert env_vars[key] == value
