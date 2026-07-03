# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

CUDA_RUNTIME_ENV_VARS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDAToolkit_ROOT",
    "CUDACXX",
    "CUDNN_HOME",
    "CUDNN_PATH",
    "CUDNN_INCLUDE_PATH",
    "CUDNN_LIBRARY_PATH",
    "NCCL_HOME",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CMAKE_PREFIX_PATH",
    "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH",
    "SGLANG_NUMA_BIND_V2",
)


def get_cuda_runtime_env_vars() -> dict[str, str]:
    return {key: value for key in CUDA_RUNTIME_ENV_VARS if (value := os.environ.get(key))}
