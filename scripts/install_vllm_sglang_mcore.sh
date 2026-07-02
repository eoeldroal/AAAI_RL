#!/usr/bin/env bash
set -euo pipefail

# Conda/uv translation of docker/Dockerfile.stable.sglang.
# Keep this script aligned with that Dockerfile instead of setup.py extras or
# older ad hoc install recipes. The only deliberate overlay is keeping FA4
# available in this single conda env: Docker x86 removes FA4 before installing
# FA2, but B200 SGLang multimodal serving auto-selects FA4 while trainer/Megatron
# paths still import FA2 APIs such as flash_attn.bert_padding.

CUDA_VERSION="${CUDA_VERSION:-13.0.2}"
CUDA_BACKEND="${CUDA_BACKEND:-cu130}"
CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION:-13.0.88}"
CUDA_CCCL_VERSION="${CUDA_CCCL_VERSION:-13.3.3.4.1}"
CUDA_CUPTI_VERSION="${CUDA_CUPTI_VERSION:-13.0.85}"
CUDA_RUNTIME_VERSION="${CUDA_RUNTIME_VERSION:-13.0.96}"
CUDA_NVRTC_VERSION="${CUDA_NVRTC_VERSION:-13.3.33}"
CUDA_NVJITLINK_VERSION="${CUDA_NVJITLINK_VERSION:-13.0.88}"
CUDA_NVML_DEV_VERSION="${CUDA_NVML_DEV_VERSION:-13.0.87}"
CUDA_CUFFT_VERSION="${CUDA_CUFFT_VERSION:-12.0.0.61}"
CUDA_CURAND_VERSION="${CUDA_CURAND_VERSION:-10.4.0.35}"
CUDA_CUSOLVER_VERSION="${CUDA_CUSOLVER_VERSION:-12.0.4.66}"
CUDA_CUSPARSE_VERSION="${CUDA_CUSPARSE_VERSION:-12.6.3.3}"
CUBLAS_VERSION="${CUBLAS_VERSION:-13.6.0.2}"
CUDNN_VERSION="${CUDNN_VERSION:-9.23.2.1}"
CUDNN_FRONTEND_VERSION="${CUDNN_FRONTEND_VERSION:-1.25.0}"
NCCL_VERSION="${NCCL_VERSION:-2.30.7}"
CUPY_VERSION="${CUPY_VERSION:-14.1.1}"
TRL_VERSION="${TRL_VERSION:-0.27.0}"
TRANSFORMER_ENGINE_VERSION="${TRANSFORMER_ENGINE_VERSION:-v2.15}"
FLASH_ATTENTION_VERSION="${FLASH_ATTENTION_VERSION:-2.8.3}"
FLASH_ATTENTION_4_SPEC="${FLASH_ATTENTION_4_SPEC:-flash-attn-4[cu13]==4.0.0b20}"
MCORE_VERSION="${MCORE_VERSION:-core_v0.16.1}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.12}"
SGLANG_KERNEL_VERSION="${SGLANG_KERNEL_VERSION:-0.4.2.post2}"
MAX_JOBS="${MAX_JOBS:-128}"
NVCC_THREADS="${NVCC_THREADS:-4}"
NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-4}"
NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-100}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
CUDAARCHS="${CUDAARCHS:-100}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
MAKEFLAGS="${MAKEFLAGS:--j${MAX_JOBS}}"
BUILD_LOG_ROOT="${BUILD_LOG_ROOT:-${PWD}/logs/build}"
BUILD_LOG_DIR="${BUILD_LOG_DIR:-${BUILD_LOG_ROOT}/sglang_mcore_$(date +%Y%m%d_%H%M%S)}"
PRESERVE_BUILD_TEMP="${PRESERVE_BUILD_TEMP:-1}"
PIP_NO_CLEAN_ARGS=()
if [ "${PRESERVE_BUILD_TEMP}" = "1" ]; then
    PIP_NO_CLEAN_ARGS=(--no-clean)
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
UV_BIN="${UV_BIN:-uv}"

ARCH="$(uname -m)"
CUDA_VERSION_MAJOR="$(echo "${CUDA_VERSION}" | cut -d "." -f 1)"
PYTHON_EXE="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
PYTHON_DIR="$(dirname "${PYTHON_EXE}")"
PYTHON_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDA_PIP_HOME="${PYTHON_SITE_PACKAGES}/nvidia/cu${CUDA_VERSION_MAJOR}"
CUDA_CCCL_INCLUDE_HOME="${CUDA_PIP_HOME}/include/cccl"
CUDNN_PIP_HOME="${PYTHON_SITE_PACKAGES}/nvidia/cudnn"
NCCL_PIP_HOME="${PYTHON_SITE_PACKAGES}/nvidia/nccl"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "uv not found; installing uv into ${PYTHON_PREFIX}"
    "${PYTHON_EXE}" -m pip install -U uv
    if [ -x "${PYTHON_DIR}/uv" ]; then
        UV_BIN="${PYTHON_DIR}/uv"
    elif command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    else
        echo "uv installation completed but no uv executable was found." >&2
        exit 1
    fi
fi

export UV_PYTHON="${PYTHON_EXE}"
export CUDA_HOME="${CUDA_PIP_HOME}"
export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CUDNN_PATH="${CUDNN_PIP_HOME}"
export CUDNN_HOME="${CUDNN_PIP_HOME}"
export CUDNN_INCLUDE_PATH="${CUDNN_PIP_HOME}/include"
export CUDNN_LIBRARY_PATH="${CUDNN_PIP_HOME}/lib"
export NCCL_HOME="${NCCL_PIP_HOME}"
export PATH="${CUDA_HOME}/bin:${PYTHON_DIR}:${PATH}"
export CPATH="${CUDA_HOME}/include:${CUDA_CCCL_INCLUDE_HOME}:${CUDNN_PIP_HOME}/include:${NCCL_PIP_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CUDA_HOME}:${CUDNN_PIP_HOME}:${NCCL_PIP_HOME}:${CMAKE_PREFIX_PATH:-}"
export CMAKE_INCLUDE_PATH="${CUDA_HOME}/include:${CUDA_CCCL_INCLUDE_HOME}:${CUDNN_PIP_HOME}/include:${NCCL_PIP_HOME}/include:${CMAKE_INCLUDE_PATH:-}"
export CMAKE_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:${CMAKE_LIBRARY_PATH:-}"
export MAX_JOBS
export NVCC_THREADS
export NVTE_BUILD_THREADS_PER_JOB
export NVTE_CUDA_ARCHS
export FLASH_ATTN_CUDA_ARCHS
export TORCH_CUDA_ARCH_LIST
export CUDAARCHS
export CMAKE_BUILD_PARALLEL_LEVEL
export MAKEFLAGS

uv_pip() {
    "${UV_BIN}" pip "$@"
}

source_pip_install() {
    "${PYTHON_EXE}" -m pip install "$@"
}

require_nvcc_version() {
    local expected_major_minor
    expected_major_minor="$(echo "${CUDA_VERSION}" | cut -d "." -f 1,2)"

    if ! command -v nvcc >/dev/null 2>&1; then
        echo "nvcc is required for Apex/TransformerEngine/FlashAttention source builds." >&2
        echo "Install or expose a CUDA ${expected_major_minor} toolchain before continuing." >&2
        exit 1
    fi

    if ! nvcc --version | grep -q "release ${expected_major_minor}"; then
        echo "nvcc version does not match Dockerfile.stable.sglang CUDA ${CUDA_VERSION}." >&2
        nvcc --version >&2
        exit 1
    fi
}

ensure_cuda_package_layout() {
    if [ ! -e "${CUDA_HOME}/lib64" ]; then
        ln -s lib "${CUDA_HOME}/lib64"
    elif [ ! -d "${CUDA_HOME}/lib64" ]; then
        echo "CUDA lib64 path exists but is not a directory: ${CUDA_HOME}/lib64" >&2
        exit 1
    fi

    ensure_unversioned_library_links "${CUDA_HOME}/lib" \
        libcudart.so \
        libcublas.so \
        libcublasLt.so \
        libnvrtc.so \
        libnvJitLink.so \
        libcupti.so \
        libcufft.so \
        libcurand.so \
        libcusolver.so \
        libcusolverMg.so \
        libcusparse.so

    ensure_unversioned_library_links "${CUDNN_PIP_HOME}/lib" \
        libcudnn.so \
        libcudnn_graph.so \
        libcudnn_engines_runtime_compiled.so \
        libcudnn_ops.so \
        libcudnn_cnn.so \
        libcudnn_adv.so \
        libcudnn_engines_precompiled.so \
        libcudnn_heuristic.so

    ensure_unversioned_library_links "${NCCL_PIP_HOME}/lib" libnccl.so
}

ensure_unversioned_library_links() {
    local directory="$1"
    shift
    local soname

    for soname in "$@"; do
        ensure_unversioned_library_link "${directory}" "${soname}"
    done
}

ensure_unversioned_library_link() {
    local directory="$1"
    local soname="$2"
    local candidate

    if [ ! -d "${directory}" ]; then
        echo "Required library directory is missing: ${directory}" >&2
        exit 1
    fi

    if [ -e "${directory}/${soname}" ]; then
        return
    fi

    candidate="$(find "${directory}" -maxdepth 1 -name "${soname}.*" -print | sort -V | tail -n 1 || true)"
    if [ -n "${candidate}" ]; then
        ln -s "$(basename "${candidate}")" "${directory}/${soname}"
    fi
}

require_cuda_package_paths() {
    local path
    require_paths \
        "${CUDA_HOME}/bin/nvcc" \
        "${CUDA_HOME}/bin/ptxas" \
        "${CUDA_HOME}/bin/nvlink" \
        "${CUDA_HOME}/include/cuda.h" \
        "${CUDA_HOME}/include/cuda_profiler_api.h" \
        "${CUDA_HOME}/include/cupti.h" \
        "${CUDA_HOME}/include/nvml.h" \
        "${CUDA_HOME}/include/curand.h" \
        "${CUDA_HOME}/include/cusolverDn.h" \
        "${CUDA_HOME}/include/cusparse.h" \
        "${CUDA_HOME}/include/nv/target" \
        "${CUDA_CCCL_INCLUDE_HOME}/cuda/std/type_traits" \
        "${CUDA_CCCL_INCLUDE_HOME}/cub/cub.cuh" \
        "${CUDA_HOME}/nvvm/libdevice" \
        "${CUDA_HOME}/lib/libcudart.so.${CUDA_VERSION_MAJOR}" \
        "${CUDA_HOME}/lib/libcudart.so" \
        "${CUDA_HOME}/lib/libcublas.so" \
        "${CUDA_HOME}/lib/libcublasLt.so" \
        "${CUDA_HOME}/lib/libcupti.so" \
        "${CUDA_HOME}/lib/libnvJitLink.so" \
        "${CUDA_HOME}/lib/libcufft.so" \
        "${CUDA_HOME}/lib/libcurand.so" \
        "${CUDA_HOME}/lib/libcusolver.so" \
        "${CUDA_HOME}/lib/libcusparse.so" \
        "${CUDA_HOME}/lib/libcudadevrt.a" \
        "${CUDA_HOME}/lib/libcudart_static.a" \
        "${CUDA_HOME}/lib64/libcudadevrt.a" \
        "${CUDA_HOME}/lib64/libcudart_static.a" \
        "${CUDNN_PIP_HOME}/include/cudnn.h" \
        "${CUDNN_PIP_HOME}/lib/libcudnn.so.9" \
        "${CUDNN_PIP_HOME}/lib/libcudnn.so" \
        "${CUDNN_PIP_HOME}/lib/libcudnn_graph.so" \
        "${CUDNN_PIP_HOME}/lib/libcudnn_engines_runtime_compiled.so" \
        "${NCCL_PIP_HOME}/include/nccl.h" \
        "${NCCL_PIP_HOME}/lib/libnccl.so.2" \
        "${NCCL_PIP_HOME}/lib/libnccl.so"
}

require_paths() {
    local path

    for path in "$@"
    do
        if [ ! -e "${path}" ]; then
            echo "Required CUDA package path is missing: ${path}" >&2
            exit 1
        fi
    done
}

install_cuda_runtime_stack() {
    uv_pip install \
        "nvidia-cuda-cccl==${CUDA_CCCL_VERSION}" \
        "nvidia-cuda-profiler-api==${CUDA_CUPTI_VERSION}" \
        "nvidia-cuda-cupti==${CUDA_CUPTI_VERSION}" \
        "nvidia-cuda-nvcc==${CUDA_NVCC_VERSION}" \
        "nvidia-cuda-crt==${CUDA_NVCC_VERSION}" \
        "nvidia-nvvm==${CUDA_NVCC_VERSION}" \
        "nvidia-cuda-runtime==${CUDA_RUNTIME_VERSION}"
    uv_pip install \
        "nvidia-cudnn-cu${CUDA_VERSION_MAJOR}==${CUDNN_VERSION}" \
        "nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION}" \
        "nvidia-cublas==${CUBLAS_VERSION}" \
        "nvidia-cuda-nvrtc==${CUDA_NVRTC_VERSION}" \
        "nvidia-nvml-dev==${CUDA_NVML_DEV_VERSION}" \
        "nvidia-nvjitlink==${CUDA_NVJITLINK_VERSION}" \
        "nvidia-cufft==${CUDA_CUFFT_VERSION}" \
        "nvidia-curand==${CUDA_CURAND_VERSION}" \
        "nvidia-cusparse==${CUDA_CUSPARSE_VERSION}" \
        "nvidia-cusolver==${CUDA_CUSOLVER_VERSION}"
    uv_pip install "cupy-cuda${CUDA_VERSION_MAJOR}x==${CUPY_VERSION}"
    uv_pip install --no-deps --upgrade "nvidia-nccl-cu${CUDA_VERSION_MAJOR}==${NCCL_VERSION}"
    ensure_cuda_package_layout
    require_cuda_package_paths
}

install_conda_activation_hooks() {
    local activate_dir="${PYTHON_PREFIX}/etc/conda/activate.d"
    local deactivate_dir="${PYTHON_PREFIX}/etc/conda/deactivate.d"
    local activate_hook="${activate_dir}/verl_cuda_stack.sh"
    local deactivate_hook="${deactivate_dir}/verl_cuda_stack.sh"

    mkdir -p "${activate_dir}" "${deactivate_dir}"

    cat > "${activate_hook}" <<EOF
# Generated by scripts/install_vllm_sglang_mcore.sh.
export VERL_CUDA_STACK_OLD_PATH="\${PATH:-}"
export VERL_CUDA_STACK_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export VERL_CUDA_STACK_OLD_LIBRARY_PATH="\${LIBRARY_PATH:-}"
export VERL_CUDA_STACK_OLD_CPATH="\${CPATH:-}"
export VERL_CUDA_STACK_OLD_CMAKE_PREFIX_PATH="\${CMAKE_PREFIX_PATH:-}"
export VERL_CUDA_STACK_OLD_CMAKE_INCLUDE_PATH="\${CMAKE_INCLUDE_PATH:-}"
export VERL_CUDA_STACK_OLD_CMAKE_LIBRARY_PATH="\${CMAKE_LIBRARY_PATH:-}"
export VERL_CUDA_STACK_OLD_SGLANG_NUMA_BIND_V2="\${SGLANG_NUMA_BIND_V2-__VERL_UNSET__}"

export CUDA_HOME="${CUDA_HOME}"
export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CUDNN_PATH="${CUDNN_PIP_HOME}"
export CUDNN_HOME="${CUDNN_PIP_HOME}"
export CUDNN_INCLUDE_PATH="${CUDNN_PIP_HOME}/include"
export CUDNN_LIBRARY_PATH="${CUDNN_PIP_HOME}/lib"
export NCCL_HOME="${NCCL_PIP_HOME}"

export PATH="${CUDA_HOME}/bin:${PYTHON_DIR}:\${PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:\${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${CUDA_CCCL_INCLUDE_HOME}:${CUDNN_PIP_HOME}/include:${NCCL_PIP_HOME}/include:\${CPATH:-}"
export CMAKE_PREFIX_PATH="${CUDA_HOME}:${CUDNN_PIP_HOME}:${NCCL_PIP_HOME}:\${CMAKE_PREFIX_PATH:-}"
export CMAKE_INCLUDE_PATH="${CUDA_HOME}/include:${CUDA_CCCL_INCLUDE_HOME}:${CUDNN_PIP_HOME}/include:${NCCL_PIP_HOME}/include:\${CMAKE_INCLUDE_PATH:-}"
export CMAKE_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:\${CMAKE_LIBRARY_PATH:-}"
export SGLANG_NUMA_BIND_V2=0
EOF

    cat > "${deactivate_hook}" <<'EOF'
# Generated by scripts/install_vllm_sglang_mcore.sh.
export PATH="${VERL_CUDA_STACK_OLD_PATH:-${PATH:-}}"
export LD_LIBRARY_PATH="${VERL_CUDA_STACK_OLD_LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${VERL_CUDA_STACK_OLD_LIBRARY_PATH:-}"
export CPATH="${VERL_CUDA_STACK_OLD_CPATH:-}"
export CMAKE_PREFIX_PATH="${VERL_CUDA_STACK_OLD_CMAKE_PREFIX_PATH:-}"
export CMAKE_INCLUDE_PATH="${VERL_CUDA_STACK_OLD_CMAKE_INCLUDE_PATH:-}"
export CMAKE_LIBRARY_PATH="${VERL_CUDA_STACK_OLD_CMAKE_LIBRARY_PATH:-}"
if [ "${VERL_CUDA_STACK_OLD_SGLANG_NUMA_BIND_V2:-__VERL_UNSET__}" = "__VERL_UNSET__" ]; then
    unset SGLANG_NUMA_BIND_V2
else
    export SGLANG_NUMA_BIND_V2="${VERL_CUDA_STACK_OLD_SGLANG_NUMA_BIND_V2}"
fi
unset VERL_CUDA_STACK_OLD_PATH
unset VERL_CUDA_STACK_OLD_LD_LIBRARY_PATH
unset VERL_CUDA_STACK_OLD_LIBRARY_PATH
unset VERL_CUDA_STACK_OLD_CPATH
unset VERL_CUDA_STACK_OLD_CMAKE_PREFIX_PATH
unset VERL_CUDA_STACK_OLD_CMAKE_INCLUDE_PATH
unset VERL_CUDA_STACK_OLD_CMAKE_LIBRARY_PATH
unset VERL_CUDA_STACK_OLD_SGLANG_NUMA_BIND_V2
EOF
}

verify_pinned_python_packages() {
    "${PYTHON_EXE}" - <<PY
import importlib.metadata as md
import sys

expected = {
    "nvidia-cublas": "${CUBLAS_VERSION}",
    "nvidia-cuda-nvrtc": "${CUDA_NVRTC_VERSION}",
    "nvidia-cudnn-cu${CUDA_VERSION_MAJOR}": "${CUDNN_VERSION}",
    "nvidia-nccl-cu${CUDA_VERSION_MAJOR}": "${NCCL_VERSION}",
    "cupy-cuda${CUDA_VERSION_MAJOR}x": "${CUPY_VERSION}",
    "transformers": "5.6.1",
    "apache-tvm-ffi": "0.1.12",
    "nvidia-cutlass-dsl": "4.6.0.dev0",
    "flash-attn-4": "4.0.0b20",
    "sglang": "${SGLANG_VERSION}",
    "sglang-kernel": "${SGLANG_KERNEL_VERSION}+cu${CUDA_VERSION_MAJOR}0",
}

failures = []
for package, expected_version in expected.items():
    try:
        actual_version = md.version(package)
    except md.PackageNotFoundError:
        failures.append(f"{package}: missing, expected {expected_version}")
        continue
    if actual_version != expected_version:
        failures.append(f"{package}: {actual_version}, expected {expected_version}")

if failures:
    print("Pinned package verification failed:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("Pinned package verification passed.")
PY
}

verify_expected_pip_check_conflicts() {
    local check_output
    check_output="$(mktemp)"

    if "${PYTHON_EXE}" -m pip check > "${check_output}" 2>&1; then
        rm -f "${check_output}"
        return
    fi

    "${PYTHON_EXE}" - "${check_output}" <<PY
from pathlib import Path
import importlib.metadata as md
import sys

path = Path(sys.argv[1])
lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

torch_version = md.version("torch")
allowed = {
    "sgl-deep-gemm 0.1.0 has requirement apache-tvm-ffi==0.1.9, but you have apache-tvm-ffi 0.1.12.",
    "sglang ${SGLANG_VERSION} has requirement apache-tvm-ffi==0.1.9, but you have apache-tvm-ffi 0.1.12.",
    "sglang ${SGLANG_VERSION} has requirement nvidia-cutlass-dsl==4.5.0, but you have nvidia-cutlass-dsl 4.6.0.dev0.",
    "sglang ${SGLANG_VERSION} has requirement transformers==5.6.0, but you have transformers 5.6.1.",
    f"torch {torch_version} has requirement nvidia-cudnn-cu${CUDA_VERSION_MAJOR}==9.19.0.56; platform_system == \\"Linux\\", but you have nvidia-cudnn-cu${CUDA_VERSION_MAJOR} ${CUDNN_VERSION}.",
    f"torch {torch_version} has requirement nvidia-nccl-cu${CUDA_VERSION_MAJOR}==2.28.9; platform_system == \\"Linux\\", but you have nvidia-nccl-cu${CUDA_VERSION_MAJOR} ${NCCL_VERSION}.",
}

unexpected = [line for line in lines if line not in allowed]
if unexpected:
    print("Unexpected pip check conflicts:")
    for line in unexpected:
        print(f"  - {line}")
    print("All pip check output:")
    for line in lines:
        print(f"  - {line}")
    sys.exit(1)

print("Only expected pip check conflicts remain:")
for line in lines:
    print(f"  - {line}")
PY
    local status=$?
    rm -f "${check_output}"
    return "${status}"
}

echo "Using Python: ${PYTHON_EXE}"
echo "Using uv: $(${UV_BIN} --version)"
echo "Target: Dockerfile.stable.sglang / sglang ${SGLANG_VERSION} / CUDA ${CUDA_VERSION}"
echo "Build parallelism: MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS}, CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}, MAKEFLAGS=${MAKEFLAGS}"
echo "CUDA arch target: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}, FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}, CUDAARCHS=${CUDAARCHS}"
mkdir -p "${BUILD_LOG_DIR}"
echo "Source build logs: ${BUILD_LOG_DIR}"
echo "Preserve pip build temp: ${PRESERVE_BUILD_TEMP}"

echo "1. Install SGLang base-image equivalent"
uv_pip install --prerelease allow --torch-backend "${CUDA_BACKEND}" "sglang[all]==${SGLANG_VERSION}"
# transformers 5.6.0 expects the pre-0.13 Hugging Face kernels API. Current
# PyPI can otherwise resolve kernels 0.16, which breaks import-time SGLang
# patches before the server can even start.
uv_pip install "kernels>=0.12,<0.13"

echo "2. Install Dockerfile base utilities and CUDA/cuDNN/NCCL runtime"
uv_pip install pybind11 nvidia-mathdx cmake ninja packaging wheel setuptools build
install_cuda_runtime_stack
install_conda_activation_hooks

echo "3. Build and install Apex as in Dockerfile.stable.sglang"
require_nvcc_version
{
    MAX_JOBS="${MAX_JOBS}" source_pip_install -v \
        "${PIP_NO_CLEAN_ARGS[@]}" \
        --disable-pip-version-check \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        git+https://github.com/NVIDIA/apex.git
} 2>&1 | tee "${BUILD_LOG_DIR}/apex.log"

echo "4. Build and install TransformerEngine ${TRANSFORMER_ENGINE_VERSION}"
{
    NVTE_FRAMEWORK=pytorch \
    MAX_JOBS="${MAX_JOBS}" \
    NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB}" \
    source_pip_install \
        "${PIP_NO_CLEAN_ARGS[@]}" \
        --resume-retries 999 \
        --no-build-isolation \
        "git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_VERSION}"
} 2>&1 | tee "${BUILD_LOG_DIR}/transformer_engine.log"
echo "4b. Restore CUDA/cuDNN/NCCL pins after TransformerEngine dependency resolution"
install_cuda_runtime_stack

echo "5. Install Dockerfile Python utilities"
uv_pip install codetiming mathruler pylatexenc cachetools pytest-asyncio

echo "6. Align FlashAttention with Dockerfile.stable.sglang plus B200 SGLang overlay"
# Install FA2 exactly like Dockerfile.stable.sglang for trainer/Megatron utility
# imports, then restore FA4's CuTe files for SGLang's Blackwell multimodal path.
# FA2 also ships a flash_attn/cute tree, so the order matters. Install the FA4
# beta20 runtime island explicitly; letting the FA4 resolver run freely would
# move torch/CUDA/cuDNN/NCCL away from the Dockerfile-aligned stack.
{
    source_pip_install \
        "${PIP_NO_CLEAN_ARGS[@]}" \
        --no-build-isolation \
        "flash_attn==${FLASH_ATTENTION_VERSION}"
} 2>&1 | tee "${BUILD_LOG_DIR}/flash_attn.log"
echo "6b. Restore CUDA/cuDNN/NCCL pins after FlashAttention dependency resolution"
install_cuda_runtime_stack
uv_pip install --prerelease allow --no-deps \
    "nvidia-cutlass-dsl==4.6.0.dev0" \
    "nvidia-cutlass-dsl-libs-base==4.6.0.dev0" \
    "nvidia-cutlass-dsl-libs-cu13==4.6.0.dev0" \
    "quack-kernels==0.5.3" \
    "apache-tvm-ffi==0.1.12" \
    "protobuf==6.33.6"
uv_pip install --prerelease allow --reinstall --no-deps "${FLASH_ATTENTION_4_SPEC}"

echo "7. Install RL/runtime utility packages"
uv_pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    "numpy>=2.0.0" \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    "ray[default]>=2.41.0" \
    torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "transformers==5.6.1" \
    wandb \
    "packaging>=20.0" \
    tensorboard
install_cuda_runtime_stack
uv_pip install --no-deps "trl==${TRL_VERSION}"
uv_pip install nvtx matplotlib liger_kernel TransferQueue==0.1.8
uv_pip install --no-deps torchcodec --index-url "https://download.pytorch.org/whl/${CUDA_BACKEND}"
uv_pip install qwen-vl-utils==0.0.14

echo "8. Install SGLang kernel wheel from the Dockerfile URL"
if [ "${ARCH}" = "aarch64" ]; then
    uv_pip show "sglang-kernel"
else
    SGLANG_KERNEL_WHL="https://github.com/sgl-project/whl/releases/download/v${SGLANG_KERNEL_VERSION}/sglang_kernel-${SGLANG_KERNEL_VERSION}+cu${CUDA_VERSION_MAJOR}0-cp310-abi3-manylinux2014_x86_64.whl#sha256=4e7ce619274234d182b20da883fcf1d20e7e55cbea90d62244e2b6a3d6c0fc85"
    uv_pip install --reinstall "${SGLANG_KERNEL_WHL}"
fi

echo "9. Install mbridge and Megatron-LM from Dockerfile.stable.sglang"
uv_pip install -U git+https://github.com/ISEEKYAN/mbridge.git@main
uv_pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_VERSION}"

echo "10. Install local verl without dependency resolution"
uv_pip install --no-deps -e .

echo "10b. Restore CUDA/cuDNN/NCCL pins before import verification"
install_cuda_runtime_stack
verify_pinned_python_packages
verify_expected_pip_check_conflicts

echo "11. Verify critical imports"
"${PYTHON_BIN}" - <<'PY'
import importlib

for name in [
    "torch",
    "sglang",
    "sgl_kernel",
    "flash_attn",
    "transformer_engine",
    "cupy",
    "megatron",
    "verl",
]:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: {version}")

from flash_attn import flash_attn_func
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input
from flash_attn.cute import flash_attn_func as flash_attn_4_func
from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
import apex
import cupy
import megatron.core
import sglang
import transformer_engine.pytorch
from verl.checkpoint_engine import CheckpointEngineRegistry
import verl.checkpoint_engine.nccl_checkpoint_engine

print("flash_attn_func:", flash_attn_func)
print("flash_attn_varlen_func:", flash_attn_varlen_func)
print("unpad_input:", unpad_input)
print("flash_attn_4_func:", flash_attn_4_func)
print("flash_attn_4_varlen_func:", flash_attn_4_varlen_func)
print("cupy:", cupy.__version__)
assert "nccl" in CheckpointEngineRegistry._registry

import torch
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
PY

echo "Successfully installed Dockerfile.stable.sglang-aligned packages"
echo "Conda activation hooks were installed under ${PYTHON_PREFIX}/etc/conda."
echo "For a shell that was already active before this install, re-run conda activate or open a new shell before training."
