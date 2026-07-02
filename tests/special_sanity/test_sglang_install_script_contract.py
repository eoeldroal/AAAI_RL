from pathlib import Path


SCRIPT = Path("scripts/install_vllm_sglang_mcore.sh")


def _script_text() -> str:
    return SCRIPT.read_text()


def test_sglang_install_script_preserves_fa4_and_installs_fa2():
    script = _script_text()

    assert 'uv_pip uninstall -y "flash-attn-4"' not in script
    fa2_install = '"flash_attn==${FLASH_ATTENTION_VERSION}"'
    fa4_restore = 'uv_pip install --prerelease allow --reinstall --no-deps "${FLASH_ATTENTION_4_SPEC}"'

    assert "flash-attn-4[cu13]==4.0.0b20" in script
    for fa4_dependency in [
        '"nvidia-cutlass-dsl==4.6.0.dev0"',
        '"nvidia-cutlass-dsl-libs-base==4.6.0.dev0"',
        '"nvidia-cutlass-dsl-libs-cu13==4.6.0.dev0"',
        '"quack-kernels==0.5.3"',
        '"apache-tvm-ffi==0.1.12"',
        '"protobuf==6.33.6"',
    ]:
        assert fa4_dependency in script
    assert fa2_install in script
    assert '"${BUILD_LOG_DIR}/flash_attn.log"' in script
    assert fa4_restore in script
    assert script.index(fa2_install) < script.index(fa4_restore)


def test_sglang_install_script_bootstraps_uv_for_one_command_install():
    script = _script_text()

    uv_bootstrap = 'echo "uv not found; installing uv into ${PYTHON_PREFIX}"'
    uv_install = '"${PYTHON_EXE}" -m pip install -U uv'
    uv_pip_function = "uv_pip() {"

    assert uv_bootstrap in script
    assert uv_install in script
    assert "UV_BIN=\"${PYTHON_DIR}/uv\"" in script
    assert script.index(uv_bootstrap) < script.index(uv_pip_function)
    assert script.index(uv_install) < script.index(uv_pip_function)


def test_sglang_install_script_pins_hf_kernels_for_transformers_v5():
    script = _script_text()

    assert '"kernels>=0.12,<0.13"' in script


def test_sglang_install_script_installs_source_build_tools():
    script = _script_text()

    for build_tool in ["cmake", "ninja", "packaging", "wheel", "setuptools", "build"]:
        assert build_tool in script


def test_sglang_install_script_preserves_source_build_diagnostics():
    script = _script_text()

    assert "BUILD_LOG_DIR=" in script
    assert "PIP_NO_CLEAN_ARGS=" in script
    assert "--no-clean" in script
    for build_log in [
        '"${BUILD_LOG_DIR}/apex.log"',
        '"${BUILD_LOG_DIR}/transformer_engine.log"',
        '"${BUILD_LOG_DIR}/flash_attn.log"',
    ]:
        assert build_log in script


def test_sglang_install_script_installs_cuda_profiler_api_headers_for_apex():
    script = _script_text()

    assert '"nvidia-cuda-profiler-api==${CUDA_CUPTI_VERSION}"' in script
    assert '"${CUDA_HOME}/include/cuda_profiler_api.h"' in script


def test_sglang_install_script_pins_source_build_cuda_toolkit_family():
    script = _script_text()

    for cuda_package in [
        '"nvidia-cuda-cupti==${CUDA_CUPTI_VERSION}"',
        '"nvidia-nvml-dev==${CUDA_NVML_DEV_VERSION}"',
        '"nvidia-nvjitlink==${CUDA_NVJITLINK_VERSION}"',
        '"nvidia-cufft==${CUDA_CUFFT_VERSION}"',
        '"nvidia-curand==${CUDA_CURAND_VERSION}"',
        '"nvidia-cusparse==${CUDA_CUSPARSE_VERSION}"',
        '"nvidia-cusolver==${CUDA_CUSOLVER_VERSION}"',
    ]:
        assert cuda_package in script

    for required_path in [
        '"${CUDA_HOME}/include/cupti.h"',
        '"${CUDA_HOME}/include/nvml.h"',
        '"${CUDA_HOME}/include/curand.h"',
        '"${CUDA_HOME}/include/cusolverDn.h"',
        '"${CUDA_HOME}/include/cusparse.h"',
        '"${CUDA_HOME}/lib/libcupti.so"',
        '"${CUDA_HOME}/lib/libnvJitLink.so"',
        '"${CUDA_HOME}/lib/libcufft.so"',
        '"${CUDA_HOME}/lib/libcurand.so"',
        '"${CUDA_HOME}/lib/libcusolver.so"',
        '"${CUDA_HOME}/lib/libcusparse.so"',
    ]:
        assert required_path in script


def test_sglang_install_script_installs_cupy_for_checkpoint_engine_backends():
    script = _script_text()

    assert '"cupy-cuda${CUDA_VERSION_MAJOR}x==${CUPY_VERSION}"' in script
    assert '"cupy-cuda${CUDA_VERSION_MAJOR}x": "${CUPY_VERSION}"' in script
    assert '"nccl" in CheckpointEngineRegistry._registry' in script


def test_sglang_install_script_installs_conda_activation_hooks_for_cuda_runtime():
    script = _script_text()

    assert "install_conda_activation_hooks() {" in script
    assert '${PYTHON_PREFIX}/etc/conda/activate.d' in script
    assert '${PYTHON_PREFIX}/etc/conda/deactivate.d' in script
    for exported_path in [
        'export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:',
        'export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDNN_PIP_HOME}/lib:${NCCL_PIP_HOME}/lib:',
        'export CPATH="${CUDA_HOME}/include:${CUDA_CCCL_INCLUDE_HOME}:${CUDNN_PIP_HOME}/include:',
        'export CUDA_HOME="${CUDA_HOME}"',
        'export CUDNN_HOME="${CUDNN_PIP_HOME}"',
        'export NCCL_HOME="${NCCL_PIP_HOME}"',
        "export SGLANG_NUMA_BIND_V2=0",
    ]:
        assert exported_path in script
    assert 'export VERL_CUDA_STACK_OLD_SGLANG_NUMA_BIND_V2="\\${SGLANG_NUMA_BIND_V2-__VERL_UNSET__}"' in script
    assert "unset VERL_CUDA_STACK_OLD_SGLANG_NUMA_BIND_V2" in script
    assert "install_cuda_runtime_stack\ninstall_conda_activation_hooks" in script


def test_sglang_install_script_restores_cuda_pins_after_source_installs():
    script = _script_text()

    restore_call = "install_cuda_runtime_stack"
    assert "install_cuda_runtime_stack() {" in script

    transformer_engine_step = 'echo "4. Build and install TransformerEngine ${TRANSFORMER_ENGINE_VERSION}"'
    transformer_engine_restore = 'echo "4b. Restore CUDA/cuDNN/NCCL pins after TransformerEngine dependency resolution"'
    flash_attention_step = 'echo "6. Align FlashAttention with Dockerfile.stable.sglang plus B200 SGLang overlay"'
    flash_attention_restore = 'echo "6b. Restore CUDA/cuDNN/NCCL pins after FlashAttention dependency resolution"'
    fa4_restore = 'uv_pip install --prerelease allow --reinstall --no-deps "${FLASH_ATTENTION_4_SPEC}"'
    final_restore = 'echo "10b. Restore CUDA/cuDNN/NCCL pins before import verification"'
    verify_pins = "\nverify_pinned_python_packages\n"
    verify_pip_conflicts = "\nverify_expected_pip_check_conflicts\n"
    import_verification = 'echo "11. Verify critical imports"'

    assert script.count(restore_call) >= 4
    assert "verify_pinned_python_packages() {" in script
    assert "verify_expected_pip_check_conflicts() {" in script
    assert script.index(transformer_engine_step) < script.index(transformer_engine_restore)
    assert script.index(transformer_engine_restore) < script.index(flash_attention_step)
    assert script.index(flash_attention_step) < script.index(flash_attention_restore)
    assert script.index(flash_attention_restore) < script.index(fa4_restore)
    assert script.index(final_restore) < script.index(import_verification)
    assert script.index(final_restore) < script.index(verify_pins)
    assert script.index(final_restore) < script.index(verify_pip_conflicts)
    assert script.index(verify_pins) < script.index(import_verification)
    assert script.index(verify_pip_conflicts) < script.index(import_verification)


def test_sglang_install_script_fails_closed_on_unexpected_package_drift():
    script = _script_text()

    for pinned_package in [
        '"nvidia-cublas": "${CUBLAS_VERSION}"',
        '"nvidia-cuda-nvrtc": "${CUDA_NVRTC_VERSION}"',
        '"nvidia-cudnn-cu${CUDA_VERSION_MAJOR}": "${CUDNN_VERSION}"',
        '"nvidia-nccl-cu${CUDA_VERSION_MAJOR}": "${NCCL_VERSION}"',
        '"transformers": "5.6.1"',
        '"apache-tvm-ffi": "0.1.12"',
        '"nvidia-cutlass-dsl": "4.6.0.dev0"',
        '"flash-attn-4": "4.0.0b20"',
        '"sglang": "${SGLANG_VERSION}"',
        '"sglang-kernel": "${SGLANG_KERNEL_VERSION}+cu${CUDA_VERSION_MAJOR}0"',
    ]:
        assert pinned_package in script

    for expected_conflict in [
        "sgl-deep-gemm 0.1.0 has requirement apache-tvm-ffi==0.1.9",
        "sglang ${SGLANG_VERSION} has requirement apache-tvm-ffi==0.1.9",
        "sglang ${SGLANG_VERSION} has requirement nvidia-cutlass-dsl==4.5.0",
        "sglang ${SGLANG_VERSION} has requirement transformers==5.6.0",
        "torch {torch_version} has requirement nvidia-cudnn-cu${CUDA_VERSION_MAJOR}==9.19.0.56",
        "torch {torch_version} has requirement nvidia-nccl-cu${CUDA_VERSION_MAJOR}==2.28.9",
        "Unexpected pip check conflicts:",
    ]:
        assert expected_conflict in script


def test_sglang_install_script_installs_verl_runtime_dependencies_before_local_install():
    script = _script_text()

    runtime_step = 'echo "7. Install RL/runtime utility packages"'
    local_install_step = 'echo "10. Install local verl without dependency resolution"'
    required_runtime_deps = [
        '"ray[default]>=2.41.0"',
        '"tensordict>=0.8.0,<=0.10.0,!=0.9.0"',
        '"transformers==5.6.1"',
        "torchdata",
        "wandb",
        "tensorboard",
        "TransferQueue==0.1.8",
    ]

    assert script.index(runtime_step) < script.index(local_install_step)
    for dependency in required_runtime_deps:
        assert dependency in script


def test_sglang_install_script_verifies_runtime_import_contracts():
    script = _script_text()

    for required_import in [
        "from flash_attn import flash_attn_func",
        "from flash_attn import flash_attn_varlen_func",
        "from flash_attn.bert_padding import unpad_input",
        "from flash_attn.cute import flash_attn_func as flash_attn_4_func",
        "from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func",
        "import transformer_engine.pytorch",
        "import apex",
        "import cupy",
        "import megatron.core",
        "import sglang",
        "from verl.checkpoint_engine import CheckpointEngineRegistry",
        "import verl.checkpoint_engine.nccl_checkpoint_engine",
    ]:
        assert required_import in script
