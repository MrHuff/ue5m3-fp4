# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:25.10-py3@sha256:5c8302e4628ac326c412368675cc462b8aee2f96326a3bf817304a83816f179a

ARG MAX_JOBS=8
ARG TRANSFORMER_ENGINE_REVISION=01aef4fc721bd12fd09cd56d53a314aee1b953d6
ARG TRANSFORMER_ENGINE_VERSION=2.16.0.dev0+01aef4fc
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=${MAX_JOBS} \
    NVTE_FRAMEWORK=pytorch \
    NVTE_CUDA_ARCHS=100a

WORKDIR /opt/ue5m3-fp4
COPY reproduce/environment/requirements-ngc-25.10.lock /tmp/requirements.lock
COPY reproduce/environment/compiled-requirements-ngc-25.10.lock /tmp/compiled.lock

RUN python -m pip install --upgrade \
      "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1" \
      "cmake==3.31.6" "ninja==1.13.0" "pybind11[global]==3.0.1" \
      "packaging==25.0" \
    && python -m pip install --no-deps --require-hashes \
      --requirement /tmp/requirements.lock \
    && python -m pip install --no-build-isolation --require-hashes \
      --requirement /tmp/compiled.lock

COPY . /opt/ue5m3-fp4
RUN test -f third_party/torchtitan/torchtitan/train.py \
    && test -f third_party/TransformerEngine/setup.py \
    && observed_te_revision="$(git -C third_party/TransformerEngine rev-parse HEAD)" \
    && test "$observed_te_revision" = "$TRANSFORMER_ENGINE_REVISION" \
    && git -C third_party/TransformerEngine submodule update --init --recursive \
    && (python -m pip uninstall --yes \
        transformer_engine transformer-engine-cu12 transformer-engine-cu13 \
        transformer-engine-torch || true)
RUN NVTE_FRAMEWORK=pytorch \
    MAX_JOBS="${MAX_JOBS}" \
    python -m pip install --no-build-isolation --force-reinstall --no-deps \
      ./third_party/TransformerEngine \
    && python -c \
      "import transformer_engine as te; assert te.__version__ == '${TRANSFORMER_ENGINE_VERSION}', te.__version__"
RUN test -f third_party/torchtitan/torchtitan/train.py \
    && python -m pip install --no-deps --editable . \
    && PYTHONPATH="/opt/ue5m3-fp4/third_party/torchtitan" \
       python reproduce/scripts/verify_runtime.py --allow-no-gpu

ENV PYTHONPATH=/opt/ue5m3-fp4/src:/opt/ue5m3-fp4/third_party/torchtitan
ENTRYPOINT ["bash"]
