FROM python:3.10-slim

WORKDIR /app

# System deps: OpenCV-headless runtime, video capture, and model build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Intel NPU + GPU (OpenCL / Level Zero) userspace runtimes -> enables NPU and GPU inference.
# The .debs are NOT vendored in the repo. If ./deps is missing from the build context we fetch
# them from Intel's upstream GitHub releases via ./fetch_deps.sh; if ./deps is present we use it
# as-is (populate it locally with ./fetch_deps.sh). Version defaults live in that script; the
# ARGs below let a one-off build override them (--build-arg).
#   NPU: https://github.com/intel/linux-npu-driver/releases
#   GPU: https://github.com/intel/compute-runtime/releases
#        https://github.com/intel/intel-graphics-compiler/releases
# /dev/accel (NPU) and /dev/dri (GPU) are provided to the container by compose.
ARG NPU_DRIVER_VERSION=
ARG NPU_DRIVER_BUILD=
ARG NPU_DRIVER_UBUNTU=
ARG GPU_CR_VERSION=
ARG GPU_IGC_VERSION=
ARG GPU_IGC_BUILD=
ARG GPU_GMM_VERSION=
RUN set -eux; \
    if ! ls ./deps/*.deb >/dev/null 2>&1; then \
        echo "[deps] ./deps not in build context -> fetching upstream"; \
        bash ./fetch_deps.sh ./deps; \
    else \
        echo "[deps] using vendored ./deps"; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends libze1 ocl-icd-libopencl1 ./deps/*.deb; \
    rm -rf ./deps /var/lib/apt/lists/*

# Ensure data directories exist for config and persistent models
RUN mkdir -p /data/models

EXPOSE 8777
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8777", "--no-access-log"]
