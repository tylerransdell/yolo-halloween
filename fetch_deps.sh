#!/usr/bin/env bash
#
# Populate a local deps directory with the Intel userspace runtime packages needed for
# OpenVINO NPU *and* GPU inference:
#
#   NPU : intel-driver-compiler-npu, intel-level-zero-npu, intel-fw-npu
#         <- intel/linux-npu-driver
#   GPU : intel-opencl-icd, libze-intel-gpu1, libigdgmm12, intel-ocloc
#         <- intel/compute-runtime
#         intel-igc-core-2, intel-igc-opencl-2
#         <- intel/intel-graphics-compiler (hard deps of the two runtime packages)
#
# These .debs are NOT vendored in the repo -- this script materialises them from Intel's
# upstream GitHub releases. The Dockerfile runs it automatically when ./deps is missing, so a
# plain `docker compose build` works on a fresh clone. Run it by hand to cache them locally
# (the Dockerfile then reuses ./deps and skips the downloads).
#
# Usage:
#   ./fetch_deps.sh [OUT_DIR]        # OUT_DIR defaults to ./deps
#   FORCE=1 ./fetch_deps.sh          # re-download even if OUT_DIR is already populated
#
# Version overrides (defaults are pinned below for reproducible builds):
#   NPU_DRIVER_VERSION  NPU_DRIVER_BUILD  NPU_DRIVER_UBUNTU
#   GPU_CR_VERSION  GPU_IGC_VERSION  GPU_IGC_BUILD  GPU_GMM_VERSION
set -euo pipefail

OUT_DIR="${1:-./deps}"

# --- pinned upstream versions: single source of truth (Dockerfile forwards --build-args) ---
NPU_DRIVER_VERSION="${NPU_DRIVER_VERSION:-1.38.0}"
NPU_DRIVER_BUILD="${NPU_DRIVER_BUILD:-20260910-34487311128}"
NPU_DRIVER_UBUNTU="${NPU_DRIVER_UBUNTU:-ubuntu2404}"    # ubuntu2204 | ubuntu2404 | ubuntu2604

GPU_CR_VERSION="${GPU_CR_VERSION:-26.35.39758.10}"      # intel/compute-runtime
GPU_IGC_VERSION="${GPU_IGC_VERSION:-2.41.5}"            # intel/intel-graphics-compiler
GPU_IGC_BUILD="${GPU_IGC_BUILD:-22716}"
GPU_GMM_VERSION="${GPU_GMM_VERSION:-22.10.0}"           # libigdgmm12, bundled with the CR release

# Idempotent: already populated -> nothing to do (unless FORCE=1).
if compgen -G "${OUT_DIR}/*.deb" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
    echo "[deps] ${OUT_DIR} already contains .debs; skipping (set FORCE=1 to refresh)"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/*.deb

# --- NPU driver: shipped as a tarball containing .deb + .ddeb + .asc ---
NPU_ASSET="linux-npu-driver-v${NPU_DRIVER_VERSION}.${NPU_DRIVER_BUILD}-${NPU_DRIVER_UBUNTU}.tar.gz"
NPU_URL="https://github.com/intel/linux-npu-driver/releases/download/v${NPU_DRIVER_VERSION}/${NPU_ASSET}"
echo "[deps] NPU driver v${NPU_DRIVER_VERSION} (${NPU_DRIVER_UBUNTU})"
echo "[deps]   ${NPU_URL}"
curl -fsSL -o "${tmp}/${NPU_ASSET}" "${NPU_URL}"
mkdir -p "${tmp}/npu"
tar -xzf "${tmp}/${NPU_ASSET}" -C "${tmp}/npu"
# '*.deb' copies only installable runtime packages (skips '.deb.asc' and '.ddeb').
find "${tmp}/npu" -name '*.deb' -exec cp -v {} "${OUT_DIR}/" \;

# --- GPU runtime: individual .deb assets ---
CR_BASE="https://github.com/intel/compute-runtime/releases/download/${GPU_CR_VERSION}"
IGC_BASE="https://github.com/intel/intel-graphics-compiler/releases/download/v${GPU_IGC_VERSION}"
GPU_ASSETS=(
    "${CR_BASE}/intel-opencl-icd_${GPU_CR_VERSION}-0_amd64.deb"
    "${CR_BASE}/libze-intel-gpu1_${GPU_CR_VERSION}-0_amd64.deb"
    "${CR_BASE}/intel-ocloc_${GPU_CR_VERSION}-0_amd64.deb"
    "${CR_BASE}/libigdgmm12_${GPU_GMM_VERSION}_amd64.deb"
    "${IGC_BASE}/intel-igc-core-2_${GPU_IGC_VERSION}+${GPU_IGC_BUILD}_amd64.deb"
    "${IGC_BASE}/intel-igc-opencl-2_${GPU_IGC_VERSION}+${GPU_IGC_BUILD}_amd64.deb"
)
echo "[deps] GPU runtime ${GPU_CR_VERSION} + IGC ${GPU_IGC_VERSION}"
for url in "${GPU_ASSETS[@]}"; do
    echo "[deps]   ${url}"
    curl -fsSL -o "${OUT_DIR}/$(basename "${url}")" "${url}"
done

echo "[deps] wrote $(find "${OUT_DIR}" -name '*.deb' | wc -l) package(s) to ${OUT_DIR}"
ls -1 "${OUT_DIR}"