#!/usr/bin/env bash
# =============================================================================
# Pull the CUDA-Q devdeps image (no sudo, via skopeo+udocker) and build the
# tensornet-mps-mgpu backend inside it -- one command, no long lines to paste.
#
#   conda activate cudaq-env
#   ./udocker_build.sh                 # fetch image + build
#   ./udocker_build.sh run             # run the VQE on GPUs (after build)
#
# Override the image:  IMG_REF=ghcr.io/nvidia/cuda-quantum-devdeps:<tag> ./udocker_build.sh
# =============================================================================
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMG_REF="${IMG_REF:-ghcr.io/nvidia/cuda-quantum-devdeps:ext-cu12-gcc11-main}"
LOCAL_REF="cudaq-devdeps:local"        # name inside udocker
CONT="${CONT:-cqdev}"                  # udocker container name
TAR="$PROJECT_DIR/devdeps.tar"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || true
export UDOCKER_DIR="${UDOCKER_DIR:-$HOME/.udocker}"

# v2 registries.conf so the conda skopeo doesn't choke on its bundled v1 file.
RC="$HOME/.config/containers/registries.conf"
mkdir -p "$(dirname "$RC")"
grep -q 'unqualified-search-registries' "$RC" 2>/dev/null || \
  printf 'unqualified-search-registries = ["docker.io"]\n' > "$RC"
SKOPEO=(skopeo --registries-conf "$RC" --insecure-policy)

command -v udocker >/dev/null || python -m pip install -q udocker
udocker install >/dev/null 2>&1 || true
command -v skopeo >/dev/null || { echo "ERROR: skopeo missing (conda install -c conda-forge skopeo)"; exit 1; }

# ---- run mode ---------------------------------------------------------------
if [ "${1:-}" = "run" ]; then
  echo ">> running VQE in container (GPUs $GPUS) ..."
  exec udocker run --nvidia -v "$PROJECT_DIR":/work "$CONT" \
       bash -lc "cd /work && bash run.sh ${*:2}"
fi

# ---- 1. fetch image (amd64) + load into udocker -----------------------------
if ! udocker inspect "$CONT" >/dev/null 2>&1; then
  echo ">> tags matching this image family (sanity check):"
  "${SKOPEO[@]}" list-tags "docker://${IMG_REF%%:*}" 2>/dev/null \
    | grep -E "cu12.*gcc11|gcc11.*cu12" | head -5 || echo "   (tag list unavailable; proceeding with $IMG_REF)"

  echo ">> fetching $IMG_REF (amd64) via skopeo -> $TAR ..."
  "${SKOPEO[@]}" copy --override-arch amd64 --override-os linux \
    "docker://$IMG_REF" "docker-archive:$TAR:$LOCAL_REF"

  echo ">> loading into udocker ..."
  udocker load -i "$TAR"
  udocker create --name="$CONT" "$LOCAL_REF"
  rm -f "$TAR"
else
  echo ">> udocker container '$CONT' already exists; reusing."
fi

# ---- 2. build the backend inside the container ------------------------------
echo ">> building CUDA-Q + tensornet-mps-mgpu inside the container ..."
udocker run -v "$PROJECT_DIR":/work "$CONT" \
  bash -lc "cd /work && bash setup_local_cudaq.sh"

cat <<EOF

============================================================================
Build finished inside container '$CONT'.
Output is on the host at: $PROJECT_DIR/cudaq_local

Run the VQE on GPUs $GPUS:
  ./udocker_build.sh run
(equivalently: udocker run --nvidia -v $PROJECT_DIR:/work $CONT bash -lc 'cd /work && bash run.sh')
============================================================================
EOF
