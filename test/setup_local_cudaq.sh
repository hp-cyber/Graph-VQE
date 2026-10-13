#!/usr/bin/env bash
# =============================================================================
# One-shot build of CUDA-Q plus the custom sharded MPS backend and persistent
# Hamiltonian-batching backends, analytic Pauli-prefix acceleration, and the
# worker-lifecycle fix needed when independent VQE changes partition widths.
# Direct cuSOLVER SVD preserves numerical stability at small singular values;
# zero-cutoff simulations reject caps that discard resolved singular directions.
# Installed ENTIRELY here; Docker and sudo are not required.
#
# The script carries a compressed, versioned copy of the GPU backend. If
# ./custom_backend is absent after moving the project to another server, it is
# reconstructed locally and then compiled with CUDA-Q. The script creates or
# modifies ./custom_backend, ./cuda-quantum-src, and ./cudaq_local.
# Application dependencies (including CUDA PyTorch for reverse gradients)
# must be installed in the active Python environment; they are checked below.
# Usage (inside a CUDA-Q devdeps container, or a server with a consistent
# clang/gcc<=13 toolchain; your documented cudaq-env provides Python, CUDA,
# cuQuantum, MPI, and the source-build utilities):
#
#     conda activate cudaq-env
#     ./setup_local_cudaq.sh
#     source ./activate_local_cudaq.sh
#
# Everything (CMake, Ninja, LLVM, prerequisites, CUDA-Q, and our backend)
# installs under ./cudaq_local; nothing is installed into /usr/local.
#
# Knobs (env): CUDAQ_VERSION_TAG / CUDAQ_COMMIT (CUDA-Q source revision),
# CUDAQ_MQPU_TEST_GPUS (GPU list for the package smoke test; default is the
# current CUDA_VISIBLE_DEVICES, then every GPU reported by nvidia-smi),
# CUDAQ_SKIP_GPU_TEST=1, CUDAQ_SETUP_PREFLIGHT_ONLY=1,
# CUDAQ_REQUIRE_GRADIENT_DEPS=0 (backend-only installation; default requires
# CUDA PyTorch and opt_einsum for the complete three-optimizer test suites).
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---- config -----------------------------------------------------------------
CUDAQ_COMMIT="${CUDAQ_COMMIT:-91ab3092e76dab8887d1fdf0c99a2478ca90581c}"   # == 0.14.2
CUDAQ_VERSION_TAG="${CUDAQ_VERSION_TAG:-0.14.2}"
CUDAQ_REPO="${CUDAQ_REPO:-https://github.com/NVIDIA/cuda-quantum.git}"
CMAKE_BOOTSTRAP_VERSION="${CUDAQ_CMAKE_VERSION:-3.26.4}"
NINJA_BOOTSTRAP_VERSION="${CUDAQ_NINJA_VERSION:-1.11.1}"

LOCAL="$PROJECT_DIR/cudaq_local"          # all installs live here
SRC="$PROJECT_DIR/cuda-quantum-src"       # CUDA-Q source tree
BACKEND="$SRC/runtime/nvqir/cutensornet"  # where our backend files go
PACKAGED_BACKEND="$PROJECT_DIR/custom_backend"
export CUDAQ_TOOLS_INSTALL_PREFIX="$LOCAL"
export XDG_CACHE_HOME="$LOCAL/cache"
export CUDA_CACHE_PATH="$LOCAL/cuda-cache"
mkdir -p "$LOCAL/bin" "$LOCAL/share" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
export PATH="$LOCAL/bin:$LOCAL/llvm/bin:$PATH"

download_file() { # $1 = URL, $2 = output file
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="$2" "$1"
  else
    echo "ERROR: curl or wget is required to download local build tools." >&2
    return 1
  fi
}

bootstrap_local_build_tools() {
  local need_cmake=0 need_ninja=0 bootstrap_dir cmake_installer ninja_tarball
  local installed_cmake installed_ninja
  installed_cmake="$("$LOCAL/bin/cmake" --version 2>/dev/null | awk 'NR == 1 {print $3}' || true)"
  installed_ninja="$("$LOCAL/bin/ninja" --version 2>/dev/null || true)"
  [ "$installed_cmake" = "$CMAKE_BOOTSTRAP_VERSION" ] || need_cmake=1
  [ "$installed_ninja" = "$NINJA_BOOTSTRAP_VERSION" ] || need_ninja=1
  if [ "$need_cmake" = 0 ] && [ "$need_ninja" = 0 ]; then
    echo ">> local build tools: $($LOCAL/bin/cmake --version | head -1); Ninja $($LOCAL/bin/ninja --version)"
    return
  fi

  command -v tar >/dev/null 2>&1 || {
    echo "ERROR: tar is required to bootstrap local CMake/Ninja." >&2
    return 1
  }
  bootstrap_dir="$(mktemp -d "$LOCAL/.tool-bootstrap.XXXXXX")"
  # This directory is generated here and is safe to remove on either success
  # or failure. Quoting through a function avoids a trap with an expanded path.
  cleanup_local_tool_bootstrap() {
    [ -n "${bootstrap_dir:-}" ] && [ -d "$bootstrap_dir" ] &&
      rm -rf -- "$bootstrap_dir"
  }
  trap cleanup_local_tool_bootstrap EXIT

  if [ "$need_cmake" = 1 ]; then
    case "$(uname -s):$(uname -m)" in
      Linux:x86_64|Linux:aarch64)
        cmake_installer="$bootstrap_dir/cmake-install.sh"
        echo ">> installing CMake $CMAKE_BOOTSTRAP_VERSION locally ..."
        download_file \
          "https://github.com/Kitware/CMake/releases/download/v${CMAKE_BOOTSTRAP_VERSION}/cmake-${CMAKE_BOOTSTRAP_VERSION}-linux-$(uname -m).sh" \
          "$cmake_installer"
        bash "$cmake_installer" --skip-license --exclude-subdir --prefix="$LOCAL"
        ;;
      *)
        echo "ERROR: automatic local CMake bootstrap supports Linux x86_64/aarch64;" >&2
        echo "       install CMake and Ninja in $LOCAL/bin on this platform." >&2
        return 1
        ;;
    esac
  fi

  if [ "$need_ninja" = 1 ]; then
    command -v make >/dev/null 2>&1 || {
      echo "ERROR: make is required to build the local Ninja executable." >&2
      return 1
    }
    ninja_tarball="$bootstrap_dir/ninja-${NINJA_BOOTSTRAP_VERSION}.tar.gz"
    echo ">> installing Ninja $NINJA_BOOTSTRAP_VERSION locally ..."
    download_file \
      "https://github.com/ninja-build/ninja/archive/refs/tags/v${NINJA_BOOTSTRAP_VERSION}.tar.gz" \
      "$ninja_tarball"
    tar -xzf "$ninja_tarball" -C "$bootstrap_dir"
    "$LOCAL/bin/cmake" \
      -S "$bootstrap_dir/ninja-${NINJA_BOOTSTRAP_VERSION}" \
      -B "$bootstrap_dir/ninja-build" \
      -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS=-static-libstdc++
    "$LOCAL/bin/cmake" --build "$bootstrap_dir/ninja-build" --parallel
    install -m 0755 "$bootstrap_dir/ninja-build/ninja" "$LOCAL/bin/ninja"
  fi

  hash -r
  "$LOCAL/bin/cmake" --version >/dev/null 2>&1 || {
    echo "ERROR: local CMake bootstrap did not produce a working executable." >&2
    return 1
  }
  "$LOCAL/bin/ninja" --version >/dev/null 2>&1 || {
    echo "ERROR: local Ninja bootstrap did not produce a working executable." >&2
    return 1
  }
  echo ">> local build tools: $($LOCAL/bin/cmake --version | head -1); Ninja $($LOCAL/bin/ninja --version)"
  cleanup_local_tool_bootstrap
  trap - EXIT
}

# ---- environment detection: conda (bare metal) vs devdeps container ---------
# Bare-metal mode: conda cudaq-env provides python/cuQuantum, and we build LLVM.
# Container mode  : the devdeps image already has LLVM + cuQuantum + toolchain,
#                   so we skip install_prerequisites and reuse them.
if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" = "base" ]; then
  CONDA_EXE="$(command -v conda || true)"
  if [ -z "$CONDA_EXE" ]; then
    for candidate in "$HOME/conda/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
      if [ -x "$candidate" ]; then
        CONDA_EXE="$candidate"
        break
      fi
    done
  fi
  if [ -n "$CONDA_EXE" ]; then
    CONDA_BASE="$($CONDA_EXE info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate cudaq-env 2>/dev/null || true
    fi
  fi
fi
command -v git >/dev/null || { echo "ERROR: git required" >&2; exit 1; }
command -v python  >/dev/null 2>&1 || alias python=python3
PY="$(command -v python || command -v python3)"
PY="$("$PY" -c 'import sys; print(sys.executable)')"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo /usr)"

# CUDA-Q's `install_prerequisites.sh -m` intentionally does not install its
# toolchain. Bootstrap the pinned versions ourselves so a fresh checkout never
# depends on CMake/Ninja from another project or on administrator access.
bootstrap_local_build_tools

# Is LLVM already provided (devdeps container)?  Then we're in PREBUILT mode.
PREBUILT=0
for d in "${LLVM_INSTALL_PREFIX:-}" /opt/llvm /usr/lib/llvm; do
  [ -n "$d" ] && [ -x "$d/bin/llvm-config" ] && { PREBUILT=1; LLVM_PREBUILT="$d"; break; }
done

# Never share LLVM's source/build directory through $HOME. CMake caches contain
# absolute paths (including CMAKE_MAKE_PROGRAM), so a shared or copied build is
# exactly what produces references to another checkout's cudaq_local/bin/ninja.
if [ "$PREBUILT" = 0 ]; then
  export LLVM_SOURCE="$LOCAL/llvm-project-src"
  export LLVM_BUILD_FOLDER="build"
fi

# Resolve CUDA before CUDACXX. The previous order expanded an unset CUDA_HOME
# when nvcc was absent, which made `set -u` abort with an unhelpful error.
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    CUDA_HOME="${CONDA_PREFIX:-/usr/local/cuda}"
  fi
  export CUDA_HOME
fi
if [ -z "${CUDACXX:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDACXX="$(command -v nvcc)"
  else
    CUDACXX="$CUDA_HOME/bin/nvcc"
  fi
elif command -v "$CUDACXX" >/dev/null 2>&1; then
  CUDACXX="$(command -v "$CUDACXX")"
fi
export CUDACXX

# Select smoke-test devices without assuming that every server has GPUs 0,1,2.
# CUDA_VISIBLE_DEVICES values are physical IDs (or UUIDs); CUDA-Q sees their
# dense logical indices and CUDAQ_MQPU_NGPUS is therefore just the list length.
if [ -n "${CUDAQ_MQPU_TEST_GPUS:-}" ]; then
  MQPU_TEST_GPUS="$(tr -d '[:space:]' <<<"$CUDAQ_MQPU_TEST_GPUS")"
elif [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|-1|NoDevFiles) MQPU_TEST_GPUS="" ;;
    *) MQPU_TEST_GPUS="$(tr -d '[:space:]' <<<"$CUDA_VISIBLE_DEVICES")" ;;
  esac
elif command -v nvidia-smi >/dev/null 2>&1; then
  MQPU_TEST_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null |
    awk 'NF {gsub(/[[:space:]]/, ""); printf "%s%s", sep, $0; sep=","}')"
else
  MQPU_TEST_GPUS=""
fi
if [ -n "$MQPU_TEST_GPUS" ]; then
  [[ "$MQPU_TEST_GPUS" != ,* && "$MQPU_TEST_GPUS" != *, &&
     "$MQPU_TEST_GPUS" != *,,* ]] || {
    echo "ERROR: invalid CUDAQ_MQPU_TEST_GPUS list: '$MQPU_TEST_GPUS'" >&2
    exit 1
  }
  MQPU_TEST_QPUS="$(awk -F, '{print NF}' <<<"$MQPU_TEST_GPUS")"
else
  MQPU_TEST_QPUS=0
fi

# The prerequisite scripts expect these variables even when -m skips their
# toolchain installer. Pick the active compiler explicitly if Conda did not.
if [ -z "${CC:-}" ]; then
  CC="$(command -v gcc || command -v cc || true)"
  export CC
fi
if [ -z "${CXX:-}" ]; then
  CXX="$(command -v g++ || command -v c++ || true)"
  export CXX
fi
[ -n "$CC" ] && [ -x "$CC" ] || {
  echo "ERROR: no working C compiler found; install the compilers listed in note.txt." >&2
  exit 1
}
[ -n "$CXX" ] && [ -x "$CXX" ] || {
  echo "ERROR: no working C++ compiler found; install the compilers listed in note.txt." >&2
  exit 1
}

# CUDA 12.4 supports GCC through version 13. Conda currently exports GCC 14 in
# cudaq-env, so prefer an installed compatible system compiler before starting
# the multi-hour LLVM build.
if [ "$PREBUILT" = 0 ]; then
  ACTIVE_CXX="${CXX:-$(command -v c++ || true)}"
  ACTIVE_CXX_BANNER="$("$ACTIVE_CXX" --version 2>/dev/null | head -1 || true)"
  ACTIVE_CXX_MAJOR="$("$ACTIVE_CXX" -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1 || true)"
  if [[ "$ACTIVE_CXX_BANNER" == *gcc* || "$ACTIVE_CXX_BANNER" == *GCC* ||
        "$ACTIVE_CXX_BANNER" == *g++* || "$ACTIVE_CXX_BANNER" == *G++* ]] &&
     [[ "$ACTIVE_CXX_MAJOR" =~ ^[0-9]+$ ]] && [ "$ACTIVE_CXX_MAJOR" -gt 13 ]; then
    COMPAT_GCC_VERSION=""
    for version in 13 12 11; do
      if command -v "gcc-$version" >/dev/null 2>&1 &&
         command -v "g++-$version" >/dev/null 2>&1; then
        COMPAT_GCC_VERSION="$version"
        break
      fi
    done
    if [ -z "$COMPAT_GCC_VERSION" ]; then
      echo "ERROR: GCC $ACTIVE_CXX_MAJOR is too new for CUDA 12.4 and no GCC 11-13 was found." >&2
      echo "       Install a compatible GCC or set CC, CXX, and CUDAHOSTCXX explicitly." >&2
      exit 1
    fi
    export CC="$(command -v "gcc-$COMPAT_GCC_VERSION")"
    export CXX="$(command -v "g++-$COMPAT_GCC_VERSION")"
    export CUDAHOSTCXX="$CXX"
    if command -v "gfortran-$COMPAT_GCC_VERSION" >/dev/null 2>&1; then
      export FC="$(command -v "gfortran-$COMPAT_GCC_VERSION")"
      export F77="$FC"
    fi
    echo ">> compiler:        GCC $ACTIVE_CXX_MAJOR is unsupported by CUDA 12.4; using GCC $COMPAT_GCC_VERSION"
  fi

  # These are the host-side utilities used by CUDA-Q 0.14.2's source
  # prerequisite builds. The environment recipe installs them in cudaq-env;
  # list every
  # omission together instead of failing hours later in LLVM or zlib.
  missing_tools=()
  for tool in wget tar make m4 autoconf autoreconf automake libtool libtoolize flex bison; do
    command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
  done
  if [ -z "${FC:-}" ]; then
    FC="$(command -v gfortran || true)"
    export FC
  elif command -v "$FC" >/dev/null 2>&1; then
    FC="$(command -v "$FC")"
    export FC
  fi
  [ -n "${FC:-}" ] && [ -x "$FC" ] || missing_tools+=("gfortran")
  if [ "${#missing_tools[@]}" -ne 0 ]; then
    echo "ERROR: missing source-build tools: ${missing_tools[*]}" >&2
    echo "       Install m4/autoconf/automake/libtool, compilers, flex, and bison" >&2
    echo "       into cudaq-env as in your environment notes, then rerun this script." >&2
    exit 1
  fi
fi

[ -x "$CUDACXX" ] || {
  echo "ERROR: CUDA compiler not found at '$CUDACXX'." >&2
  echo "       Activate cudaq-env from your environment notes, or set CUDA_HOME and CUDACXX." >&2
  exit 1
}

echo ">> install root:    $LOCAL"
echo ">> CUDA-Q source:   $SRC  @ ${CUDAQ_COMMIT}"
echo ">> python:          $PY   (site: $SITE)"
echo ">> mode:            $([ "$PREBUILT" = 1 ] && echo "PREBUILT/container (LLVM @ $LLVM_PREBUILT)" || echo "bare-metal (will build LLVM)")"
echo ">> CUDA toolkit:    $CUDA_HOME  (nvcc: $CUDACXX)"
if [ "$PREBUILT" = 0 ]; then
  echo ">> LLVM source:     $LLVM_SOURCE  (checkout-local)"
fi
if [ -n "$MQPU_TEST_GPUS" ]; then
  echo ">> smoke-test GPUs: $MQPU_TEST_GPUS  ($MQPU_TEST_QPUS logical QPU(s))"
else
  echo ">> smoke-test GPUs: none detected (GPU smoke test will be skipped)"
fi

# Reverse differentiation is implemented by the application, using CUDA PyTorch.
# Do not import CUDA-Q or allocate GPU memory in this dependency check.
if "$PY" -c 'import torch, opt_einsum; assert torch.version.cuda' >/dev/null 2>&1; then
  echo ">> Reverse gradients: CUDA PyTorch and opt_einsum available"
else
  echo "ERROR: complete VQE suites require CUDA-enabled PyTorch and opt_einsum in $PY." >&2
  echo "Install a CUDA PyTorch build compatible with your server and opt_einsum==3.4.0 in that environment." >&2
  if [ "${CUDAQ_REQUIRE_GRADIENT_DEPS:-1}" != 0 ]; then exit 1; fi
  echo ">> CUDAQ_REQUIRE_GRADIENT_DEPS=0: continuing with backend-only installation."
fi

# ---- self-contained custom backend package ----------------------------------
# A copied project may intentionally omit generated CUDA-Q/build directories
# and custom_backend. Recreate the exact versioned backend sources carried by
# this installer before the source-build preflight validates them.
materialize_packaged_backend() {
  mkdir -p "$PACKAGED_BACKEND"
  "$PY" - "$PACKAGED_BACKEND" <<'PYEOF'
import base64
import hashlib
import io
import math
import os
import sys
import tarfile
import tempfile

destination = os.path.realpath(sys.argv[1])
payload = """
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
"""
expected = {'analytic_batch.cuh': '4c962ad92fd0170b5f5acfd94e0607101a38fdd21c346a2155f64f138fca39f0', 'dist_svd.cuh': 'ef1ee6ee0b69665dcb839f40c6373ce660323d2db4e884deeb1dbdc7ec3ed0e6', 'sharded_mps.cuh': '7164197f6f2507631f6bcbe3c2b879c896a5c2399fb21db1377df7f4dd758b66', 'simulator_mps_dist.cu': 'cb70da9e8fd0126f3d469ab0e91fcd574dda0a96cabf8e8d5a452202f29ca1c4'}

compressed = base64.b64decode("".join(payload.split()), validate=True)
with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as archive:
    archive_names = {
        member.name for member in archive.getmembers() if member.isfile()
    }
    if archive_names != set(expected):
        raise SystemExit(
            "ERROR: bundled custom_backend archive has unexpected contents: "
            + ", ".join(sorted(archive_names))
        )
    for filename, wanted_hash in expected.items():
        source = archive.extractfile(filename)
        if source is None:
            raise SystemExit(
                f"ERROR: bundled custom_backend file is missing: {filename}"
            )
        bundled_data = source.read()
        actual_hash = hashlib.sha256(bundled_data).hexdigest()
        if actual_hash != wanted_hash:
            raise SystemExit(
                f"ERROR: bundled custom_backend checksum mismatch: {filename}"
            )

        output_path = os.path.join(destination, filename)
        replacing = os.path.isfile(output_path)
        if replacing:
            with open(output_path, "rb") as existing:
                existing_hash = hashlib.sha256(existing.read()).hexdigest()
            if existing_hash == wanted_hash:
                continue

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{filename}.", dir=destination
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(bundled_data)
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        action = "updated" if replacing else "materialized"
        print(f"   {action} custom_backend/{filename}")
PYEOF
}

materialize_packaged_backend

# The versioned archive above is retained for backward-compatible, standalone
# reconstruction. Apply source-level migrations after extraction so a copied
# installer always materializes the current backend implementation even when
# no external custom_backend directory accompanies it.
apply_packaged_backend_migrations() {
  "$PY" - "$PACKAGED_BACKEND/simulator_mps_dist.cu" <<'PYEOF'
import os
import sys
import tempfile

path = os.path.realpath(sys.argv[1])
with open(path, encoding="utf-8") as source:
    text = source.read()

old = """\
    const auto zValues = state.sampleZExpectationsIndependentGpu(
        zWords, shots, zSeeds);
"""
new = """\
    // The Hamiltonian target partitions terms across physical GPUs, but a
    // commuting diagonal partition must still reuse one measurement stream.
    // Every worker deliberately uses the same deterministic stream seed: the
    // prepared replicas therefore produce the same global `shots` bitstrings,
    // while each GPU evaluates only its assigned term masks.  This makes the
    // one-GPU workload identical to the shared-shot target and prevents the
    // old terms*shots sampling explosion.  Other MQPU targets retain their
    // existing independent-term fallback for mixed-basis observations.
#if defined(CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND)
    const auto zValues = state.sampleZExpectationsGpu(
        zWords, shots, hbatchSeedForShotShard(evaluationSeed, 0));
#else
    const auto zValues = state.sampleZExpectationsIndependentGpu(
        zWords, shots, zSeeds);
#endif
"""
if new not in text:
    if old not in text:
        raise SystemExit(
            "ERROR: cannot migrate packaged Hamiltonian shared-sampling source"
        )
    text = text.replace(old, new, 1)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".simulator_mps_dist.cu.", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("   migrated custom_backend/simulator_mps_dist.cu: shared diagonal Hamiltonian samples")
PYEOF
}

apply_packaged_backend_migrations

for backend_source in analytic_batch.cuh dist_svd.cuh sharded_mps.cuh simulator_mps_dist.cu; do
  [ -s "$PACKAGED_BACKEND/$backend_source" ] || {
    echo "ERROR: packaged GPU backend is missing: $PACKAGED_BACKEND/$backend_source" >&2
    exit 1
  }
done

if [ "${CUDAQ_SETUP_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo ">> preflight complete; CUDAQ_SETUP_PREFLIGHT_ONLY=1, so no source build was started."
  exit 0
fi

# ---- 1. fetch CUDA-Q source (shallow, pinned) -------------------------------
if [ ! -d "$SRC/.git" ]; then
  echo ">> cloning CUDA-Q ..."
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$CUDAQ_REPO"
  if ! git -C "$SRC" fetch --depth 1 origin "$CUDAQ_COMMIT" 2>/dev/null; then
    git -C "$SRC" fetch --depth 1 origin "$CUDAQ_VERSION_TAG"
  fi
  git -C "$SRC" checkout -q FETCH_HEAD
else
  echo ">> CUDA-Q source already present."
fi

# CUDA-Q's prerequisite installer defaults bootstrapped CMake/Ninja to
# /usr/local, and its AWS s2n dependency uses legacy CMake policies. Redirect
# tools into cudaq_local and make the AWS configure compatible with CMake 4.
# Keep these source patches here so a newly cloned tree is fixed immediately.
python3 - "$SRC/scripts/install_prerequisites.sh" <<'PYEOF'
import re
import sys

path = sys.argv[1]
s = open(path).read()
replacements = {
    'bash cmake-install.sh --skip-licence --exclude-subdir --prefix=/usr/local':
        'bash cmake-install.sh --skip-licence --exclude-subdir '
        '--prefix="${CUDAQ_TOOLS_INSTALL_PREFIX:-/usr/local}"',
    'mv build/ninja /usr/local/bin/':
        'mv build/ninja "${CUDAQ_TOOLS_INSTALL_PREFIX:-/usr/local}/bin/"',
    '      -DCMAKE_INSTALL_PREFIX="${AWS_INSTALL_PREFIX}" \\\n'
    '      -DCMAKE_BUILD_TYPE=Release \\\n':
        '      -DCMAKE_INSTALL_PREFIX="${AWS_INSTALL_PREFIX}" \\\n'
        '      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \\\n'
        '      -DCMAKE_BUILD_TYPE=Release \\\n',
}
patched_markers = (
    '--prefix="${CUDAQ_TOOLS_INSTALL_PREFIX:-/usr/local}"',
    'mv build/ninja "${CUDAQ_TOOLS_INSTALL_PREFIX:-/usr/local}/bin/"',
    '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
    'Local-only build: do not invoke a system package manager.',
    'function remove_temp_installs {\n  :\n}',
)
changed = False
for old, new in replacements.items():
    if old in s:
        s = s.replace(old, new, 1)
        changed = True
if 'Local-only build: do not invoke a system package manager.' not in s:
    pattern = (
        r'function temp_install_if_command_unknown \{.*?\n\}'
        r'\n\nfunction remove_temp_installs'
    )
    replacement = '''function temp_install_if_command_unknown {
  if [ ! -x "$(command -v "$1")" ]; then
    # Local-only build: do not invoke a system package manager.
    echo "ERROR: required build tool '$1' is missing; install it before running setup_local_cudaq.sh." >&2
    return 1
  fi
}

function remove_temp_installs'''
    s, count = re.subn(pattern, replacement, s, count=1, flags=re.DOTALL)
    if count != 1:
        print("   ERROR: prerequisite package-manager function not found",
              file=sys.stderr)
        sys.exit(1)
    changed = True
if 'function remove_temp_installs {\n  :\n}' not in s:
    pattern = r'function remove_temp_installs \{.*?\n\}'
    replacement = '''function remove_temp_installs {
  :
}'''
    s, count = re.subn(pattern, replacement, s, count=1, flags=re.DOTALL)
    if count != 1:
        print("   ERROR: prerequisite cleanup function not found",
              file=sys.stderr)
        sys.exit(1)
    changed = True
if not all(marker in s for marker in patched_markers):
    print("   ERROR: prerequisite tool install anchors not found", file=sys.stderr)
    sys.exit(1)
elif changed:
    open(path, "w").write(s)
    print("   patched prerequisite installer compatibility")
else:
    print("   prerequisite installer compatibility already patched")
PYEOF

# ---- 2. write the multi-GPU MPS backend into the cutensornet dir ------------
echo ">> writing tensornet-mps-mgpu backend sources ..."
cat > "$BACKEND/dist_svd.cuh" <<'MGPU_DIST_SVD_CUH_EOF'
// Reusable distributed randomized truncated SVD (cuBLAS/cuSOLVER/NCCL).
// Header form so both the SVD self-test and the sharded-MPS engine can use it.
// Works for G>=1 GPUs: with G==1 it is a single-GPU randomized truncated SVD
// (NCCL allreduce is skipped).  All GPU matrices are COLUMN-MAJOR.
#pragma once

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <algorithm>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)
#define CUSOLVER_CHECK(x) do { cusolverStatus_t s=(x); if(s){fprintf(stderr,"cuSOLVER %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)
#define NCCL_CHECK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r));exit(1);} }while(0)
#endif

using cdouble = cuDoubleComplex;
static const cdouble C_ONE  = make_cuDoubleComplex(1.0, 0.0);
static const cdouble C_ZERO = make_cuDoubleComplex(0.0, 0.0);

struct GpuCtx {
  int dev;
  cudaStream_t stream;
  cublasHandle_t blas;
  cusolverDnHandle_t solver;
  ncclComm_t comm;   // unused when the call uses a single GPU
};

static inline void allreduceSum(std::vector<GpuCtx>& ctx, std::vector<cdouble*>& bufs,
                                size_t count) {
  if (ctx.size() == 1) return;   // single GPU: buffer already holds the full sum
  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < ctx.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    NCCL_CHECK(ncclAllReduce(bufs[i], bufs[i], count * 2, ncclDouble, ncclSum,
                             ctx[i].comm, ctx[i].stream));
  }
  NCCL_CHECK(ncclGroupEnd());
  for (auto& c : ctx) { CUDA_CHECK(cudaSetDevice(c.dev)); CUDA_CHECK(cudaStreamSynchronize(c.stream)); }
}

static inline void qrColumns(GpuCtx& g, int m, int l, const cdouble* A, cdouble* Qout) {
  CUDA_CHECK(cudaSetDevice(g.dev));
  CUDA_CHECK(cudaMemcpyAsync(Qout, A, sizeof(cdouble)*(size_t)m*l, cudaMemcpyDeviceToDevice, g.stream));
  cdouble* tau; CUDA_CHECK(cudaMalloc(&tau, sizeof(cdouble)*l));
  int lwork = 0; CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(g.solver, m, l, Qout, m, &lwork));
  int lwork2 = 0; CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(g.solver, m, l, l, Qout, m, tau, &lwork2));
  if (lwork2 > lwork) lwork = lwork2;
  cdouble* work; CUDA_CHECK(cudaMalloc(&work, sizeof(cdouble)*lwork));
  int* info;     CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnZgeqrf(g.solver, m, l, Qout, m, tau, work, lwork, info));
  CUSOLVER_CHECK(cusolverDnZungqr(g.solver, m, l, l, Qout, m, tau, work, lwork, info));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  CUDA_CHECK(cudaFree(tau)); CUDA_CHECK(cudaFree(work)); CUDA_CHECK(cudaFree(info));
}

static inline void heig(GpuCtx& g, int l, cdouble* C, double* w_host) {
  CUDA_CHECK(cudaSetDevice(g.dev));
  double* w; CUDA_CHECK(cudaMalloc(&w, sizeof(double)*l));
  int lwork=0;
  CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(g.solver, CUSOLVER_EIG_MODE_VECTOR,
                  CUBLAS_FILL_MODE_LOWER, l, C, l, w, &lwork));
  cdouble* work; CUDA_CHECK(cudaMalloc(&work, sizeof(cdouble)*lwork));
  int* info; CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnZheevd(g.solver, CUSOLVER_EIG_MODE_VECTOR,
                  CUBLAS_FILL_MODE_LOWER, l, C, l, w, work, lwork, info));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  CUDA_CHECK(cudaMemcpy(w_host, w, sizeof(double)*l, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(w)); CUDA_CHECK(cudaFree(work)); CUDA_CHECK(cudaFree(info));
}

static inline cdouble* peerCopy(int srcDev, const cdouble* src, int dstDev, size_t n) {
  cdouble* dst; CUDA_CHECK(cudaSetDevice(dstDev)); CUDA_CHECK(cudaMalloc(&dst, sizeof(cdouble)*n));
  CUDA_CHECK(cudaMemcpyPeer(dst, dstDev, src, srcDev, sizeof(cdouble)*n));
  return dst;
}

__global__ static void scaleRowsKernel(cdouble* B, int r, int ncols, const double* inv) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= r*ncols) return;
  int row = idx % r;
  B[idx].x *= inv[row]; B[idx].y *= inv[row];
}

struct SvdResult {
  std::vector<cdouble> U;   // m x k  (host, column-major)
  std::vector<double>  S;   // k
  std::vector<cdouble> Vh;  // k x n  (host, column-major, gathered)
  int k;
};

// Distributed (or single-GPU) randomized truncated SVD of a column-sharded M.
static inline SvdResult distSvd(std::vector<GpuCtx>& ctx, std::vector<cdouble*>& blocks,
                  const std::vector<int>& ng, int m, int n,
                  int maxBond, double absCut, double relCut, int oversample,
                  unsigned seed) {
  int G = (int)ctx.size();
  int root = 0;
  int l = std::min(std::min(maxBond + oversample, m), n);

  std::vector<cdouble*> Y(G);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int i = 0; i < G; ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    std::vector<cdouble> omh((size_t)ng[i]*l);
    for (auto& z : omh) z = make_cuDoubleComplex(nd(rng), nd(rng));
    cdouble* om; CUDA_CHECK(cudaMalloc(&om, sizeof(cdouble)*omh.size()));
    CUDA_CHECK(cudaMemcpy(om, omh.data(), sizeof(cdouble)*omh.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&Y[i], sizeof(cdouble)*(size_t)m*l));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_N, CUBLAS_OP_N, m, l, ng[i],
                  &C_ONE, blocks[i], m, om, ng[i], &C_ZERO, Y[i], m));
    CUDA_CHECK(cudaStreamSynchronize(ctx[i].stream));
    CUDA_CHECK(cudaFree(om));
  }
  allreduceSum(ctx, Y, (size_t)m*l);

  std::vector<cdouble*> Q(G);
  CUDA_CHECK(cudaSetDevice(ctx[root].dev));
  CUDA_CHECK(cudaMalloc(&Q[root], sizeof(cdouble)*(size_t)m*l));
  qrColumns(ctx[root], m, l, Y[root], Q[root]);
  for (int i = 0; i < G; ++i) if (i != root)
    Q[i] = peerCopy(ctx[root].dev, Q[root], ctx[i].dev, (size_t)m*l);

  std::vector<cdouble*> B(G), Cpart(G);
  for (int i = 0; i < G; ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    CUDA_CHECK(cudaMalloc(&B[i], sizeof(cdouble)*(size_t)l*ng[i]));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_C, CUBLAS_OP_N, l, ng[i], m,
                  &C_ONE, Q[i], m, blocks[i], m, &C_ZERO, B[i], l));
    CUDA_CHECK(cudaMalloc(&Cpart[i], sizeof(cdouble)*(size_t)l*l));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_N, CUBLAS_OP_C, l, l, ng[i],
                  &C_ONE, B[i], l, B[i], l, &C_ZERO, Cpart[i], l));
    CUDA_CHECK(cudaStreamSynchronize(ctx[i].stream));
  }
  allreduceSum(ctx, Cpart, (size_t)l*l);

  std::vector<double> w(l);
  heig(ctx[root], l, Cpart[root], w.data());
  std::vector<double> S(l);
  for (int i = 0; i < l; ++i) S[i] = std::sqrt(std::max(0.0, w[l-1-i]));
  double smax = S.empty()?0.0:S[0];
  double thresh = std::max(absCut, relCut*smax);
  int k = 0; for (int i=0;i<l;++i) if (S[i] > thresh) ++k;
  if (k < 1) k = 1; if (k > maxBond) k = maxBond; if (k > l) k = l;

  std::vector<cdouble> Cvec((size_t)l*l);
  CUDA_CHECK(cudaSetDevice(ctx[root].dev));
  CUDA_CHECK(cudaMemcpy(Cvec.data(), Cpart[root], sizeof(cdouble)*(size_t)l*l, cudaMemcpyDeviceToHost));
  std::vector<cdouble> Uhat((size_t)l*k);
  for (int c = 0; c < k; ++c) { int src = l-1-c;
    for (int rrow = 0; rrow < l; ++rrow) Uhat[(size_t)c*l + rrow] = Cvec[(size_t)src*l + rrow]; }
  cdouble* dUhat; CUDA_CHECK(cudaMalloc(&dUhat, sizeof(cdouble)*(size_t)l*k));
  CUDA_CHECK(cudaMemcpy(dUhat, Uhat.data(), sizeof(cdouble)*(size_t)l*k, cudaMemcpyHostToDevice));

  SvdResult res; res.k = k; res.S.assign(S.begin(), S.begin()+k);
  cdouble* dU; CUDA_CHECK(cudaMalloc(&dU, sizeof(cdouble)*(size_t)m*k));
  CUBLAS_CHECK(cublasZgemm(ctx[root].blas, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l,
                &C_ONE, Q[root], m, dUhat, l, &C_ZERO, dU, m));
  res.U.resize((size_t)m*k);
  CUDA_CHECK(cudaMemcpy(res.U.data(), dU, sizeof(cdouble)*(size_t)m*k, cudaMemcpyDeviceToHost));

  std::vector<double> invS(k); for (int i=0;i<k;++i) invS[i] = S[i]>0? 1.0/S[i] : 0.0;
  res.Vh.assign((size_t)k*n, C_ZERO);
  int colOffset = 0;
  for (int i = 0; i < G; ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    cdouble* UhatI = (i==root)? dUhat : peerCopy(ctx[root].dev, dUhat, ctx[i].dev, (size_t)l*k);
    double* dInv; CUDA_CHECK(cudaMalloc(&dInv, sizeof(double)*k));
    CUDA_CHECK(cudaMemcpy(dInv, invS.data(), sizeof(double)*k, cudaMemcpyHostToDevice));
    cdouble* Vhg; CUDA_CHECK(cudaMalloc(&Vhg, sizeof(cdouble)*(size_t)k*ng[i]));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_C, CUBLAS_OP_N, k, ng[i], l,
                  &C_ONE, UhatI, l, B[i], l, &C_ZERO, Vhg, k));
    int threads=256, blocks_=((k*ng[i])+threads-1)/threads;
    scaleRowsKernel<<<blocks_,threads,0,ctx[i].stream>>>(Vhg, k, ng[i], dInv);
    CUDA_CHECK(cudaStreamSynchronize(ctx[i].stream));
    std::vector<cdouble> Vhh((size_t)k*ng[i]);
    CUDA_CHECK(cudaMemcpy(Vhh.data(), Vhg, sizeof(cdouble)*(size_t)k*ng[i], cudaMemcpyDeviceToHost));
    for (int c=0;c<ng[i];++c) for (int r=0;r<k;++r)
      res.Vh[(size_t)(colOffset+c)*k + r] = Vhh[(size_t)c*k + r];
    colOffset += ng[i];
    CUDA_CHECK(cudaFree(dInv)); CUDA_CHECK(cudaFree(Vhg));
    if (i!=root) CUDA_CHECK(cudaFree(UhatI));
  }
  for (int i=0;i<G;++i){ CUDA_CHECK(cudaSetDevice(ctx[i].dev)); CUDA_CHECK(cudaFree(Y[i])); CUDA_CHECK(cudaFree(Q[i])); CUDA_CHECK(cudaFree(B[i])); CUDA_CHECK(cudaFree(Cpart[i])); }
  CUDA_CHECK(cudaSetDevice(ctx[root].dev)); CUDA_CHECK(cudaFree(dUhat)); CUDA_CHECK(cudaFree(dU));
  return res;
}
MGPU_DIST_SVD_CUH_EOF

cat > "$BACKEND/sharded_mps.cuh" <<'MGPU_SHARDED_MPS_CUH_EOF'
// Sharded-MPS engine (header): eager explicit chain across GPUs.
// Used by the self-test (sharded_mps.cu) and the NVQIR wrapper (milestone 3).
//  - chain sharded by contiguous segments across GPUs (memory pooling)
//  - 1q gates local; adjacent 2q gates via custom contraction + truncated SVD
//  - arbitrary-pair 2q gates via SWAP routing
//  - <psi|Pauli-string|psi> expectation (for VQE energy)
#pragma once
#include "dist_svd.cuh"
#include <string>
#include <algorithm>
#include <cstdlib>

__global__ static void k_apply1q(const cdouble* A, const cdouble* U, cdouble* out,
                                 int Dl, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int total = Dl*2*Dr; if (idx>=total) return;
  int c = idx % Dr; int t = idx/Dr; int x = t%2; int l = t/2;
  cdouble acc = make_cuDoubleComplex(0.0,0.0);
  for (int i=0;i<2;++i) acc = cuCadd(acc, cuCmul(U[x*2+i], A[(l*2+i)*Dr + c]));
  out[(l*2+x)*Dr + c] = acc;
}

__global__ static void k_buildM(cdouble* M, const cdouble* A, const cdouble* B,
                               const cdouble* G, int Dl, int Dc, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int total = Dl*2*2*Dr; if (idx>=total) return;
  int r = idx % Dr; int t = idx/Dr; int J = t%2; t/=2; int I = t%2; int l = t/2;
  cdouble acc = make_cuDoubleComplex(0.0,0.0);
  for (int i=0;i<2;++i) for (int j=0;j<2;++j) {
    cdouble g = G[(I*2+J)*4 + (i*2+j)];
    if (g.x==0.0 && g.y==0.0) continue;
    cdouble s = make_cuDoubleComplex(0.0,0.0);
    for (int c=0;c<Dc;++c) s = cuCadd(s, cuCmul(A[(l*2+i)*Dc + c], B[(c*2+j)*Dr + r]));
    acc = cuCadd(acc, cuCmul(g, s));
  }
  int row = l*2+I, col = J*Dr + r, m = 2*Dl;
  M[row + (size_t)col*m] = acc;
}

// ---- GPU-side Pauli expectation (stage 1) ----------------------------------
// Left-to-right transfer: E_{s+1} = A_s^dagger (P_s) E_s A_s, kept on-device.
// Two O(chi^3) sweeps per site (vs the old host O(chi^4) quadruple loop), all
// on the GPU that owns the site; the environment E is peer-copied across GPU
// boundaries. Indexing mirrors the validated host version exactly.
//   k_transL:  R[a,ip,bp] = sum_ap E[a,ap] A[ap,ip,bp]          (contract left bond)
//   k_applyP:  S[a,i,bp]  = sum_ip P[i,ip] R[a,ip,bp]           (apply Pauli)
//   k_transR:  newE[b,bp] = sum_{a,i} conj(A[a,i,b]) S[a,i,bp]  (contract bra)
__global__ static void k_transL(const cdouble* E, const cdouble* A, cdouble* R,
                                int Dl, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= Dl*2*Dr) return;
  int bp = idx % Dr; int t = idx / Dr; int ip = t % 2; int a = t / 2;
  cdouble acc = make_cuDoubleComplex(0,0);
  for (int ap = 0; ap < Dl; ++ap)
    acc = cuCadd(acc, cuCmul(E[a*Dl + ap], A[(ap*2+ip)*Dr + bp]));
  R[(a*2+ip)*Dr + bp] = acc;
}
__global__ static void k_applyP(const cdouble* R, cdouble* S, int Dl, int Dr,
                                cdouble p00, cdouble p01, cdouble p10, cdouble p11) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= Dl*2*Dr) return;
  int bp = idx % Dr; int t = idx / Dr; int i = t % 2; int a = t / 2;
  cdouble r0 = R[(a*2+0)*Dr + bp], r1 = R[(a*2+1)*Dr + bp];
  cdouble pi0 = (i==0)?p00:p10, pi1 = (i==0)?p01:p11;
  S[(a*2+i)*Dr + bp] = cuCadd(cuCmul(pi0, r0), cuCmul(pi1, r1));
}
__global__ static void k_transR(const cdouble* A, const cdouble* S, cdouble* newE,
                                int Dl, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= Dr*Dr) return;
  int bp = idx % Dr; int b = idx / Dr;
  cdouble acc = make_cuDoubleComplex(0,0);
  int rows = Dl*2;
  for (int t = 0; t < rows; ++t)
    acc = cuCadd(acc, cuCmul(cuConj(A[t*Dr + b]), S[t*Dr + bp]));
  newE[b*Dr + bp] = acc;
}
// 2x2 Pauli matrix elements P[i][ip] (bra index i, ket index ip)
static inline void pauliMat(char p, cdouble& p00, cdouble& p01,
                            cdouble& p10, cdouble& p11) {
  p00 = p01 = p10 = p11 = make_cuDoubleComplex(0,0);
  switch (p) {
    case 'X': p01 = make_cuDoubleComplex(1,0);  p10 = make_cuDoubleComplex(1,0);  break;
    case 'Y': p01 = make_cuDoubleComplex(0,-1); p10 = make_cuDoubleComplex(0,1);  break;
    case 'Z': p00 = make_cuDoubleComplex(1,0);  p11 = make_cuDoubleComplex(-1,0); break;
    default:  p00 = make_cuDoubleComplex(1,0);  p11 = make_cuDoubleComplex(1,0);  break; // I
  }
}

struct Site { cdouble* d; int Dl, Dr, dev; };   // physical dim = 2

// Per-thread persistent GPU context pool: NCCL communicators + cuBLAS / cuSOLVER
// handles + streams, created ONCE per thread for the (fixed) GPU set and reused
// by every ShardedMPS on that thread. Previously these were created per circuit
// execution and never destroyed, leaking NCCL communicators (and their proxy
// threads) until pthread_create failed with EAGAIN ("Resource temporarily
// unavailable") partway through a long VQE run. thread_local keeps it correct
// for the multi-threaded SubspaceVQE path (NCCL comms aren't safe to share
// across threads), while bounding total comms to one set per worker thread.
static std::vector<GpuCtx>& shardedGpuPool(const std::vector<int>& gpu_ids) {
  static thread_local std::vector<GpuCtx> pool;
  if (pool.empty()) {
    int G = (int)gpu_ids.size();
    pool.resize(G);
    if (G > 1) {
      // Multi-GPU: init NCCL communicators for cross-GPU allreduce / peer copies.
      std::vector<int> devs = gpu_ids;
      ncclComm_t* comms = new ncclComm_t[G];
      NCCL_CHECK(ncclCommInitAll(comms, G, devs.data()));
      for (int i = 0; i < G; ++i) {
        pool[i].dev = gpu_ids[i]; CUDA_CHECK(cudaSetDevice(pool[i].dev));
        CUDA_CHECK(cudaStreamCreate(&pool[i].stream));
        CUBLAS_CHECK(cublasCreate(&pool[i].blas)); CUBLAS_CHECK(cublasSetStream(pool[i].blas, pool[i].stream));
        CUSOLVER_CHECK(cusolverDnCreate(&pool[i].solver)); CUSOLVER_CHECK(cusolverDnSetStream(pool[i].solver, pool[i].stream));
        pool[i].comm = comms[i];
      }
      delete[] comms;
    } else {
      // Single GPU: no NCCL needed. allreduceSum and peer-copies in dist_svd.cuh
      // and expectPauli/expectPauliBatch are already guarded by G==1 checks.
      pool[0].dev = gpu_ids[0]; CUDA_CHECK(cudaSetDevice(pool[0].dev));
      CUDA_CHECK(cudaStreamCreate(&pool[0].stream));
      CUBLAS_CHECK(cublasCreate(&pool[0].blas)); CUBLAS_CHECK(cublasSetStream(pool[0].blas, pool[0].stream));
      CUSOLVER_CHECK(cusolverDnCreate(&pool[0].solver)); CUSOLVER_CHECK(cusolverDnSetStream(pool[0].solver, pool[0].stream));
      pool[0].comm = nullptr;
    }
  }
  return pool;
}

struct ShardedMPS {
  int n;
  std::vector<int> gpu_ids;
  std::vector<Site> site;
  std::vector<int> owner;
  std::vector<GpuCtx> ctx;
  int maxBond; double absCut, relCut; unsigned seedCounter = 1;
  bool distributedBoundary = false;   // true -> cross-GPU bonds use G>1 distSvd

  ShardedMPS(int nqubits, std::vector<int> gids, int maxB, double aC, double rC)
      : n(nqubits), gpu_ids(gids), maxBond(maxB), absCut(aC), relCut(rC) {
    int G = gpu_ids.size();
    // Share the persistent comms/handles (copies handle values; not owned here).
    ctx = shardedGpuPool(gpu_ids);
    owner.resize(n);
    for (int s=0;s<n;++s) owner[s] = gpu_ids[(long long)s*G/n];
    site.resize(n);
    for (int s=0;s<n;++s){ int dev=owner[s]; CUDA_CHECK(cudaSetDevice(dev));
      cdouble h[2]={make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0)};
      cdouble* d; CUDA_CHECK(cudaMalloc(&d, sizeof(cdouble)*2));
      CUDA_CHECK(cudaMemcpy(d,h,sizeof(cdouble)*2,cudaMemcpyHostToDevice));
      site[s]={d,1,1,dev}; }
  }

  // Free only this instance's site tensors. The comms/cuBLAS/cuSOLVER handles in
  // ctx belong to the persistent pool (shardedGpuPool) and are reused, so they
  // must NOT be destroyed here. No CUDA_CHECK (never throw from a destructor).
  ~ShardedMPS(){
    for (auto& st : site) if (st.d){ cudaSetDevice(st.dev); cudaFree(st.d); st.d=nullptr; }
  }

  GpuCtx& ctxOf(int dev){ for(auto&c:ctx) if(c.dev==dev) return c; return ctx[0]; }

  void resetZero(){
    for (int s=0;s<n;++s){ CUDA_CHECK(cudaSetDevice(site[s].dev)); CUDA_CHECK(cudaFree(site[s].d));
      cdouble h[2]={make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0)};
      cdouble* d; CUDA_CHECK(cudaMalloc(&d,sizeof(cdouble)*2));
      CUDA_CHECK(cudaMemcpy(d,h,sizeof(cdouble)*2,cudaMemcpyHostToDevice));
      site[s]={d,1,1,owner[s]}; }
  }

  void apply1q(const std::vector<std::complex<double>>& U, int s){
    int dev=site[s].dev; CUDA_CHECK(cudaSetDevice(dev));
    cdouble* Ud; CUDA_CHECK(cudaMalloc(&Ud,sizeof(cdouble)*4));
    CUDA_CHECK(cudaMemcpy(Ud,U.data(),sizeof(cdouble)*4,cudaMemcpyHostToDevice));
    int Dl=site[s].Dl, Dr=site[s].Dr; size_t sz=(size_t)Dl*2*Dr;
    cdouble* out; CUDA_CHECK(cudaMalloc(&out,sizeof(cdouble)*sz));
    int th=256, bl=(Dl*2*Dr+th-1)/th;
    k_apply1q<<<bl,th>>>(site[s].d, Ud, out, Dl, Dr); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(site[s].d)); CUDA_CHECK(cudaFree(Ud)); site[s].d=out;
  }

  // adjacent 2q gate on (s,s+1); G4 row-major 4x4 [(I*2+J)*4+(i*2+j)]
  void apply2q(const std::vector<std::complex<double>>& G4, int s){
    int ldev=site[s].dev, rdev=site[s+1].dev, work=ldev;
    int Dl=site[s].Dl, Dc=site[s].Dr, Dr=site[s+1].Dr;
    CUDA_CHECK(cudaSetDevice(work));
    cdouble* A = site[s].d;
    cdouble* B = (rdev==work)? site[s+1].d : peerCopy(rdev, site[s+1].d, work, (size_t)Dc*2*Dr);
    cdouble* Gd; CUDA_CHECK(cudaMalloc(&Gd,sizeof(cdouble)*16));
    CUDA_CHECK(cudaMemcpy(Gd,G4.data(),sizeof(cdouble)*16,cudaMemcpyHostToDevice));
    int m=2*Dl, ncol=2*Dr;
    cdouble* M; CUDA_CHECK(cudaMalloc(&M,sizeof(cdouble)*(size_t)m*ncol));
    int th=256, bl=(Dl*2*2*Dr+th-1)/th;
    k_buildM<<<bl,th>>>(M,A,B,Gd,Dl,Dc,Dr); CUDA_CHECK(cudaDeviceSynchronize());

    // Single-GPU truncated SVD. (Stage 3 -- a distributed-SVD path scattering M
    // across all GPUs -- was implemented and validated to 1.4e-15, but on PCIe
    // (no NVLink) at chi<=512 the per-gate NCCL communication outweighed the
    // O(chi^3/G) compute saving, so it is DISABLED. The multi-GPU distSvd routine
    // remains in dist_svd.cuh if revisited on NVLink / very large chi.)
    std::vector<GpuCtx> one{ctxOf(work)};
    std::vector<cdouble*> blk{M}; std::vector<int> ng{ncol};
    SvdResult r = distSvd(one, blk, ng, m, ncol, maxBond, absCut, relCut, 10, seedCounter++);
    int k=r.k;

    std::vector<cdouble> hA((size_t)Dl*2*k);
    for (int l=0;l<Dl;++l) for(int I=0;I<2;++I) for(int c=0;c<k;++c)
      hA[(size_t)(l*2+I)*k + c] = r.U[(size_t)c*m + (l*2+I)];
    std::vector<cdouble> hB((size_t)k*2*Dr);
    for (int c=0;c<k;++c) for(int J=0;J<2;++J) for(int rr=0;rr<Dr;++rr){
      cdouble v = r.Vh[(size_t)(J*Dr+rr)*k + c];
      hB[(size_t)(c*2+J)*Dr + rr] = make_cuDoubleComplex(r.S[c]*v.x, r.S[c]*v.y);
    }
    CUDA_CHECK(cudaFree(site[s].d));
    CUDA_CHECK(cudaSetDevice(ldev)); cdouble* dA; CUDA_CHECK(cudaMalloc(&dA,sizeof(cdouble)*hA.size()));
    CUDA_CHECK(cudaMemcpy(dA,hA.data(),sizeof(cdouble)*hA.size(),cudaMemcpyHostToDevice));
    site[s].d=dA; site[s].Dl=Dl; site[s].Dr=k;
    CUDA_CHECK(cudaFree(site[s+1].d));
    CUDA_CHECK(cudaSetDevice(rdev)); cdouble* dB; CUDA_CHECK(cudaMalloc(&dB,sizeof(cdouble)*hB.size()));
    CUDA_CHECK(cudaMemcpy(dB,hB.data(),sizeof(cdouble)*hB.size(),cudaMemcpyHostToDevice));
    site[s+1].d=dB; site[s+1].Dl=k; site[s+1].Dr=Dr;
    CUDA_CHECK(cudaSetDevice(work)); CUDA_CHECK(cudaFree(M)); CUDA_CHECK(cudaFree(Gd));
    if (rdev!=work) CUDA_CHECK(cudaFree(B));
  }

  static std::vector<std::complex<double>> swapMat(){
    std::vector<std::complex<double>> S(16,0.0);
    for(int I=0;I<2;++I)for(int J=0;J<2;++J)for(int i=0;i<2;++i)for(int j=0;j<2;++j)
      S[(I*2+J)*4+(i*2+j)] = (I==j && J==i)?1.0:0.0;
    return S;
  }
  static std::vector<std::complex<double>> swapOperands(const std::vector<std::complex<double>>& G){
    std::vector<std::complex<double>> H(16);
    for(int I=0;I<2;++I)for(int J=0;J<2;++J)for(int i=0;i<2;++i)for(int j=0;j<2;++j)
      H[(I*2+J)*4+(i*2+j)] = G[(J*2+I)*4+(j*2+i)];
    return H;
  }

  // arbitrary-pair 2q gate on qubits (a,b) via SWAP routing
  void applyTwoQubit(const std::vector<std::complex<double>>& G4, int a, int b){
    if (a==b) throw std::runtime_error("2q gate on same qubit");
    if (a>b){ applyTwoQubit(swapOperands(G4), b, a); return; }
    static const std::vector<std::complex<double>> SW = swapMat();
    for (int x=b; x>a+1; --x) apply2q(SW, x-1);   // bring qubit b down to a+1
    apply2q(G4, a);
    for (int x=a+1; x<b; ++x) apply2q(SW, x);      // restore order
  }

  // GPU-side <psi|P|psi> for a Pauli string (exact, any chi). Environment E is
  // kept on-device; two O(chi^3) kernel sweeps per site on the owning GPU, with
  // E peer-copied across GPU boundaries.
  std::complex<double> expectPauli(const std::string& paulis){
    int Edev = site[0].dev;
    CUDA_CHECK(cudaSetDevice(Edev));
    cdouble* E; CUDA_CHECK(cudaMalloc(&E, sizeof(cdouble)));
    cdouble one = make_cuDoubleComplex(1,0);
    CUDA_CHECK(cudaMemcpy(E, &one, sizeof(cdouble), cudaMemcpyHostToDevice));
    int dimE = 1;
    const int TH = 256;
    for (int s = 0; s < n; ++s) {
      int dev = site[s].dev, Dl = site[s].Dl, Dr = site[s].Dr;
      if (Edev != dev) {  // bring environment to the GPU that owns this site
        CUDA_CHECK(cudaSetDevice(dev));
        cdouble* E2; CUDA_CHECK(cudaMalloc(&E2, sizeof(cdouble)*(size_t)dimE*dimE));
        CUDA_CHECK(cudaMemcpyPeer(E2, dev, E, Edev, sizeof(cdouble)*(size_t)dimE*dimE));
        CUDA_CHECK(cudaSetDevice(Edev)); CUDA_CHECK(cudaFree(E));
        E = E2; Edev = dev;
      }
      CUDA_CHECK(cudaSetDevice(dev));
      cdouble *R, *S, *newE;
      CUDA_CHECK(cudaMalloc(&R,    sizeof(cdouble)*(size_t)Dl*2*Dr));
      CUDA_CHECK(cudaMalloc(&S,    sizeof(cdouble)*(size_t)Dl*2*Dr));
      CUDA_CHECK(cudaMalloc(&newE, sizeof(cdouble)*(size_t)Dr*Dr));
      char p = paulis.empty() ? 'I' : paulis[s];
      cdouble p00,p01,p10,p11; pauliMat(p, p00,p01,p10,p11);
      k_transL<<<(Dl*2*Dr+TH-1)/TH, TH>>>(E, site[s].d, R, Dl, Dr);
      k_applyP<<<(Dl*2*Dr+TH-1)/TH, TH>>>(R, S, Dl, Dr, p00,p01,p10,p11);
      k_transR<<<(Dr*Dr+TH-1)/TH, TH>>>(site[s].d, S, newE, Dl, Dr);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(R)); CUDA_CHECK(cudaFree(S)); CUDA_CHECK(cudaFree(E));
      E = newE; dimE = Dr;  // Edev stays = dev
    }
    cdouble res; CUDA_CHECK(cudaSetDevice(Edev));
    CUDA_CHECK(cudaMemcpy(&res, E, sizeof(cdouble), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(E));
    return std::complex<double>(res.x, res.y);
  }

  // Stage 2: batched Pauli expectations via a GPU-segment pipeline. Each
  // contiguous GPU segment is a pipeline stage; the per-term environment flows
  // left->right, so after the pipeline fills, every segment (GPU) is working on
  // a different term concurrently. Returns <psi|P_k|psi> for each word.
  std::vector<std::complex<double>> expectPauliBatch(const std::vector<std::string>& words){
    int T = (int)words.size();
    std::vector<std::complex<double>> result(T, std::complex<double>(0,0));
    if (T == 0) return result;

    // ---- contiguous segments from owner[] (one pipeline stage per segment) --
    struct Seg { int dev, s0, s1, dIn, dOut, maxRS, maxB; };
    std::vector<Seg> segs;
    for (int s = 0; s < n; ) {
      int d = site[s].dev, s0 = s;
      while (s < n && site[s].dev == d) ++s;
      Seg sg; sg.dev = d; sg.s0 = s0; sg.s1 = s;
      sg.dIn = site[s0].Dl; sg.dOut = site[s-1].Dr;
      sg.maxRS = 0; sg.maxB = std::max(sg.dIn, sg.dOut);
      for (int k = s0; k < s; ++k) {
        sg.maxRS = std::max(sg.maxRS, site[k].Dl*2*site[k].Dr);
        sg.maxB  = std::max(sg.maxB, std::max(site[k].Dl, site[k].Dr));
      }
      segs.push_back(sg);
    }
    int G = (int)segs.size();

    // ---- preallocate per-segment scratch + boundary buffers (reused) --------
    std::vector<cdouble*> inbuf(G), outbuf(G), Rb(G), Sb(G), Eb0(G), Eb1(G);
    for (int g = 0; g < G; ++g) { auto& sg = segs[g]; CUDA_CHECK(cudaSetDevice(sg.dev));
      CUDA_CHECK(cudaMalloc(&inbuf[g],  sizeof(cdouble)*(size_t)sg.dIn*sg.dIn));
      CUDA_CHECK(cudaMalloc(&outbuf[g], sizeof(cdouble)*(size_t)sg.dOut*sg.dOut));
      CUDA_CHECK(cudaMalloc(&Rb[g],     sizeof(cdouble)*(size_t)sg.maxRS));
      CUDA_CHECK(cudaMalloc(&Sb[g],     sizeof(cdouble)*(size_t)sg.maxRS));
      CUDA_CHECK(cudaMalloc(&Eb0[g],    sizeof(cdouble)*(size_t)sg.maxB*sg.maxB));
      CUDA_CHECK(cudaMalloc(&Eb1[g],    sizeof(cdouble)*(size_t)sg.maxB*sg.maxB));
    }
    // every term enters segment 0 with the 1x1 identity environment; set once.
    { cdouble one = make_cuDoubleComplex(1,0); CUDA_CHECK(cudaSetDevice(segs[0].dev));
      CUDA_CHECK(cudaMemcpy(inbuf[0], &one, sizeof(cdouble), cudaMemcpyHostToDevice)); }

    const int TH = 256;
    // launch one segment's transfer (async on its GPU's default stream; no sync)
    auto segTransfer = [&](int g, const std::string& word){
      auto& sg = segs[g]; CUDA_CHECK(cudaSetDevice(sg.dev));
      cdouble* E = inbuf[g]; int pp = 0;
      for (int s = sg.s0; s < sg.s1; ++s) {
        int Dl = site[s].Dl, Dr = site[s].Dr; bool last = (s == sg.s1 - 1);
        cdouble* dst = last ? outbuf[g] : (pp ? Eb1[g] : Eb0[g]);
        char p = word.empty() ? 'I' : word[s];
        cdouble p00,p01,p10,p11; pauliMat(p, p00,p01,p10,p11);
        k_transL<<<(Dl*2*Dr+TH-1)/TH, TH>>>(E, site[s].d, Rb[g], Dl, Dr);
        k_applyP<<<(Dl*2*Dr+TH-1)/TH, TH>>>(Rb[g], Sb[g], Dl, Dr, p00,p01,p10,p11);
        k_transR<<<(Dr*Dr+TH-1)/TH, TH>>>(site[s].d, Sb[g], dst, Dl, Dr);
        E = dst; pp ^= 1;
      }
    };

    // ---- wavefront: at step t, segment g handles term (t-g) ----------------
    for (int t = 0; t < T + G - 1; ++t) {
      for (int g = 0; g < G; ++g) { int k = t - g; if (k < 0 || k >= T) continue; segTransfer(g, words[k]); }
      for (int g = 0; g < G; ++g) { int k = t - g; if (k < 0 || k >= T) continue;
        CUDA_CHECK(cudaSetDevice(segs[g].dev)); CUDA_CHECK(cudaDeviceSynchronize()); }
      for (int g = 0; g < G; ++g) { int k = t - g; if (k < 0 || k >= T) continue;
        if (g == G-1) { cdouble r; CUDA_CHECK(cudaSetDevice(segs[g].dev));
          CUDA_CHECK(cudaMemcpy(&r, outbuf[g], sizeof(cdouble), cudaMemcpyDeviceToHost));
          result[k] = std::complex<double>(r.x, r.y);
        } else { size_t sz = sizeof(cdouble)*(size_t)segs[g].dOut*segs[g].dOut;
          CUDA_CHECK(cudaMemcpyPeer(inbuf[g+1], segs[g+1].dev, outbuf[g], segs[g].dev, sz)); }
      }
    }
    for (int g = 0; g < G; ++g) { CUDA_CHECK(cudaSetDevice(segs[g].dev));
      cudaFree(inbuf[g]); cudaFree(outbuf[g]); cudaFree(Rb[g]); cudaFree(Sb[g]); cudaFree(Eb0[g]); cudaFree(Eb1[g]); }
    return result;
  }

  // Reference host implementation (O(chi^4)); kept for validation/fallback.
  std::complex<double> expectPauliHost(const std::string& paulis){
    using cd = std::complex<double>;
    auto P = [&](char p, int i, int ip)->cd{
      switch(p){ case 'I': return i==ip?cd(1,0):cd(0,0);
        case 'X': return i!=ip?cd(1,0):cd(0,0);
        case 'Y': return (i==0&&ip==1)?cd(0,-1):((i==1&&ip==0)?cd(0,1):cd(0,0));
        case 'Z': return i==ip?(i==0?cd(1,0):cd(-1,0)):cd(0,0);
        default: return i==ip?cd(1,0):cd(0,0);} };
    // E: (chiL x chiL) transfer; start 1x1 = [1]
    std::vector<cd> E(1, cd(1,0)); int dimE=1;
    for (int s=0;s<n;++s){
      int Dl=site[s].Dl, Dr=site[s].Dr;
      std::vector<cdouble> h((size_t)Dl*2*Dr);
      CUDA_CHECK(cudaSetDevice(site[s].dev));
      CUDA_CHECK(cudaMemcpy(h.data(),site[s].d,sizeof(cdouble)*h.size(),cudaMemcpyDeviceToHost));
      auto A=[&](int a,int i,int c)->cd{ cdouble z=h[(a*2+i)*Dr+c]; return cd(z.x,z.y); };
      char p = paulis.empty()? 'I' : paulis[s];
      std::vector<cd> newE((size_t)Dr*Dr, cd(0,0));
      for (int b=0;b<Dr;++b) for(int bp=0;bp<Dr;++bp){ cd acc(0,0);
        for (int a=0;a<Dl;++a) for(int ap=0;ap<Dl;++ap){ cd e=E[(size_t)a*dimE+ap]; if(e==cd(0,0))continue;
          for(int i=0;i<2;++i) for(int ip=0;ip<2;++ip){ cd pv=P(p,i,ip); if(pv==cd(0,0))continue;
            acc += std::conj(A(a,i,b)) * pv * A(ap,ip,bp) * e; } }
        newE[(size_t)b*Dr+bp]=acc; }
      E.swap(newE); dimE=Dr;
    }
    return E[0];   // 1x1
  }

  std::vector<std::complex<double>> toStatevector(){
    std::vector<std::complex<double>> psi;
    int Dr0=site[0].Dr;
    std::vector<cdouble> h((size_t)1*2*Dr0);
    CUDA_CHECK(cudaSetDevice(site[0].dev));
    CUDA_CHECK(cudaMemcpy(h.data(),site[0].d,sizeof(cdouble)*h.size(),cudaMemcpyDeviceToHost));
    int rows=2, bond=Dr0; psi.resize((size_t)rows*bond);
    for(int i=0;i<2;++i) for(int c=0;c<Dr0;++c) psi[(size_t)i*bond+c]={h[(0*2+i)*Dr0+c].x,h[(0*2+i)*Dr0+c].y};
    for(int s=1;s<n;++s){ int Dc=site[s].Dl, Dr=site[s].Dr;
      std::vector<cdouble> hs((size_t)Dc*2*Dr);
      CUDA_CHECK(cudaSetDevice(site[s].dev));
      CUDA_CHECK(cudaMemcpy(hs.data(),site[s].d,sizeof(cdouble)*hs.size(),cudaMemcpyDeviceToHost));
      std::vector<std::complex<double>> np((size_t)rows*2*Dr, 0.0);
      for(int rIdx=0;rIdx<rows;++rIdx) for(int j=0;j<2;++j) for(int rr=0;rr<Dr;++rr){
        std::complex<double> acc=0;
        for(int c=0;c<Dc;++c) acc += psi[(size_t)rIdx*bond+c]*std::complex<double>(hs[(c*2+j)*Dr+rr].x,hs[(c*2+j)*Dr+rr].y);
        np[((size_t)rIdx*2+j)*Dr+rr]=acc; }
      rows*=2; bond=Dr; psi.swap(np); }
    return psi;
  }

  // Exact MPS perfect sampling (Ferris-Vidal): returns `shots` bitstrings in
  // CUDA-Q qubit order (qubit 0 = leftmost). Site tensors are pulled to host
  // (tiny at the chi used here); precompute right environments, then sample
  // left->right from the exact conditional P(s_k | s_0..s_{k-1}). O(shots*n*chi^2).
  std::vector<std::string> sampleBits(int shots, unsigned seed){
    using cd = std::complex<double>;
    std::vector<std::vector<cd>> A(n); std::vector<int> Dl(n), Dr(n);
    for (int k=0;k<n;++k){ Dl[k]=site[k].Dl; Dr[k]=site[k].Dr;
      std::vector<cdouble> h((size_t)Dl[k]*2*Dr[k]);
      CUDA_CHECK(cudaSetDevice(site[k].dev));
      CUDA_CHECK(cudaMemcpy(h.data(), site[k].d, sizeof(cdouble)*h.size(), cudaMemcpyDeviceToHost));
      A[k].resize(h.size());
      for (size_t t=0;t<h.size();++t) A[k][t]=cd(h[t].x, h[t].y);
    }
    // right environments R[k] (Dl_k x Dl_k); R[n] = [1].
    std::vector<std::vector<cd>> R(n+1);
    R[n].assign(1, cd(1,0));
    for (int k=n-1;k>=0;--k){
      int dl=Dl[k], dr=Dr[k];
      std::vector<cd> T((size_t)dl*2*dr, cd(0,0));   // T[a,i,b'] = sum_b A[a,i,b] R[k+1][b,b']
      for (int a=0;a<dl;++a) for (int i=0;i<2;++i) for (int bp=0;bp<dr;++bp){
        cd s(0,0); for (int b=0;b<dr;++b) s += A[k][(a*2+i)*dr+b]*R[k+1][(size_t)b*dr+bp];
        T[(a*2+i)*dr+bp]=s; }
      std::vector<cd> Rk((size_t)dl*dl, cd(0,0));
      for (int a=0;a<dl;++a) for (int ap=0;ap<dl;++ap){
        cd s(0,0); for (int i=0;i<2;++i) for (int bp=0;bp<dr;++bp) s += T[(a*2+i)*dr+bp]*std::conj(A[k][(ap*2+i)*dr+bp]);
        Rk[(size_t)a*dl+ap]=s; }
      R[k].swap(Rk);
    }
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> uni(0.0,1.0);
    std::vector<std::string> out; out.reserve(shots);
    for (int sh=0; sh<shots; ++sh){
      std::vector<cd> v(1, cd(1,0));   // left vector over Dl_0 = 1
      std::string bits(n, '0');
      for (int k=0;k<n;++k){
        int dl=Dl[k], dr=Dr[k];
        std::vector<cd> w0(dr,cd(0,0)), w1(dr,cd(0,0));   // w_i[b] = sum_a v[a] A[a,i,b]
        for (int a=0;a<dl;++a){ cd va=v[a];
          for (int b=0;b<dr;++b){ w0[b]+=va*A[k][(a*2+0)*dr+b]; w1[b]+=va*A[k][(a*2+1)*dr+b]; } }
        // P(prefix,s_k=i) = sum_{b,bp} w[b] R[k+1][b,bp] conj(w[bp]), where
        // R[b,bp] = sum_tail t[b] conj(t[bp]). The conj must be on the SECOND
        // factor: MPS tensors are complex (from the SVD) even for real states.
        auto quad=[&](std::vector<cd>& w)->double{ cd s(0,0);
          for (int b=0;b<dr;++b){ cd t(0,0); for (int bp=0;bp<dr;++bp) t += R[k+1][(size_t)b*dr+bp]*std::conj(w[bp]);
            s += w[b]*t; } return s.real(); };
        double p0=quad(w0), p1=quad(w1), tot=p0+p1; if (tot<=0){ p0=p1=0.5; tot=1.0; }
        int si = (uni(rng) < p0/tot) ? 0 : 1;
        bits[k] = si ? '1' : '0';                         // MPS site order
        std::vector<cd>& ws = si ? w1 : w0;
        double nrm = std::sqrt(si ? p1 : p0); if (nrm>0) for (auto& z : ws) z/=nrm;   // renormalize (avoid underflow)
        v.swap(ws);
      }
      std::reverse(bits.begin(), bits.end());   // MPS site k = CUDA-Q qubit n-1-k -> reverse to qubit order
      out.push_back(std::move(bits));
    }
    return out;
  }
};
MGPU_SHARDED_MPS_CUH_EOF

# The heredocs above retain the historical backend for script readability, but
# the versioned package is authoritative. It contains the GPU-resident MQPU
# SVD, gate, and finite-shot sampling paths and makes a moved checkout rebuild
# exactly the backend that was tested here.
install -m 0644 "$PACKAGED_BACKEND/analytic_batch.cuh" "$BACKEND/analytic_batch.cuh"
install -m 0644 "$PACKAGED_BACKEND/dist_svd.cuh" "$BACKEND/dist_svd.cuh"
install -m 0644 "$PACKAGED_BACKEND/sharded_mps.cuh" "$BACKEND/sharded_mps.cuh"

cat > "$BACKEND/simulator_mps_dist.cu" <<'MGPU_SIMULATOR_MPS_DIST_CU_EOF'
// NVQIR wrappers around the tested custom MPS engine (sharded_mps.cuh).
//
// This source intentionally builds two plugins:
//   tensornet-mps-mgpu: the original MPS-chain sharding backend.
//   tensornet-mps-mqpu: one independent, non-sharded custom MPS per CUDA-Q
//                       MQPU worker. CUDA-Q batches Hamiltonian terms across
//                       those workers.
//
// IMPORTANT: this file compiles ONLY inside an in-tree CUDA-Q build (it needs
// nvqir/CircuitSimulator.h and cudaq/spin_op.h). It is NOT built by the local
// build.sh. Drop it into runtime/nvqir/cutensornet/ and add it to that dir's
// CMakeLists (see distributed_mps/cpp_backend/CMake_snippet.txt). Spots marked
// [VERIFY] use CUDA-Q APIs whose exact form may need a small fixup for your
// release; the underlying engine is already validated.
//
// Design: replaces cuTensorNet's lazy state with our eager explicit MPS chain.
//   applyGate  -> ShardedMPS::apply1q / applyTwoQubit (SWAP-routed)
//   observe    -> sum_k coeff_k * <P_k>  via ShardedMPS::expectPauli
//   bond dim   <- CUDAQ_MPS_MAX_BOND (same env var as stock tensornet-mps)
//
// CUDAQ_MPS_MQPU_BACKEND is defined only for the tensornet-mps-mqpu plugin.
// In that build, ensureEngine() inherits the CUDA device selected by the MQPU
// worker and passes a singleton device list to ShardedMPS. Consequently it
// initializes no NCCL communicator, does no cross-GPU MPS sharding, and makes
// no percentage-of-free-memory scratch reservation.
//
// Noise support (quantum trajectory / Monte Carlo wavefunction method):
//   - applyNoiseChannel() stochastically selects one Kraus operator per gate.
//   - Unitary-mixture channels (depolarizing, bit-flip, phase-flip, Pauli):
//     selected with given probabilities; no renormalization needed.
//   - General 1-qubit Kraus (amplitude damping): probability of each operator
//     computed from the local reduced density matrix via <X>,<Y>,<Z> on the
//     target qubit; state renormalized after selection.
//   - observe()/sample() with noise run a trajectory loop: gate history is
//     replayed from |0>^n for each trajectory, each with a fresh Kraus
//     realization. Number of trajectories = numberTrajectories if set, else
//     shots, else DEFAULT_OBSERVE_TRAJECTORIES.

#ifdef CUDAQ_MPS_MQPU_BACKEND
// Both custom plugins can be loaded into the same process. Give the MQPU build
// distinct C++ symbols so ELF interposition cannot bind its factory/vtable or
// MPS methods to the already-loaded tensornet-mps-mgpu implementation.
#define ShardedMPS ShardedMPSMQPU
#define SimulatorMPSDist SimulatorMPSMQPU
#endif

#include "sharded_mps.cuh"
#include "nvqir/CircuitSimulator.h"
#include "common/NoiseModel.h"
#include "cudaq/spin_op.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <random>

#ifdef CUDAQ_MPS_MQPU_BACKEND
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mqpu"
#else
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mgpu"
#endif

// Round-robin GPU assignment for single-GPU MPS mode (CUDAQ_MPS_NUM_GPUS=1).
// Each new thread that calls ensureEngine() is assigned the next available GPU
// so that parallel SubspaceVQE blocks land on different physical GPUs.
#ifndef CUDAQ_MPS_MQPU_BACKEND
static std::atomic<int> g_next_gpu_rr{0};
#endif

namespace nvqir {

class SimulatorMPSDist : public nvqir::CircuitSimulatorBase<double> {
  using Base = nvqir::CircuitSimulatorBase<double>;
  using GateApplicationTask = typename Base::GateApplicationTask;

  std::unique_ptr<ShardedMPS> mps;
  std::size_t allocated = 0;     // qubits allocated so far
  int maxBond = 64;
  double absCut = 1e-5, relCut = 1e-5;
  std::vector<int> gpuIds;

  // ---- Noise / trajectory support ------------------------------------------
  // Each gate applied during circuit execution is recorded so that the full
  // circuit can be replayed (from |0>^n) for each noisy trajectory.
  struct GateRecord {
    std::string operationName;
    std::vector<std::complex<double>> matrix;
    std::vector<std::size_t> controls;
    std::vector<std::size_t> targets;
    std::vector<double> parameters;
  };
  std::vector<GateRecord> m_gateHistory;
  bool m_recordingGates = true;   // false during trajectory replays
  std::mt19937_64 m_rng{std::random_device{}()};

  // Default trajectories when noise model is set but no explicit count given.
  static constexpr std::size_t DEFAULT_OBSERVE_TRAJECTORIES = 1000;

  // Build the engine once the qubit count is known.
  //
  // tensornet-mps-mqpu:
  //   CUDA-Q's MQPU platform selects the worker device before simulator use.
  //   Inherit that device and force a singleton MPS. This is deliberately
  //   independent of CUDAQ_MPS_NUM_GPUS.
  //
  // tensornet-mps-mgpu, CUDAQ_MPS_NUM_GPUS (default: all visible GPUs):
  //   Set to 1 to use a single GPU per circuit, enabling SubspaceVQE to run
  //   multiple blocks in parallel on different GPUs via ThreadPoolExecutor.
  //   Each new thread gets the next GPU in round-robin order so blocks on
  //   GPU 0, 1, 2, ... run concurrently without NCCL overhead.
  //   Set to N>1 to shard the MPS chain across N GPUs (original behavior).
  void ensureEngine() {
    if (mps) return;
    if (allocated < 2)
      throw std::runtime_error(CUDAQ_CUSTOM_MPS_BACKEND_NAME
                               " needs >= 2 qubits");

    gpuIds.clear();
#ifdef CUDAQ_MPS_MQPU_BACKEND
    // GPUEmulatedQPU has already called cudaSetDevice(qpu_id). Capture that
    // logical device rather than assigning by thread creation order.
    int selectedDevice = 0;
    CUDA_CHECK(cudaGetDevice(&selectedDevice));
    gpuIds.push_back(selectedDevice);
#else
    int G_total = 1;
    CUDA_CHECK(cudaGetDeviceCount(&G_total));
    const char* ng_env = std::getenv("CUDAQ_MPS_NUM_GPUS");
    int num_mps_gpus = ng_env ? std::max(1, std::atoi(ng_env)) : G_total;
    num_mps_gpus = std::min(num_mps_gpus, G_total);

    if (num_mps_gpus == 1) {
      // Single-GPU mode: assign this thread a GPU via round-robin so parallel
      // block workers land on different devices.
      thread_local static int t_gpu = -1;
      if (t_gpu < 0)
        t_gpu = g_next_gpu_rr.fetch_add(1) % G_total;
      gpuIds.push_back(t_gpu);
    } else {
      for (int i = 0; i < num_mps_gpus; ++i) gpuIds.push_back(i);
    }
#endif

    if (auto *b = std::getenv("CUDAQ_MPS_MAX_BOND")) maxBond = std::max(1, atoi(b));
    if (auto *a = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) absCut = atof(a);
    if (auto *r = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) relCut = atof(r);
    mps = std::make_unique<ShardedMPS>((int)allocated, gpuIds, maxBond, absCut, relCut);
    mps->distributedBoundary = (gpuIds.size() > 1);
  }

  // controlled-U (1 control c, 1 target t) -> 4x4 row-major [(C*2+T)*4+(c*2+t)]
  static std::vector<std::complex<double>>
  controlledGate(const std::vector<std::complex<double>> &U) {
    std::vector<std::complex<double>> G(16, 0.0);
    for (int C=0;C<2;++C) for(int T=0;T<2;++T) for(int c=0;c<2;++c) for(int t=0;t<2;++t){
      if (C != c) continue;                       // control qubit unchanged
      std::complex<double> v = (c==0) ? (T==t ? 1.0 : 0.0) : U[T*2+t];
      G[(C*2+T)*4 + (c*2+t)] = v;
    }
    return G;
  }

  // CUDA-Q qubit index -> our site index (msb ordering, like Qpp's helper).
  int conv(std::size_t i) const { return (int)(allocated - 1 - i); }

  // ---- Internal gate application (no history recording) --------------------
  void applyGateInternal(const GateApplicationTask &task) {
    const auto &m = task.matrix;
    std::vector<std::size_t> ctr, tgt;
    for (auto i : task.controls) ctr.push_back(conv(i));
    for (auto i : task.targets)  tgt.push_back(conv(i));

    if (ctr.empty() && tgt.size() == 1) {
      std::vector<std::complex<double>> U{m[0], m[1], m[2], m[3]};
      mps->apply1q(U, (int)tgt[0]); return;
    }
    if (ctr.empty() && tgt.size() == 2) {
      mps->applyTwoQubit(m, (int)tgt[0], (int)tgt[1]); return;
    }
    if (ctr.size() == 1 && tgt.size() == 1) {
      std::vector<std::complex<double>> U{m[0], m[1], m[2], m[3]};
      mps->applyTwoQubit(controlledGate(U), (int)ctr[0], (int)tgt[0]); return;
    }
    throw std::runtime_error(
        CUDAQ_CUSTOM_MPS_BACKEND_NAME
        ": >2-qubit or multi-control gates not yet supported "
        "(decompose, or extend applyGate).");
  }

  // ---- Kraus channel application helpers -----------------------------------

  // Apply one Kraus channel to the MPS via quantum trajectory selection.
  void applyKrausChannel(const std::vector<std::size_t> &targets,
                         const cudaq::kraus_channel &ch) {
    using cd = std::complex<double>;
    if (ch.is_unitary_mixture()) {
      // Sample one unitary from the mixture with the given probabilities.
      std::discrete_distribution<int> dist(ch.probabilities.begin(),
                                           ch.probabilities.end());
      int idx = dist(m_rng);
      const auto &U = ch.unitary_ops[idx]; // std::vector<cd>, row-major
      if (targets.size() == 1) {
        mps->apply1q(U, conv(targets[0]));
      } else if (targets.size() == 2) {
        mps->applyTwoQubit(U, conv(targets[0]), conv(targets[1]));
      } else {
        throw std::runtime_error(
            CUDAQ_CUSTOM_MPS_BACKEND_NAME
            ": noise channel on >2 qubits not supported");
      }
      // Unitary preserves norm; no renormalization needed.
    } else {
      // General (non-unitary-mixture) Kraus channel.
      if (targets.size() == 1) {
        applyGeneralKraus1q(targets[0], ch.get_ops());
      } else {
        throw std::runtime_error(
            CUDAQ_CUSTOM_MPS_BACKEND_NAME
            ": general (non-unitary-mixture) Kraus channels "
            "on 2-qubit targets are not supported; use depolarization or "
            "Pauli2 channels instead.");
      }
    }
  }

  // Quantum trajectory application of a general 1-qubit Kraus channel.
  // Computes p_m = Tr(K_m†K_m · ρ_q) from the local reduced density matrix,
  // samples m, applies K_m, and renormalizes the MPS.
  void applyGeneralKraus1q(std::size_t cudaq_qubit,
                            const std::vector<cudaq::kraus_op> &ops) {
    using cd = std::complex<double>;
    const int site = conv(cudaq_qubit);

    // Build single-qubit Pauli words for the target qubit.
    std::string wZ(mps->n, 'I'); wZ[site] = 'Z';

    // <Z> always needed for diagonal rho elements.
    double Iz = mps->expectPauli(wZ).real();
    double rho00 = (1.0 + Iz) / 2.0;
    double rho11 = (1.0 - Iz) / 2.0;

    // Check if any K†K has off-diagonal elements (needs <X>, <Y> too).
    bool needOffDiag = false;
    for (const auto &op : ops) {
      const auto &d = op.data;
      cd KdK01 = std::conj(d[0])*d[1] + std::conj(d[2])*d[3];
      if (std::abs(KdK01) > 1e-10) { needOffDiag = true; break; }
    }
    cd rho10{0.0, 0.0};
    if (needOffDiag) {
      std::string wX(mps->n, 'I'); wX[site] = 'X';
      std::string wY(mps->n, 'I'); wY[site] = 'Y';
      double Ix = mps->expectPauli(wX).real();
      double Iy = mps->expectPauli(wY).real();
      // ρ_q = (I + Ix*X + Iy*Y + Iz*Z) / 2
      // ρ[1][0] = (Ix + i*Iy) / 2  (ket=0, bra=1 convention)
      rho10 = cd(Ix, Iy) / 2.0;
    }

    // Compute p_m = Tr(K_m†K_m · ρ_q) for each Kraus op.
    std::vector<double> probs;
    probs.reserve(ops.size());
    for (const auto &op : ops) {
      const auto &d = op.data;
      // K†K diagonal elements
      cd KdK00 = std::conj(d[0])*d[0] + std::conj(d[2])*d[2];
      cd KdK11 = std::conj(d[1])*d[1] + std::conj(d[3])*d[3];
      double p = KdK00.real()*rho00 + KdK11.real()*rho11;
      if (needOffDiag) {
        cd KdK01 = std::conj(d[0])*d[1] + std::conj(d[2])*d[3];
        // 2*Re(K†K[0][1] * ρ[1][0])  (off-diagonal contribution)
        p += 2.0 * (KdK01 * rho10).real();
      }
      probs.push_back(std::max(0.0, p));
    }

    // Normalize (completeness: sum should be 1 up to numerical drift).
    double total = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (total > 1e-15)
      for (auto &p : probs) p /= total;
    else
      std::fill(probs.begin(), probs.end(), 1.0 / probs.size()); // fallback

    // Sample Kraus operator index.
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int idx = dist(m_rng);

    // Apply selected Kraus operator.
    const auto &d = ops[idx].data;
    std::vector<cd> K{cd(d[0]), cd(d[1]), cd(d[2]), cd(d[3])};
    mps->apply1q(K, site);

    // Renormalize state by 1/sqrt(p_idx).
    double nrm = std::sqrt(probs[idx]);
    if (nrm > 1e-15) {
      cd s = 1.0 / nrm;
      std::vector<cd> scaleOp{s, 0.0, 0.0, s};   // diagonal scaling matrix
      mps->apply1q(scaleOp, site);
    }
  }

  // ---- Trajectory replay ---------------------------------------------------
  // Re-execute the saved gate history from |0>^n, applying noise after each
  // gate. Used by observe() and sample() for multi-trajectory averaging.
  // Must be called with m_recordingGates = false and mps already resetZero().
  void replayCircuitWithNoise(const std::vector<GateRecord> &history,
                              cudaq::ExecutionContext *ctx) {
    for (const auto &g : history) {
      GateApplicationTask task(g.operationName, g.matrix, g.controls,
                               g.targets, g.parameters);
      applyGateInternal(task);
      // Manually invoke noise channel lookup (mirrors base class flushGateQueue).
      if (ctx && ctx->noiseModel && !ctx->noiseModel->empty()) {
        auto channels = ctx->noiseModel->get_channels(
            g.operationName, g.targets, g.controls, g.parameters);
        for (const auto &ch : channels)
          applyKrausChannel(g.targets, ch);
      }
    }
  }

  // ---- Pauli word construction for observe ---------------------------------
  struct PauliTermData {
    std::vector<std::string> words;
    std::vector<std::complex<double>> coeffs;
    std::vector<std::string> termIds;
  };

  PauliTermData buildPauliTermData(const cudaq::spin_op &op) const {
    PauliTermData pd;
    const int nq = mps->n;
    for (const auto &term : op) {
      std::string word = term.get_pauli_word(nq);
      // Gates route CUDA-Q qubit q to MPS site conv(q)=n-1-q, so reverse.
      std::reverse(word.begin(), word.end());
      pd.words.push_back(std::move(word));
      pd.coeffs.push_back(term.evaluate_coefficient());
      pd.termIds.push_back(term.get_term_id());
    }
    return pd;
  }

public:
  SimulatorMPSDist() = default;

  // Bring both base addQubitsToState overloads into scope (we only specialize
  // the (count, ptr) one) to avoid a partial-override error.
  using nvqir::CircuitSimulatorBase<double>::addQubitsToState;

  // ---- required CircuitSimulatorBase overrides ----
  void addQubitToState() override { addQubitsToState(1); }

  void addQubitsToState(std::size_t count, const void * = nullptr) override {
    if (mps) throw std::runtime_error(
        CUDAQ_CUSTOM_MPS_BACKEND_NAME
        ": adding qubits after gates is not supported");
    allocated += count;
  }

  void deallocateStateImpl() override {
    mps.reset(); allocated = 0;
    m_gateHistory.clear(); m_recordingGates = true;
  }

  void setToZeroState() override {
    if (mps) mps->resetZero();
    // Clear history: a new circuit is being set up from |0>.
    if (m_recordingGates) m_gateHistory.clear();
  }

  // Apply gate and record to history for later trajectory replay.
  void applyGate(const GateApplicationTask &task) override {
    ensureEngine();
    applyGateInternal(task);
    if (m_recordingGates)
      m_gateHistory.push_back({
          task.operationName,
          std::vector<std::complex<double>>(task.matrix.begin(), task.matrix.end()),
          task.controls, task.targets,
          std::vector<double>(task.parameters.begin(), task.parameters.end())});
  }

  // ---- Noise channel application ------------------------------------------
  // Called by the base class gate loop after each gate when a noise model is
  // active. Stochastically applies one Kraus operator from each matching
  // channel to the current MPS state.
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override {
    auto *ctx = cudaq::getExecutionContext();
    if (!ctx || !ctx->noiseModel || ctx->noiseModel->empty()) return;

    auto channels = ctx->noiseModel->get_channels(
        std::string(gateName), targets, controls, params);
    for (const auto &ch : channels)
      applyKrausChannel(targets, ch);
  }

  // ---- Advertise analytic observe (handled even with noise via trajectories)
  bool canHandleObserve() override { return true; }

  // ---- observe() -----------------------------------------------------------
  // 1. No noise, no shots  → exact analytic GPU expectation (fastest).
  // 2. No noise, shots > 0 → shot noise: sample bitstrings from |psi|^2,
  //    estimate each Z/I Pauli term by parity average (mimics standard CUDA-Q
  //    shot-based observe). X/Y terms fall back to analytic.
  // 3. Noise model set     → multi-trajectory loop (quantum Monte Carlo).
  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    ensureEngine();
    auto *ctx = cudaq::getExecutionContext();
    const bool hasNoise = ctx && ctx->noiseModel && !ctx->noiseModel->empty();
    // The C++ async/parallel observe API uses -1 for analytic execution. Since
    // ExecutionContext stores shots as size_t, that sentinel arrives here as
    // SIZE_MAX and must not be interpreted as an enormous shot request.
    const bool finiteShots =
        ctx && ctx->shots > 0 &&
        ctx->shots != std::numeric_limits<std::size_t>::max();
    const bool useShots = !hasNoise && finiteShots;

    auto pd = buildPauliTermData(op);

    static const bool useHost = []{
      const char *e = std::getenv("CUDAQ_MPS_EXPECT_HOST");
      return e && atoi(e) != 0;
    }();

    auto computeExpVals = [&]() -> std::vector<std::complex<double>> {
      if (useHost) {
        std::vector<std::complex<double>> ev(pd.words.size());
        for (size_t k = 0; k < pd.words.size(); ++k)
          ev[k] = mps->expectPauliHost(pd.words[k]);
        return ev;
      }
      return mps->expectPauliBatch(pd.words);
    };

    // Helper: build and return an observe_result from per-term expectation values.
    auto makeResult = [&](const std::vector<std::complex<double>> &ev)
        -> cudaq::observe_result {
      std::complex<double> total = 0.0;
      std::vector<cudaq::ExecutionResult> results;
      for (size_t k = 0; k < pd.words.size(); ++k) {
        std::complex<double> contrib = pd.coeffs[k] * ev[k];
        total += contrib;
        // Per-term raw data stores <P_k>, not c_k<P_k>. CUDA-Q uses the term
        // ID to reapply c_k when an async result is serialized/reconstructed.
        results.emplace_back(
            cudaq::ExecutionResult({}, pd.termIds[k], ev[k].real()));
      }
      cudaq::sample_result data(total.real(), results);
      return cudaq::observe_result(total.real(), op, data);
    };

    // --- Path 1: exact analytic (no noise, no shots) ---
    if (!hasNoise && !useShots)
      return makeResult(computeExpVals());

    // --- Path 2: shot noise (no noise model, shots > 0) ---
    // Sample `shots` bitstrings from |psi|^2. For Z/I-only Pauli terms,
    // estimate <P> = mean parity over shots — this is the same statistical
    // noise standard CUDA-Q produces with shots_count > 0. X/Y terms are
    // computed analytically (protein-folding Hamiltonians are Z-only).
    if (useShots) {
      const int nq = mps->n;
      int nshots = ctx->shots;
      auto bitstrings = mps->sampleBits(nshots, sampleSeed++);

      std::vector<std::complex<double>> ev(pd.words.size());
      for (size_t k = 0; k < pd.words.size(); ++k) {
        const auto &word = pd.words[k];  // MPS site order

        bool zOnly = true;
        for (char c : word) if (c != 'I' && c != 'Z') { zOnly = false; break; }

        if (zOnly) {
          // bs is CUDA-Q qubit order: bs[q] is qubit q.
          // word[s] is MPS site s = CUDA-Q qubit nq-1-s.
          double sum = 0.0;
          for (const auto &bs : bitstrings) {
            double parity = 1.0;
            for (int s = 0; s < nq; ++s)
              if (word[s] == 'Z')
                parity *= (bs[nq - 1 - s] == '0') ? 1.0 : -1.0;
            sum += parity;
          }
          ev[k] = sum / nshots;
        } else {
          // X or Y present — fall back to exact analytic for this term.
          ev[k] = mps->expectPauli(word);
        }
      }
      return makeResult(ev);
    }

    // --- Path 3: noisy trajectory loop ---
    std::size_t numTraj = DEFAULT_OBSERVE_TRAJECTORIES;
    if (ctx->numberTrajectories.has_value())
      numTraj = ctx->numberTrajectories.value();
    else if (finiteShots)
      numTraj = ctx->shots;

    auto savedHistory = m_gateHistory;
    m_recordingGates = false;

    std::vector<std::complex<double>> termExpVals(pd.words.size(), 0.0);
    for (std::size_t traj = 0; traj < numTraj; ++traj) {
      mps->resetZero();
      replayCircuitWithNoise(savedHistory, ctx);
      auto ev = computeExpVals();
      for (size_t k = 0; k < pd.words.size(); ++k)
        termExpVals[k] += ev[k] / static_cast<double>(numTraj);
    }

    m_recordingGates = true;
    mps->resetZero();
    for (const auto &g : savedHistory) {
      GateApplicationTask task(g.operationName, g.matrix, g.controls,
                               g.targets, g.parameters);
      applyGateInternal(task);
    }

    return makeResult(termExpVals);
  }

  unsigned sampleSeed = 1234567u;

  bool measureQubit(const std::size_t) override {
    throw std::runtime_error(CUDAQ_CUSTOM_MPS_BACKEND_NAME
                             ": mid-circuit measureQubit not "
                             "implemented; use observe() or sample().");
  }
  void resetQubit(const std::size_t) override {
    throw std::runtime_error(CUDAQ_CUSTOM_MPS_BACKEND_NAME
                             ": resetQubit not yet implemented.");
  }

  // ---- sample() ------------------------------------------------------------
  // Without noise: exact MPS perfect sampling from |psi|^2.
  // With noise: one trajectory per shot — each shot re-runs the circuit from
  // |0> with an independent Kraus realization before sampling one bitstring.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    ensureEngine();
    auto *ctx = cudaq::getExecutionContext();
    const bool hasNoise = ctx && ctx->noiseModel && !ctx->noiseModel->empty();
    int nshots = shots > 0 ? shots : 1000;

    if (!hasNoise) {
      // Fast exact-sampling path (unchanged from before).
      std::vector<std::string> bs = mps->sampleBits(nshots, sampleSeed++);
      cudaq::CountsDictionary counts;
      std::string key; key.reserve(qubits.size());
      for (const auto &full : bs) {
        key.clear();
        for (auto q : qubits) key.push_back(full[(int)q]);
        counts[key]++;
      }
      return cudaq::ExecutionResult(counts);
    }

    // --- Noisy path: one trajectory per shot ---
    auto savedHistory = m_gateHistory;
    m_recordingGates = false;

    cudaq::CountsDictionary counts;
    std::string key; key.reserve(qubits.size());
    for (int sh = 0; sh < nshots; ++sh) {
      mps->resetZero();
      replayCircuitWithNoise(savedHistory, ctx);
      auto bs = mps->sampleBits(1, sampleSeed++);   // one bitstring per trajectory
      key.clear();
      for (auto q : qubits) key.push_back(bs[0][(int)q]);
      if (includeSequentialData)
        counts[key]++;
      else
        counts[key]++;
    }

    m_recordingGates = true;
    // Restore deterministic final state (noise-free).
    mps->resetZero();
    for (const auto &g : savedHistory) {
      GateApplicationTask task(g.operationName, g.matrix, g.controls,
                               g.targets, g.parameters);
      applyGateInternal(task);
    }

    return cudaq::ExecutionResult(counts);
  }

  // Lift the base class's 63-qubit limit. The base computes a "state dimension"
  // of 2^n and throws at n>=64 (overflow). An MPS never materializes 2^n, and
  // this value is only used for FLOP/memory accounting, so return a
  // non-overflowing placeholder for large n -- enabling proteins >63 qubits.
  std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits < 64 ? (1ULL << numQubits) : numQubits;
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }
  std::string name() const override { return CUDAQ_CUSTOM_MPS_BACKEND_NAME; }

  CircuitSimulator *clone() override {
    thread_local static auto sim = std::make_unique<SimulatorMPSDist>();
    return sim.get();
  }
};

} // namespace nvqir

#ifdef CUDAQ_MPS_MQPU_BACKEND
/// Register the singleton-device custom MPS under the MQPU backend name.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mqpu)
#else
/// Register the original sharded custom MPS backend.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mgpu)
#endif

#undef CUDAQ_CUSTOM_MPS_BACKEND_NAME
#ifdef CUDAQ_MPS_MQPU_BACKEND
#undef SimulatorMPSDist
#undef ShardedMPS
#endif
MGPU_SIMULATOR_MPS_DIST_CU_EOF

# Keep the wrapper synchronized too: its finite-shot observe path evaluates
# diagonal Hamiltonian parities on the GPU and returns only reduced results.
install -m 0644 "$PACKAGED_BACKEND/simulator_mps_dist.cu" \
  "$BACKEND/simulator_mps_dist.cu"

cat > "$BACKEND/tensornet-mps-mgpu.yml" <<'MGPU_TENSORNET-MPS-MGPU_YML_EOF'
# ============================================================================ #
# Target config for the multi-GPU memory-pooled MPS backend.                   #
# Drop into runtime/nvqir/cutensornet/ alongside tensornet-mps.yml.            #
# ============================================================================ #

name: tensornet-mps-mgpu
description: "Multi-GPU memory-pooled matrix product state simulator (sharded chain + distributed truncated SVD)"
gpu-requirements: true

target-arguments:
  - key: option
    required: false
    type: option-flags
    help-string: "Comma-separated options. Supported: 'fp64' (default)."

configuration-matrix:
  - name: multi-gpu-fp64
    option-flags: [fp64]
    default: true
    config:
      nvqir-simulation-backend: tensornet-mps-mgpu
      preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
MGPU_TENSORNET-MPS-MGPU_YML_EOF

cat > "$BACKEND/tensornet-mps-mqpu.yml" <<'MQPU_TENSORNET-MPS_YML_EOF'
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

name: tensornet-mps-mqpu
description: "Block-first MQPU execution with strong prepared-state MPS Hamiltonian batching inside disjoint GPU groups."
gpu-requirements: true

config:
  # CUDA-Q schedules independent blocks onto logical QPUs first. Each selected
  # QPU then batches its full block Hamiltonian inside a disjoint GPU group.
  nvqir-simulation-backend: tensornet-mps-mqpu
  preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
  platform-library: mqpu
MQPU_TENSORNET-MPS_YML_EOF

cat > "$BACKEND/tensornet-mps-mqpu-shots.yml" <<'MQPU_SHOTS_YML_EOF'
name: tensornet-mps-mqpu-shots
description: "Block-first MQPU execution with fixed-budget shared-distribution shot sharding inside disjoint GPU groups."
gpu-requirements: true

config:
  nvqir-simulation-backend: tensornet-mps-mqpu-shots
  preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
  platform-library: mqpu
MQPU_SHOTS_YML_EOF

cat > "$BACKEND/tensornet-mps-mqpu-hamiltonian.yml" <<'MQPU_HAMILTONIAN_YML_EOF'
name: tensornet-mps-mqpu-hamiltonian
description: "Block-first MQPU execution with finite-shot Hamiltonian-term batching inside disjoint GPU groups."
gpu-requirements: true

config:
  nvqir-simulation-backend: tensornet-mps-mqpu-hamiltonian
  preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
  platform-library: mqpu
MQPU_HAMILTONIAN_YML_EOF

cat > "$BACKEND/tensornet-mps-mqpu-replay.yml" <<'MQPU_REPLAY_YML_EOF'
name: tensornet-mps-mqpu-replay
description: "Persistent per-GPU streamed MPS ansatz replay with Hamiltonian-term batching."
gpu-requirements: true

config:
  nvqir-simulation-backend: tensornet-mps-mqpu-replay
  preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
MQPU_REPLAY_YML_EOF

cat > "$BACKEND/tensornet-mps-mqpu-strong.yml" <<'MQPU_STRONG_YML_EOF'
name: tensornet-mps-mqpu-strong
description: "Persistent MPS replicas with pipelined independent Hamiltonian-term sampling."
gpu-requirements: true

config:
  nvqir-simulation-backend: tensornet-mps-mqpu-strong
  preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
MQPU_STRONG_YML_EOF


# ---- 3. patch CMakeLists to build the backend and install the MQPU target ----
python3 - "$BACKEND/CMakeLists.txt" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
marker = "nvqir_create_cutn_plugin(tensornet-mps TRUE ${BASE_TENSOR_BACKEND_SRS} simulator_mps_fp64_register.cpp)"
if marker not in src:
    print("   ERROR: expected CMake marker not found", file=sys.stderr); sys.exit(1)
changed = False
if "nvqir_create_cutn_plugin(tensornet-mps-mgpu TRUE simulator_mps_dist.cu)" not in src:
    patch = marker + "\n" + "\n".join([
      "  # --- custom MPS backends (added by setup_local_cudaq.sh) ---",
      "  if (NOT DEFINED NCCL_ROOT AND DEFINED ENV{NCCL_ROOT})",
      "    set(NCCL_ROOT $ENV{NCCL_ROOT})",
      "  endif()",
      "  nvqir_create_cutn_plugin(tensornet-mps-mgpu TRUE simulator_mps_dist.cu)",
      "  target_include_directories(nvqir-tensornet-mps-mgpu PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
      "  target_link_libraries(nvqir-tensornet-mps-mgpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
      "  nvqir_create_cutn_plugin(tensornet-mps-mqpu FALSE simulator_mps_dist.cu)",
      "  target_compile_definitions(nvqir-tensornet-mps-mqpu PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_STRONG_PIPELINE=1)",
      "  target_include_directories(nvqir-tensornet-mps-mqpu PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
      "  target_link_libraries(nvqir-tensornet-mps-mqpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
      "  nvqir_create_cutn_plugin(tensornet-mps-mqpu-replay FALSE simulator_mps_dist.cu)",
      "  target_compile_definitions(nvqir-tensornet-mps-mqpu-replay PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_REPLAY_BACKEND=1)",
      "  target_include_directories(nvqir-tensornet-mps-mqpu-replay PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
      "  target_link_libraries(nvqir-tensornet-mps-mqpu-replay PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
      "  nvqir_create_cutn_plugin(tensornet-mps-mqpu-strong FALSE simulator_mps_dist.cu)",
      "  target_compile_definitions(nvqir-tensornet-mps-mqpu-strong PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_STRONG_BACKEND=1)",
      "  target_include_directories(nvqir-tensornet-mps-mqpu-strong PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
      "  target_link_libraries(nvqir-tensornet-mps-mqpu-strong PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
    ])
    src = src.replace(marker, patch, 1)
    changed = True
if "add_target_config(tensornet-mps-mqpu)" not in src:
    target_patch = marker + "\n" + "\n".join([
      "  # Persistent custom MPS Hamiltonian-batching targets.",
      "  add_target_config(tensornet-mps-mqpu)",
      "  add_target_config(tensornet-mps-mqpu-replay)",
      "  add_target_config(tensornet-mps-mqpu-strong)",
    ])
    src = src.replace(marker, target_patch, 1)
    changed = True
mqpu_plugin = "  nvqir_create_cutn_plugin(tensornet-mps-mqpu FALSE simulator_mps_dist.cu)"
if mqpu_plugin not in src:
    mgpu_link = "  target_link_libraries(nvqir-tensornet-mps-mgpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)"
    if mgpu_link not in src:
        print("   ERROR: expected custom MPS link marker not found", file=sys.stderr); sys.exit(1)
    src = src.replace(mgpu_link, mgpu_link + "\n" + "\n".join([
      mqpu_plugin,
      "  target_compile_definitions(nvqir-tensornet-mps-mqpu PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_STRONG_PIPELINE=1)",
      "  target_include_directories(nvqir-tensornet-mps-mqpu PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
      "  target_link_libraries(nvqir-tensornet-mps-mqpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
    ]), 1)
    changed = True
replay_plugin = "  nvqir_create_cutn_plugin(tensornet-mps-mqpu-replay FALSE simulator_mps_dist.cu)"
strong_plugin = "  nvqir_create_cutn_plugin(tensornet-mps-mqpu-strong FALSE simulator_mps_dist.cu)"
if replay_plugin not in src or strong_plugin not in src:
    mqpu_link = "  target_link_libraries(nvqir-tensornet-mps-mqpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)"
    if mqpu_link not in src:
        print("   ERROR: expected MQPU link marker not found", file=sys.stderr); sys.exit(1)
    additions = []
    if replay_plugin not in src:
        additions += [
          replay_plugin,
          "  target_compile_definitions(nvqir-tensornet-mps-mqpu-replay PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_REPLAY_BACKEND=1)",
          "  target_include_directories(nvqir-tensornet-mps-mqpu-replay PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
          "  target_link_libraries(nvqir-tensornet-mps-mqpu-replay PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
        ]
    if strong_plugin not in src:
        additions += [
          strong_plugin,
          "  target_compile_definitions(nvqir-tensornet-mps-mqpu-strong PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_STRONG_BACKEND=1)",
          "  target_include_directories(nvqir-tensornet-mps-mqpu-strong PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
          "  target_link_libraries(nvqir-tensornet-mps-mqpu-strong PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
        ]
    src = src.replace(mqpu_link, mqpu_link + "\n" + "\n".join(additions), 1)
    changed = True
variant_plugins = {
    "tensornet-mps-mqpu-shots": "CUDAQ_MPS_MQPU_SHOTS_BACKEND=1",
    "tensornet-mps-mqpu-hamiltonian":
        "CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND=1",
}
mqpu_link = "  target_link_libraries(nvqir-tensornet-mps-mqpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)"
for target, variant_definition in variant_plugins.items():
    plugin = f"  nvqir_create_cutn_plugin({target} FALSE simulator_mps_dist.cu)"
    if plugin not in src:
        if mqpu_link not in src:
            print("   ERROR: expected MQPU link marker not found", file=sys.stderr)
            sys.exit(1)
        additions = "\n".join([
            plugin,
            f"  target_compile_definitions(nvqir-{target} PRIVATE CUDAQ_MPS_MQPU_BACKEND=1 {variant_definition} CUDAQ_MPS_MQPU_STRONG_PIPELINE=1)",
            f"  target_include_directories(nvqir-{target} PRIVATE ${{CMAKE_SOURCE_DIR}}/runtime ${{NCCL_ROOT}}/include)",
            f"  target_link_libraries(nvqir-{target} PRIVATE CUDA::cublas CUDA::cusolver ${{NCCL_ROOT}}/lib/libnccl.so)",
        ])
        src = src.replace(mqpu_link, mqpu_link + "\n" + additions, 1)
        changed = True
for target in (
    "tensornet-mps-mqpu-shots",
    "tensornet-mps-mqpu-hamiltonian",
    "tensornet-mps-mqpu-replay",
    "tensornet-mps-mqpu-strong",
):
    config = f"  add_target_config({target})"
    if config not in src:
        base_config = "  add_target_config(tensornet-mps-mqpu)"
        if base_config not in src:
            print("   ERROR: expected MQPU target-config marker not found", file=sys.stderr); sys.exit(1)
        src = src.replace(base_config, base_config + "\n" + config, 1)
        changed = True
legacy_mqpu_definition = (
    "  target_compile_definitions(nvqir-tensornet-mps-mqpu PRIVATE "
    "CUDAQ_MPS_MQPU_BACKEND=1)"
)
strong_mqpu_definition = (
    "  target_compile_definitions(nvqir-tensornet-mps-mqpu PRIVATE "
    "CUDAQ_MPS_MQPU_BACKEND=1 CUDAQ_MPS_MQPU_STRONG_PIPELINE=1)"
)
if legacy_mqpu_definition in src:
    src = src.replace(legacy_mqpu_definition, strong_mqpu_definition, 1)
    changed = True
if src.count(strong_mqpu_definition) != 1:
    print("   ERROR: standard MQPU target is missing the strong-pipeline definition", file=sys.stderr)
    sys.exit(1)
open(path, "w").write(src)
print("   patched CMakeLists.txt" if changed else "   CMakeLists already patched")
PYEOF

# ---- 3a. preserve MQPU-selected device in the stock cuTensorNet simulator ---
# CUDA-Q 0.14.2 resets every non-MPI simulator instance to CUDA device 0 in its
# constructor. Under the MQPU platform this makes the QPU-1 thread create its
# cuTensorNet handle and scratch memory on QPU 0, then execute on QPU 1. Preserve
# the current CUDA device selected by GPUEmulatedQPU instead.
python3 - "$BACKEND/simulator_cutensornet.inc" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = """  // we assume that the processes are mapped to nodes in contiguous chunks
  const int deviceId =
      cudaq::mpi::is_initialized() ? cudaq::mpi::rank() % numDevices : 0;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));"""
new = """  // MPI ranks are mapped to devices in contiguous chunks. Outside MPI, retain
  // the device selected by the platform. This is required by the MQPU
  // platform, whose worker thread calls cudaSetDevice(qpu_id) before lazily
  // constructing its thread-local simulator.
  int deviceId{0};
  const bool mqpuOwnsDevice = std::getenv("CUDAQ_MQPU_NGPUS") != nullptr;
  if (cudaq::mpi::is_initialized() && !mqpuOwnsDevice)
    deviceId = cudaq::mpi::rank() % numDevices;
  else
    HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));"""
if new in src:
    print("   simulator_cutensornet.inc already MQPU-device-aware")
elif old in src:
    open(path, "w").write(src.replace(old, new, 1))
    print("   patched simulator_cutensornet.inc (preserve MQPU device)")
else:
    print("   ERROR: expected cuTensorNet device-selection code not found",
          file=sys.stderr)
    sys.exit(1)
PYEOF

# ---- 3aa. handle identity-only MQPU Hamiltonian partitions safely -----------
# The MPS tensor-network operator constructor assumes at least one non-identity
# degree. Hamiltonian batching can legitimately assign a constant identity term
# by itself to a worker, so return that known expectation without contracting.
python3 - "$BACKEND/simulator_cutensornet.inc" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = """  assert(cudaq::spin_op::canonicalize(ham) == ham);
  LOG_API_TIME();
  prepareQubitTensorState();"""
replacement = """  assert(cudaq::spin_op::canonicalize(ham) == ham);
  LOG_API_TIME();
  // A Hamiltonian-batching worker can receive an identity-only partition.
  // TensorNetworkSpinOp derives its network extent from non-identity degrees,
  // so constructing it for that partition dereferences an empty degree list.
  // The expectation of an identity-only operator is just its coefficient sum
  // and does not require a tensor-network contraction.
  bool identityOnly = true;
  std::complex<double> identityExpectation = 0.0;
  std::vector<cudaq::ExecutionResult> identityResults;
  identityResults.reserve(ham.num_terms());
  for (const auto &term : ham) {
    if (!term.is_identity()) {
      identityOnly = false;
      break;
    }
    const auto coefficient = term.evaluate_coefficient();
    identityExpectation += coefficient;
    identityResults.emplace_back(cudaq::ExecutionResult(
        {}, term.get_term_id(), coefficient.real()));
  }
  if (identityOnly) {
    cudaq::sample_result perTermData(identityExpectation.real(),
                                     identityResults);
    return cudaq::observe_result(identityExpectation.real(), ham, perTermData);
  }

  prepareQubitTensorState();"""
if replacement in src:
    print("   simulator_cutensornet.inc already handles identity-only observe")
elif anchor in src:
    open(path, "w").write(src.replace(anchor, replacement, 1))
    print("   patched simulator_cutensornet.inc (identity-only observe)")
else:
    print("   ERROR: expected cuTensorNet observe entry not found",
          file=sys.stderr)
    sys.exit(1)
PYEOF

# ---- 3ab. select the MQPU device before lazy simulator construction ---------
python3 - "$SRC/runtime/cudaq/platform/mqpu/custatevec/GPUEmulatedQPU.cpp" <<'PYEOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
anchor = """    CUDAQ_INFO("MultiQPUPlatform::configureExecutionContext QPU {}", qpu_id);
    if (noiseModel)"""
replacement = """    CUDAQ_INFO("MultiQPUPlatform::configureExecutionContext QPU {}", qpu_id);
    // The execution manager constructs its thread-local simulator here, before
    // launchKernel. Select the QPU device before that lazy construction.
    cudaSetDevice(qpu_id);
    if (noiseModel)"""
already_patched = re.search(
    r"configureExecutionContext\(.*?cudaSetDevice\(qpu_id\);.*?if \(noiseModel\)",
    src,
    flags=re.DOTALL,
)
if already_patched:
    print("   GPUEmulatedQPU.cpp already selects device during configuration")
elif anchor in src:
    open(path, "w").write(src.replace(anchor, replacement, 1))
    print("   patched GPUEmulatedQPU.cpp (early MQPU device selection)")
else:
    print("   ERROR: expected MQPU configuration code not found", file=sys.stderr)
    sys.exit(1)
PYEOF

# ---- 3ac. normalize analytic shots for Python parallel observe ---------------
python3 - "$SRC/python/cudaq/runtime/observe.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = """    if execution != None:
        return observe_parallel(kernel,
                                spin_operator,
                                execution,
                                *args,
                                shots_count=shots_count,
                                noise_model=noise_model)"""
new = """    if execution != None:
        # Python uses 0 for analytic observe; C++ parallel observe uses -1.
        parallel_shots = shots_count if shots_count > 0 else -1
        return observe_parallel(kernel,
                                spin_operator,
                                execution,
                                *args,
                                shots_count=parallel_shots,
                                noise_model=noise_model)"""
if "parallel_shots = shots_count if shots_count > 0 else -1" in src:
    print("   python observe.py already normalizes parallel analytic shots")
elif old in src:
    open(path, "w").write(src.replace(old, new, 1))
    print("   patched observe.py (parallel analytic shots: 0 -> -1)")
else:
    print("   ERROR: expected Python parallel-observe code not found",
          file=sys.stderr)
    sys.exit(1)
PYEOF

# ---- 3ac. aggregate MQPU ObserveResult values, not raw-data sentinels --------
python3 - "$SRC/runtime/cudaq/algorithms/observe.h" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = """    result += incomingData.expectation();
    data += incomingData;"""
new = """    // Direct-observe simulators such as cuTensorNet MPS need the
    // precomputed ObserveResult value; raw data may have no global register.
    result += res.expectation();
    data += incomingData;"""
if "result += res.expectation();" in src:
    print("   observe.h already aggregates ObserveResult expectations")
elif old in src:
    open(path, "w").write(src.replace(old, new, 1))
    print("   patched observe.h (correct MQPU direct-observe aggregation)")
else:
    print("   ERROR: expected MQPU aggregation code not found", file=sys.stderr)
    sys.exit(1)
PYEOF

# ---- 3b. fix const_iterator symbol localization (so `import cudaq` works) ----
# conda's default CXXFLAGS inject -fvisibility-inlines-hidden. That makes the
# inline members of sum_op<matrix_handler>::const_iterator HIDDEN in TUs that
# implicitly instantiate them (e.g. super_op.cpp); the linker's weak-merge then
# localizes the symbol in libcudaq-operator.so, so the python bindings
# (_quakeDialects) fail at import with an undefined-symbol error. Override the
# flag for the operator library and mark the iterator types default-visible.
python3 - "$SRC/runtime/cudaq/operators/CMakeLists.txt" "$SRC/runtime/cudaq/operators.h" <<'PYEOF'
import sys
cml, hdr = sys.argv[1], sys.argv[2]
s = open(cml).read()
if "-fno-visibility-inlines-hidden" not in s:
    anchor = '-Wno-ctad-maybe-unsupported")'
    if anchor in s:
        s = s.replace(anchor, '-Wno-ctad-maybe-unsupported -fno-visibility-inlines-hidden")', 1)
        open(cml, "w").write(s); print("   patched operators/CMakeLists.txt (inlines-hidden override)")
    else:
        print("   ERROR: CXX_FLAGS anchor not found in operators/CMakeLists.txt", file=sys.stderr); sys.exit(1)
else:
    print("   operators/CMakeLists.txt already patched")
h = open(hdr).read()
if 'struct const_iterator {' in h and '__attribute__((visibility("default"))) const_iterator' not in h:
    h = h.replace('struct const_iterator {',
                  'struct __attribute__((visibility("default"))) const_iterator {')
    open(hdr, "w").write(h); print("   patched operators.h (const_iterator default visibility)")
else:
    print("   operators.h already patched / pattern absent")
PYEOF

# ---- 4. install prefixes + dependency locations -----------------------------
export CUDAQ_INSTALL_PREFIX="$LOCAL/cudaq"   # our build always installs here
export CUDAQ_BUILD_TESTS=FALSE
mkdir -p "$CUDAQ_INSTALL_PREFIX"

if [ "$PREBUILT" = 1 ]; then
  # Container: reuse the image's LLVM + cuQuantum + cuTensor (don't rebuild).
  export LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-$LLVM_PREBUILT}"
  : "${CUQUANTUM_INSTALL_PREFIX:=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}}"
  : "${CUTENSOR_INSTALL_PREFIX:=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}}"
  if [ ! -d "$CUQUANTUM_INSTALL_PREFIX" ]; then
    [ -d "$SITE/cuquantum" ] \
      && CUQUANTUM_INSTALL_PREFIX="$SITE/cuquantum" \
      || CUQUANTUM_INSTALL_PREFIX="$LOCAL/cuquantum"
  fi
  if [ ! -d "$CUTENSOR_INSTALL_PREFIX" ]; then
    [ -d "$SITE/cutensor" ] \
      && CUTENSOR_INSTALL_PREFIX="$SITE/cutensor" \
      || CUTENSOR_INSTALL_PREFIX="$LOCAL/cutensor"
  fi
  export CUQUANTUM_INSTALL_PREFIX CUTENSOR_INSTALL_PREFIX
else
  # Bare metal: build everything into $LOCAL.
  export LLVM_INSTALL_PREFIX="$LOCAL/llvm"
  export PYBIND11_INSTALL_PREFIX="$LOCAL/pybind11" BLAS_INSTALL_PREFIX="$LOCAL/blas"
  export ZLIB_INSTALL_PREFIX="$LOCAL/zlib" OPENSSL_INSTALL_PREFIX="$LOCAL/ssl"
  export CURL_INSTALL_PREFIX="$LOCAL/curl" AWS_INSTALL_PREFIX="$LOCAL/aws" QRMI_INSTALL_PREFIX="$LOCAL/qrmi"
  if [ -z "${CUQUANTUM_INSTALL_PREFIX:-}" ] ||
     [ ! -d "$CUQUANTUM_INSTALL_PREFIX" ]; then
    [ -d "$SITE/cuquantum" ] \
      && CUQUANTUM_INSTALL_PREFIX="$SITE/cuquantum" \
      || CUQUANTUM_INSTALL_PREFIX="$LOCAL/cuquantum"
  fi
  if [ -z "${CUTENSOR_INSTALL_PREFIX:-}" ] ||
     [ ! -d "$CUTENSOR_INSTALL_PREFIX" ]; then
    [ -d "$SITE/cutensor" ] \
      && CUTENSOR_INSTALL_PREFIX="$SITE/cutensor" \
      || CUTENSOR_INSTALL_PREFIX="$LOCAL/cutensor"
  fi
  export CUQUANTUM_INSTALL_PREFIX CUTENSOR_INSTALL_PREFIX
  mkdir -p "$LLVM_INSTALL_PREFIX" "$PYBIND11_INSTALL_PREFIX" "$BLAS_INSTALL_PREFIX" \
           "$ZLIB_INSTALL_PREFIX" "$OPENSSL_INSTALL_PREFIX" "$CURL_INSTALL_PREFIX" \
           "$AWS_INSTALL_PREFIX" "$QRMI_INSTALL_PREFIX"
fi

# Installed CMake/pkg-config metadata and helper scripts can also retain the
# old server's absolute paths. Reuse portable libraries, but force a clean
# local reinstall for a component whose metadata points at another checkout or
# another cudaq-env. This check matters when the whole project was copied.
installed_component_is_relocated() { # remaining arguments are metadata files
  "$PY" - "$LOCAL" "$SRC" "${LLVM_SOURCE:-}" "${CONDA_PREFIX:-}" "$@" <<'PYEOF'
import os
import sys

allowed = [os.path.realpath(path) for path in sys.argv[1:5] if path]
markers = ("/cudaq_local", "/cuda-quantum-src", "/.llvm-project", "/envs/cudaq-env")
for filename in sys.argv[5:]:
    if not os.path.isfile(filename):
        continue
    try:
        with open(filename, "rb") as stream:
            content = stream.read()
        if b"\0" in content:
            continue
        for line in content.decode("utf-8", errors="ignore").splitlines():
            if any(marker in line for marker in markers) and not any(
                path in line for path in allowed
            ):
                print(f"   stale metadata: {filename}: {line.strip()}")
                raise SystemExit(0)
    except OSError:
        continue
raise SystemExit(1)
PYEOF
}

if [ "$PREBUILT" = 0 ]; then
  if installed_component_is_relocated \
       "$ZLIB_INSTALL_PREFIX/lib/pkgconfig/zlib.pc" \
       "$ZLIB_INSTALL_PREFIX/lib/pkgconfig/minizip.pc" \
       "$ZLIB_INSTALL_PREFIX/lib/libminizip.la"; then
    echo ">> relocated zlib/minizip install detected; rebuilding locally ..."
    rm -rf "$ZLIB_INSTALL_PREFIX"
  fi
  if installed_component_is_relocated \
       "$OPENSSL_INSTALL_PREFIX/lib/pkgconfig/libssl.pc" \
       "$OPENSSL_INSTALL_PREFIX/lib64/pkgconfig/libssl.pc" \
       "$OPENSSL_INSTALL_PREFIX/bin/c_rehash"; then
    echo ">> relocated OpenSSL install detected; rebuilding locally ..."
    rm -rf "$OPENSSL_INSTALL_PREFIX"
  fi
  if installed_component_is_relocated \
       "$CURL_INSTALL_PREFIX/bin/curl-config" \
       "$CURL_INSTALL_PREFIX/lib/pkgconfig/libcurl.pc"; then
    echo ">> relocated Curl install detected; rebuilding locally ..."
    rm -rf "$CURL_INSTALL_PREFIX"
  fi
  if installed_component_is_relocated \
       "$AWS_INSTALL_PREFIX/lib/cmake/AWSSDK/platformDeps.cmake"; then
    echo ">> relocated AWS SDK install detected; rebuilding locally ..."
    rm -rf "$AWS_INSTALL_PREFIX"
  fi
  if installed_component_is_relocated \
       "$LLVM_INSTALL_PREFIX/lib/cmake/llvm/LLVMConfig.cmake" \
       "$LLVM_INSTALL_PREFIX/lib/cmake/mlir/MLIRConfig.cmake" \
       "$LLVM_INSTALL_PREFIX/bin/llvm-lit"; then
    echo ">> relocated LLVM install detected; rebuilding locally ..."
    rm -rf "$LLVM_INSTALL_PREFIX" "$LLVM_SOURCE/$LLVM_BUILD_FOLDER"
  fi
fi
if installed_component_is_relocated \
     "$CUDAQ_INSTALL_PREFIX/build_config.xml" \
     "$CUDAQ_INSTALL_PREFIX/bin/nvq++" \
     "$CUDAQ_INSTALL_PREFIX/lib/pkgconfig/fmt.pc" \
     "$CUDAQ_INSTALL_PREFIX/include/nvqpp/nvqpp_config.h"; then
  echo ">> relocated CUDA-Q install detected; reinstalling locally ..."
  rm -rf "$CUDAQ_INSTALL_PREFIX"
  mkdir -p "$CUDAQ_INSTALL_PREFIX"
fi

# NCCL for our backend (include/nccl.h + lib/libnccl.so). Find it; else pip-install.
export NCCL_ROOT="$LOCAL/nccl"; mkdir -p "$NCCL_ROOT/include" "$NCCL_ROOT/lib"
_link_nccl() {  # $1 = a prefix containing include/nccl.h + lib/libnccl.so*
  [ -e "$1/include/nccl.h" ] || return 1
  ln -sf "$1/include/nccl.h" "$NCCL_ROOT/include/nccl.h"
  for so in "$1"/lib/libnccl.so*; do [ -e "$so" ] && ln -sf "$so" "$NCCL_ROOT/lib/$(basename "$so")"; done
  [ -e "$NCCL_ROOT/lib/libnccl.so" ] || { v="$(ls "$NCCL_ROOT"/lib/libnccl.so.* 2>/dev/null|head -1)"; [ -n "$v" ] && ln -sf "$v" "$NCCL_ROOT/lib/libnccl.so"; }
  [ -e "$NCCL_ROOT/lib/libnccl.so" ]
}
_link_nccl "$SITE/nvidia/nccl" || _link_nccl /usr/local/cuda || _link_nccl /usr || {
  echo ">> NCCL not found; installing a private copy under cudaq_local ..."
  "$PY" -m pip install -q --no-cache-dir --target "$LOCAL/python-deps" \
    nvidia-nccl-cu12 2>/dev/null || true
  _link_nccl "$LOCAL/python-deps/nvidia/nccl" ||
    echo ">> WARNING: NCCL still missing; place it under $NCCL_ROOT."
}
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-}:$NCCL_ROOT:${PYBIND11_INSTALL_PREFIX:-}:${ZLIB_INSTALL_PREFIX:-}:${OPENSSL_INSTALL_PREFIX:-}:${CURL_INSTALL_PREFIX:-}:${BLAS_INSTALL_PREFIX:-}:${CMAKE_PREFIX_PATH:-}"

# ---- 5. build CUDA-Q + our backend ------------------------------------------
cd "$SRC"
# A copied or moved project may contain a CMake cache whose absolute source,
# install, and NCCL paths still point at the old checkout.  Reusing it silently
# installs or links against that other tree.  The build directory is generated
# state owned by this setup script, so discard it before configuring locally.
if [ -f "$SRC/build/CMakeCache.txt" ] &&
   { ! grep -Fqx "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC" "$SRC/build/CMakeCache.txt" ||
     ! grep -Fqx "CMAKE_INSTALL_PREFIX:PATH=$CUDAQ_INSTALL_PREFIX" "$SRC/build/CMakeCache.txt" ||
     ! grep -Fqx "NCCL_ROOT:UNINITIALIZED=$NCCL_ROOT" "$SRC/build/CMakeCache.txt"; }; then
  echo ">> stale/relocated CUDA-Q CMake cache detected; reconfiguring locally ..."
  rm -rf "$SRC/build"
fi
if [ "$PREBUILT" = 1 ]; then
  echo ">> PREBUILT mode: skipping install_prerequisites (using container LLVM/cuQuantum)."
else
  LLVM_BUILD_DIR="$LLVM_SOURCE/$LLVM_BUILD_FOLDER"
  LLVM_CACHE="$LLVM_BUILD_DIR/CMakeCache.txt"
  LOCAL_NINJA="$(command -v ninja)"
  # A project copied from another server can contain its former LLVM build.
  # Detect every important absolute path before allowing CMake to reuse it.
  if [ -f "$LLVM_CACHE" ] &&
     { ! grep -Fqx "CMAKE_HOME_DIRECTORY:INTERNAL=$LLVM_SOURCE/llvm" "$LLVM_CACHE" ||
       ! grep -Fqx "CMAKE_INSTALL_PREFIX:PATH=$LLVM_INSTALL_PREFIX" "$LLVM_CACHE" ||
       ! grep -Eq "^CMAKE_MAKE_PROGRAM:(FILEPATH|INTERNAL)=${LOCAL_NINJA//\//\\/}$" "$LLVM_CACHE"; }; then
    echo ">> stale/relocated LLVM CMake cache detected; reconfiguring locally ..."
    # LLVM_BUILD_DIR is fixed above to cudaq_local/llvm-project-src/build;
    # source and installed artifacts are intentionally preserved.
    rm -rf "$LLVM_BUILD_DIR"
  fi
  echo ">> building prerequisites (LLVM etc.) -> $LOCAL  (long; ~1-3h) ..."
  # -m builds only dependencies whose install prefixes were explicitly set
  # above. CMake and Ninja were bootstrapped before -m because upstream skips
  # its toolchain section in this mode. The patched helper fails on other
  # missing host tools instead of invoking a system package manager.
  bash scripts/install_prerequisites.sh -m
fi
echo ">> building CUDA-Q + persistent MPS Hamiltonian-batching targets -> $CUDAQ_INSTALL_PREFIX ..."
bash scripts/build_cudaq.sh

# Record the installed acceleration feature and exact library hashes.
"$PY" - "$CUDAQ_INSTALL_PREFIX" "$PACKAGED_BACKEND" <<'PYMANIFEST'
import hashlib, json, sys
from pathlib import Path
root, source=map(Path,sys.argv[1:])
files=('analytic_batch.cuh','dist_svd.cuh','sharded_mps.cuh','simulator_mps_dist.cu')
libraries=[root/'lib'/f'libnvqir-tensornet-mps-{suffix}.so'
           for suffix in ('mgpu','mqpu','mqpu-shots','mqpu-hamiltonian','mqpu-replay','mqpu-strong')]
missing=[str(p) for p in libraries if not p.is_file() or p.stat().st_size==0]
if missing:
    raise SystemExit('ERROR: incomplete backend install: '+', '.join(missing))
manifest={'feature':'analytic-pauli-prefix-v1',
          'features':['analytic-pauli-prefix-v1','hbatch-worker-lifecycle-v1','shared-diagonal-shots-v1','stable-direct-svd-v1','zero-cutoff-bond-guard-v1'],
          'reverse_gradients':'application CUDA PyTorch + opt_einsum; backend does not calculate gradients',
          'sources':{p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in files},
          'libraries':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}}
(root/'analytic_backend_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PYMANIFEST

# Generate a sourceable environment for standalone and campaign use.
"$PY" - "$PROJECT_DIR" "$PY" <<'PYACTIVATE'
import shlex, sys
from pathlib import Path
root=Path(sys.argv[1])
text = """#!/usr/bin/env bash
# Generated by setup_local_cudaq.sh; source this file before importing CUDA-Q.
export CUDAQ_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cudaq_local"
export CUDAQ_SOURCE="${CUDAQ_LOCAL%/cudaq_local}/cuda-quantum-src"
export CUDAQ_MPS_ANALYTIC_MODE=shared
"""
python=sys.argv[2]
for name in ('ANALYTIC_MPS_PYTHON','GRADIENT_PYTHON','SPSA_PYTHON','GRAPH_VQE_PYTHON'):
    text += 'export '+name+'='+shlex.quote(python)+'\n'
text += 'export PATH='+shlex.quote(str(Path(python).parent))+':"$CUDAQ_LOCAL/cudaq/bin:$PATH"\n'
text += 'export PYTHONPATH="$CUDAQ_LOCAL/cudaq:${PYTHONPATH:-}"\n'
text += 'export LD_LIBRARY_PATH="$CUDAQ_LOCAL/cudaq/lib:$CUDAQ_LOCAL/nccl/lib:'+str(Path(python).parent.parent/'lib')+':${LD_LIBRARY_PATH:-}"\n'
(root/'activate_local_cudaq.sh').write_text(text)
PYACTIVATE

# ---- 6. package-level MQPU smoke test ---------------------------------------
# Exercise every requested logical QPU after a rebuild and compare explicitly
# assigned Hamiltonian pieces with CUDA-Q's parallel.thread aggregate. This test
# is reusable and has no dependency on a consuming application.
if [ "${CUDAQ_SKIP_GPU_TEST:-0}" = 1 ]; then
  echo ">> CUDAQ_SKIP_GPU_TEST=1; skipping package-level GPU smoke test."
elif [ "$MQPU_TEST_QPUS" = 0 ]; then
  echo ">> no visible GPUs detected; build completed without the package-level GPU smoke test."
else
  echo ">> validating tensornet-mps-mqpu on physical GPUs $MQPU_TEST_GPUS ..."
  unset CUDAQ_MPS_HBATCH_GROUPS
  CUDA_VISIBLE_DEVICES="$MQPU_TEST_GPUS" \
  CUDAQ_MQPU_NGPUS=1 \
  CUDAQ_MPS_HBATCH_NGPUS="$MQPU_TEST_QPUS" \
  PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$CUDAQ_INSTALL_PREFIX/lib:$NCCL_ROOT/lib:${CONDA_PREFIX:-/usr/lib}/lib:${LD_LIBRARY_PATH:-}" \
  "$PY" - <<'PYEOF'
import math
import os
import cudaq

visible_devices = [
    item.strip()
    for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if item.strip()
]
expected_qpus = len(visible_devices)
assert cudaq.num_available_gpus() == expected_qpus

kernel, theta = cudaq.make_kernel(float)
q = kernel.qalloc(2)
kernel.ry(theta, q[0])
kernel.x(q[1])
kernel.cx(q[0], q[1])

paulis = (
    cudaq.spin.i(0),
    cudaq.spin.z(0),
    cudaq.spin.z(1),
    cudaq.spin.x(0) * cudaq.spin.x(1),
    cudaq.spin.y(0) * cudaq.spin.y(1),
    cudaq.spin.z(0) * cudaq.spin.z(1),
    cudaq.spin.x(1),
)
hamiltonian = 0.11 * paulis[0]
for index, pauli in enumerate(paulis[1:], start=1):
    hamiltonian += (0.11 + 0.03 * index) * pauli
angle = 0.37
results = {}
for backend in (
    "tensornet-mps-mqpu-shots",
    "tensornet-mps-mqpu-hamiltonian",
    "tensornet-mps-mqpu",
    "tensornet-mps-mqpu-replay",
    "tensornet-mps-mqpu-strong",
):
    cudaq.set_target(backend)
    target = cudaq.get_target()
    assert target.name == backend
    assert target.num_qpus() == 1
    results[backend] = cudaq.observe(
        kernel, hamiltonian, angle, shots_count=-1
    ).expectation()
# Independently known expectation for this two-qubit circuit. This checks
# the shared analytic contraction, not just agreement between backend variants.
expected_energy = -0.15 + 0.43 * math.sin(angle) - 0.03 * math.cos(angle)
reference = results["tensornet-mps-mqpu-shots"]
assert abs(reference - expected_energy) < 1.0e-10, (reference, expected_energy)
assert all(abs(value - reference) < 1.0e-10 for value in results.values())

# Exercise the corrected path itself: all I/Z terms must share one sampled
# computational-basis distribution, with the fixed total shot budget sharded
# across the configured physical helpers.
diagonal = (
    0.17 * cudaq.spin.i(0)
    + 0.23 * cudaq.spin.z(0)
    - 0.31 * cudaq.spin.z(1)
    + 0.19 * cudaq.spin.z(0) * cudaq.spin.z(1)
)
for backend in (
    "tensornet-mps-mqpu-shots",
    "tensornet-mps-mqpu-hamiltonian",
):
    cudaq.set_target(backend)
    sampled_diagonal = cudaq.observe(
        kernel, diagonal, angle, shots_count=4096
    ).expectation()
    assert math.isfinite(sampled_diagonal)
    sample_counts = cudaq.sample(kernel, angle, shots_count=4096)
    assert sum(int(count) for count in sample_counts.values()) == 4096

# Independent graph VQE changes register width between local partitions and
# the combined global circuit. Repeatedly exercise that worker lifecycle.
cudaq.set_target("tensornet-mps-mqpu-shots")
for width in (2, 5, 3, 2):
    local, theta = cudaq.make_kernel(float)
    register = local.qalloc(width)
    local.ry(theta, register[0])
    observable = cudaq.spin.z(0) + 0.25 * cudaq.spin.z(width - 1)
    value = cudaq.observe(local, observable, angle, shots_count=-1).expectation()
    assert abs(value - (math.cos(angle) + 0.25)) < 1.0e-10
    counts = cudaq.sample(local, angle, shots_count=1000)
    assert sum(int(count) for count in counts.values()) == 1000
print(
    f"   verified {len(results)} persistent backends on "
    f"{expected_qpus} visible GPU(s); expectations match; "
    "shots and Hamiltonian modes passed"
)
PYEOF
fi

cat <<EOF

============================================================================
Done. Installed under: $LOCAL
  Analytic:    shared Pauli-prefix contractions (shots=-1)
  Lifecycle:   worker state reset when local/global circuit widths change
  Manifest:    $CUDAQ_INSTALL_PREFIX/analytic_backend_manifest.json

Activate this installation in your shell:
  source "$PROJECT_DIR/activate_local_cudaq.sh"
Then run the desired campaign launcher. Reverse gradients require CUDA PyTorch.
  CUDA-Q:   $CUDAQ_INSTALL_PREFIX
  shots:       $CUDAQ_INSTALL_PREFIX/lib/libnvqir-tensornet-mps-mqpu-shots.so
  Hamiltonian: $CUDAQ_INSTALL_PREFIX/lib/libnvqir-tensornet-mps-mqpu-hamiltonian.so
  targets:     $CUDAQ_INSTALL_PREFIX/targets/tensornet-mps-mqpu-{shots,hamiltonian}.yml

The package is application-independent. A consuming process selects devices
before importing cudaq, for example:
  export CUDA_VISIBLE_DEVICES=${MQPU_TEST_GPUS:-0}
  export VQE_CUDAQ_TARGET=tensornet-mps-mqpu-shots
  export CUDAQ_MQPU_NGPUS=1
  export CUDAQ_MPS_HBATCH_NGPUS=$([ "$MQPU_TEST_QPUS" -gt 0 ] && echo "$MQPU_TEST_QPUS" || echo 1)
============================================================================
EOF
