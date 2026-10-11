#!/usr/bin/env bash
# =============================================================================
# One-shot build of CUDA-Q plus the custom sharded MPS backend and persistent
# Hamiltonian-batching backends, analytic Pauli-prefix acceleration, and the
# worker-lifecycle fix needed when independent VQE changes partition widths.
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
expected = {'analytic_batch.cuh': '4c962ad92fd0170b5f5acfd94e0607101a38fdd21c346a2155f64f138fca39f0', 'dist_svd.cuh': 'aba925435a29ed8dbc39624dd552389dd43823f6e5d8662e4be6ff4818920fb5', 'sharded_mps.cuh': '7164197f6f2507631f6bcbe3c2b879c896a5c2399fb21db1377df7f4dd758b66', 'simulator_mps_dist.cu': 'cb70da9e8fd0126f3d469ab0e91fcd574dda0a96cabf8e8d5a452202f29ca1c4'}

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
          'features':['analytic-pauli-prefix-v1','hbatch-worker-lifecycle-v1','shared-diagonal-shots-v1'],
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
