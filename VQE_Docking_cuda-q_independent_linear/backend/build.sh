#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDAQ_SOURCE="${CUDAQ_SOURCE:-/home/fengy46/cuda-quantum-src}"
CUDAQ_BUILD="${CUDAQ_BUILD:-$CUDAQ_SOURCE/build}"
CUDAQ_LOCAL="${CUDAQ_LOCAL:-/home/fengy46/cudaq_local}"
NINJA="${NINJA:-$CUDAQ_LOCAL/bin/ninja}"
SOURCE_FILE="$HERE/source/simulator_mps_dist.cu"
OUTPUT_ROOT="$HERE/backend_variants"
LOCAL_INSTALL="$CUDAQ_LOCAL/cudaq"
TARGET_LIBRARY="libnvqir-tensornet-mps-mqpu-shots.so"
NINJA_TARGET="lib/$TARGET_LIBRARY"
UPSTREAM_SOURCE="$CUDAQ_SOURCE/runtime/nvqir/cutensornet/simulator_mps_dist.cu"
PRODUCTION_OBJECT="runtime/nvqir/cutensornet/CMakeFiles/nvqir-tensornet-mps-mqpu-shots.dir/simulator_mps_dist.cu.o"

printf '\n===== backend build started: %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for required in "$NINJA" "$SOURCE_FILE" "$HERE/source/sharded_mps.cuh" \
                "$HERE/source/dist_svd.cuh" "$CUDAQ_BUILD/build.ninja" \
                "$LOCAL_INSTALL/lib/libcudaq-common.so"; do
  if [ ! -e "$required" ]; then
    echo "ERROR: required build input is missing: $required" >&2
    exit 1
  fi
done

normalize_backend_templates() {
  "${GPU_ABLATION_PYTHON:-python3}" - "$1" "$CUDAQ_BUILD" "$UPSTREAM_SOURCE" \
    "$PRODUCTION_OBJECT" "$NINJA_TARGET" "${CUDAQ_LOCAL:-}" <<'PY_TEMPLATES'
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import sys

raw, build, expected_source, expected_object, expected_library, local = sys.argv[1:]
build = Path(build).resolve()
# Keep raw spans so shell operators and escaped $ORIGIN in the link command
# are preserved exactly. Only identified source/outputs/definitions change.
word = re.compile(r'''(?:[^\s"'\\]|\\[\s\S]|"(?:[^"\\]|\\[\s\S])*"|'[^']*')+''')
required = {'CUDAQ_MPS_MQPU_BACKEND', 'CUDAQ_MPS_MQPU_SHOTS_BACKEND',
            'CUDAQ_MPS_MQPU_STRONG_PIPELINE'}

def fail(message):
    raise ValueError(message)

def tokens(line):
    spans, values, end = [], [], 0
    for match in word.finditer(line):
        if line[end:match.start()].strip():
            fail('Cannot parse a quoted compiler argument')
        decoded = shlex.split(match.group())
        if len(decoded) != 1:
            fail('Cannot isolate a compiler argument')
        spans.append(match.span())
        values.append(decoded[0])
        end = match.end()
    if line[end:].strip():
        fail('Unclosed quote in compiler command')
    return spans, values

def path(value):
    return (build / value).resolve()

def argument(values, flag):
    indices = [i + 1 for i, value in enumerate(values) if value == flag]
    if len(indices) != 1 or indices[0] >= len(values):
        fail(f'Expected exactly one {flag} argument')
    return indices[0]

def render(line, spans, changes):
    for i in sorted(changes, reverse=True):
        start, end = spans[i]
        line = line[:start] + changes[i] + line[end:]
    return line

def relocate(lines):
    """Rebase a copied CMake build using its recorded original directories."""
    compiles = [tokens(line)[1] for line in lines if '-c' in tokens(line)[1]]
    if len(compiles) != 1:
        return lines, False
    values = compiles[0]
    original_source = path(values[argument(values, '-c')])
    if original_source == Path(expected_source).resolve():
        return lines, False
    cache_path = build / 'CMakeCache.txt'
    if not cache_path.is_file():
        fail(f'Source mismatch: {original_source}; no CMakeCache.txt to verify a relocated build')
    cache = {}
    for entry in cache_path.read_text().splitlines():
        if entry and not entry.startswith(('#', '//')) and ':' in entry and '=' in entry:
            key, value = entry.split('=', 1)
            cache[key.split(':', 1)[0]] = value
    old_source = cache.get('CMAKE_HOME_DIRECTORY')
    suffix = 'runtime/nvqir/cutensornet/simulator_mps_dist.cu'
    if not old_source or original_source != (Path(old_source) / suffix).resolve():
        fail(f'Source mismatch: Ninja uses {original_source}; CMake recorded source root is {old_source}')
    new_source = Path(expected_source).resolve().parents[3]
    old_local = Path(cache.get('CMAKE_INSTALL_PREFIX', ''))
    if old_local.name != 'cudaq' or not old_local.is_absolute() or not local:
        fail('Cannot identify the original cudaq_local directory from CMAKE_INSTALL_PREFIX')
    old_compiler = Path(cache.get('CMAKE_CUDA_COMPILER', ''))
    mappings = [
        (cache.get('CMAKE_CACHEFILE_DIR', str(Path(old_source) / 'build')), str(build)),
        (old_source, str(new_source)),
        (str(old_local.parent), str(Path(local).resolve())),
    ]
    # The reported archive was built with a Conda CUDA toolkit. Its replacement
    # must be the selected environment's toolkit, not an arbitrary nvcc in PATH.
    if old_compiler.name == 'nvcc' and old_compiler.parent.name == 'bin':
        old_toolchain = str(old_compiler.parent.parent)
        override = os.environ.get('CUDAQ_TOOLCHAIN_PREFIX')
        if '/envs/' in old_toolchain or override:
            mappings.append((old_toolchain, str(Path(override or sys.prefix).resolve())))
    mappings = [(old, new) for old, new in mappings if old != new]
    # Replace original roots simultaneously; do not rewrite a replacement a
    # second time when old and new directories overlap.
    roots = dict(mappings)
    safe_root = re.compile(r'^/[A-Za-z0-9_./+\-]+$')
    if any(not safe_root.fullmatch(root) for pair in mappings for root in pair):
        fail('Relocated build roots contain shell-special characters; reconfigure CMake at these paths')
    pattern = re.compile(r'(?:(?<=-I)|(?<=-L)|(?<![A-Za-z0-9_./]))(' +
                         '|'.join(re.escape(root) for root in sorted(roots, key=len, reverse=True)) +
                         r')(?=/|$|[\s\"\x27,:])')
    print('>> Relocating copied Ninja commands (CMake files remain unchanged):', file=sys.stderr)
    for old, new in mappings:
        print(f'   {old} -> {new}', file=sys.stderr)
    rewritten = [pattern.sub(lambda match: roots[match.group(1)], line) for line in lines]
    return rewritten, True

def select_host_compiler(lines):
    """Use a user-installed GCC 11 for both nvcc's host and the linker."""
    compiles = [(line, *tokens(line)) for line in lines if '-c' in tokens(line)[1]]
    if len(compiles) != 1:
        return lines, False
    _, _, values = compiles[0]
    hosts = []
    for i, value in enumerate(values):
        if value.startswith(('-ccbin=', '--compiler-bindir=')):
            hosts.append(value.split('=', 1)[1])
        elif value in ('-ccbin', '--compiler-bindir') and i + 1 < len(values):
            hosts.append(values[i + 1])
    if not hosts:
        return lines, False
    if len(hosts) != 1:
        fail('Expected exactly one nvcc host compiler')
    original = hosts[0]
    override = os.environ.get('GPU_ABLATION_CXX')
    if not override and path(original).is_file() and os.access(path(original), os.X_OK):
        return lines, False
    candidate_paths = [override] if override else [
        Path.home() / '.local/share/cudaq-gcc11/bin/x86_64-conda-linux-gnu-g++',
        Path(sys.prefix) / 'bin/x86_64-conda-linux-gnu-g++',
        Path(sys.prefix).parent / 'cudaq-gcc11/bin/x86_64-conda-linux-gnu-g++',
        shutil.which('g++-11'),
    ]
    selected, version, rejected = None, '', []
    for candidate in candidate_paths:
        if not candidate:
            continue
        candidate = Path(shutil.which(str(candidate)) or candidate).expanduser().absolute()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            rejected.append(f'{candidate}: not executable')
            continue
        result = subprocess.run([str(candidate), '-dumpfullversion', '-dumpversion'],
                                capture_output=True, text=True)
        version = result.stdout.strip()
        if result.returncode or not re.fullmatch(r'11(?:\.\d+)*', version):
            rejected.append(f'{candidate}: expected GCC 11, got {version!r}')
            continue
        selected = str(candidate)
        break
    if selected is None:
        if override:
            fail('Invalid GPU_ABLATION_CXX: ' + '; '.join(rejected))
        # The regular input check reports the missing original compiler along
        # with any other missing dependencies.
        return lines, False
    rewritten, link_replacements = [], 0
    for line in lines:
        spans, values = tokens(line)
        changes = {}
        if '-c' in values:
            for i, value in enumerate(values):
                if value.startswith(('-ccbin=', '--compiler-bindir=')):
                    changes[i] = shlex.quote(value.split('=', 1)[0] + '=' + selected)
                elif i > 0 and values[i - 1] in ('-ccbin', '--compiler-bindir'):
                    changes[i] = shlex.quote(selected)
        else:
            for i, value in enumerate(values):
                if value == original:
                    changes[i] = shlex.quote(selected)
                    link_replacements += 1
        rewritten.append(render(line, spans, changes))
    if link_replacements != 1:
        fail('Cannot identify the same C++ compiler in the link command; refusing a partial compiler switch')
    print(f'>> Host C++ compiler: {selected} (GCC {version}); replaces {original} in compile and link',
          file=sys.stderr)
    return rewritten, True

def check_relocated_inputs(parsed):
    """Report all missing referenced inputs before compiling any variant."""
    missing = set()
    for _, _, values in parsed:
        skipped = set()
        for flag in ('-o', '-MT', '-MF'):
            for i, value in enumerate(values[:-1]):
                if value == flag:
                    skipped.add(i + 1)
        for i, value in enumerate(values):
            if i in skipped or path(value) == path(expected_object):
                continue
            candidate, executable = '', False
            if value.startswith(('-I', '-L')) and len(value) > 2:
                candidate = value[2:]
            elif value.startswith('-ccbin='):
                candidate, executable = value.split('=', 1)[1], True
            elif value.startswith('/') or value.startswith('lib/lib'):
                candidate = value
                executable = Path(value).name in ('nvcc', 'g++-11', 'g++', 'c++', 'clang++')
            if candidate:
                target = path(candidate)
                if not target.exists():
                    missing.add(f'Missing: {target}')
                elif executable and not os.access(target, os.X_OK):
                    missing.add(f'Not executable: {target}')
    if missing:
        fail('Copied build needs these inputs on this server:\n  ' + '\n  '.join(sorted(missing)) +
             '\nInstall the matching build dependencies or rebuild CUDA-Q locally; '
             'changing the source path alone is insufficient.')

try:
    lines = [line for line in raw.splitlines() if 'simulator_mps_dist.cu' in line]
    lines, relocated = relocate(lines)
    lines, compiler_changed = select_host_compiler(lines)
    parsed = [(line, *tokens(line)) for line in lines]
    compiles = [item for item in parsed if '-c' in item[2]]
    links = [item for item in parsed if '-c' not in item[2]]
    if len(compiles) != 1 or len(links) != 1:
        fail(f'Expected one compile and one link command; found {len(compiles)} and {len(links)}')
    line, spans, values = compiles[0]
    changes, definitions = {}, {}
    for i, value in enumerate(values):
        if not value.startswith('-D'):
            continue
        j = i + 1 if value == '-D' else i
        if j >= len(values):
            fail('Missing value after -D')
        definition = values[j] if j != i else value[2:]
        name, _, setting = definition.partition('=')
        if name.startswith(('CUDAQ_MPS_MQPU_', 'CUDAQ_MPS_ABLATION_', 'CUDAQ_MPS_MGPU')):
            if name not in required:
                fail(f'Unexpected backend control {name}; cannot assume equivalent ablation settings')
            if name in definitions:
                fail(f'Duplicate backend definition: {name}')
            definitions[name] = setting or '1'
            changes[i] = '__ABLATION_DEFINITIONS__' if len(definitions) == 1 else ''
            if j != i:
                changes[j] = ''
    if set(definitions) != required or any(value != '1' for value in definitions.values()):
        fail(f'Shots backend requires {sorted(required)} enabled; found {definitions}')
    source_index = argument(values, '-c')
    if path(values[source_index]) != Path(expected_source).resolve():
        fail(f'Source mismatch: Ninja uses {values[source_index]} (resolved {path(values[source_index])}); '
             f'CUDAQ_SOURCE expects {expected_source}')
    changes[source_index] = '__ABLATION_SOURCE__'
    object_index = argument(values, '-o')
    object_path = path(values[object_index])
    if object_path != path(expected_object):
        fail(f'Unexpected shots object output: {values[object_index]}')
    for i, value in enumerate(values):
        if path(value) == object_path:
            changes[i] = '__ABLATION_OBJECT__'
        elif path(value) == Path(str(object_path) + '.d'):
            changes[i] = '__ABLATION_DEPFILE__'
    compile_template = render(line, spans, changes)

    line, spans, values = links[0]
    library_index = argument(values, '-o')
    if path(values[library_index]) != path(expected_library):
        fail(f'Unexpected shots library output: {values[library_index]}')
    objects = [i for i, value in enumerate(values) if path(value) == object_path]
    if len(objects) != 1:
        fail('Link command does not contain exactly one matching shots object')
    link_template = render(line, spans, {library_index: '__ABLATION_LIBRARY__',
                                        objects[0]: '__ABLATION_OBJECT__'})
    if relocated or compiler_changed:
        check_relocated_inputs(parsed)
    print(compile_template)
    print(link_template)
except (ValueError, OSError) as error:
    print(f'ERROR: {error}', file=sys.stderr)
    print('Ninja commands for diagnosis:', file=sys.stderr)
    for line in raw.splitlines():
        if 'simulator_mps_dist.cu' in line:
            print(line, file=sys.stderr)
    sys.exit(1)
PY_TEMPLATES
}

echo ">> CUDA-Q source: $CUDAQ_SOURCE; build: $CUDAQ_BUILD"
# Command substitution propagates Ninja/parser failures instead of masking
# them inside mapfile's process substitution. No ripgrep dependency.
NINJA_COMMANDS="$("$NINJA" -C "$CUDAQ_BUILD" -t commands "$NINJA_TARGET" |
  grep -F 'simulator_mps_dist.cu')"
NORMALIZED_TEMPLATES="$(normalize_backend_templates "$NINJA_COMMANDS")"
mapfile -t BACKEND_COMMANDS <<< "$NORMALIZED_TEMPLATES"
COMPILE_TEMPLATE="${BACKEND_COMMANDS[0]}"
LINK_TEMPLATE="${BACKEND_COMMANDS[1]}"

SOURCE_HASHES="$(cd "$HERE/source" && sha256sum dist_svd.cuh sharded_mps.cuh analytic_batch.cuh simulator_mps_dist.cu)"
TEMPLATE_HASH="$(printf '%s\n%s\n' "$COMPILE_TEMPLATE" "$LINK_TEMPLATE" | sha256sum | awk '{print $1}')"
CUDAQ_COMMIT="$(git -C "$CUDAQ_SOURCE" rev-parse HEAD)"

make_overlay() {
  local install_dir="$1"
  # An extracted archive still contains links and Python extensions from the
  # original server. Refresh these against this server, including broken links.
  "${GPU_ABLATION_PYTHON:-python3}" - "$LOCAL_INSTALL" "$install_dir" "$TARGET_LIBRARY" <<'PY'
from pathlib import Path
import os
import shutil
import sys
import tempfile

source, destination = map(Path, sys.argv[1:3])
source = source.resolve()
plugin = sys.argv[3]
(destination / 'lib').mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='.refresh-', dir=destination) as temporary:
    temporary = Path(temporary)
    for dependency in [*(source / 'lib').iterdir(), source / 'targets']:
        if dependency.name in (plugin, 'libcudaq-common.so'):
            continue
        target = destination / ('targets' if dependency.name == 'targets'
                                else 'lib/' + dependency.name)
        link = temporary / 'dependency'
        link.symlink_to(dependency)
        os.replace(link, target)

    # A real common library is required for overlay-relative plugin discovery.
    common = temporary / 'libcudaq-common.so'
    shutil.copy2(source / 'lib/libcudaq-common.so', common)
    os.replace(common, destination / 'lib/libcudaq-common.so')

    # Copy to support different filesystems and replace extensions from the old
    # server. Do not merge packages with potentially different Python ABIs.
    package = temporary / 'cudaq'
    shutil.copytree(source / 'cudaq', package, symlinks=True)
    previous = temporary / 'previous_cudaq'
    target = destination / 'cudaq'
    if target.exists() or target.is_symlink():
        target.replace(previous)
    try:
        package.replace(target)
    except BaseException:
        if previous.exists() or previous.is_symlink():
            previous.replace(target)
        raise
PY
}

build_variant() {
  local variant="$1"
  local definitions="$2"
  local variant_dir="$OUTPUT_ROOT/$variant"
  local install_dir="$variant_dir/install"
  local object_file="$variant_dir/simulator_mps_dist.cu.o"
  local library_file="$install_dir/lib/$TARGET_LIBRARY"
  local provenance_file="$variant_dir/provenance.txt"
  local compile_file="$variant_dir/compile_command.txt"
  local link_file="$variant_dir/link_command.txt"

  if [[ "$definitions" == *"MGPU"* ]] || [[ "$definitions" == *"HAMILTONIAN"* ]]; then
    echo "ERROR: excluded backend family appeared in $variant definitions." >&2
    exit 1
  fi
  if [[ "$definitions" != *"CUDAQ_MPS_MQPU_SHOTS_BACKEND=1"* ]]; then
    echo "ERROR: $variant must register only the shots target." >&2
    exit 1
  fi

  mkdir -p "$variant_dir"
  make_overlay "$install_dir"

  local source_quoted object_quoted depfile_quoted library_quoted
  printf -v source_quoted '%q' "$SOURCE_FILE"
  printf -v object_quoted '%q' "$object_file"
  printf -v depfile_quoted '%q' "$object_file.d"
  printf -v library_quoted '%q' "$library_file"
  local compile_command="${COMPILE_TEMPLATE//__ABLATION_DEFINITIONS__/$definitions}"
  compile_command="${compile_command//__ABLATION_SOURCE__/$source_quoted}"
  compile_command="${compile_command//__ABLATION_OBJECT__/$object_quoted}"
  compile_command="${compile_command//__ABLATION_DEPFILE__/$depfile_quoted}"
  local link_command="${LINK_TEMPLATE//__ABLATION_OBJECT__/$object_quoted}"
  link_command="${link_command//__ABLATION_LIBRARY__/$library_quoted}"

  local expected_provenance
  expected_provenance="variant=$variant
definitions=$definitions
cudaq_commit=$CUDAQ_COMMIT
template_sha256=$TEMPLATE_HASH
$SOURCE_HASHES
compile_command_sha256=$(printf '%s' "$compile_command" | sha256sum | awk '{print $1}')
link_command_sha256=$(printf '%s' "$link_command" | sha256sum | awk '{print $1}')"

  if [ -s "$library_file" ] && [ -f "$provenance_file" ] &&
     [ "$(<"$provenance_file")" = "$expected_provenance" ]; then
    echo ">> current: $variant"
    return
  fi

  printf '%s\n' "$compile_command" > "$compile_file"
  printf '%s\n' "$link_command" > "$link_file"
  echo ">> building: $variant"
  (cd "$CUDAQ_BUILD" && bash -c "$compile_command")
  (cd "$CUDAQ_BUILD" && bash -c "$link_command")
  printf '%s\n' "$expected_provenance" > "$provenance_file"
  sha256sum "$library_file"
}

build_variant "analytic" "-DCUDAQ_MPS_MQPU_BACKEND=1 -DCUDAQ_MPS_MQPU_SHOTS_BACKEND=1 -DCUDAQ_MPS_MQPU_STRONG_PIPELINE=1"
echo ">> built project-local shots backend with worker lifecycle fix"
