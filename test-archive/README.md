# Test archive

This archive preserves the complete local `Graph-VQE/test/` directory captured
on November 16, 2026. It includes experiment logs, checkpoints, source files,
validation results, plots, and local runtime configuration. No files within
`test/` were excluded. The original directory was retained on the source machine.

The archive uses tar and Zstandard compression and is split into ordered parts
of at most 1,900,000,000 bytes. All parts are stored with Git LFS. Part sizes,
SHA-256 hashes, and source entry counts are recorded in `manifest.json`.

## Download and verify

Install Git LFS and Zstandard (`zstd`). From the repository root:

```bash
git lfs install
git lfs pull --include="test-archive/test.tar.zst.part-*"
cd test-archive
sha256sum -c SHA256SUMS
```

All parts must report `OK` before extraction. The parts form one compressed
stream and cannot be extracted individually.

## Extract

Run these commands in Bash from `test-archive/`. Use a new or empty destination
with at least 674 GiB of available space, plus filesystem overhead:

```bash
set -o pipefail
mkdir -p ../restored-test
cat test.tar.zst.part-* | zstd -d -c | tar -xf - -C ../restored-test
```

The restored directory is `restored-test/test/`. The repository already has
tracked `test/` sources that may differ from this snapshot; keeping the restored
snapshot in a separate directory preserves both versions. Runtime configuration
and stored paths describe the original machine and may need adjustment before
resuming experiments elsewhere.
