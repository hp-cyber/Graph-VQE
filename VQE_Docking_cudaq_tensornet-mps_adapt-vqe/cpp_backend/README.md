# tensornet-mps-mgpu backend sources

These are the **source-of-truth** files for the multi-GPU sharded-MPS NVQIR
backend, extracted from `../setup_local_cudaq.sh` (which embeds byte-identical
copies as heredocs and is what actually builds them).

- `dist_svd.cuh`          — distributed randomized truncated SVD (cuBLAS/cuSOLVER/NCCL)
- `sharded_mps.cuh`       — ShardedMPS engine (sites sharded across GPUs)
- `simulator_mps_dist.cu` — NVQIR backend `tensornet-mps-mgpu` (registers the target)
- `tensornet-mps-mgpu.yml`— target config (fp64)

## IMPORTANT: keep in sync with the build script
The build compiles the copies **embedded in `setup_local_cudaq.sh`**, not these
files directly. If you edit these, re-embed them into the script's heredocs
(`MGPU_*_EOF` blocks) before rebuilding, or the change won't take effect.

Compiled artifact in use: `~/cudaq_local/cudaq/lib/libnvqir-tensornet-mps-mgpu.so`.
