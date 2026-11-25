# [BugFix][Worker] Resolve the npu-smi card ID in `check_health`

## What this PR does / why we need it?

`NPUWorker.check_health()` passes `self.local_rank` to
`npu-smi info -i`. A local rank identifies a vLLM logical chip, while
`npu-smi -i` expects the host NPU ID (card ID). The IDs diverge on
multi-chip cards, with `ASCEND_RT_VISIBLE_DEVICES`, and in containers.
The health RPC therefore either fails with `Invalid card id` or queries a
different card's health.

This change is scoped to explicit calls of the worker health RPC
(`executor.check_health()` or `executor.collective_rpc("check_health")`).
The stock HTTP `/health` endpoint does not call the worker health RPC in
the current vLLM implementation; wiring it to do so is outside this PR.

This PR resolves the bound runtime device to its chip Phy-ID using the
existing platform mapping, then maps that ID to the NPU ID through the
`Chip Phy-ID` column of `npu-smi info -m`. The health command receives the
resolved card ID, and the log messages report the card actually queried.

The fallback keeps the previous best-effort behavior for platforms whose
`npu-smi info -m` output does not expose `Chip Phy-ID`.

On the reproduction host (cards 1 and 4), logical rank 0 resolves to chip
Phy-ID 2 and card 1. Before the change, `npu-smi -i 0` returns 215
(`Invalid card id`); after the change, `npu-smi -i 1 -t health` returns 0.

Fixes #<issue-number>

## Does this PR introduce _any_ user-facing change?

Yes. Explicit worker health checks now monitor the card containing the
worker's bound chip on remapped and multi-chip hosts. The previous behavior
could monitor a different card or fail every time. A release-note fragment
is included.

## How was this patch tested?

Focused CPU unit tests:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python -m pytest -q \
  tests/ut/device_allocator/test_cpu_binding.py \
  tests/ut/worker/test_worker_check_health.py
```

Result: `87 passed`.

Run against the vLLM `main` commit pinned by `.github/vllm-main-verified.commit`
and `transformers==5.19.0` from `requirements.txt`.

The added coverage checks:

- parsing `Chip Phy-ID -> NPU ID` with multi-chip cards;
- using Phy-ID rather than `Chip Logic ID` when they differ;
- fallback when `npu-smi info -m` is unavailable;
- the exact `npu-smi` command invoked by `check_health`.

Real NPU validation with `ASCEND_RT_VISIBLE_DEVICES=0,1`:

```text
rank 0: visible=0 chip_phy_id=2 npu-smi card=1
rank 1: visible=1 chip_phy_id=3 npu-smi card=1
check_health_npu: PASS
```

Direct pre-fix and post-fix commands:

```text
npu-smi info -i 0 -t health -> rc=215, Invalid card id
npu-smi info -i 1 -t health -> rc=0, NPU ID: 1
```

Ruff:

```bash
ruff check vllm_ascend/cpu_binding.py vllm_ascend/worker/worker.py \
  tests/ut/device_allocator/test_cpu_binding.py \
  tests/ut/worker/test_worker_check_health.py
ruff format --check vllm_ascend/cpu_binding.py \
  vllm_ascend/worker/worker.py \
  tests/ut/device_allocator/test_cpu_binding.py \
  tests/ut/worker/test_worker_check_health.py
```

Result: all checks passed; all files formatted.

CI/NPU coverage still to be confirmed by the repository workflows: the
full worker suite and any hardware-specific health-check path.

## Checklist

### Testing

- [x] **Boundary / edge-case coverage**: covers multi-chip cards, a
  non-identity logical-to-Phy-ID mapping, and missing `Chip Phy-ID` output.
- [x] **Regression test**: the old implementation would call
  `npu-smi -i 0`; the new test requires the resolved card ID.
- [x] **Error / failure path coverage**: covers an unavailable
  `npu-smi info -m` and falls back without crashing.
- [x] **Meaningful & reliable tests**: tests assert parsed mappings and
  the actual subprocess arguments, not just mock calls in isolation.
- [x] **Performance impact**: health checks are not a serving hot path;
  mapping is limited to the existing health-RPC invocation.

- vLLM version: main
- vLLM main: c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15
