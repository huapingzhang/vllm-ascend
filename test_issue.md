# [Bug]: `check_health` uses a vLLM logical rank as an `npu-smi` card ID

## Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
OS: Ubuntu 22.04.5 LTS (aarch64)
Python: 3.12.16
torch: 2.10.0+cpu
torch_npu: 2.10.0.post7
npu-smi: 25.5.0
CANN: 9.2.1
vllm-ascend: main @ 4617b5a4bbcafb6272489e766eb60ab34484825e
vLLM main: c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15
Hardware: Ascend 910; physical cards 1 and 4 are visible
```

The key `npu-smi` output is embedded below.

</details>

## 🐛 Describe the bug

### Code path

The bug is in `NPUWorker.check_health()` in
`vllm_ascend/worker/worker.py`. On the current reproduction revision it is:

```python
def check_health(self) -> None:
    import subprocess

    logger.debug("check_health starting for rank %s...", self.local_rank)
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-i", str(self.local_rank), "-t", "health"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parse_text_output(result.stdout)
            logger.debug("check_health success for rank %s.", self.local_rank)
        else:
            logger.warning("query NPU card %s fail: %s", self.local_rank, result.stderr)
    except subprocess.TimeoutExpired:
        logger.warning("query NPU card %s timeout.", self.local_rank)
    except FileNotFoundError:
        logger.warning("npu-smi tool not found.")
    except Exception as e:
        logger.error("query NPU card %s fail: %s", self.local_rank, e)
```

There are three distinct ID spaces:

- `self.local_rank` is a vLLM logical device ID.
- `_init_device()` converts it to the process-visible device with
  `current_platform.logical_device_id_to_visible_device_id(self.local_rank)`
  before calling `torch.npu.set_device(...)`.
- `npu-smi info -i` expects the host NPU ID (card ID).

`check_health()` skips both logical-to-visible and CANN/runtime
physical-device resolution. It passes the first ID directly to an API that
expects the third. These IDs coincide only in a no-remapping,
one-card-per-chip setup. They differ with
`ASCEND_RT_VISIBLE_DEVICES`, container device remapping, and multi-chip
cards.

### Actual vllm-ascend scenario

The stock HTTP `/health` endpoint does not currently call the worker health
RPC: `AsyncLLM.check_health()` only checks the engine's own error state.
This bug affects code or external monitoring that explicitly invokes the
worker health RPC, for example:

```text
MultiprocExecutor.check_health()
  -> collective_rpc("check_health")
  -> NPUWorker.check_health()
```

No model output is needed to trigger the bug. Start a real `LLM` engine on
one visible device and invoke `check_health` through vLLM's collective-RPC
entry point:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_BATCH_INVARIANT=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_LOGGING_LEVEL=DEBUG \
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
python - <<'PY'
import os

from vllm import LLM

llm = LLM(
    model=os.environ["MODEL_PATH"],
    max_model_len=512,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
)

# Reaches NPUWorker.check_health() through the worker collective RPC.
llm.collective_rpc("check_health")
PY
```

Observed on the reproduction host:

```text
WARNING ... [worker.py:1435] query NPU card 0 fail:
```

The RPC still returns normally, so a caller that only relies on "no
exception" sees a healthy worker. This is the same collective-RPC path used
by `MultiprocExecutor.check_health()`:

```python
self.collective_rpc("check_health", timeout=10)
```

This is the real failure mode: the engine is serving, and an explicit worker
health RPC reports success, but its device health check queried a card that
does not exist.

### Why rank 0 is the wrong card

On this host, `npu-smi info -m` reports cards 1 and 4:

```text
NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name
1       0        0              2            Ascend910
1       1        1              3            Ascend910
4       0        2              8            Ascend910
4       1        3              9            Ascend910
```

The CANN runtime resolves bound device 0 to chip Phy-ID 2:

```text
logical=0 visible=0 chip_phy_id=2
```

Phy-ID 2 belongs to card 1. The card with the serving process's HBM usage
is card 1. The code nevertheless runs:

```text
$ npu-smi info -i 0 -t health
Invalid card id.
$ echo $?
215

$ npu-smi info -i 1 -t health
NPU ID                         : 1
$ echo $?
0
```

`npu-smi` writes `Invalid card id.` to stdout rather than stderr. The
existing code logs only `result.stderr`, which is why the warning is
truncated after `fail:`.

### Expected behavior

`check_health()` should resolve the bound runtime device to its physical
chip ID, map that ID to the hosting card through the `Chip Phy-ID` column
of `npu-smi info -m`, and query that card. It must not assume
`local_rank == npu-smi card ID`.

## Additional context

- This report is limited to the explicit worker health-RPC path. Wiring
  executor health into the HTTP `/health` endpoint would be a separate
  behavior change and is not included here.
- `npu-smi info -i 1` can also silently inspect the wrong card if card 1
  exists but is not the card hosting the worker. Avoiding this requires the
  same chip-to-card resolution.
- I searched the repository's issues and PRs on 2026-11-25 for
  `check_health`, `npu-smi`, `Invalid card id`, `local_rank`, and
  `Chip Phy-ID`; no matching bug or open implementation PR was found.
- Patch and CPU regression tests are ready.
