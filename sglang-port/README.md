# DeltaServe — sglang Port

This directory holds the **port of DeltaServe to sglang**. It is the
parallel of the existing
[DeltaServe-vLLM](https://github.com/852866031/DeltaServe-vLLM) fork, but
targeted at [sglang](https://github.com/sgl-project/sglang) instead of vLLM.

## Where the working port lives → [`v046-port/`](v046-port/)

**[`v046-port/`](v046-port/) is the current, benchmarked port** against
`sglang==0.4.6.post5`. Everything you need is there:

- **[`v046-port/INSTALL.md`](v046-port/INSTALL.md)** — one-command install
  (`bash v046-port/install.sh`). The repo ships sglang itself as a vendored
  wheel + a portable patch + the new `deltaserve/` runtime; the installer
  overlays them onto a stock install.
- **[`v046-port/README.md`](v046-port/README.md)** — the co-serving benchmark
  report: apples-to-apples results vs DeltaServe-vLLM on the same H200, plot
  index, optimization status (5 of 14 `CO_SERVING_OPTIMIZATIONS` sections),
  and the roadmap.

```bash
cd sglang-port/v046-port
bash install.sh        # vendored sglang + deps + DeltaServe overlay
```

---

## Design / history docs (Phase-1 reconnaissance & plan)

The markdown files in this directory are the **original recon + design
record** from the first pass of the port (driven by `pair-cli`). They are
archival — they describe the planning snapshot and reference scratch paths
(`/tmp/...`) that were never committed. The code they planned has since been
superseded by the implementation in `v046-port/`. Kept for context on *why*
the port is shaped the way it is.

| File | Purpose |
|---|---|
| `D1_SGLANG_RECON.md` | sglang architecture reconnaissance (7 themes: Scheduler / ModelRunner / Worker / LoRA / process-model / KV-cache / batch-scheduling, each with file:line refs and porting surprises). |
| `D2_MAPPING.md` | Component-by-component map: each DeltaServe-vLLM component → where its sglang equivalent should live + sglang-specific risks. |
| `D3_PLAN.md` | The original phased implementation plan. |
| `PORT_COMPLETE.md` | Phase-1 completion log (per-phase acceptance tests, surface area, preserved invariants). Historical — the live surface area is now what `v046-port/sglang-046-port.patch` + `v046-port/new-files/` define. |
| `SESSION_TRANSCRIPT.md` | Full transcript of the original Phase-1 pair-cli session. Audit trail. |
| `phase1_task.yaml` | Original Phase-1 pair-cli config. |

> Reading order if you care about the design rationale: `D1` → `D2` → `D3`.
> If you just want to run it, skip straight to
> [`v046-port/INSTALL.md`](v046-port/INSTALL.md).

---

## Why a separate branch (not in DeltaServe-vLLM)?

The vLLM fork is upstream-vendored; the sglang port targets a different
upstream and needs its own deviation history. Keeping it under this repo
(DeltaServe) on a dedicated branch lets the design docs sit next to the
existing co-serving research notes (`CLAUDE.md`, `docs/`) without polluting
either fork's diff against its upstream.
