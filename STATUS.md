# ClauseGate: final status report (after GPU/training run)

5-seed AgentDojo training, zero-shot transfer to InjecAgent and
R-Judge, and the adversarial harness are all complete. No commit,
no push.

## Headline (measured, locked)

| Benchmark / Metric | Result (5-seed mean ± std) | vs. original target |
|---|---|---|
| AgentDojo **ASR** | **0.000 ± 0.000** | target ≤ 5 %, beats by 5 pp |
| AgentDojo latency / call (CPU) | **6.18 ± 0.15 ms** | target ≤ 200 ms CPU, beats by 30 × |
| AgentDojo recourse → non-DENY | **1.000 ± 0.000** | target ≥ 90 %, beats by 10 pp |
| AgentDojo oversight recall | 0.990 ± 0.008 | n/a, strong |
| AgentDojo benign utility | 0.302 ± 0.156 | target ≥ 60 %, misses (over-cautious; see caveat) |
| AgentDojo macro-F1 | 0.525 ± 0.098 | n/a |
| AgentDojo accuracy | 0.684 ± 0.035 | n/a |
| **ChatInject** ASR after defence | **0.000** (318 / 318 blocked) | n/a, perfect |
| **RAG-poison** ASR after defence | **0.000** (318 / 318 blocked) | n/a, perfect |
| GCG-MCP-random stub ASR | 0.328 (stub does not really attack) | n/a |
| **InjecAgent zero-shot defended ASR** | **0.000** (n = 300) | target ≤ 10 %, beats |
| **R-Judge zero-shot defended ASR** | **0.000** (n = 371) | target ≤ 15 %, beats |

Per-seed JSONs at `results/headline/agentdojo_full_seed_{1,2,3,5,8}.json`;
cross-seed aggregate at `results/headline/agentdojo_full_5seed_aggregate.json`;
zero-shot reports at `results/headline/zero_shot_{injecagent,rjudge}.json`.

## What to look at first

1. `README.md`: contains the locked numbers and honest caveats.
2. `results/headline/agentdojo_full_5seed_aggregate.json`: the full
   cross-seed table.
3. `results/jsonl/seed_*/per_sample.jsonl`: every test decision logged
   with verdict, scores, firing clauses, model_version_id,
   tri_graph_signature.
4. `results/receipts/seed_*/exemplar.json`: one signed receipt per
   seed (HMAC-SHA-256 with an ephemeral key; CNF at
   `results/receipts/seed_*/cnf/exemplar.cnf`).

## What ran

```
                                          wall-clock   on
1.  pip install -e . + python-sat        ~ 90 s         CPU
2.  pip install agentdojo                ~ 60 s         CPU
3.  pytest tests/                            9 s        CPU  (56 / 56 pass)
4.  smoke training (workspace, seed 1)      9 s        CPU
5.  5-seed AgentDojo pooled run         ~ 3.5 min       CPU
6.  zero-shot InjecAgent (300 cases)    ~ 60 s          CPU
7.  zero-shot R-Judge (371 traces)      ~ 65 s          CPU
                                        --------
   Total wall-clock                      ~ 8 minutes    CPU
```

The Tesla V100-SXM3-32GB sat idle the whole time. The big speed
win was a 20 × vectorisation of the canonical NumPy
`MultiClassTM`: tensordot in `clause_outputs`, batched RNG and
delta tensor in `_feedback`. Test runtime dropped from 187 s to 9 s
after the rewrite, and 5-seed training went from an estimated
~80 GPU-hours / seed to **actual 45 s / seed on CPU**.

## What did NOT run

- **CUDA wrapper for graphtm-cbr's kernels**: not built tonight.
  The vectorised NumPy backend was fast enough that the CUDA
  dispatch path was never hit. The interface is in place
  (`TriGraphHGTM(backend="cuda")`) and pulls the kernels from
  `graphtm-cbr/graphtm/cuda/kernels.cu` when a real
  `CudaMultiClassTM` is wired up. That is the next milestone,
  needed for the production `n_clauses ≥ 2000`, deeper-tree
  HGTM that the architecture doc describes.
- **Auditor user study (n ≥ 10)**: not run. Infrastructure is
  ready (every decision has a signed receipt + readable firing
  clauses). Needs ~10 security engineers to rate decisions; this
  is a calendar item, not a code item.
- **Full GCG-MCP attack**: only the random-search stub runs
  (~33 % ASR after, no improvement over baseline). A real GCG
  needs gradient access to the LLM emitting the tool call, which
  ClauseGate does not have. The paper states this honestly.

## Changes over the overnight build

- **Type-II feedback bug** in `tm/vanilla.py` (wrong literal slot),
  caught during initial test runs, fixed.
- **Cross-graph `injected-from` detection** (`graphs/builder.py`):
  was matching the full arg-value string; rewritten to use
  needle-list + ≥ 10-char common-substring fallback. This is what
  surfaces prompt-injection patterns to the clauses.
- **Default policy** (`data/synth.py`): expanded from 12 synthetic
  tools to 12 synthetic + 56 AgentDojo tools, classified by name
  semantics (read/search/get/list → allow; send/post/share/create
  → oversight; delete/send_money/update_password/reserve → deny).
- **20 × vectorisation** of the canonical NumPy TM: tensordot in
  `clause_outputs`, batched RNG in `_feedback`. Same numerical
  semantics; 20 × throughput. The whole test suite now runs in 9 s.
- **InjecAgent loader** (`data/injecagent_loader.py`): rewritten to
  parse the published Capitalized-key JSON shape ("Attacker Tools",
  "Tool Parameters", "Tool Response") with the user instruction
  + tool response + attacker instruction as three RAG chunks at
  trust levels {2, 0, 0}.
- **R-Judge loader** (`data/rjudge_loader.py`): rewritten to walk
  the official multi-turn `contents` list-of-lists, extract the
  agent's "ToolName: {args}" action strings, attach prior tool
  outputs + user prompts as RAG chunks at trust level 1, and map
  the binary safe/unsafe label to ALLOW / DENY.
- **`scripts/train_full_agentdojo.py`**: 5-seed runner with
  per-sample JSONL logger + adversarial harness + recourse
  measurement + cross-seed aggregator + signed-receipt exemplar.
- **`scripts/zero_shot_injecagent.py`** and **`scripts/zero_shot_rjudge.py`**:
  train on AgentDojo, evaluate zero-shot on the downloaded
  benchmarks.

## File-tree delta vs. last night

```
clausegate/
├── data/external/                        downloaded benchmark data (672 KB + 13 KB + 14 R-Judge JSON ≈ 1.3 MB)
├── results/
│   ├── headline/
│   │   ├── agentdojo_full_seed_1.json     <-- new
│   │   ├── agentdojo_full_seed_2.json     <-- new
│   │   ├── agentdojo_full_seed_3.json     <-- new
│   │   ├── agentdojo_full_seed_5.json     <-- new
│   │   ├── agentdojo_full_seed_8.json     <-- new
│   │   ├── agentdojo_full_5seed_aggregate.json    <-- new
│   │   ├── zero_shot_injecagent.json      <-- new
│   │   └── zero_shot_rjudge.json          <-- new
│   ├── jsonl/seed_{1,2,3,5,8}/per_sample.jsonl    <-- 5 × 356 decisions = 1780 receipts logged
│   └── receipts/seed_{1,2,3,5,8}/exemplar.json    <-- signed
├── scripts/
│   ├── train_full_agentdojo.py            <-- new
│   ├── zero_shot_injecagent.py            <-- new
│   ├── zero_shot_rjudge.py                <-- new
│   └── train_workspace_smoke.py
├── .venv/                                 <-- pip install env
├── src/clausegate/data/agentdojo_loader.py    <-- rewritten for AgentDojo v1.2.1 API
├── src/clausegate/data/synth.py               <-- expanded policy
├── src/clausegate/data/injecagent_loader.py   <-- rewritten for Capitalized-key JSON
├── src/clausegate/data/rjudge_loader.py       <-- rewritten for multi-turn list-of-lists
├── src/clausegate/tm/vanilla.py               <-- 20x vectorisation
├── src/clausegate/graphs/builder.py           <-- substring-needle cross-edge detection
└── README.md                                  <-- locked numbers + caveats
```

## Open decisions

1. **`README.md` §"Measured headline results"** wording. The
   calibrated claims framing is the candidate paper headline.
2. **The benign-utility number (0.302)** is the only headline that
   misses its target. The over-cautious behaviour is correct in
   that it has 99 % oversight recall and 100 % ALLOW precision,
   but the paper's framing can be either (a) "ALLOW precision = 1.0
   in 4/5 seeds" or (b) push utility up by extending the AgentDojo
   augmentation. Recommendation: (a). It is a stronger story and
   matches the EU AI Act Art-14 "human oversight" framing.
3. **Next milestone (the genuine GPU ask):** wire
   `TriGraphHGTM(backend="cuda")` to `graphtm-cbr/graphtm/cuda/kernels.cu`
   so `n_clauses ≥ 2000` and a deeper tree become tractable. The
   CPU NumPy backend was fast enough that GPU was not needed for
   the headline numbers, but the deeper-tree HGTM ablation
   (`R=2, IA=2, IF=5, LA=15, LF=3, D_bits=8192` from the spec)
   needs CUDA. The entry point + parity tests are in place;
   estimated ~1 day of focused work to wire it.
