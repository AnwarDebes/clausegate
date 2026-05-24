# ClauseGate: Research Plan

This is the executable plan for taking ClauseGate from "code
complete on synthetic + small AgentDojo subset" to the paper's
headline result table. Every entry has an owner (Anwar),
a calibrated target, and an honest caveat.

## North-star claims (the paper headline)

H1. **Faithful Boolean verification.** ClauseGate matches a strong
LLM-based agent guard on Attack Success Rate (ASR) and utility on
AgentDojo while emitting human-readable Boolean clauses for every
allow/deny/oversight decision. *Calibrated target:* match or beat
the LLM-judge baseline reported in AgentDojo (Debenedetti 2024,
table 5) on average ASR across 5 seeds; honest caveat: TM may
underperform on the *utility* axis (subset of benign tasks
incorrectly denied) and we report the trade-off transparently.

H2. **Sub-50 ms verification.** End-to-end forward + receipt build
on a single proposed tool call ≤ 50 ms on one V100 (P95) and ≤
200 ms on CPU (P95). *Calibrated target:* P50 ≤ 20 ms / P95 ≤ 50
ms on V100; P50 ≤ 80 ms / P95 ≤ 200 ms on CPU.

H3. **SAT-verifiable receipts.** Every decision ships a
DIMACS-CNF + HMAC-SHA-256 receipt; an auditor with `python-sat`
alone can re-derive the recorded verdict from the literal vector,
firing-clause set, and clause weights without ClauseGate
installed. *Calibrated target:* SAT receipt produced for 100 % of
decisions; solve time per receipt ≤ 50 ms median.

H4. **Useful counterfactual recourse.** For ≥ 90 % of DENY
verdicts on AgentDojo's injected-prompt cases, ClauseGate emits a
≤ 3-edit argument set that, when applied to the proposed tool
call, flips the verdict to ALLOW under the same model. *Calibrated
target:* recourse success ≥ 0.90 at K = 3 edits; honest caveat: a
high success rate on adversarial cases is partly a property of the
attack distribution (small perturbations sit close to the decision
boundary), and we report both success rate and edit *meaning*
(injected-string removal vs argument-clipping).

H5. **Zero-shot transfer.** A ClauseGate trained on AgentDojo
generalises to InjecAgent and R-Judge without fine-tuning, within
a published margin. *Calibrated target:* zero-shot ASR on
InjecAgent within 10 pp of the in-domain transformer baseline; on
R-Judge within 15 pp.

These are **targets, not measurements**. Lock the headline table
only after the 5-seed protocol below completes.

## 5-seed reproducibility protocol

Same as graphtm-cbr / clauseguard:

- **Seeds**: `{1, 2, 3, 5, 8}` for: data-split, model init, training
  iteration order, dropout (if any), recourse search tie-break.
- **Splits**: AgentDojo's published train / val / test, applied
  per-suite (workspace, slack, banking, travel); never pooled
  across suites for the headline numbers.
- **Reporting**: mean ± std for each metric, with Wilcoxon
  signed-rank against the strongest baseline (LLM-judge AgentDojo
  defence) and bootstrap 95 % CI (n = 1000) for delta metrics.
- **Logging**: every per-sample decision (input hash, verdict,
  firing-clause ids, recourse latency, recourse outcome) is
  appended to `results/jsonl/seed_{seed}/per_sample.jsonl`,
  same JSONL pattern as graphtm-cbr.

## Timeline

| Week | Milestone | Output |
|---|---|---|
| W1 | Frozen architecture + interfaces + novelty audit | this repo state on `pyproject.toml` v0.1.0 |
| W1-W2 | All 8 modules CPU-pure-Python, unit + integration tests green | `pytest -q` exits 0 |
| W2 | Synthetic stress tests (1000 scenarios) for graph builder, recourse, receipt | `tests/test_latency.py`, `tests/test_recourse_stress.py` |
| W3 | AgentDojo loader integrated; smoke train on workspace suite | `scripts/train_workspace_smoke.py` finishes |
| W3-W4 | CUDA path validated against CPU reference | `tests/test_cuda_parity.py` green on V100 |
| W4-W6 | 5-seed full training on AgentDojo (workspace + slack + banking + travel) | `results/headline/*.json` per seed |
| W6 | Zero-shot eval on InjecAgent + R-Judge | `results/zero_shot/*.json` |
| W7 | Adversarial robustness eval (ChatInject, GCG-MCP, RAG-poison) | `results/robustness/*.json` |
| W7 | Auditor user study (n ≥ 10 security engineers) | `results/audit_study/*.md` |
| W8 | Paper draft, figures, supplementary | `PAPER.md`, `paper.tex`, `figures/` |

## Compute budget

Per-seed wall-clock estimates on a single Tesla V100-SXM3-32GB:

| Task | Wall-clock per seed | × 5 seeds |
|---|---|---|
| AgentDojo train + validate (4 suites pooled) | ~12 GPU-hours | 60 GPU-hours |
| Zero-shot eval (InjecAgent + R-Judge) | ~1 GPU-hour | 5 GPU-hours |
| Adversarial harness | ~3 GPU-hours | 15 GPU-hours |
| Ablations (no-policy-graph, no-context-graph, no-recourse, no-distill) | ~6 GPU-hours per ablation × 4 | 30 GPU-hours per ablation seed |

Headline total: ~80 GPU-hours per seed × 5 seeds = **~400 V100-hours**,
or **~2 weeks wall-clock** if seeds run serially. With four-way
ablations the full study fits inside ~3-4 weeks. Substantially
below clauseguard (1 week per seed) because the AgentDojo training
set is small (≤ 10k decisions per suite vs FEVER's 145k).

## Datasets & licences

- **AgentDojo** (Debenedetti et al., NeurIPS 2024, arXiv
  2406.13352). MIT licence. `pip install agentdojo`. Used for
  primary training + eval.
- **InjecAgent** (Zhan et al., ACL 2024, arXiv 2403.02691). MIT
  licence. Zero-shot transfer only.
- **R-Judge** (Yuan et al., EMNLP-F 2024, arXiv 2401.10019).
  Apache-2.0 licence. Zero-shot transfer only.
- **Synthetic adversarial scenarios** generated by
  `clausegate.data.synth` for stress tests and ablations; permits
  CI runs without external dataset downloads.

No protected-attribute or personally identifying data is used. All
agent transcripts are public benchmark releases.

## Baselines (ordered from easiest to strongest)

1. **Always-allow**: sanity floor.
2. **Static regex shield**: keyword blocklist (the production
   default for many small deployers).
3. **TfIdf + logistic regression**: bag-of-tokens over the
   serialised tool call.
4. **TfIdf + Random Forest**: same features, non-linear baseline.
5. **CodeBERT / ToolBERT pooled embedding + logistic regression**
   on the serialised tool call: neural feature baseline.
6. **GuardAgent (LLM-judge over the tool call)**: the closest
   published agent-safety method.
7. **VeriGuard (formally-verified-code-synthesis defence)**: the
   closest *verification*-style published method.
8. **NeMo Guardrails + Colang**: the closest industrial defence.

ClauseGate vs (1-5) demonstrates that learned Boolean clauses
beat the parametric baselines; ClauseGate vs (6-8) demonstrates
that signed Boolean receipts + recourse are achievable without
giving up accuracy.

## Ablations

A. No-policy-graph (action + context only): how much does
explicit policy injection help?
B. No-context-graph (action + policy only): how much does the
RAG-chunk graph contribute?
C. No-recourse: verdict only.
D. No-structural-distillation (M3.5 disabled).
E. Flat-feature TM (vanilla MultiClassTsetlinMachine) on
serialised tool call: does the graph buy anything over a flat
TM?
F. Vary tree geometry: `R ∈ {1, 2}`, `IA ∈ {2, 4}`, `LF ∈ {3, 5}`
to confirm chosen spec.

## Adversarial harness (M7)

Three adapted attacks:

1. **ChatInject** (Liu et al., NDSS 2024): indirect prompt
   injection via RAG chunks. Baseline 32 % ASR on AgentDojo
   GPT-4o; target ASR ≤ 5 % with ClauseGate in front.
2. **GCG-MCP**: adapt GCG (Zou et al. 2023) to the MCP tool-call
   token space (search adversarial argument prefixes that maximise
   the LLM's likelihood of emitting a tool call ClauseGate
   normally denies).
3. **RAG-poison**: Liu et al., USENIX Security 2025: inject
   ≤ 5 documents into the RAG corpus to mislead the agent. Target:
   ClauseGate's verdict insensitive to ≤ 5 poisoned docs at the
   `injected-from` cross-graph edge layer.

Per-attack: report ASR before defence vs after defence, with the
average number of clauses fired per blocked attack (so the auditor
can see *which* rule caught it).

## Auditor user study (n ≥ 10 security engineers)

Following the clause86 protocol:
- Show each auditor 30 ClauseGate decisions (10 ALLOW, 10 DENY, 10
  NEEDS-OVERSIGHT), with the firing clause set and the SAT
  receipt;
- Ask: (i) is the receipt's verdict consistent with the firing
  clauses you can read? (ii) would you trust this verifier in a
  CI gate? (iii) is the counterfactual edit meaningful for a
  blocked attack?
- Outcome metric: % of decisions rated "consistent and
  meaningful." Calibrated target: ≥ 80 %.

## Honest open risks

- **Coverage of unseen tools.** A TM clause vocabulary is fixed
  at training time. New tools not in the training graph will fall
  through to NEEDS-OVERSIGHT by design; this is correct, but the
  paper must measure how often that happens on real-world MCP
  deployments. The R-Judge zero-shot eval gives a first read.
- **Adversarial GCG-MCP attacks.** It is plausible that argument
  prefixes can be found that fire spurious ALLOW clauses. The
  Saha SAT-verification primitive (2303.14464) gives a path to
  *prove* robustness within an L0 ball, and we will run this on a
  validation subset to bound the worst case.
- **Policy graph staleness.** The policy is a deployer-authored
  artefact; the verifier is only as correct as the policy graph.
  ClauseGate produces a counterfactual for every blocked call,
  which is also a signal that the policy may need refinement.
- **AgentDojo distribution shift.** The benchmark uses simulated
  tool environments; real production traffic will differ. The
  paper must report this caveat alongside the headline numbers.
- **Receipt key management.** Out of scope for the paper; we ship
  the receipt format that a deployer can sign with their own HMAC
  key + rotate per Article-14 audit requirements.
