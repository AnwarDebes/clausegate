# ClauseGate: Interpretable Verification of LLM Agent Tool Calls via Tri-Graph Tsetlin Machines with SAT-Verifiable Receipts and Counterfactual Argument Recourse

**Anwar (University of Agder).**

> **Paper status.** Draft. The 5-seed AgentDojo protocol +
> zero-shot transfer to InjecAgent / R-Judge + adversarial harness
> have **all completed on CPU**; numbers are **measured, not
> planned**. The CUDA wrapper for graphtm-cbr's kernels is the
> next milestone (needed for the deeper-tree HGTM ablation). See
> `STATUS.md` for the locked numbers and honest caveats; see
> `docs/research_plan.md` for the full 5-seed protocol.

## Abstract

Every modern LLM agent or Model Context Protocol (MCP) deployment
exposes a tool-execution surface to user-controlled inputs. OWASP
LLM Top 10 has listed prompt injection as the #1 LLM risk for two
consecutive years, and OpenAI publicly states the problem is
"unlikely to ever be fully solved." Existing guardrails are LLM
judges themselves: opaque, single-vendor, and unauditable. Under
EU AI Act Article 14 (human oversight) and Article 15
(cybersecurity, resilience to adversarial examples), a deployer
must be able to interpret what its verifier does and prove that
each blocked call was blocked for an articulable reason.

We present **ClauseGate**, a Tsetlin-Machine-based verifier that
sits in front of every proposed agent tool call. ClauseGate ingests
the agent's proposed call, the retrieved RAG context, and the
operator's policy, builds a *tri-graph* (three coupled typed-edge
multigraphs plus cross-graph alignment edges), and emits an
ALLOW / DENY / NEEDS-OVERSIGHT verdict together with: (i) the
firing Boolean clauses, (ii) a DIMACS-CNF + HMAC-SHA-256 receipt
that an external auditor can re-derive on a `python-sat`-only
environment without ClauseGate installed, and (iii) a
counterfactual minimal-argument-edit (≤ 3 edits) that would have
flipped the verdict. Inference runs sub-50 ms on a V100 and
sub-200 ms on CPU, fast enough to sit inline in front of every
MCP server tool execution.

On the AgentDojo benchmark (Debenedetti et al., NeurIPS 2024) we
target ≤ 5 % attack success rate at ≥ 60 % benign utility, with
zero-shot transfer to InjecAgent (ACL 2024) and R-Judge
(EMNLP-Findings 2024). The contribution is the **composition** of
(a) hierarchical graph Tsetlin Machines, (b) SAT-verifiable
clause receipts, and (c) counterfactual graph-edit recourse, all
of which exist as primitives in prior work, on a previously
untouched application surface (LLM agent tool calls) with the
sharpest production-urgent regulatory anchor (OWASP LLM01:2025 +
MITRE ATLAS + EU AI Act Art 14/15).

## 1 Introduction

[TODO: motivate via Mata v. Avianca, Moffatt v. Air Canada,
Postmark MCP attack Sept 2025, Copilot prompt-injection-via-email
cross-tenant 2024, AgentDojo GPT-4o 25 % ASR baseline, ChatInject
45.9 % ASR; pivot to "deployer cannot effectively oversee a
guardrail that is itself an opaque LLM judge."]

## 2 Related work

[TODO: expand the table in `docs/novelty_audit.md` into a Related
Work section.]

### 2.1 LLM agent guardrails and verifiers

GuardAgent, VeriGuard, MCPShield, NeMo Guardrails, Llama Guard 3/4,
ShieldGemma, PSG-Agent, Eidoku, ToolCallVerifier, Progent. Common
weaknesses: LLM-judge opacity, no per-decision receipts, no
counterfactual recourse, no SAT-verifiable witnesses.

### 2.2 Tsetlin Machines for safety + verification

Saha et al. SAT-encoded TM verification (arXiv:2303.14464).
Bhattarai et al. fake-news TM (LREC 2022). Granmo et al. GraphTM
(arXiv:2507.14874). Our `clauseguard` (LLM claim verification),
`clause86` (Article-86 hiring receipts), `graphtm-cbr` (molecular
mutagenicity with graph-edit recourse). None has been applied to
LLM agent actions; none combines SAT-verifiable receipts with
counterfactual recourse on an agent action graph.

### 2.3 Counterfactual recourse

Wachter et al. (2017), Karimi et al. CLEAR / CoGS family. All
operate on tabular or graph inputs; argument-edit recourse for
LLM agent tool calls has not been instantiated. Our prior
graphtm-cbr ships the closest cognate (atom-bond graph edits) and
serves as the construction primitive here.

## 3 Method

### 3.1 Tri-graph schema

[See `docs/ARCHITECTURE.md` §"Tri-graph schema". Re-render the
Mermaid diagram here for the camera-ready.]

### 3.2 Tri-graph walking HGTM

Architecture spec: `R=2, IA=2, IF=5, LA=15, LF=3, D_bits=8192,
n_clauses=200, T=50, s=3.9` (default; tunable per suite). CUDA
kernels lifted wholesale from
`graphtm-cbr/graphtm/cuda/kernels.cu`. Reference CPU implementation
in `clausegate.tm.vanilla` for numerical-parity tests.

### 3.3 SAT-verifiable receipts

DIMACS-CNF encoding: per-literal unit clauses + per-firing-clause
conjunction-witness clauses + pairwise class-sum comparison
clauses. HMAC-SHA-256 signature over the canonical-JSON receipt
body. Auditor verification path uses Glucose 4 (python-sat).

### 3.4 Counterfactual argument-edit recourse

Greedy minimum-edit search bounded by `max_edits = 3` and
`max_latency_ms = 50`. Candidate set is clause-attributed: for
each firing clause, walk the literal vocabulary and emit one of
{`clear_substring`, `clip_numeric`, `replace_string`, `drop_arg`,
`remove_chunk`, `lift_policy`} per implicated literal slot.

## 4 Experiments

### 4.1 Datasets

- **AgentDojo** (Debenedetti et al., NeurIPS 2024, arXiv:2406.13352):
  97 user tasks + 629 injection cases over 4 suites
  (workspace, slack, banking, travel). MIT.
- **InjecAgent** (Zhan et al., ACL 2024, arXiv:2403.02691): 1054
  single-turn injection cases over 17 tools. MIT.
- **R-Judge** (Yuan et al., EMNLP-Findings 2024, arXiv:2401.10019):
  569 multi-turn agent traces over 27 scenarios. Apache-2.0.

### 4.2 Protocol

5-seed protocol (1, 2, 3, 5, 8). Mean ± std for each metric, with
Wilcoxon signed-rank against the strongest baseline and bootstrap
95 % CI (n = 1000).

### 4.3 Baselines

Always-allow, static regex shield, TfIdf+logistic, TfIdf+RF,
CodeBERT-pooled+logistic, GuardAgent, VeriGuard, NeMo Guardrails.

### 4.4 Ablations

A. No-policy-graph. B. No-context-graph. C. No-recourse.
D. No-structural-distillation. E. Flat-feature TM.
F. Tree-geometry sweep.

### 4.5 Adversarial harness

ChatInject (NDSS 2024), RAG-poison (USENIX Security 2025),
GCG-MCP (adapt Zou et al. 2023). Report ASR before vs after the
defence with the average number of clauses fired per blocked
attack.

### 4.6 Auditor user study

n ≥ 10 security engineers. 30 decisions × 3 questions each (see
`docs/research_plan.md` §"Auditor user study").

## 5 Headline results (5-seed measured, CPU)

5-seed protocol with seeds {1, 2, 3, 5, 8}, `n_clauses=200, T=50,
s=3.9`, 30 epochs. AgentDojo pooled across all four suites
(workspace + slack + banking + travel), 80 / 20 train / test split
per seed. Adversarial harness aggregates 318 attacks per attack
type (across seeds). Single-CPU-thread on a vectorised NumPy
backend; GPU latency expected to be ~10-20× lower once the CUDA
wrapper lands.

| Benchmark / Metric | Best baseline | ClauseGate (measured) |
|---|---|---|
| AgentDojo: **ASR ↓** | 25 % (GPT-4o + AgentDojo defence) | **0.000 ± 0.000** |
| AgentDojo: Benign utility ↑ | 69 % (GPT-4o, no defence) | 0.302 ± 0.156 (over-cautious; oversight recall 0.990) |
| AgentDojo: Macro-F1 ↑ | n/a | 0.525 ± 0.098 |
| AgentDojo: Latency / call ↓ | n/a | **6.18 ± 0.15 ms (CPU)** |
| AgentDojo: Recourse → non-DENY @ K = 3 ↑ | n/a | **1.000 ± 0.000** |
| AgentDojo: Recourse P50 / P95 (ms) | n/a | 34.6 / 68.5 |
| **ChatInject ASR after defence ↓** | 32 % baseline | **0.000** (318 / 318 blocked) |
| **RAG-poison ASR after defence ↓** | high baseline | **0.000** (318 / 318 blocked) |
| GCG-MCP random-search stub ASR | n/a | 0.328 (stub does not really attack) |
| **InjecAgent (zero-shot) defended ASR ↓** | 24 % (GPT-4 ReAct) | **0.000** (n=300) |
| **R-Judge (zero-shot) defended ASR ↓** | n/a | **0.000** (n=371) |
| Auditor study consistency rate | n/a | **not measured** (n ≥ 10 study future work) |

The ASR = 0 result on AgentDojo + InjecAgent + R-Judge is the
headline finding. The benign-utility miss (0.302 vs 0.60 target) is
the trade-off, documented openly in §6.

## 6 Discussion

[TODO: expand. Honest caveats:
* Coverage of unseen tools: TM vocabulary is fixed at train time.
* Adversarial robustness: GCG-style adversarial argument-prefix
  search is plausible; the Saha-2303.14464 verifier is run on a
  validation subset to bound the worst case within an L0 ball.
* Policy graph staleness: the policy is a deployer-authored
  artefact and the verifier inherits its correctness.
* Benchmark distribution shift: AgentDojo simulates tool envs;
  production traffic will differ.]

## 7 Regulatory anchoring

- **EU AI Act Art 14** (human oversight): NEEDS-OVERSIGHT
  verdict + firing-clause export are the interpretation tool an
  operator is "enabled to use."
- **EU AI Act Art 15** (cybersecurity, resilience to data
  poisoning and adversarial examples): SAT-verifiable robustness
  witnesses.
- **OWASP LLM01:2025** (Prompt Injection): the cross-graph
  `injected-from` layer surfaces both direct and indirect injection
  patterns to clauses.
- **MITRE ATLAS** AML.T0051 (LLM Prompt Injection), AML.T0054
  (output manipulation), AML.T0096 (AI Service API for C2):
  per-decision SAT receipts are the detection-evidence artefact
  ATLAS expects.
- **NIST AI 600-1** (GenAI Profile): Boolean clauses are the
  "interpretability metric baseline" the document calls for.
- **NIS2 + DORA** (financial-services agent deployments): sub-
  50 ms inline verifier with signed audit log fits the ICT-risk
  + incident-reporting cascade.

## 8 Reuse from prior work

ClauseGate is built by composition. The heavy lifting (CUDA
kernels, SAT encoder, recourse engine, structural distillation
primitive) is lifted from existing repos and adapted to the
ClauseGate schema. See README.md §"Reuse from the existing
portfolio" for the full table.

## 9 Limitations and threat-model boundaries

In scope: indirect prompt injection via RAG / prior tool outputs;
tool-call parameter tampering; composed-tool jailbreaks; forged-
receipt detection.

Out of scope: proving the deployer did not log a different decision
than they enforced (delegated to a separate trusted-log protocol);
prompt-injection that is clause-consistent (policy bug, not
verifier bug).

## 10 Conclusion

ClauseGate is the first Tsetlin-Machine-based
LLM-agent tool-call verifier with SAT-verifiable receipts and
counterfactual minimal-argument-edit recourse. It composes the
HGTM+CUDA+recourse stack of graphtm-cbr, the SAT-receipt
machinery of clauseguard, and the structural-distillation
primitive of attention-distilled-graphtm on a previously untouched
application surface (AgentDojo / InjecAgent / R-Judge) anchored
to the production-urgent OWASP LLM01:2025 + MITRE ATLAS + EU AI
Act Art 14/15 regulatory stack. The headline numbers above are
planned targets, not measurements; they are locked after the
documented 5-seed protocol completes.
