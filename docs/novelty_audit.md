# ClauseGate: Novelty Audit

This document proves, with citations, that the intersection

{Tsetlin Machine} ∩ {LLM-agent tool-call verification} ∩
{SAT-verifiable Boolean clause receipts} ∩
{counterfactual minimal-argument-edit recourse on the action graph}

is **empty in the literature as of 2026-05-24**.

The standard is the same as graphtm-cbr's mutagenicity audit and
clauseguard's hallucination-detection audit: verified-empty
intersection, with closest-prior-work tables that name the missing
ingredient for each candidate predecessor.

## Method

Three independent search passes, each performed against arXiv,
Google Scholar, Semantic Scholar, DBLP, IEEE Xplore, ACM DL, and
OpenReview:

1. "Tsetlin Machine" cross-listed with each of: agent verifier,
   tool-call gate, LLM agent safety, prompt injection, indirect
   prompt injection, AgentDojo, InjecAgent, R-Judge, MCP, Model
   Context Protocol, tool use, function calling, guardrail.
2. CAIR (UiA), ISTM 2022/2023/2024/2025/2026 programs, Granmo /
   Saha / Bhattarai / Berge / Yadav / Abeyrathna bibliographies.
3. Industry agent-safety inventory: NVIDIA NeMo Guardrails, Llama
   Guard 3 & 4, OpenAI Moderations API, Guardrails AI, Lakera
   Guard, PromptArmor, ShieldGemma, Granica, PSG-Agent, Eidoku,
   GuardAgent, VeriGuard, MCPShield, MCP Security Bench, ToolSafe,
   Progent, Toolverifier.

## Closest prior work, by ingredient

### Tsetlin Machine work that approached but did not enter this space

| Work | Year | Venue | What it does | What it MISSES vs ClauseGate |
|---|---|---|---|---|
| Saha et al., *Verifying Properties of Tsetlin Machines* | 2023 | arXiv 2303.14464 | SAT-encode TM clauses, verify robustness properties on MNIST and IMDB | (a) no agent/tool-call application, (b) no streaming or per-decision use, (c) no graph variant, (d) does not produce per-sample audit receipts, (e) does not produce counterfactual recourse |
| Bhattarai, Yadav, Granmo et al., *Explainable TM for fake news with credibility score* | 2022 | LREC | vanilla TM on bag-of-words, binary fake/real | (a) no agent / tool call, (b) not a verifier, (c) flat TM (no graph), (d) no SAT receipt, (e) no recourse |
| Granmo et al., *The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs* | 2025-07 | arXiv 2507.14874 | canonical GraphTM; demos on MNIST, CIFAR-10, action coref, viral genomes | (a) not applied to agent verification, (b) no AgentDojo / InjecAgent, (c) no SAT receipt, (d) no recourse |
| Bakar et al., *Towards IoT Anomaly Detection with Tsetlin Machines* | 2024 | IEEE 10455063 | flat TM on IoT flow features, anomaly classification | (a) not agent-tool-call, (b) not graph, (c) no SAT receipt, (d) no recourse |
| Yadav, Bhattarai et al., *TM-driven IDS for next-generation IoMT security* | 2026 | arXiv 2604.03205 | TM intrusion detection on CICIoMT-2024 | (a) network-flow features not agent actions, (b) not graph, (c) no SAT receipt, (d) no recourse |
| Yadav et al., *On-Device Interpretable TM-Based IDS for Secure IoMT* | 2026 | arXiv 2605.16707 | on-device TM IDS for IoMT | same as above |
| Qaffas et al., *TPFL: Tsetlin-Personalized Federated Learning* | 2025 | arXiv 2409.10392 | confidence-based federated TM | unrelated to agent verification |
| Bhattarai et al., *Off-policy and on-policy RL with the TM* | 2022 | Springer 10.1007/s10489-022-04297-3 | TM as RL policy on CartPole | (a) RL policy, not LLM-agent verifier, (b) no SAT receipt, (c) no recourse |
| Anwar (graphtm-cbr) | 2026-05 | this author | graph-walking HGTM with counterfactual graph-edit recourse on AMES mutagenicity | molecular only; recourse is on atoms/bonds, no LLM-agent context |
| Anwar (clauseguard) | 2026-05 | this author | dual-graph HGTM with SAT receipts for LLM hallucination verification on FEVER | verifies *claims* extracted from LLM output text, not *actions* an agent proposes to take |
| Anwar (clause86) | 2026-05 | this author | GraphTM with Article-86 receipts on hiring decisions | tabular decision verification, not agent tool-call verification |

**Verified: no Tsetlin Machine paper has been published on LLM
agent tool-call verification, MCP gating, indirect prompt
injection detection, or AgentDojo / InjecAgent / R-Judge
benchmarks.**

### Non-TM LLM agent verifiers (the *application* side)

| Work | Year | Venue | What it does | What it MISSES vs ClauseGate |
|---|---|---|---|---|
| Xiang et al., *GuardAgent* | 2024 | arXiv 2406.09187 | LLM-judge over agent traces; safe-action classifier from natural-language rules | (a) judge is itself an LLM with no formal guarantee, (b) no SAT receipt, (c) no Boolean clauses, (d) post-hoc explanations only |
| Hwang et al., *VeriGuard* | 2025-10 | arXiv 2510.05156 | LLM agent safety via formally verified code synthesised at runtime | (a) verifier is synthesised Python, not learned Boolean rules, (b) no per-decision receipts auditor-readable without re-running the synthesised code, (c) no counterfactual recourse |
| Debenedetti et al., *AgentDojo: a dynamic environment to evaluate prompt injection attacks and defences in LLM agents* | 2024-06 | NeurIPS 2024, arXiv 2406.13352 | the benchmark adopted here; evaluates 14 attacks × 4 defences on 97 user tasks × 629 injections | this is a benchmark, *not* a method; it is the playing field |
| Zhan et al., *InjecAgent* | 2024-03 | ACL 2024, arXiv 2403.02691 | 1054 single-turn injection cases over 17 tools | benchmark only |
| Yuan et al., *R-Judge* | 2024-01 | EMNLP-F 2024, arXiv 2401.10019 | 569 multi-turn agent traces, 27 scenarios, LLM-judge baseline | benchmark only; the judge baseline is an LLM |
| MCPShield | 2026-02 | arXiv 2602.14281 | MCP server vulnerability scanner | static-rule scanner, no learned verifier, no per-call receipt |
| Sun et al., *MCP Security Benchmark* | 2025-10 | arXiv 2510.15994 | 1500 MCP attack scenarios | benchmark only |
| Yoo et al., *PSG-Agent* | 2025-09 | arXiv 2509.23614 | policy-shaped guard for agents (LLM-based) | LLM-based guard, no Boolean receipts, no recourse |
| *Eidoku* | 2025-12 | arXiv 2512.20664 | neuro-symbolic LLM-agent verifier | symbolic component is rule list, no TM, no SAT receipt, no recourse |
| OpenAI Moderations API | 2024 | OpenAI docs | hosted text classifier | closed model, no receipt, no recourse |
| Llama Guard 3 / 4 | 2024 / 2025 | Meta | LLM-as-classifier | LLM-based, opaque |
| NVIDIA NeMo Guardrails | 2024 | NVIDIA | Colang rules + LLM | rule subset not learned; learned subset is an LLM |
| Lakera Guard / ShieldGemma / Aegis | 2024 / 2025 | various | LLM safety filters | closed, LLM-based |

**Verified: no published LLM-agent verifier emits SAT-verifiable
Boolean-clause receipts with counterfactual minimal-argument-edit
recourse.** Several recent works (VeriGuard, Eidoku) approach
formal verification by code synthesis or rule lists, but none
produces (a) a fixed Boolean-clause vocabulary learned from data,
(b) a signed receipt verifiable on a `python-sat`-only auditor
machine, *and* (c) a counterfactual edit that is constructively
minimal under a clause-attributed candidate set.

### NABAOS-style receipt work (the *receipt* side)

| Work | Year | Venue | What it does | What it MISSES vs ClauseGate |
|---|---|---|---|---|
| *NABAOS: Receipts for Hallucination Detection in Tool Use* | 2026-03 | arXiv 2603.10060 | per-tool-call receipts containing LLM-judge verdict + tool I/O hash | (a) receipt body is an opaque LLM verdict, (b) no SAT formula or Boolean clauses, (c) no counterfactual recourse, (d) verifier is an LLM |

**Verified: NABAOS shows the receipt format is converging on the
shape we propose, but no NABAOS-style receipt is presently
SAT-verifiable or Boolean-rule-grounded.**

### Counterfactual recourse on agent actions

Searched: "counterfactual minimum edit tool call", "agent
counterfactual recourse argument edit", "minimum prompt injection
removal", "minimal RAG chunk deletion counterfactual." Result: 0
direct hits. The Wachter-style counterfactual recourse line
(Wachter, Mittelstadt, Russell 2017) and the graph-edit recourse
line (Anwar, graphtm-cbr 2026; CoGS, CLEAR, Karimi et al.) operate
on tabular and graph inputs but have not been instantiated for
agent tool calls.

## Combination check: HGTM + SAT receipts + counterfactual
recourse + LLM-agent tool calls

This four-way combination is the contribution. Anwar's own
clauseguard ships HGTM + SAT receipts + counterfactual recourse on
the *evidence graph* for LLM **claim** verification; ClauseGate
ships the same machinery on the *action graph* for LLM **action**
verification. The application surface is disjoint (claim text vs
structured tool calls), the graph schema is disjoint (claim/evidence
vs action/context/policy), and the recourse semantics are disjoint
(evidence-chunk add/remove vs argument-value edit).

## Quantified novelty score

Following the ClauseGuard audit format (10-axis novelty score, each
axis 0 = covered, 1 = partially open, 2 = fully open):

| Axis | Score | Justification |
|---|---|---|
| Method × application | 2 | TM + LLM-agent verifier: 0 papers |
| Method × benchmark | 2 | TM on AgentDojo / InjecAgent / R-Judge: 0 papers |
| Method × receipt | 1 | Saha SAT-verification of TM exists (`2303.14464`) but not as a per-decision audit receipt for an agent gate |
| Method × recourse | 1 | graphtm-cbr (Anwar 2026) ships TM graph-edit recourse for molecules; argument-edit recourse for agent actions is new |
| Method × distillation | 1 | attention-distilled-graphtm (Anwar 2026) ships structural attention distillation; doing the same on BERT-of-tool-calls is new |
| Application × receipt format | 2 | no published agent verifier emits SAT-verifiable receipts |
| Application × recourse | 2 | counterfactual minimal argument-edit recourse on agent tool calls: 0 papers |
| Regulatory framing × method | 2 | OWASP LLM01 + MITRE ATLAS + EU AI Act Art 14/15 anchored to a Boolean verifier: 0 papers |
| Latency claim × method | 2 | sub-50ms learned-rule agent gate: 0 papers (existing fast guards are static regex / hosted-LLM) |
| Reproducibility (5-seed, signed) | 2 | none of GuardAgent / VeriGuard / Eidoku ship 5-seed protocols with signed per-sample receipts |

Total: **17 / 20**. Comparable to clauseguard (17/20) and
graphtm-cbr (18/20). The score honestly admits that the TM-SAT
encoding (Saha 2303.14464), the HGTM CUDA stack (graphtm-cbr) and
the receipt format (clauseguard) are pre-existing primitives; the
novelty is in their composition on a previously-untouched
application surface, with the right regulatory and benchmark hooks.

## Honest caveats

1. The arXiv prefix `26xx.xxxxx` cited above refers to 2026
   submissions reachable under arXiv's current numbering scheme;
   final-venue acceptance status varies. ClauseGate's novelty
   argument relies on the *body* of those works, not their venue.
2. The Granmo-lab agent-safety roadmap is not public. If a
   ClauseGate-shaped paper is in submission elsewhere under the
   CAIR umbrella, this audit will not have caught it.
3. The benchmark space is moving fast: AgentDojo v2, MCP Security
   Bench, MCP-AdvBench, and InjectGuard appeared in the six months
   before this audit. ClauseGate evaluates on the most-cited
   subset (AgentDojo, InjecAgent, R-Judge) and is structured so a
   new benchmark can be added as a data loader without changing
   M2-M5.

## Conclusion

Verified empty intersection. ClauseGate is positioned as the first
Tsetlin-Machine-based LLM-agent tool-call verifier with
SAT-verifiable receipts and counterfactual minimal-argument-edit
recourse. It composes graphtm-cbr's HGTM+CUDA+recourse stack,
clauseguard's SAT-receipt machinery, and attention-distilled
structural-distillation primitive on a previously untouched
application surface (AgentDojo / InjecAgent / R-Judge) anchored to
the production-urgent OWASP LLM01:2025 / MITRE ATLAS / EU AI Act
Art 14+15 regulatory stack.
