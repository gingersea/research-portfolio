# Recommended Reading for the Agentic AI Interview: Complete English Guide

Updated: 18 November 2026 · [中文版](reading-guide.zh.md) · [Index](README.md)

This guide prepares you for a research conversation: understand a question, explain how a method addresses it, and propose an experiment that could challenge your interpretation. Read NewtonBench, The Curse of CoT, LogiDynamics, AbsInstruct, and MMLongBench first. MemLens and AgentIdeaBench extend that foundation; LMPNN and COLA provide supporting background.

Each chapter separates **paper evidence** from **conceptual explanations, teaching examples, proposed experiments, and interview practice**. The latter are instructional contributions of this guide, not experiments reported by the papers. None of the proposed experiments has been run for this guide. Publication status and reading scope appear in [SOURCES.md](SOURCES.md).

## 1. NewtonBench: How can an agent discover a law through experiments?

**Source:** [NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](https://arxiv.org/html/2510.07172v3) · ICLR 2026 · Highest priority.

### Paper evidence: question, design, findings, and limits

NewtonBench changes familiar laws and lets agents query simulated systems to discover hidden relationships. Its 324 tasks span 12 physics domains; evaluation distinguishes symbolic structure from numerical fit. Complexity and noise make discovery harder. Code assistance sometimes reduces performance, with premature fitting offered as an explanation. These modified laws are stress tests, not newly established laws of nature.

### Concepts to understand

**Static fitting versus active exploration.** Static fitting starts with a fixed dataset. An active agent also chooses what to measure next. Its actions determine the information it receives, so experimental selection is itself part of the capability being evaluated.

**Identifiability.** Several models can agree on the observations already collected. Improving the fit to those observations does not necessarily resolve the ambiguity. You may need a different measurement range, intervention, or sensor.

**Exploration versus exploitation.** Exploration seeks observations that distinguish hypotheses. Exploitation refines a promising hypothesis, for example by estimating its parameters. Committing too early can lead to experiments that repeatedly confirm the same mistaken assumption.

**Symbolic versus numerical correctness.** A polynomial can approximate a relationship over a narrow interval without recovering its governing structure. Conversely, a structurally correct expression can predict poorly when its constants are badly estimated. Evaluate these separately.

### Teaching example: choosing the next measurement

Suppose two hypotheses are $F=k/r^2$ and $F=k/r^3$, each calibrated at a reference distance. Doubling the distance predicts one-quarter versus one-eighth of the reference force. That measurement is more discriminative than repeatedly measuring close to the reference point.

However, moving too far away may make the force smaller than the instrument can measure reliably. A useful experiment balances predicted disagreement, measurement uncertainty, and cost. This is an instructional example, not a reported NewtonBench result.

### Proposed experiment and evaluation

Compare random sampling, a uniform grid, and hypothesis-disagreement sampling. Fix initial observations, tool access, and the number of experiments. Across noise levels, measure formula recovery, prediction error on unseen inputs, and experiments required.

Use the same fitting procedure first, so selection quality is not confounded with fitter quality. Then hold observations fixed to compare fitters. Repeat stochastic runs and report uncertainty rather than selecting the best run.

### Interview practice and portfolio connection

**Question:** Why might additional tools reduce an agent's performance?

**Answer outline:** Tools enlarge the action space without guaranteeing better decisions. Check whether the agent spends its budget optimizing an incorrect hypothesis, lacks falsifying experiments, or fails to detect tool errors.

**Practice statement:** “I would separate hypothesis generation, experiment selection, parameter estimation, and evidence verification. My video project also taught me that observations must be reliable before they can support a physical conclusion.”

Do not present physical-consistency assessment as completed autonomous scientific discovery. Discovery additionally requires active experimentation and hypothesis search.

## 2. The Curse of CoT: Why can an explicit explanation still fail?

**Source:** [The Curse of CoT: On the Limitations of Chain-of-Thought in In-Context Learning](https://arxiv.org/html/2504.05081v2) · TMLR · Highest priority.

### Paper evidence: question, design, findings, and limits

This study compares direct answering and CoT on pattern-based ICL. Its diagnostic controls examine intermediate text, contextual distance, and rule inference versus execution. Incorrect rationales can accompany correct answers. The findings do not establish that CoT fails on every reasoning task; the authors' account is not a complete identification of internal model computation.

### Concepts to understand

**ICL versus training.** In-context learning uses demonstrations in the current prompt, usually without changing model weights. Supervised fine-tuning changes weights through optimization. Giving a model examples is not the same intervention as training it on those examples.

**Rule discovery versus execution.** Inferring a transformation from examples differs from applying a supplied transformation. Separating these stages makes failure diagnosis more informative.

**Correctness versus faithfulness.** Correctness asks whether an answer matches the target. Faithfulness asks whether an explanation reflects what actually determined the answer. Plausible text alone cannot establish the latter.

**CoT versus ReAct.** CoT organizes intermediate reasoning text. ReAct also interleaves actions and environmental observations. External feedback can help correct errors, but only if the feedback is informative and interpreted appropriately.

### Teaching example: demonstrations may not determine a unique rule

The examples $1\mapsto2$ and $2\mapsto4$ fit both $f(x)=2x$ and $g(x)=2x+(x-1)(x-2)$. At $x=3$, the predictions are 6 and 8.

A fluent explanation saying “multiply by two” therefore does not prove unique rule identification. Ask whether the task provides enough evidence, whether several hypotheses should remain plausible, and whether another example could resolve the ambiguity.

### Proposed experiment and evaluation

Compare direct answering, free-form explanation, execution with the correct rule supplied, and additional text of comparable length but no task information. First measure natural operating costs, then include budget-matched comparisons. Record answer accuracy, inferred-rule correctness, execution accuracy with an oracle rule, and generation cost.

If supplying the correct rule helps, investigate rule discovery. If it does not, inspect execution, formatting, or perception. A length control can isolate some interference, but it cannot by itself establish a complete cognitive mechanism.

### Interview practice and portfolio connection

**Question:** How would you show that a reasoning component actually contributes?

**Answer outline:** Specify the observable stage it should affect. Remove it, replace it with a simple equal-cost alternative, and supply an ideal intermediate result. Interpret each intervention separately.

Your optimization project offers a useful analogy: a component appearing in code is different from it actually affecting an update. Similarly, an explanation appearing in an output does not establish that it improved the decision.

## 3. LogiDynamics: How do induction, abduction, and deduction work together?

**Source:** [LogiDynamics: Unraveling the Dynamics of Inductive, Abductive and Deductive Logical Inferences in LLM Reasoning](https://aclanthology.org/2025.emnlp-main.1045/) · EMNLP 2025 · Highest priority.

### Paper evidence: question, design, findings, and limits

LogiDynamics compares direct induction with abductive–deductive pipelines in controlled analogy tasks, varying modality, difficulty, and answer format. Staged reasoning helps in some difficult, visual, or symbolic settings, but its advantage changes with conditions. The paper's System 1/2 labels describe operational pipelines, not demonstrated human-like cognitive systems inside a model.

### Concepts to understand

| Inference | Starting point and objective | Physical-observation example |
|---|---|---|
| Induction | Instances → a general relationship | Summarize repeated trajectories as approximately uniformly accelerated |
| Abduction | An observation → possible explanations | A trajectory change could reflect collision, force, or tracking error |
| Deduction | A rule and conditions → a prediction | Derive consequences assuming an elastic collision occurred |

Abduction does not guarantee a unique explanation. Deduction depends on the validity of its premises. Induction requires care when generalizing from limited samples. These processes can form a cycle in which each constrains the others.

### Teaching example: a plausible explanation needs a discriminating prediction

A ball appears to reverse direction. Candidate explanations include a collision, camera motion, and an identity switch in the tracker. Choosing the most familiar story can overlook observation failure.

Derive different predictions: a collision should have a contact event; camera motion may affect background points; an identity switch may coincide with an appearance discontinuity. Search for evidence that distinguishes these explanations instead of merely producing a coherent narrative.

### Proposed experiment and evaluation

Express the same underlying rules in text, images, and symbols. Compare direct answers, hypothesis-then-execution, and hypotheses with an independent verification stage. Keep available information as similar as possible and separately label perception, hypothesis, and execution failures.

Provide an oracle symbolic state as an additional control for visual tasks. Check multiple-choice options for unintended clues. For free-form answers, account for equivalent expressions so formatting differences do not masquerade as reasoning failures.

### Interview practice and portfolio connection

**Question:** Does the usefulness of staged reasoning contradict evidence that CoT can hurt?

**Answer outline:** Not necessarily. Tasks, pipeline design, hypothesis filtering, and compute budgets differ. The research question is which conditions determine the benefit.

**Practice statement:** “I would maintain several explanations and derive predictions that distinguish them. In video analysis, tracking failure should remain a candidate explanation rather than immediately concluding that physics was violated.”

## 4. AbsInstruct: How can fine-tuning improve abstraction?

**Source:** [AbsInstruct: Eliciting Abstraction Ability from LLMs through Explanation Tuning with Plausibility Estimation](https://aclanthology.org/2024.acl-long.55/) · ACL 2024 · High priority.

### Paper evidence: question, design, findings, and limits

AbsInstruct combines explanation-bearing abstraction instructions, plausibility-based selection, and general instructions for fine-tuning, using LoRA in its experiments. It reports gains on evaluated abstraction and generalization tasks while preserving general instruction following. It is not a new LoRA optimization algorithm; model-assessed plausibility should not be equated with objective truth.

### Concepts to understand

**Abstraction is more than choosing a broader word.** Moving from “opening a door with a key” to “using a tool to accomplish a goal” preserves useful structure while discarding incidental details. An abstraction can be too broad to guide action or too specific to transfer.

**Explanation supervision.** A training target may contain an answer and its explanation. Because explanations add supervised tokens, an experiment should distinguish the value of their content from the effect of additional training.

**LoRA.** For a linear layer:

$$W=W_0+\frac{\alpha}{r}BA$$

Here $W_0\in\mathbb{R}^{d_{out}\times d_{in}}$ is frozen, while $B\in\mathbb{R}^{d_{out}\times r}$ and $A\in\mathbb{R}^{r\times d_{in}}$ are trainable. The added parameter count is $r(d_{out}+d_{in})$, compared with $d_{out}d_{in}$ entries in the full matrix. Actual memory and speed also depend on activations, sequence length, and implementation.

### Teaching example: abstraction versus lexical association

Train on opening a door with a key and tightening a screw with a screwdriver. Test on lifting a bucket from a well with a rope. This asks whether the model transfers a tool–goal relationship.

Add a negative example, such as walking while holding a key, so the presence of a tool noun does not suffice. These are illustrative examples, not samples or results attributed to AbsInstruct.

### Proposed experiment and evaluation

Compare answer-only supervision, answers plus explanations, filtered explanations, and a mixture with general instructions. Fix the base model and adaptation method. Run both example-matched and training-token-matched comparisons.

Split evaluation by concepts or relations to reduce lexical overlap. Test the target skill, general instruction following, abstention when evidence is absent, and transfer to unseen tasks. Record loss masking, padding, seeds, learning rates, and training time.

### Interview practice and portfolio connection

**Question:** Are an optimizer and LoRA alternative choices on the same axis?

**Answer outline:** No. LoRA defines an update parameterization; the optimizer defines how trainable parameters change. A parameterization-by-optimizer experiment can study their interaction under controlled data and compute budgets.

**Practice statement:** “My optimization project taught me that update implementation and evaluation settings affect attribution. For agent fine-tuning, I would measure task gains, retention of general capabilities, and appropriate uncertainty expression.”

Training on explanations and requesting explanations at inference time are different interventions. Their outcomes need not coincide.

## 5. MMLongBench: Does fitting information into context mean using it effectively?

**Source:** [MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](https://arxiv.org/html/2505.10610v3) · NeurIPS 2025 Datasets and Benchmarks · High priority.

### Paper evidence: question, design, findings, and limits

MMLongBench standardizes cross-modal context length across several task families, including visual retrieval, in-context learning, and summarization. Performance on one task is an inadequate proxy for overall long-context ability. Results concern evaluated settings, not guaranteed success for arbitrary long-video agents. MMLongBench and MMLongBench-Doc are different works.

### Concepts to understand

**Capacity versus effective use.** Accepting a long input does not establish that a model can locate every relevant fact or combine several facts correctly.

**A multimodal processing chain.** Visual information is encoded and connected to language representations. Failure can occur during detail recognition, evidence localization, cross-page association, or reasoning. End-to-end accuracy merges these causes.

**RAG.** Retrieval narrows the evidence available to the answering model. High recall does not ensure correct answers, and an accidentally correct answer does not prove appropriate evidence use.

**Resource costs.** Pairwise interactions in standard dense self-attention grow quadratically with sequence length. For a fixed architecture, inference KV storage grows with length. Implementation and architecture affect actual cost; maximum context length alone does not determine it.

### Teaching example: locating the failure

An agent reads equipment manuals and must identify a model's safety threshold. The answer may depend on a table heading, a footnote, and a curve's color. OCR may retain numbers but lose color; a summary may discard the threshold; retrieval may return a different device in the same product family.

These need different remedies: better perception, retained original evidence, or better retrieval. Asking the model to “think more carefully” cannot reliably reconstruct information that is unavailable.

### Proposed experiment and evaluation

Compare full original documents, OCR text, retrieve-then-answer, and oracle evidence. Vary evidence position, distractor volume, and the number of facts that must be combined.

Report retrieval recall, answer accuracy given correct evidence, end-to-end accuracy, latency, and cost. If performance remains poor with oracle evidence, investigate perception and reasoning rather than only tuning retrieval.

### Interview practice and portfolio connection

**Question:** Can a million-token context window replace RAG?

**Answer outline:** Compare quality, cost, latency, and evidence traceability for the task. The approaches can also be combined; capacity alone is not sufficient.

Apply the same diagnosis to the video project: check visibility, identity continuity, and coordinate reliability before interpreting physical behavior. Observation failure is not itself evidence of a physical violation.

## 6. MemLens: Why does an agent's memory lose fidelity?

**Source:** [MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models](https://arxiv.org/html/2605.14906v1) · 2026 preprint, work in progress · Recent extension.

### Paper evidence: question, design, findings, and limits

MemLens compares long-context models and memory agents on abilities including cross-session reasoning, temporal reasoning, updates, and abstention. Its analysis distinguishes evidence dilution in long contexts from visual information lost during memory compression. System adaptations and evaluation samples matter; the results do not establish that every memory architecture is inferior to raw context.

### Concepts to understand

**Memory is a pipeline.** Writing selects what to retain, storage determines representation, retrieval selects what to recover, updating handles obsolete information, and answering uses the evidence. Each stage can fail.

**Lossy compression.** If an image contains four cups but its summary says only “cups on a table,” retrieving that summary perfectly still cannot answer a later question about the count.

**Temporal validity and provenance.** “The device was in room A on Monday” and “it moved to room B on Tuesday” should not be merged into simultaneous locations. Preserve time, identity, and sources for current-state and historical queries.

**Abstention and calibration.** Admitting insufficient evidence is part of competence. Distinguish information never supplied, information not retrieved, and information discarded during summarization.

### Teaching example and proposed experiment

Create several sessions: first show a device and its serial number, later report a repair, then replace the device. Ask for the current serial number, the old device's repair history, and a purchase date that was never provided.

Compare text summaries, retrieval with original images, structured memory with time and provenance, and direct history access. Hold the answering model fixed. Measure retrieval success, update correctness, obsolete-information use, and unsupported answers.

Supply the correct original image in an additional condition to distinguish a memory bottleneck from an answering-model bottleneck. This is a suggested study, not a reproduction claim.

### Interview practice and portfolio connection

**Question:** How should we evaluate a cautious agent that frequently refuses to answer?

**Answer outline:** Report coverage alongside error among accepted answers. High accepted accuracy can conceal near-total refusal.

Define coverage as answered questions divided by all questions, and selective risk as errors divided by answered questions. Examine their trade-off as the abstention threshold changes. Your video measurement gates require a similar account of reliability and usable coverage.

## 7. AgentIdeaBench: How should AI-generated research ideas be evaluated?

**Source:** [AgentIdeaBench: Benchmarking Scientific Ideation in the Agent Era](https://arxiv.org/html/2609.07611v1) · 2026 preprint · Recent extension.

### Paper evidence: question, design, findings, and limits

The paper compares ideation with fixed literature and active retrieval. Its critics find gains mainly in feasibility and specificity, without a significant originality gain. The authors acknowledge limited expert validation for originality and bundled factors in the active condition. Controlled analyses do not establish the exploratory Scientific World Modeling benefit.

### Concepts to understand

**Novelty** is a substantive difference from prior work, not different terminology or large textual distance.

**Feasibility** concerns available data, compute, tools, and validation. A valuable objective can still be untestable under current resources.

**Testability** requires observations that would support or undermine a hypothesis. “Make agents smarter” does not yet specify a falsifiable research claim.

**Evaluator validity** concerns whether a score measures the intended construct. LLM judges need checks for consistency, preferences, and agreement with experts. Agreement among models does not automatically establish truth.

### Teaching example: turning a broad idea into a testable question

Broad idea: “Give a video agent a better memory module.”

Testable version: “Under equal storage and retrieval budgets, do timestamped keyframes and object identities reduce cross-session identity confusion and improve abstention relative to text-only summaries?”

This identifies an intervention, comparator, constraint, and outcomes. Novelty still requires a prior-art search. Specific wording does not make an idea unprecedented.

### Proposed experiment and evaluation

Compare no retrieval, fixed relevant references, agent-selected references, and those same selected references supplied without the interactive workflow. Control reference counts and generation budgets to distinguish evidence quality, retrieval control, and multi-turn interaction.

Use blinded domain reviewers for novelty, feasibility, and testability. Record disagreements and examine the closest prior work. Quantify uncertainty with limited samples; a small score gain is not a scientific breakthrough.

### Interview practice and portfolio connection

**Question:** How would you demonstrate scientific creativity in an agent?

**Answer outline:** Define an operational criterion, then examine prior-art novelty, hypothesis quality, and reproducible experimental outcomes. Idea scores are one measurement layer, not a substitute for validation.

Use “I would like to test…” for future work. Your measurement and optimization experience can motivate a concrete hypothesis without turning unfinished work into a completed achievement.

## 8. LMPNN: How can knowledge graphs support logical queries?

**Source:** [Logical Message Passing Networks with One-hop Inference on Atomic Formulas](https://arxiv.org/abs/2301.08859) · ICLR 2023 · Supporting background.

### Paper evidence

LMPNN represents complex queries as graphs and aggregates local inferences based on existing knowledge-graph representations. It addresses complex queries over incomplete graphs. Neural predictions should not be treated as formally guaranteed true propositions.

### Concepts and teaching example

A knowledge graph represents entities and relations, such as a researcher working at an institution. “Who works at X and has authored a paper on Y?” combines conditions:

$$worksAt(x,X)\land\exists p\,[authored(x,p)\land topic(p,Y)]$$

Here $\land$ means conjunction and $\exists$ means existence. Evidence along different relations must refer to the same candidate entity. Combining one person's employment with another person's publication would produce an invalid answer.

Incomplete graphs introduce a further issue: absence of a recorded publication does not necessarily establish that the publication never existed. Negation requires explicit assumptions about missing facts.

### Proposed experiment, interview practice, and portfolio connection

Build a small graph with missing relationships. Compare strict symbolic queries, neural predictions, and a hybrid that attaches sources. Evaluate correctness, handling of unknown facts, and explanation paths separately.

**Question:** Does Graph RAG automatically eliminate hallucinations?

**Answer outline:** No. Extraction errors, mistaken entity merges, and obsolete relations can propagate. Bind answers to evidence and distinguish recorded facts from predicted completions.

The connection to video research is identity consistency. Splitting one object into several identities, or merging different objects, compromises subsequent relational reasoning.

## 9. COLA: How does causal reasoning differ from association?

**Source:** [COLA: Contextualized Commonsense Causal Reasoning from the Causal Inference Perspective](https://aclanthology.org/2023.acl-long.288/) · ACL 2023 · Supporting background.

### Paper evidence

COLA evaluates commonsense causal relations within event sequences, using temporal information and covariate balancing to address confounding. Better prediction on this task does not imply that a language model can identify arbitrary real-world causal effects.

### Concepts and teaching example

**Association versus causation.** Wet streets and umbrellas can share a common cause: rain. Their co-occurrence does not establish that umbrellas make streets wet.

The observational quantity $P(Y\mid X)$ concerns outcomes when X is observed. The interventional quantity $P(Y\mid do(X))$ concerns outcomes when X is actively set. Equating them requires additional assumptions.

**Temporal order is insufficient.** A preceding B does not rule out common causes, selection effects, or omitted variables. Covariate adjustment also requires causal knowledge; controlling for more variables is not automatically better.

### Proposed experiment, interview practice, and portfolio connection

In a simulated environment with a known mechanism, let a common cause influence two variables. Compare an association-based conclusion with an estimate using the known causal structure. Add intervention data and test predictions of intervention outcomes.

**Question:** Does a velocity change after a visible collision prove the collision caused it?

**Answer outline:** Check synchronization, identity, external forces, and camera motion. Explain what is observed and what remains unobserved. Physical theory constrains explanations but does not replace valid measurement.

Changing a governing law in a benchmark and intervening on a variable within a fixed mechanism are different operations. State clearly whether a proposed experiment changes the mechanism or one of its inputs.

## 10. Synthesis: turn reading into research competence

### A map of research questions

| Stage | Question | Relevant reading |
|---|---|---|
| Explanation | What hypotheses and predictions are possible? | LogiDynamics; The Curse of CoT |
| Evidence acquisition | Which experiment or retrieval action comes next? | NewtonBench; AgentIdeaBench |
| Evidence preservation | Are visual details and updates retained? | MMLongBench; MemLens |
| Learning | How should training and transfer be evaluated? | AbsInstruct |
| Knowledge organization | Are identities, relations, and sources consistent? | LMPNN |
| Causal judgment | Can association, mechanisms, and interventions be distinguished? | COLA |

### Connect the portfolio accurately

**Optimization and LoRA:** Emphasize experimental attribution, whether update components actually execute effectively, the distinction between parameterization and optimization, and configuration-sensitive evaluation. Do not convert historical metrics into a validated claim of a superior optimizer. See the [current project introduction](../01_优化与LoRA_完整介绍.md).

**Video physical consistency:** Emphasize observable evidence, identity tracking, conditions for valid physical judgments, and accepted precision versus coverage. The project remains incomplete; do not claim reliable generator ranking or a completed general scientific agent. See the [current project introduction](../02_视频物理一致性_完整介绍.md).

### A two-minute oral template for each paper

1. State the research question and why it matters.
2. Explain the method, comparator, and metric in a few sentences.
3. Give one finding with its task boundary rather than an isolated leaderboard number.
4. Identify a limitation and an experiment that could investigate it.
5. Connect it to actual experience, separating completed work from future ideas.

### Final self-check

- Can you distinguish induction, abduction, and deduction with one example?
- Can you explain why a correct answer does not prove faithful reasoning?
- Can you select a more informative experiment for a scientific agent?
- Can you distinguish perception, retrieval, memory, and reasoning failures?
- Can you explain LoRA versus optimization and design a fair comparison?
- Can you report reliability, coverage, and cost together?
- Can you turn an idea into a hypothesis with an intervention, baseline, metric, and falsifying observation?

[Index](README.md) · [中文版](reading-guide.zh.md) · [Sources and reading scope](SOURCES.md)
