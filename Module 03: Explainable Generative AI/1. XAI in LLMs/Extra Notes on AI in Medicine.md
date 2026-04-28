# Extra Notes on AI in Medicine

## Why XAI matters more in medicine

Medical use of LLMs (clinical decision support, radiology report drafting, triage, summarization of patient records, medical Q&A) sits under stricter constraints than most domains:

- **High stakes** — a wrong diagnosis or missed contraindication can harm patients.
- **Regulation** — FDA AI/ML guidance, EU AI Act (high-risk category), HIPAA, and GDPR's "right to explanation" demand more than a single confidence score.
- **Clinician trust** — physicians won't act on a recommendation they can't audit.
- **Distribution shift** — clinical language is full of abbreviations, negation, and rare events that pre-training rarely covers well.

Fine-tuning a base LLM (LLaMA, Mistral, GPT-OSS, Med-PaLM, BioGPT, ClinicalBERT/BioBERT-style decoders) on medical data is the standard first step. The XAI question is: **once fine-tuned, how do we show that the model's outputs are grounded, calibrated, and faithful to medical reasoning?**

---

## Part A — Techniques drawn from these module notes

Below are the techniques covered in `1. XAI in LLM Challenges`, `2. XAI in LLM Fine-tuning`, `3. XAI in LLM Prompting`, and `4. XAI in Knowledge Augmentation (RAG)`, mapped to medical fine-tuning relevance.

### A.1 Feature attribution (from §2 Fine-tuning → Local)

The single most useful family for clinicians, because it answers *"which words in the note drove this prediction?"*.

- **Integrated Gradients** — token-level attribution for a fine-tuned medical LLM. Useful for "which symptoms in the discharge summary led to the model flagging sepsis?" Robust on transformer architectures and is the dominant gradient method noted in §2.
- **TransSHAP** (Chen et al., 2023) — sub-word SHAP attributions on transformer LMs. Especially good in medicine because clinical terms tokenize into many sub-words ("hepatosplenomegaly" → multiple tokens) and SHAP gives a principled way to aggregate them.
- **Layer-wise Relevance Propagation (LRP)** — decomposition-based, gives per-token contribution. Has a longer track record in medical imaging and clinical text classification than the newer methods.
- **Perturbation-based attribution** — mask one drug name, one negation cue ("no fever"), or one ICD term and see how the output shifts. Particularly important in medicine because **negation flips meaning** ("no chest pain" vs "chest pain"), and perturbation reveals whether the fine-tuned model actually encodes the negation.

### A.2 Example-based explanations (from §2)

- **Counterfactual explanations** — "if the patient's troponin had been normal, would the model still have flagged ACS?". This is exactly the form of reasoning clinicians already use, so it transfers naturally.
- **Influential instances** — measure which fine-tuning examples most affect a given test prediction. Critical for medical for two reasons: (1) **debugging** — find the bad-label or biased patient note pulling the model in a direction; (2) **audit / regulatory** — show that a sensitive prediction is driven by clinically relevant training cases, not spurious ones.
- **Adversarial examples (TextFooler, SemAttack)** — robustness probing. Medical NLP is brittle: swapping a brand name for its generic, paraphrasing a complaint, or inserting filler should not flip a diagnosis. These methods both expose failures and produce data for adversarial fine-tuning.

### A.3 Natural language explanations (from §2)

Train the model on `(input, label, human-written explanation)` triples so it generates a justification alongside its prediction. CoS-style datasets exist for general reasoning; in medicine the analogue is **rationale-annotated clinical notes** (e.g. MedQA / MedMCQA with explanations, PubMedQA's long answers). Useful, but with one caveat: the generated rationale is **not guaranteed to be faithful** to the model's internal reasoning — see Part B on faithfulness.

### A.4 Concept-based, probing, and neuron activation (from §2 Global)

- **TCAV** — supply concept sets ("anticoagulation", "pediatric", "renal failure") and measure how much each concept influences predictions. Maps cleanly to clinical ontologies (UMLS, SNOMED, ICD-10).
- **Classifier-based probing** — freeze the fine-tuned model, train shallow classifiers on its hidden states to ask "does this layer encode negation? temporal expressions? drug-drug interactions?". Standard tool in BioNLP.
- **Neuron activation analysis** — identify "negation neurons", "dosage neurons", "anatomical-location neurons" and verify by ablation. Heavier lift but high payoff for safety-critical deployments.

### A.5 Prompting-side explainability (from §3)

Even after fine-tuning, the model is still queried via prompts. The §3 techniques apply directly:

- **In-Context Learning probes (Liu 2023, Wei 2023)** — flipped labels, semantically unrelated labels, contrastive demonstrations. Tells you whether your few-shot medical demonstrations are doing real work or whether the fine-tuned prior is overriding them.
- **Chain-of-Thought (CoT)** with saliency analysis (Wu 2023) — clinically natural, since differential diagnosis is already step-by-step. Use saliency to check that the CoT *actually shifts attention to clinically relevant tokens* and isn't just decorative text.
- **Counterfactual CoT prompts (Madonna et al.)** — perturb symbols, patterns, or text in CoT exemplars to see which components are load-bearing.

### A.6 Uncertainty estimation (from §3)

Medicine is the canonical example used in §3 of why overconfidence is dangerous. Two methods carry over directly:

- **Consistency-based estimation (Xiong et al., 2024)** — sample multiple answers, measure agreement. A confidently consistent answer to "drug interaction between warfarin and NSAID?" is a safer signal than a single high-logit answer.
- **Token-level uncertainty (Duan et al.)** — aggregate per-token confidence. Useful for flagging hallucinated entities (drug names, dosages, ICD codes) where an unexpected token-level dip indicates the model is making something up.

> The §3 warning applies with extra force in medicine: **do not trust self-reported confidence**. An LLM saying "I am 95% confident" is not an XAI signal.

### A.7 RAG and source grounding (from §4)

For medical work, RAG is arguably **the most practical "explainability" mechanism**, because clinicians are already trained to weigh evidence by source.

- Retrieve from a curated corpus (UpToDate, PubMed, internal clinical guidelines, formulary, the patient's own EHR) so every claim can be traced to a citation.
- The §4 point that "RAG is not inherently XAI but its source grounding gives users a way to verify factual accuracy" is the core argument for citation-grounded medical assistants (Med-PaLM 2, OpenEvidence, Glass Health).
- **Embedding-space visualization** (PCA / t-SNE / UMAP, §4 + §5) — sanity-check that the medical embedding space clusters by clinical concept rather than by surface features (e.g. by hospital, by note template, by physician author). If clusters track template instead of disease, retrieval is broken.

### A.8 The four core challenges (from §1) re-applied

- **Scale and complexity** — same problem, but FDA submissions still require *some* mechanistic argument; this is why fine-tuned smaller medical models (7B–13B) are popular over frontier closed models.
- **Entangled representations** — concepts like "renal failure" are distributed; concept-based probes (TCAV) and circuit-level analysis are how you partially recover them.
- **No ground truth for explanations** — in medicine, the closest ground truth we have is **clinician-written rationales**; faithfulness is then evaluated as agreement with those rationales (with all the caveats noted in §1).
- **Lack of transparency** — closed-weights models are harder to deploy in regulated settings; open-weights medical fine-tunes (Meditron, Clinical Camel, BioMistral) win on auditability even when they lose on raw capability.

---

## Part B — Latest knowledge beyond the module notes

These methods and considerations aren't covered in the section files but matter for medical fine-tuning today.

### B.1 Parameter-efficient fine-tuning interacts with explainability

- **LoRA / QLoRA / DoRA** — most medical fine-tunes today are PEFT, not full fine-tunes. The interpretability advantage is direct: you can attribute behavior shifts to the **low-rank update matrices** rather than the whole network. Diffing pre- vs post-LoRA attention heads tells you what fine-tuning actually changed.
- **Adapter probing** — train probes on adapter activations specifically, isolating "what fine-tuning added" from "what the base model already knew".
- **Prefix / prompt tuning** — the learned prefix vectors can be visualized in the same embedding space as the corpus (per §4 / §5), giving a partial picture of the soft-prompt's "concept".

### B.2 Faithfulness, not just plausibility

A clinically plausible CoT or natural-language rationale may **not reflect the model's actual computation**. The literature now distinguishes:

- **Plausibility** — does the explanation look reasonable to a human?
- **Faithfulness** — does it actually correspond to what the model did?

Tests for faithfulness include:

- **Counterfactual edits to the rationale** — if changing the rationale doesn't change the answer, the rationale is not load-bearing.
- **Self-consistency between CoT and answer** — Turpin et al. (2023) showed CoTs can be systematically biased while still producing the right answer for the wrong reason.
- **Activation patching / causal tracing** — replace activations from one run with another's and measure output change (mechanistic interpretability tooling: TransformerLens, nnsight).

For medicine, this matters because a clinician-acceptable rationale that doesn't match the model is *more dangerous* than no rationale — it manufactures false trust.

### B.3 Calibration as a first-class XAI deliverable

Beyond §3's uncertainty methods:

- **Temperature scaling** and **Platt scaling** post-hoc on a held-out clinical validation set — cheap, effective, and required for any meaningful "probability of diagnosis X" output.
- **Conformal prediction** — produces a *set* of admissible answers with a guaranteed coverage probability, e.g. "with 95% coverage, the diagnosis is in {DKA, HHS, sepsis-induced hyperglycemia}". Increasingly the recommended uncertainty wrapper for clinical NLP because the coverage guarantee is distribution-free.
- **Selective prediction / abstention** — fine-tune the model to produce an "I don't know / refer to specialist" output when calibrated confidence is below threshold. Particularly important for rare diseases.

### B.4 Hallucination detection specific to clinical text

- **Entity-level grounding checks** — extract every drug, dose, ICD code, lab value from the output and verify each against a structured source (RxNorm, LOINC, SNOMED, the actual patient record). If an entity isn't grounded, surface it.
- **SelfCheckGPT / consistency sampling at the entity level** — sample N completions, flag entities that don't appear in the majority.
- **Retrieval-grounding faithfulness (RAGAS, ARES, TruLens)** — automated metrics that score whether each generated sentence is supported by the retrieved passages. Direct fit for citation-grounded medical RAG.

### B.5 Mechanistic interpretability and sparse features

- **Sparse autoencoders (SAEs)** trained on activations of a fine-tuned medical model are starting to reveal interpretable monosemantic features ("anticoagulant prescribing context", "pediatric dosing", "negation under hedging"). Generalizes the §2 neuron-activation idea beyond single neurons to learned linear directions.
- **Circuit-level analysis** — find the minimal subnetwork responsible for a specific behavior (e.g. negation handling). Not yet routine clinical tooling, but the methodology is maturing.
- **Activation steering / representation engineering** — directly add a "be more cautious" or "stay within evidence" vector to hidden states at inference. Doubles as both a safety mechanism and an interpretability probe.

### B.6 Concept Bottleneck Models and clinical concept layers

Insert an intermediate layer of **interpretable clinical concepts** (UMLS CUIs, ICD codes, lab abnormalities) between the encoder and the prediction head. The model is forced to predict via these concepts, and:

- The concept layer is human-inspectable and editable.
- Clinicians can intervene ("the patient does not have hypertension") and re-run the prediction.
- Trades some accuracy for verifiability — usually acceptable in clinical decision support.

This generalizes TCAV from a post-hoc probe to a built-in architectural commitment.

### B.7 RLHF, RLAIF, and constitutional methods for clinical alignment

- **Clinician-in-the-loop RLHF** — the §2 mention of RLHF, but with annotators who are physicians or pharmacists. Expensive but necessary; the reward model encodes clinical norms.
- **Constitutional AI / RLAIF with a medical constitution** — codify principles ("never recommend off-label dosing without flagging it", "always cite a guideline when stating a treatment threshold") and use the model itself, with critique-and-revise, to generate alignment data. Cheaper than full physician RLHF for the long tail.
- The reward model itself becomes an XAI artifact: **what does it score highly?** is a question you can study directly.

### B.8 Bias, fairness, and subgroup auditing

Fine-tuning on real clinical data inherits real clinical bias (race, sex, socioeconomic status, language). Required additions to the XAI stack:

- **Subgroup performance reports** — accuracy, calibration, refusal rate broken out by protected attribute.
- **Counterfactual fairness** — flip a demographic mention in the input and check the prediction. A direct application of §2's counterfactual / adversarial machinery, used in a fairness frame.
- **Influential-instance audits** focused on whether biased training notes are over-represented in driving subgroup-specific predictions.

### B.9 Regulatory and deployment-shaped requirements

XAI choices in medicine are partly *driven* by regulation, not just by curiosity:

- **FDA SaMD / AI-ML Action Plan** — expects monitoring, performance reporting, and transparency on intended use.
- **EU AI Act (high-risk category)** — explicit requirements for technical documentation, human oversight, and traceability.
- **GDPR Article 22 / "right to explanation"** — local explanations (per-prediction) are legally meaningful.
- **HIPAA** — constrains where you can run the model and where retrieved documents can live; this is why on-prem PEFT-tuned open-weights models are common in hospital deployments.

The practical consequence: an XAI program for a medical LLM is rarely a single technique. A typical deployment combines (a) RAG with citation grounding, (b) per-token feature attribution for clinician inspection, (c) calibration + conformal/abstention for safe uncertainty, (d) entity-level hallucination checks, (e) subgroup audits, and (f) influential-instance / counterfactual tooling for incident review.

---

## TL;DR — recommended stack for a medical fine-tune

If forced to pick the minimum useful set:

1. **RAG with citations** (from §4) — for source grounding and verifiability.
2. **Integrated Gradients or TransSHAP** (from §2) — per-prediction token attribution clinicians can read.
3. **Counterfactual explanations** (from §2) — matches clinical reasoning style.
4. **Consistency-based uncertainty + conformal prediction** (§3 + Part B) — calibrated, with abstention.
5. **Entity-level hallucination check against RxNorm / LOINC / SNOMED** (Part B) — catches the failure mode that hurts patients most.
6. **Influential-instance and subgroup auditing** (§2 + Part B) — for debugging, fairness, and regulatory review.

Everything else (probing, neuron analysis, SAEs, concept bottlenecks, mechanistic interpretability) is high-value research-grade tooling that supplements but does not replace the six items above.
