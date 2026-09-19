# Polyglot's Shorthand

> An efficient linguistic engine for noisy Romanized and code-mixed South Asian text.

## 🚧 Project Status

**Current Status: Experimental evaluation complete.**

This project is being developed for the **Inter-IIT Tech Meet 2026**.

The repository contains the implemented retrieval-based normalization experiments, ByT5 generative normalization experiment, latency benchmarks, and error analysis used in the final presentation.

The guiding principle is:

> **Design → Implement → Measure → Identify Bottleneck → Modify → Measure Again**

Complexity must earn its place through experiments.

---

## 🎯 Problem

Romanized and code-mixed South Asian text is highly variable.

The same underlying content can appear in many surface forms:

```text
kya kar rahe ho
kya kr rhe ho
kya krre ho
kya karhe ho
```

Users may also introduce:

- phonetic spelling variations
- typos
- abbreviations
- slang
- repeated characters
- English/Hindi code-mixing
- emojis
- punctuation-based emphasis

For example:

```text
bhai order cancel krdo please
bhai order cancel kar do please
bhaiii order cancel krdo!!! 😭
```

A robust system should recognize the shared underlying content while preserving meaningful expressive information.

---

## 💡 Core Hypothesis

The central problem is a tension between:

> **Invariance to irrelevant variation vs. sensitivity to meaningful variation.**

The system should become invariant to spelling/phonetic variation that does not change meaning, while preserving distinctions that do.

For example:

```text
mujhe
muje
mujhy
mughe
```

may represent the same underlying word in noisy Romanized Hindi.

However:

```text
mama
mamma
```

must not automatically collapse into the same representation if the distinction changes lexical meaning.

Therefore:

> **Normalize irrelevant spelling variation while preserving information that can change lexical identity or pragmatic meaning.**

---

# 🧠 System Architecture

The current conceptual architecture is:

```text
                    RAW INPUT
                        │
                        ▼
          ┌─────────────────────────┐
          │ Content / Pragmatic     │
          │       Separation        │
          └────────────┬────────────┘
                       │
                ┌──────┴──────┐
                ▼             ▼
             CONTENT       PRAGMATIC
                │             │
                ▼             │
        CONTENT NORMALISATION │
                │             │
                ▼             │
            TOKENIZER         │
                │             │
                ▼             │
          CONTENT MODEL       │
                │             │
                └──────┬──────┘
                       ▼
             Structured Interpretation
```

### Architectural Boundary

The current design deliberately separates responsibilities:

> **Normalization solves linguistic surface variation. Tokenization solves efficient representation of the normalized content.**

The tokenizer should not be responsible for learning every possible Romanized spelling variation.

Likewise, normalization should not be responsible for understanding the complete semantic meaning of a sentence.

---

# 🧩 Representation

The system conceptually maintains **two persistent representations**:

### 1. Content Vector

Represents the normalized underlying content/meaning.

Example:

```text
"kya kar rahe ho"
"kya kr rhe ho"
"kya krre ho"
```

should produce similar content representations.

---

### 2. Pragmatic Meaning Vector

Represents expressive/contextual signals such as:

- emotion
- emphasis
- frustration
- sarcasm
- teasing
- urgency
- slang/pragmatic cues
- emoji meaning
- punctuation-based emphasis

Example:

```text
Bohot badhiya service
```

vs.

```text
Bohot badhiya service 😒
```

The content may be similar, while the pragmatic interpretation differs substantially.

---

## 🔎 Context

Context is **not treated as a third persistent vector**.

If the current message contains insufficient information to determine the intended meaning, the system can request relevant conversational context.

Example:

```text
haan bhai cancel krdo
```

The message does not specify what should be cancelled.

If previous context contains:

```text
Order galat address pe ja raha hai
```

the system can infer that the user is referring to the order.

If multiple interpretations remain possible, the system should preserve the ambiguity rather than inventing information.

### Principle

> **Preserve uncertainty. Do not hide or invent missing information.**

---

# 🔤 Romanized Hindi Normalization

The normalization layer is intended to handle the large variation introduced by Romanized Hindi.

The current direction is a **compositional phonetic representation** based on Hindi consonant and vowel sound units rather than a giant word-level spelling dictionary.

The goal is to move combinatorial phonetic variation into a deterministic/structured layer so that the neural model does not have to independently learn every possible spelling variation.

### Important Constraint

The normalizer must not be an irreversible phonetic hash.

It should be able to preserve distinctions where phonetic/orthographic information affects lexical identity.

---

# 🧪 Experimental Philosophy

We do not assume that the proposed normalization layer is automatically better.

We will test it against progressively stronger baselines.

## Baseline 1 — Character Tokenization

The simplest possible tokenizer:

```text
"mujhe"
    ↓
[m, u, j, h, e]
```

This establishes a minimal baseline.

---

## Baseline 2 — Standard Subword Tokenization

Evaluate standard approaches such as:

- BPE
- SentencePiece

This tests whether a conventional tokenizer already handles enough of the variation.

---

## Baseline 3 — Indic Tokenization

Evaluate an Indic-oriented tokenizer / SuperBPE-style approach.

The relevant reference is:

**IndicSuperTokenizer**

https://arxiv.org/html/2511.03237v1

The goal is not to assume that it solves Romanized Hindi normalization, but to establish a strong multilingual/Indic tokenization baseline.

---

## Proposed Approach

```text
Raw Romanized Input
        ↓
Content / Pragmatic Separation
        ↓
Content Normalization
        ↓
Tokenizer
        ↓
Content Model
```

The proposed approach only survives if experiments demonstrate that the normalization stage provides useful information beyond strong generic tokenization.

---

# 🧪 Initial Benchmark

The benchmark should contain several classes of difficult examples.

### 1. Romanization Variation

```text
kya kar rahe ho
kya kr rhe ho
kya krre ho
kya karhe ho
```

Expected behavior:

```text
Similar content representation
```

---

### 2. Lexical Distinction

```text
mama
mamma
```

Expected behavior:

```text
Different representations when the distinction changes meaning
```

This is a critical test against overly aggressive normalization.

---

### 3. Phonetic Reconstruction

Examples:

```text
mujhe
muje
mujhy
mughe
```

Expected behavior:

```text
High lexical similarity / same intended word
```

---

### 4. Pragmatic Variation

```text
Bohot badhiya service
Bohot badhiya service 😒
```

Expected behavior:

```text
Similar content
Different pragmatic representation
```

---

### 5. Context Dependence

```text
haan bhai cancel krdo
```

Without context:

```text
Ambiguous
```

With:

```text
Order galat address pe ja raha hai
```

Expected interpretation:

```text
Cancel the order
```

---

### 6. Code-Mixing / Slang / Noise

The benchmark will also include:

- Hindi-English code-mixing
- slang
- abbreviations
- repeated characters
- punctuation
- emojis
- realistic chat-style spelling errors

---

# 🤖 Lexical Helper

A lightweight lexical plausibility component may be used after phonetic normalization.

Its purpose is **not necessarily to hard-correct text**.

Instead, it can rank plausible lexical candidates.

Conceptually:

```text
Romanized Input
      ↓
Phonetic Units
      ↓
Candidate Generation
      ↓
Tiny Lexical Plausibility Model
      ↓
Candidate Ranking
      ↓
Main Semantic Model
```

For example:

```text
mughe
```

could generate candidates where:

```text
mujhe
```

receives a high plausibility score.

The system should avoid aggressively correcting:

- names
- slang
- product names
- English words
- unknown words

when the model is uncertain.

---

# 📚 Data Strategy

A larger language model may be used as a **teacher** to help construct training data.

Potential labels include:

- normalized content
- pragmatic signals
- context dependence
- `need_context`
- candidate interpretations

The teacher can also generate realistic variations involving:

- Romanization drift
- phonetic spelling
- typos
- slang
- code-mixing
- emojis
- punctuation

Teacher-generated labels should **not automatically be treated as ground truth**.

A validation subset and quality checks are required.

---

# ⚡ Efficiency Constraints

The original problem places strict efficiency constraints:

- **Maximum model size: 500M parameters**
- **Single-digit millisecond latency target**
- **High throughput**

Parameter count alone does not determine latency.

Actual performance depends on factors including:

- architecture
- sequence length
- tokenizer efficiency
- quantization
- hardware
- memory bandwidth
- batching
- implementation overhead

Therefore the objective is:

> **Find the smallest/fastest system that meets the required quality, rather than maximizing model size.**

---

# 📊 Evaluation

The system will be evaluated along multiple dimensions.

## Linguistic Quality

Measure:

- normalization accuracy
- robustness to spelling variation
- lexical distinction preservation
- pragmatic signal preservation
- context-dependent interpretation
- code-mixing robustness

## Representation Quality

Test whether equivalent inputs produce similar representations while meaningfully different inputs remain separable.

## Efficiency

Measure:

- latency
- throughput
- memory usage
- parameter count
- tokenization cost

## Ablation Studies

Important comparisons include:

```text
Character tokenizer
        vs
BPE
        vs
Indic tokenizer
        vs
Normalization + tokenizer
```

Additional ablations can test:

```text
Normalization only
Normalization + lexical helper
Normalization + pragmatic separation
Full system
```

---

# 🔬 Research Loop

Every major component follows:

```text
        ┌──────────────┐
        │   Hypothesis │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Implement    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │   Measure    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Find Failure │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Ask "Why?"   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │   Modify     │
        └──────┬───────┘
               │
               └──────────────→ Measure Again
```

### Core Rule

> **Complexity must earn its place through experiments.**

If a simpler method performs similarly, prefer the simpler method.

If a proposed component does not improve the benchmark, remove it.

---

# 📁 Repository Structure

Current intended structure:

```text
polyglot-shorthand/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
│
├── src/
│   ├── normalization/
│   ├── tokenization/
│   ├── models/
│   └── evaluation/
│
├── experiments/
│
├── tests/
│
└── docs/
```

---

# 🛠️ Development

Create and activate the project environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🧭 Current Development Plan

### Phase 1 — Establish Baselines

- [ ] Character tokenizer
- [ ] BPE tokenizer
- [ ] SentencePiece tokenizer
- [ ] Indic/SuperBPE-style tokenizer
- [ ] Initial benchmark

### Phase 2 — Content / Pragmatic Separation

- [ ] Define separation task
- [ ] Create seed examples
- [ ] Evaluate extraction quality

### Phase 3 — Content Normalization

- [ ] Define normalization contract
- [ ] Build compositional phonetic mapping
- [ ] Generate candidate representations
- [ ] Test `mama` vs `mamma`
- [ ] Test noisy Romanized variants

### Phase 4 — Lexical Helper

- [ ] Establish n-gram baseline
- [ ] Generate synthetic Romanized variants
- [ ] Train tiny neural alternative
- [ ] Compare accuracy vs latency

### Phase 5 — Main Model

- [ ] Content representation
- [ ] Pragmatic representation
- [ ] Context detection
- [ ] Context retrieval experiments
- [ ] End-to-end evaluation

### Phase 6 — Optimization

- [ ] Parameter reduction
- [ ] Quantization
- [ ] Latency measurement
- [ ] Throughput measurement
- [ ] Ablation studies

### Phase 7 — Final Evaluation

- [ ] Error analysis
- [ ] Robustness testing
- [ ] Efficiency benchmark
- [ ] Reproducibility
- [ ] Whitepaper

---

# 📌 Current North-Star

The project is not trying to build another giant language model.

The goal is to build a **small, efficient linguistic front-end** that can transform messy Romanized/code-mixed South Asian text into a representation that preserves what matters:

```text
               Surface Variation
                      ↓
        ┌──────────────────────────┐
        │  Remove irrelevant noise │
        │  Preserve meaningful     │
        │  linguistic signals      │
        └────────────┬─────────────┘
                     ↓
              Efficient Model
                     ↓
        High-throughput understanding
```

The central research question is:

> **Can explicit linguistic structure make a small model robust to the variability of Romanized and code-mixed South Asian text without sacrificing meaningful distinctions?**

The answer should come from experiments, not assumptions.

---

## 📜 License

License to be determined.

---

## 🧪 Project Philosophy

> **Don't build what sounds impressive. Build what the experiments justify.**

---

## 🔁 Steps to Reproduce Results

To reproduce the experimental results (accuracy, latency, and error profiles) described in the presentation and project documents:

1. **Set up the environment:**
   Install dependencies from the provided requirements list.
   ```bash
   python -m venv interIIT
   # Windows: interIIT\Scripts\activate
   # Linux/Mac: source interIIT/bin/activate
   pip install -r requirements.txt
   ```

   **Note on Data and Checkpoints:**
   The raw evaluation datasets (`data/raw/comi_lingua/...`) and trained model checkpoints (`scratch/byt5_best_model.pt`) are excluded from this repository due to size constraints. To reproduce the exact numbers, place the benchmark test sets in `data/raw/comi_lingua/` and run the data preparation scripts in `scripts/`, followed by `scripts/train_byt5_normalizer.py` to generate the checkpoint.

2. **Run Latency Benchmarks:**
   Evaluate the inference speeds of the ByT5, MLP, and Cross-Encoder modules.
   ```bash
   python scripts/benchmark_latency.py
   ```

3. **Run Retrieval Benchmarks:**
   Measure the performance of standard lexical retrieval approaches.
   ```bash
   python scripts/benchmark_retrieval.py
   ```

4. **Evaluate Generative Normalizer (ByT5):**
   Run the evaluation script to measure exact-match normalization accuracy, noisy-word correction, clean-word preservation, retrieval-covered/missed performance, and inference latency.
   ```bash
   python scripts/evaluate_byt5_normalizer.py
   ```

5. **Run Error Analysis:**
   Sample model errors for manual linguistic categorization.
   ```bash
   python scripts/error_analysis.py
   ```
