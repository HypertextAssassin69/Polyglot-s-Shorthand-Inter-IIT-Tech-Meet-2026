# Polyglot's Shorthand

> An efficient linguistic engine for noisy Romanized and code-mixed South Asian text.

## 🚧 Project Status

**Current Status: Experimental evaluation complete.**

This project was developed for the **Inter-IIT Tech Meet 2026**.

The repository contains the implemented retrieval-based normalization experiments, ByT5 generative normalization experiment, latency benchmarks, and error analysis used in the final presentation.

The guiding principle is:

> **Design → Implement → Measure → Identify Bottleneck → Modify → Measure Again**

Complexity must earn its place through experiments.

---

## 🎯 Problem

Romanized and code-mixed South Asian text is highly variable. The same underlying content can appear in many surface forms (e.g. `kya kar rahe ho`, `kya kr rhe ho`, `kya krre ho`, `kya karhe ho`). A robust system should recognize the shared underlying content while preserving meaningful expressive information.

---

## 💡 Core Hypothesis

> **Normalize irrelevant spelling variation while preserving information that can change lexical identity or pragmatic meaning.**

---

# 🧠 System Architecture

The conceptual architecture separates responsibilities:

> **Normalization solves linguistic surface variation. Tokenization solves efficient representation of the normalized content.**

The tokenizer should not be responsible for learning every possible Romanized spelling variation. Likewise, normalization should not be responsible for understanding the complete semantic meaning of a sentence.

---

# 🧩 Representation

The system conceptually maintains **two persistent representations**:
1. **Content Vector**: Represents the normalized underlying content/meaning.
2. **Pragmatic Meaning Vector**: Represents expressive/contextual signals (emotion, emphasis, sarcasm, emojis, etc.).

Context is **not treated as a third persistent vector**. If the current message contains insufficient information, the system can request relevant conversational context to preserve uncertainty rather than inventing missing information.

---

# 🧪 Verified Experimental Results

The broader Hybrid candidate pipeline exposed retrieval coverage as a bottleneck. The ByT5 experiment showed that direct generation can bypass the candidate-space limitation, but the tested ByT5-small latency is far above the single-digit-ms target.

### Original V1
- Test N = 582
- R@1 = 95.02%
- Edit-distance baseline R@1 = 88.29%
- Improvement = +6.73 percentage points
- Non-minimum ED = 97.26%
- ED tie = 96.67%
- Minimal pairs = 100%
- Latency ≈ 0.12 ms/word

### Hybrid V1
- Retrieval vocabulary = 23,408
- Candidate coverage = 47.61%
- Final R@1 = 35.69%
- Ranker accuracy given gold present = 76.92%
- Retriever failure = 53.60%
- Ranker failure = 10.71%
- Mean candidates = 6.36
- Median = 6
- Maximum = 8

### MLP
- R@1 = 35.59%
- Gold-present accuracy = 76.70%
- Ranker failure = 10.81%
- Parameters ≈ 17.35M

### Cross-Encoder
- R@1 = 23.37%
- Gold-present accuracy = 50.36%
- Ranker failure = 23.03%
- Parameters ≈ 17.23M

### Retrieval Benchmark
- Strict = 9.15%, 0.01 ms mean
- Full weighted = 35.94%, 13.99 ms mean, 16.61 ms P95
- Hybrid baseline = 43.59%, 14.00 ms mean, 16.62 ms P95
- Indexed weighted M=100 = 36.18%, 0.18 ms mean, 0.22 ms P95
- Indexed weighted M=1000 = 37.14%, 0.83 ms mean, 1.01 ms P95
- Indexed hybrid M=100 = 45.21%, 0.19 ms mean, 0.24 ms P95
- Indexed hybrid M=1000 = 45.03%, 0.85 ms mean, 1.02 ms P95

### ByT5
- Model = ByT5-small
- Parameters = 299,637,760
- Deduplicated instances = 239,475
- Test = 23,948
- Noisy test = 4,767
- Clean test = 19,181
- Best evaluated checkpoint = epoch 4
- Validation loss = 0.0890
- Overall exact match = 90.22%
- Noisy correction = 63.48%
- Clean preservation = 96.87%
- Retrieval-covered noisy = 65.79%
- Retrieval-missed noisy = 60.96%
- Tokenization = 2.52 ms
- Generation = 39.43 ms
- Decoding = 24.81 ms
- End-to-end = 66.76 ms
- P50 = 63.89 ms
- P95 = 95.81 ms
- Average output = 6.28 tokens

### Manual Audit of 50 ByT5 Errors
- Vowel / schwa / nasal = 21
- Phonetic / spelling = 13
- English / code-mixed = 6
- Annotation ambiguity = 4
- Clean preservation = 3
- Abbreviation = 2
- Contextual / semantic = 1

---

# 📁 Repository Structure

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

2. **Data Preparation:**
   Prepare the dataset formats required by different pipelines:
   ```bash
   python scripts/prepare_v1_data.py
   python scripts/prepare_hybrid_data.py
   python scripts/prepare_byt5_data.py
   ```

3. **Train Models:**
   Train the required models:
   ```bash
   python scripts/train_v1_model.py
   python scripts/train_hybrid_model.py
   python scripts/train_mlp_model.py
   python scripts/train_cross_encoder.py
   python scripts/train_byt5_normalizer.py
   ```

4. **Run Latency Benchmarks:**
   Evaluate the inference speeds of the ByT5, MLP, and Cross-Encoder modules.
   ```bash
   python scripts/benchmark_latency.py
   ```

5. **Run Retrieval Benchmarks:**
   Measure the performance of standard lexical retrieval approaches.
   ```bash
   python scripts/benchmark_retrieval.py
   ```

6. **Evaluate Generative Normalizer (ByT5):**
   Run the evaluation script to measure exact-match normalization accuracy, noisy-word correction, clean-word preservation, retrieval-covered/missed performance, and inference latency.
   ```bash
   python scripts/evaluate_byt5_normalizer.py
   ```

7. **Run Error Analysis:**
   Sample model errors for manual linguistic categorization.
   ```bash
   python scripts/error_analysis.py
   ```
