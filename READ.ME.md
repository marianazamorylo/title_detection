# Hybrid Dual-Contour Intelligent System for Media Manipulation and Clickbait Detection in Ukrainian Media Space

## 📌 Project Overview
This software system was developed as a graduation-level practical project within the **"Computational Linguistics"** curriculum for the **124 "Systems Analysis"** major at Lviv Polytechnic National University.

The project implements a high-performance, **dual-contour pipeline architecture** designed for the automated detection of clickbait, emotional manipulation, panic-inducing headlines, and potential information operations in Ukrainian news headlines. 

The core engineering objective is to synergetically combine lightweight linguistic feature extractors and a Naive Bayes classifier with a deep contextual transformer model (BERT). This approach delivers near-perfect classification metrics while optimizing compute footprint and reducing server-side hardware load.

---

## Architecture & Pipeline Workflow

The architecture is managed by a central orchestrator (`app.py`) that coordinates data routing across two isolated analytical contours:

[Incoming News Headline]
│
▼
┌────────────────────────────────────────────────────────┐
│ FIRST CONTOUR: Linguistic Processor & Expert System    │
├────────────────────────────────────────────────────────┤
│ • Text cleaning, tokenization, & lemmatization        │
│ • Production rules check (CapsLock, !!!, ...)          │
│ • Naive Bayes odds evaluation (SQLite Knowledge Base)  │
└─────────────────────────┬──────────────────────────────┘
│
[Short-Circuit Flag Check]
Is score in the Grey Zone [0.4, 0.75]?
│
┌────────────────┴────────────────┐
No  │                                 │ Yes
(stop_inference = True)             (stop_inference = False)
▼                                 ▼
┌────────────────────────┐       ┌────────────────────────┐
│ IMMEDIATE VERDICT      │       │ SECOND CONTOUR: BERT   │
│ (Saves ~71.8% CPU time)│       ├────────────────────────┤
│                        │       │ • Contextual Embedding │
│ Zones: Green / Red     │       │ • Deep Subword Analysis│
└────────────────────────┘       └─────────┬──────────────┘
│
▼
┌────────────────────────┐
│ FINAL ARBITRATION      │
│ Zones: Green / Yellow /│
│        Red             │
└────────────────────────┘

```

1. **First Contour (Expert-Statistical Filter):**
   * **Preprocessing:** `linguistic_processor.py` strips web noise, stop words, and punctuation. Morphological lemmatization is performed via `pymorphy3`.
   * **Rule Validation:** `expert_system.py` scans visual shape anomalies via predefined base rules (`RULE_CAPS`, `RULE_EXCESSIVE_PUNCT`, `RULE_ELLIPSIS`).
   * **Probability Engine:** The system calculates posterior target odds using a Naive Bayes framework over historical weights fetched via rapid queries from the SQLite repository (`clickbait_detector.db`).
   * **Short-Circuit Logic:** If the statistical signal strongly points to clean text or raw clickbait, the flag `stop_inference` is set to `True`. The engine outputs an immediate result, bypassing heavy neural nets.

2. **Second Contour (Deep Semantic Transformer):**
   * If the first contour flags the input as ambiguous (score lands inside the "Grey Zone" boundary: `0.4 <= score < 0.75`), `stop_inference` switches to `False`.
   * The text is routed to the deep semantic network (`semantic_core.py`), powered by a fine-tuned local instance of **BERT** under PyTorch, which performs a holistic structural arbitration.

## Setup & Execution Guide

### 1. Environment Setup
Clone the repository, initialize your environment, and isolate runtime dependencies:
```bash
git clone [https://github.com/your-username/clickbait-detector-ua.git](https://github.com/your-username/clickbait-detector-ua.git)
cd clickbait-detector-ua

# Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/macOS
# venv\Scripts\activate   # For Windows

```

### 2. Dependency Installation

Install verified package pins:

```bash
pip install -r requirements.txt

```

### 3. Launching the Web Orchestrator

Ensure your localized weights database `clickbait_detector.db` is present in the root directory, then boot up the Gradio backend:

```bash
python app.py

```

Once the BERT weights allocation and lexical indexes complete initialization, click the dynamic local URL outputted in your console shell (typically `http://127.0.0.1:7860`).


##  Repository Structure

* `app.py` — Core manager engine, hybrid pipeline orchestrator, and Gradio interface shell.
* `linguistic_processor.py` — String cleaning pipelines, lexical transformations, and token normalization via `pymorphy3`.
* `expert_system.py` — Statistical Naive Bayes loops and production rule mapping against relational constraints.
* `semantic_core.py` — Subword contextual execution wrapper for the fine-tuned local BERT transformer.
* `clickbait_detector.db` — Relational SQLite knowledge base tracking token probability logs and penalty weight tables.
* `requirements.txt` — Python runtime library requirements environment specification file.

---

##  Defensive Programming Paradigm

The system code is designed under a strict fault-tolerant engineering philosophy. Any potentially unstable operations (empty text streams, unmapped foreign subwords, corrupted SQLite rows, or memory bottlenecks inside PyTorch attention frames) are locked within resilient `try-except` wrappers in `app.py`. In the event of an anomalous structural leak, the runtime falls back gracefully, logging the full execution trace to the developer debug stream without triggering server crashes or application reboots.
