# LLM Unlearning for Copyright Infringement Prevention

Implementation of **Stable Sequential Unlearning (SSU)** for removing copyrighted content from Large Language Models, with enhanced evaluation metrics for measuring unlearning effectiveness.

## Overview

This project implements the SSU methodology to make LLMs "forget" copyrighted books they were trained on, while preserving general language capabilities. The implementation includes:

1. **SSU Training Pipeline** - Sequential unlearning with composite loss and weight saliency
2. **Enhanced Evaluation** - Beyond ROUGE: semantic similarity and membership inference metrics
3. **Inference-Only Evaluation** - Evaluate pre-trained models without retraining

## Method

### Stable Sequential Unlearning (SSU)

The SSU approach unlearns copyrighted content sequentially, one book at a time:

```
Step 0: Fine-tune base model on all books (D_f) → Memorized Model
Step 1: Unlearn Book 1 using SSU loss
Step 2: Unlearn Book 2 using SSU loss
...
Step N: Unlearn Book N → Final Unlearned Model
```

**Key Components:**
- **Composite Loss**: `L = λ₁ * L_fgt + λ₂ * L_rnd` (Equation 4 from paper)
- **Weight Saliency**: Dynamic threshold γ = μ + σ on LoRA gradients
- **Task Vector Negation**: Flip LoRA deltas to reverse memorization

### Datasets

| Dataset | Description |
|---------|-------------|
| **D_f** | 10 copyrighted books from Project Gutenberg (Sherlock Holmes, Frankenstein, Pride & Prejudice, etc.) |
| **D_f^t** | Book to unlearn at time step t |
| **D_nor** | Retention data - 200 chunks from 100 other books (for evaluation) |

## Evaluation Metrics

### 1. Regurgitation (ROUGE)
- **ROUGE-1 / ROUGE-L**: Lexical overlap between model generations and original text
- Lower scores = less regurgitation = better unlearning

### 2. Semantic Similarity (New)
- **Cosine Similarity**: Using `sentence-transformers/all-MiniLM-L6-v2`
- Captures semantic overlap even when lexical overlap is low
- Important for detecting paraphrased regurgitation

### 3. Membership Inference (New)
- **NLL-based Analysis**: Compare negative log-likelihood on forget vs. retain chunks
- **Δ NLL**: `mean(NLL_forget) - mean(NLL_retain)`
  - Before unlearning: Δ NLL ≤ 0 (model "knows" the content)
  - After unlearning: Δ NLL ≥ 0 (content is "forgotten")
- **ROC-AUC**: Discrimination ability between forget/retain chunks

### 4. Utility Preservation
- **Perplexity**: On retention dataset and general text
- Ensures model maintains language capabilities after unlearning

## Project Structure

```
├── SSU_Unlearning.ipynb      # Main notebook with full pipeline
├── gutenberg_books/          # Downloaded book data
│   ├── all_books/            # All 10 books for initial fine-tuning
│   ├── time_step_1/          # Book for step 1 unlearning
│   ├── ...
│   ├── retention_books/      # D_nor retention data
│   └── evaluation_books/     # Books for evaluation
├── ssu_unlearned_models/     # Model checkpoints
│   ├── memorized_model/      # After initial fine-tuning (optional)
│   ├── step_X_unlearned_model/  # After each unlearning step
│   └── step_10_unlearned_model/ # Final unlearned model
└── README.md
```

## Quick Start

### Installation

```bash
pip install torch transformers peft datasets accelerate rouge-score sentence-transformers scikit-learn
```

### Option 1: Full Training Pipeline

```python
# 1. Run initial fine-tuning (makes model memorize books)
memorized_model = run_initial_finetuning()

# 2. Run sequential unlearning (10 steps)
final_model = run_sequential_unlearning()
```

### Option 2: Inference-Only Evaluation

If you already have trained models, evaluate without retraining:

```python
# Compare memorized vs unlearned model
results = run_before_after_evaluation(
    memorized_model_path="path/to/memorized_model",
    unlearned_model_path="path/to/step_10_unlearned_model"
)
```

### Option 3: Memory-Efficient Evaluation (Colab)

For limited GPU memory:

```python
results = run_fast_evaluation(
    memorized_model_path="path/to/memorized_model",
    unlearned_model_path="path/to/unlearned_model"
)
```

## Configuration

Key settings in `Config` class:

```python
# Prototyping mode (faster runs)
PROTOTYPE_MODE = True  # Set False for full runs

if PROTOTYPE_MODE:
    NUM_UNLEARNING_STEPS = 3    # vs 10 for full
    NUM_CHUNKS_PER_STEP = 30    # vs 50
    EVAL_MAX_PAIRS = 5          # vs 10
    EVAL_NUM_SAMPLES = 3        # vs 10
```

## Running on Google Colab

1. Upload models to Google Drive
2. Mount Drive in notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Use Drive paths:
   ```python
   results = run_before_after_evaluation(
       memorized_model_path="/content/drive/MyDrive/models/memorized_model",
       unlearned_model_path="/content/drive/MyDrive/models/step_10_unlearned_model"
   )
   ```

## Expected Results

After successful unlearning:

| Metric | Memorized | Unlearned | Change |
|--------|-----------|-----------|--------|
| ROUGE-L | ~0.4-0.6 | ~0.1-0.3 | ↓ (good) |
| Semantic Sim | ~0.6-0.8 | ~0.3-0.5 | ↓ (good) |
| Δ NLL | ≤ 0 | ≥ 0 | ↑ (good) |
| Perplexity | ~X | ~X±10% | ~ (stable) |

## Model

- **Base Model**: `google/gemma-3-1b-it`
- **Fine-tuning**: LoRA (r=8, α=16) on q_proj, v_proj
- **Training**: 1 epoch per step, dynamic LR (1e-5 for steps 1-5, 1e-6 for 6-10)

## References

- SSU Paper: Stable Sequential Unlearning methodology
- Project Gutenberg: Source of public domain books
- PEFT/LoRA: Parameter-efficient fine-tuning

## License

This project is for research and educational purposes. The unlearning methodology can be applied to remove any memorized content from LLMs.

