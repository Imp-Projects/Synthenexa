Relative paths that start with **`Dataset/`** are written under **`synthenexa_pipeline/Dataset/`**, not the repository root, so you can run commands from the repo root without splitting outputs across two folders.

## Phase 1: Generate Training Set

### Step 1.1: Generate Initial Training Set (HumanEval-style problems)

```bash
python synthenexa_pipeline/0_train_set_generate.py \
  --output_path Dataset/train_set.jsonl \
  --total_tasks 100 \
  --model groq:llama-3.1-8b-instant
```

**Output**: `Dataset/train_set.jsonl`
- Contains 100+ coding problems with task_id, prompt, entry_point, canonical_solution, test

### Step 1.2: Validate & Clean Training Set

```bash
python synthenexa_pipeline/0_train_set_select.py \
  --input Dataset/train_set.jsonl \
  --output_path Dataset/selected_promts.jsonl
```

**Output**: `Dataset/selected_promts.jsonl`
- Filtered training set (duplicates removed, valid syntax only)

## Phase 2: Prompt Mutation & Evaluation

### Step 2.1: Mutate Prompts (Generate variations)

```bash
python synthenexa_pipeline/1_prompt_mutate.py \
  --model groq:llama-3.1-8b-instant \
  --prompt_path Dataset/selected_promts.jsonl \
  --output_path Dataset/mutated_prmts.jsonl
```

**Output**: `Dataset/mutated_prmts.jsonl`
- Contains mutated (improved) prompts from the original ones
- Falls back to Ollama if Groq is unavailable or request fails

### Step 2.2: Evaluate Mutated Prompts (Generate solutions with each prompt)

**Option A: Use Groq (with Ollama fallback)**
```bash
python synthenexa_pipeline/2_prompt_evaluate.py \
  --model groq:llama-3.1-8b-instant \
  --trainset_path Dataset/train_set.jsonl \
  --mutated_prompt_path Dataset/mutated_prmts.jsonl \
  --output_path Dataset/evaluatedprmts
```

**Option B: Use ONLY Ollama (skip Gemini quota)**
```bash
# Make sure Ollama is running first:
ollama serve

# In another terminal:
python synthenexa_pipeline/2_prompt_evaluate.py \
  --model ollama:qwen:latest \
  --trainset_path Dataset/train_set.jsonl \
  --mutated_prompt_path Dataset/mutated_prmts.jsonl \
  --output_path Dataset/evaluatedprmts
```

**Output**: `Dataset/evaluatedprmts/train_set_<model>_*.jsonl`
- Solutions generated using each mutated prompt
- One file per mutated prompt

**Files produced**:
- `Dataset/evaluatedprmts/train_set_groq_llama-3.1-8b-instant_0.jsonl` (solutions for prompt_id 0)
- `Dataset/evaluatedprmts/train_set_groq_llama-3.1-8b-instant_1.jsonl` (solutions for prompt_id 1)
- ... (one for each mutated prompt)

---

## Phase 3: Score & Select Best Prompts

### Step 3.1: Calculate Pass@K Scores (Test solutions)

For each generated solution file:

```bash
python synthenexa_pipeline/human_eval/evaluate_functional_correctness.py \
  Dataset/evaluatedprmts/train_set_groq_llama-3.1-8b-instant_0.jsonl \
  --k "1,10,100" \
  --timeout 3.0
```

**Output**: `Dataset/evaluatedprmts/train_set_groq_llama-3.1-8b-instant_0.jsonl_results.jsonl.gz`
- Pass@1, pass@10, pass@100 metrics for each prompt

---

### Step 3.2: Calculate Scores & Select Best Prompt

```bash
python synthenexa_pipeline/3_reinforcement_cal_score_and_select.py \
  --evaluate_path Dataset/evaluatedprmts \
  --testset_path Dataset/train_set.jsonl \
  --origin_prompt Dataset/mutated_prmts.jsonl \
  --best_prompt Dataset/best_prompts.jsonl
```

**Output**: `Dataset/best_prompts.jsonl`
- Ranked prompts by performance score
- Best performing prompts selected

---

## Phase 4: Reinforcement (Optional - Iterative Improvement)

### Step 4.1: Run Full Reinforcement Loop (Steps 2-3 combined)

```bash
python synthenexa_pipeline/2+3+4_reinfocement.py
```

**This file:**
- ✅ Generates solutions using mutated prompts
- ✅ Evaluates them against test cases
- ✅ Selects best performing prompt
- ✅ Generates optimized prompts from best ones
- ❌ **Currently hardcoded paths - needs CLI arguments**

**Current hardcoded inputs**:
- `data/train_set.jsonl`
- `data/mutated_prompts.jsonl`
- `data/evaluation_output`

---

## Complete Pipeline Summary

```
[1] 0_train_set_generate.py
    ↓ Output: train_set.jsonl
    
[2] 0_train_set_select.py
    ↓ Output: selected_promts.jsonl
    
[3] 1_prompt_mutate.py
    ↓ Output: mutated_prmts.jsonl
    
[4] 2_prompt_evaluate.py
    ↓ Output: evaluatedprmts/train_set_gemini-2.5-flash_*.jsonl
    
[5] evaluate_functional_correctness.py (run for each solution file)
    ↓ Output: *_results.jsonl.gz
    
[6] 3_cal_score_and_extract_best_prompt.py
    ↓ Output: best_prompts.jsonl
    
[LOOP] 2+3+4_reinfocement.py (iterate to improve)
    ↓ Output: optimized_prompts.jsonl
```

---

## Environment Setup

Before running any commands:

```bash
# Activate virtual environment
source venv/Scripts/activate

# Ensure .env has required keys
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "OLLAMA_MODEL=mistral" >> .env  # Optional
```

---

## File Structure

```
Dataset/
├── train_set.jsonl                          # Raw training set (Step 1.1)
├── selected_promts.jsonl                    # Cleaned training set (Step 1.2)
├── mutated_prmts.jsonl                      # Mutated prompts (Step 2.1)
├── evaluatedprmts/                          # Solutions per prompt (Step 2.2)
│   ├── train_set_gemini-2.5-flash_0.jsonl
│   ├── train_set_gemini-2.5-flash_0.jsonl_results.jsonl.gz
│   ├── train_set_gemini-2.5-flash_1.jsonl
│   └── train_set_gemini-2.5-flash_1.jsonl_results.jsonl.gz
└── best_prompts.jsonl                       # Selected best prompts (Step 3.2)
```

---

## Quick Start (One Command Each)

```bash
# Generate training set
python synthenexa_pipeline/0_train_set_generate.py --output_path Dataset/train_set.jsonl --total_tasks 100 --model groq:llama-3.1-8b-instant

# Clean & select
python synthenexa_pipeline/0_train_set_select.py --input Dataset/train_set.jsonl --output Dataset/selected_promts.jsonl

# Mutate prompts
python synthenexa_pipeline/1_prompt_mutate.py --model groq:llama-3.1-8b-instant --prompt_path Dataset/selected_promts.jsonl --output_path Dataset/mutated_prmts.jsonl

# Evaluate with mutated prompts
python synthenexa_pipeline/2_prompt_evaluate.py --model groq:llama-3.1-8b-instant --trainset_path Dataset/train_set.jsonl --mutated_prompt_path Dataset/mutated_prmts.jsonl --output_path Dataset/evaluatedprmts

# Score & select best
# (First run HumanEval for each output file, then run step 3 script)
```

---

## Notes

- **Groq limits**: If Groq throttles/fails, requests automatically fall back to Ollama.
- You can force Ollama by setting model as `ollama:<ollama_model_name>`.
  
- **Ollama**: Make sure it's running in another terminal:
  ```bash
  ollama serve
  ```
  Default model: `mistral`. Change with `export OLLAMA_MODEL=llama2`

- **Reinforcement Loop**: Run phases 2-3 repeatedly to improve prompts iteratively
- **Evaluation**: HumanEval tests code against hidden test cases within 3-second timeout

