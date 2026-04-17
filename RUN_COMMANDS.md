# Run Commands

Centralized run commands for this repository.

## Start / Dev / Build

This repository does not define JavaScript-style script entries (`start`, `dev`, `build`) in `package.json`.

Equivalent service run commands for `ml-service`:

- Dev:
  - `uvicorn app:app --reload --app-dir ml-service`
- Start (production-like):
  - `uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir ml-service`
- Build:
  - No build step is defined in this repository.

## Environment Setup

- `pip install -r requirements.txt`

## Core Pipeline Commands

From repository docs:

- `python core-pipeline/0_train_set_generate.py --output_path <output_path>`
- `python core-pipeline/utils/train_set_postprocessing.py --file_path <file_path>`
- `python core-pipeline/0_train_set_select.py --input <input_path> --output <output_path> --sample_size <n>`
- `python core-pipeline/1_prompt_mutate.py --model <model> --prompt_path <prompt_path> --output_path <output_path>`
- `python core-pipeline/2_prompt_evaluate.py --model <model> --trainset_path <trainset_path> --mutated_prompt_path <mutated_prompt_path> --output_path <output_path>`
- `python core-pipeline/3_reinforcement_cal_score_and_select.py --evaluate_path <evaluate_path> --testset_path <testset_path> --origin_prompt <origin_prompt> --best_prompt <best_prompt>`
- `evaluate_functional_correctness ./path_to_sanitized_result_file --problem_file=./corresponding_testset`

Additional runnable entrypoints in `core-pipeline`:

- `python core-pipeline/2+3+4_reinfocement.py`
- `python core-pipeline/3_cal_pass1_score_and_select_best_prompt.py`
- `python core-pipeline/duplicate_cleaner.py`
- `python core-pipeline/post_processing.py`
- `python core-pipeline/post_processing_modified.py`
- `python core-pipeline/prompt_test_humaneval_ET.py`
- `python core-pipeline/train_set_generate_new.py`
- `python core-pipeline/human_eval/evaluate_pass_at_k.py`
- `python core-pipeline/evalplus/evaluate.py`
- `python core-pipeline/evalplus/evalperf.py`
- `python core-pipeline/evalplus/inputgen.py`
- `python core-pipeline/evalplus/lecacy_sanitize.py`
- `python core-pipeline/evalplus/sanitize.py`
- `python core-pipeline/evalplus/syncheck.py`
- `python core-pipeline/evalplus/perf/sampling.py`
- `python core-pipeline/evalplus/perf/sas.py`
- `python core-pipeline/evalplus/perf/select_pe_inputs.py`
- `python core-pipeline/evalplus/perf/select_pe_tasks.py`

## Code Translation Commands

Documented/evaluation flow commands:

- `python code_translation/evaluate_prompt_java2python.py`
- `python code_translation/evaluate_prompt_python2java.py`
- `python code_translation/run_python_testcases_avatar.py`
- `python code_translation/run_java_testcases_avatar.py`

Additional runnable entrypoints:

- `python code_translation/Java2Python/1_prompt_mutate_java2python.py`
- `python code_translation/Java2Python/2_prompt_evaluate_java2python.py`
- `python code_translation/Java2Python/3_cal_score_and_extract_best_prompt_java2python.py`
- `python code_translation/Java2Python/4_update_prompt_java2python.py`
- `python code_translation/Python2Java/1_prompt_mutate_python2java.py`
- `python code_translation/Python2Java/2_prompt_evaluate_python2java.py`
- `python code_translation/Python2Java/3_cal_score_cal_and_extract_best_prompt_python2java.py`
- `python code_translation/Python2Java/4_update_prompt_python2java.py`
