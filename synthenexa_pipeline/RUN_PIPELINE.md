python synthenexa_pipeline/0_train_set_generate.py --output_path Dataset/testrun-full-1/train_set.jsonl --total_tasks 150 --model ollama:qwen3-coder:30b

python synthenexa_pipeline/0_train_set_select.py --input Dataset/testrun-full-1/train_set.jsonl --output_path Dataset/testrun-full-1/selected_promts.jsonl --sample_size 70

python synthenexa_pipeline/1_prompt_mutate.py --model ollama:qwen3-coder:30b --prompt_path Dataset/testrun-full-1/selected_promts.jsonl --output_path Dataset/testrun-full-1/mutated_prmts.jsonl --num_mutations=100

python synthenexa_pipeline/2_prompt_evaluate.py --model ollama:qwen3-coder:30b --trainset_path Dataset/testrun-full-1/train_set.jsonl --mutated_prompt_path Dataset/testrun-full-1/mutated_prmts.jsonl --output_path Dataset/testrun-full-1/evaluatedprmts

python synthenexa_pipeline/3_cal_pass1_score_and_select_best_prompt.py --evaluate_path Dataset/testrun-full-1/evaluatedprmts --origin_prompt Dataset/testrun-full-1/mutated_prmts.jsonl --best_prompt Dataset/testrun-full-1/best_prompts.jsonl --testset_path Dataset/testrun-full-1/train_set.jsonl

pkill ollama && OLLAMA_HOST=0.0.0.0:11434 ollama serve & sleep 2 && curl -s http://localhost:11434