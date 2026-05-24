python synthenexa_pipeline/0_train_set_generate.py --output_path Dataset/testrun-full/train_set.jsonl --total_tasks 700 --model ollama:qwen3-coder:30b

python synthenexa_pipeline/0_train_set_select.py --input Dataset/testrun-full/train_set.jsonl --output_path Dataset/testrun-full/selected_promts.jsonl --sample_size 500

python synthenexa_pipeline/1_prompt_mutate.py --model ollama:qwen3-coder:30b --prompt_path Dataset/testrun-full/selected_promts.jsonl --output_path Dataset/testrun-full/mutated_prmts.jsonl

python synthenexa_pipeline/2_prompt_evaluate.py --model ollama:qwen3-coder:30b --trainset_path Dataset/testrun-full/train_set.jsonl --mutated_prompt_path Dataset/testrun-full/mutated_prmts.jsonl --output_path Dataset/testrun-full/evaluatedprmts

python synthenexa_pipeline/3_cal_pass1_score_and_select_best_prompt.py --evaluate_path Dataset/testrun-full/evaluatedprmts --origin_prompt Dataset/testrun-full/mutated_prmts.jsonl --best_prompt Dataset/testrun-full/best_prompts.jsonl --testset_path Dataset/testrun-full/train_set.jsonl