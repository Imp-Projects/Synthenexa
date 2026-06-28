from __future__ import annotations

import argparse

from pipelines.prompt_evolution.stages.base import StageContext
from pipelines.prompt_evolution.workflow import PromptEvolutionWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Synthenexa prompt evolution pipeline.")
    parser.add_argument("--stage", choices=["mutate", "evaluate", "select"], default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_prompt_path", default="Dataset/base_prompt.jsonl")
    parser.add_argument("--feedback_path", default="Dataset/best_prompt_final.jsonl")
    parser.add_argument("--mutated_prompt_path", default="Dataset/gen_1_mutated_prompts.jsonl")
    parser.add_argument("--evaluate_path", default="Dataset/gen_1_eval_output")
    parser.add_argument("--output_path", default="Dataset/best_prompt_final.jsonl")
    parser.add_argument("--testset_path", default="Dataset/benchmark_mbpp_micro.jsonl")
    parser.add_argument("--num_mutations", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=30)
    args = parser.parse_args()

    context = StageContext(
        model=args.model,
        base_prompt_path=args.base_prompt_path,
        feedback_path=args.feedback_path,
        mutated_prompt_path=args.mutated_prompt_path,
        evaluate_path=args.evaluate_path,
        output_path=args.output_path,
        testset_path=args.testset_path,
        num_mutations=args.num_mutations,
        max_workers=args.max_workers,
    )
    workflow = PromptEvolutionWorkflow()
    if args.full:
        workflow.run_full(context)
    elif args.stage == "mutate":
        workflow.mutation.run(context)
    elif args.stage == "evaluate":
        workflow.evaluation.run(context)
    elif args.stage == "select":
        workflow.selection.run(context)


if __name__ == "__main__":
    main()
