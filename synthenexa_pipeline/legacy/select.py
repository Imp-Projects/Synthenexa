from __future__ import annotations

import argparse

from pipelines.prompt_evolution.stages.base import StageContext
from pipelines.prompt_evolution.stages.selection import PromptSelectionStage


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best prompts for the prompt evolution pipeline.")
    parser.add_argument("--evaluate_path", required=True)
    parser.add_argument("--origin_prompt", required=True)
    parser.add_argument("--best_prompt", required=True)
    parser.add_argument("--testset_path", default=None)
    args = parser.parse_args()

    context = StageContext(
        evaluate_path=args.evaluate_path,
        mutated_prompt_path=args.origin_prompt,
        output_path=args.best_prompt,
        testset_path=args.testset_path,
    )
    PromptSelectionStage().run(context)


if __name__ == "__main__":
    main()

