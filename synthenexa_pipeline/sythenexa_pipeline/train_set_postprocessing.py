from utils.train_set_postprocessing import process_file
from utils.paths import resolve_pipeline_path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Post-process generated training set JSONL.")
    parser.add_argument("--file_path", required=True, help="Path to generated JSONL file.")
    args = parser.parse_args()
    process_file(str(resolve_pipeline_path(args.file_path)))


if __name__ == "__main__":
    main()
