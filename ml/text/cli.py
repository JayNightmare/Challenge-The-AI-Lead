from __future__ import annotations

import argparse

from ml.text import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="UCI SMS spam model workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--model-id", choices=["starter", "ai-lead"], required=True
    )
    train_parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace the fixed local artifact",
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--model-id", choices=["starter", "ai-lead"], required=True
    )
    evaluate_parser.add_argument(
        "--split", choices=["validation", "test"], required=True
    )
    evaluate_parser.add_argument("--organiser", action="store_true")
    subparsers.add_parser("summarize")
    subparsers.add_parser("export-student")
    args = parser.parse_args()
    if args.command == "prepare":
        print(pipeline.prepare_data())
    elif args.command == "train":
        print(pipeline.train(args.model_id, replace=args.replace))
    elif args.command == "evaluate":
        report = pipeline.evaluate(args.model_id, args.split, organiser=args.organiser)
        print(
            {
                "task": report["task"],
                "model_id": report["model_id"],
                "split": report["split"],
                "metrics": report["metrics"],
                "reference": report["reference"],
            }
        )
    elif args.command == "export-student":
        print(pipeline.export_student_data())
    else:
        print(pipeline.summarize())


if __name__ == "__main__":
    main()
