from __future__ import annotations

import argparse

from ml.vision import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locally collected equipment image workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--model-id", choices=["starter", "ai-lead"], required=True
    )
    train_parser.add_argument("--epochs", type=int, default=8)
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
    subparsers.add_parser("export-student")
    args = parser.parse_args()
    if args.command == "prepare":
        print(pipeline.prepare_data())
    elif args.command == "train":
        print(pipeline.train(args.model_id, epochs=args.epochs, replace=args.replace))
    elif args.command == "evaluate":
        print(pipeline.evaluate(args.model_id, args.split, organiser=args.organiser))
    else:
        print(pipeline.export_student_data())


if __name__ == "__main__":
    main()
