"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-game-dialogue-simulator", description="Game developers use this to build a small, quantized GPT for in-game dialogue, then simulate its memory footprint and generation speed for resource-constrained devices.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("prepare-data", help="clean, tokenize, and split the original corpus")
    train_parser = sub.add_parser("train", help="train the local GPT on CPU")
    train_parser.add_argument("--steps", type=int, default=700)
    train_parser.add_argument("--seed", type=int, default=42)
    generate_parser = sub.add_parser("generate", help="sample dialogue from a trained checkpoint")
    generate_parser.add_argument("--prompt", default="Guard:")
    generate_parser.add_argument("--tokens", type=int, default=200)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=12)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--variant", choices=("fp32", "dynamic", "static"), default="fp32")
    sub.add_parser("quantize", help="create dynamic and calibrated static INT8 checkpoints")
    sub.add_parser("report-size", help="compare tensor and checkpoint sizes")
    sub.add_parser("simulate-speed", help="compare modeled device speed and measured laptop speed")
    pipeline_parser = sub.add_parser("run-all", help="run preparation, training, and both reports")
    pipeline_parser.add_argument("--steps", type=int, default=700)
    pipeline_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "simulate-speed":
        from app.storage import report_speed

        print(report_speed())
    if args.command == "run-all":
        from app.storage import prepare_data, quantize, report_size, report_speed, save_samples, train

        prepare_data()
        train(args.steps, args.seed)
        save_samples()
        quantize()
        print(report_size())
        print(report_speed())
    if args.command == "report-size":
        from app.storage import report_size

        print(report_size())
    if args.command == "quantize":
        from app.storage import quantize

        quantize()
    if args.command == "generate":
        from app.storage import generate

        print(generate(args.prompt, args.tokens, args.temperature, args.top_k, args.seed, args.variant))
    if args.command == "train":
        from app.storage import train

        train(args.steps, args.seed)
    if args.command == "prepare-data":
        from app.storage import prepare_data

        data = prepare_data()
        print({name: len(lines) for name, lines in data["splits"].items()})
    if args.command == "hello":
        print("Quantized Game Dialogue Simulator is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
