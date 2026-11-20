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
    args = parser.parse_args(argv)
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
