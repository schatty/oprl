import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--env", type=str, default="cartpole-balance", help="Name of the environment."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of parallel processes launched with different random seeds.",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="Number of the first seed. Following seeds will be incremented from it.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to perform training on."
    )
    return parser.parse_args()

