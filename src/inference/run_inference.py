"""
run_inference.py
----------------
Command-Line Interface for the DQN Dynamic Pricing Inference module.

This script is the standalone entry point for price recommendations.
It can be run:
  1. Interactively — prompts the user for state values one at a time.
  2. With explicit flags — ``--price``, ``--inventory``, ``--days``.
  3. In batch mode — reads multiple states from a JSON file.

The script is entirely independent of the training pipeline.
The only artefact it requires is a checkpoint file produced by training.

Usage
-----
    # Interactive prompt (state values entered one at a time)
    python -m src.inference.run_inference checkpoints/dqn_final.pt

    # Single prediction via CLI flags
    python -m src.inference.run_inference checkpoints/dqn_final.pt \\
           --price 60 --inventory 40 --days 7

    # Batch prediction from a JSON file
    python -m src.inference.run_inference checkpoints/dqn_final.pt \\
           --batch states.json

    # Demo mode (no checkpoint — uses random weights for demonstration)
    python -m src.inference.run_inference --demo

JSON batch file format
----------------------
The batch JSON file must contain a list of state objects::

    [
      {"current_price": 50, "remaining_inventory": 80, "remaining_days": 30},
      {"current_price": 80, "remaining_inventory": 20, "remaining_days": 5}
    ]

or a list of plain arrays::

    [[50, 80, 30], [80, 20, 5]]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument object.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.inference.run_inference",
        description=(
            "DQN Dynamic Pricing — Inference CLI.\n"
            "Load a trained checkpoint and get a price recommendation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to the .pt checkpoint file (e.g. checkpoints/dqn_final.pt). "
             "Required unless --demo is set.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        metavar="PRICE",
        help="Current price being offered (feature 0). Range: 10–100.",
    )
    parser.add_argument(
        "--inventory",
        type=float,
        default=None,
        metavar="INVENTORY",
        help="Remaining inventory (feature 1). Range: 0–100.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        metavar="DAYS",
        help="Remaining days in the booking horizon (feature 2). Range: 0–100.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a JSON file containing a list of states for batch inference.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run in demo mode with random network weights. "
            "Useful for testing the inference pipeline without a trained checkpoint."
        ),
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        default=True,
        help="Suppress the formatted display panel (print only the price).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional: save inference result(s) to a JSON file.",
    )

    return parser.parse_args()


def _prompt_state() -> list:
    """Interactively prompt the user to enter state feature values.

    Returns:
        list[float]: Three-element state vector.
    """
    print()
    print("  Enter the current market state (press Enter to use the default):")
    print()

    def _ask(label: str, default: float, lo: float, hi: float) -> float:
        while True:
            raw = input(f"    {label} [{lo}–{hi}] (default {default}): ").strip()
            if not raw:
                return default
            try:
                val = float(raw)
                if lo <= val <= hi:
                    return val
                print(f"      ✗ Value must be between {lo} and {hi}. Try again.")
            except ValueError:
                print("      ✗ Please enter a number.")

    current_price       = _ask("Current price       ", default=50.0, lo=10.0, hi=100.0)
    remaining_inventory = _ask("Remaining inventory ", default=50.0, lo=0.0,  hi=100.0)
    remaining_days      = _ask("Remaining days      ", default=15.0, lo=0.0,  hi=100.0)

    return [current_price, remaining_inventory, remaining_days]


def _load_batch_file(path: str) -> list:
    """Load a list of states from a JSON file.

    Supports two formats:
      - List of dicts: ``[{"current_price": 50, "remaining_inventory": 80, "remaining_days": 30}]``
      - List of arrays: ``[[50, 80, 30], [80, 20, 5]]``

    Args:
        path (str): Path to the JSON file.

    Returns:
        list[list[float]]: List of state vectors.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file content cannot be parsed as states.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Batch file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Batch JSON must be a non-empty list of states.")

    states = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            try:
                state = [
                    float(item["current_price"]),
                    float(item["remaining_inventory"]),
                    float(item["remaining_days"]),
                ]
            except KeyError as exc:
                raise ValueError(
                    f"State {i} is missing key {exc}. "
                    "Expected keys: 'current_price', 'remaining_inventory', 'remaining_days'."
                ) from exc
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            state = [float(v) for v in item]
        else:
            raise ValueError(
                f"State {i} has unexpected format: {item!r}. "
                "Provide a dict with 3 keys or a list of 3 numbers."
            )
        states.append(state)

    return states


def _build_demo_predictor():
    """Build a predictor with random weights for demonstration purposes.

    No checkpoint is required.  The Q-values will be random, but the
    full inference pipeline (state → action → price → display) works
    exactly as it would with a trained model.

    Returns:
        PricingPredictor: Demo predictor backed by random weights.
    """
    from src.agents.dqn_agent import DQNAgent
    from src.config.dqn_config import DQNConfig
    from src.inference.predictor import PricingPredictor

    cfg   = DQNConfig()
    agent = DQNAgent.from_config(cfg)     # random initial weights
    print("  [inference] DEMO MODE — using random network weights")
    print("  [inference] Train the agent first for meaningful recommendations.")
    print()
    return PricingPredictor(agent=agent, checkpoint_path="<demo>", cfg=cfg)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main function for the inference CLI.

    Returns:
        int: Exit code (0 = success, 1 = error).
    """
    args = _parse_args()

    # ── Resolve checkpoint / demo mode ─────────────────────────────────
    from src.inference.predictor import PricingPredictor

    if args.demo:
        predictor = _build_demo_predictor()
    elif args.checkpoint is None:
        # Auto-discover most recent checkpoint
        default_ckpt = "checkpoints/dqn_final.pt"
        if os.path.isfile(default_ckpt):
            print(f"  [inference] No checkpoint specified — using {default_ckpt}")
            args.checkpoint = default_ckpt
        else:
            print(
                "\n  [ERROR] No checkpoint path provided and no default checkpoint found.\n"
                "  Run training first:  python -m src.training.train\n"
                "  Or use demo mode:    python -m src.inference.run_inference --demo\n"
            )
            return 1
    else:
        pass  # checkpoint was provided explicitly

    if not args.demo:
        try:
            predictor = PricingPredictor.from_checkpoint(args.checkpoint)
        except FileNotFoundError as exc:
            print(f"\n  [ERROR] {exc}\n")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [ERROR] Could not load checkpoint: {exc}\n")
            return 1

    # ── Determine inference mode ────────────────────────────────────────

    if args.batch:
        # ── Batch mode ────────────────────────────────────────────────
        try:
            states = _load_batch_file(args.batch)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n  [ERROR] {exc}\n")
            return 1

        print(f"\n  Running batch inference on {len(states)} state(s)…\n")
        results = predictor.predict_batch(states)

        for i, result in enumerate(results):
            print(f"  ── State {i + 1} ──")
            if args.display:
                result.display()
            else:
                print(f"  Recommended price: £{result.recommended_price:.2f}")
            print()

        if args.json_out:
            payload = [r.as_dict() for r in results]
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"  [inference] Results saved → {args.json_out}")

    else:
        # ── Single prediction mode ────────────────────────────────────
        all_provided = all(v is not None for v in [args.price, args.inventory, args.days])

        if all_provided:
            # State provided via CLI flags
            state = [args.price, args.inventory, args.days]
            print(
                f"\n  State: price={args.price}  "
                f"inventory={args.inventory}  "
                f"days={args.days}\n"
            )
        else:
            # Interactive prompt
            state = _prompt_state()

        try:
            result = predictor.predict(state)
        except ValueError as exc:
            print(f"\n  [ERROR] {exc}\n")
            return 1

        print()
        if args.display:
            result.display()
        else:
            print(f"  Recommended price: £{result.recommended_price:.2f}")
            print(f"  Action index     : {result.action}")
            print(f"  Confidence       : {result.confidence * 100:.2f} %")

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(result.as_dict(), fh, indent=2)
            print(f"\n  [inference] Result saved → {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
