"""Main entry point for Parity Checker framework."""

import argparse
import sys
from pathlib import Path
from agents.parity_agent import ParityAgent


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parity Checker - Compare JAX and PyTorch module implementations"
    )

    parser.add_argument(
        "--jax-file",
        required=True,
        help="Path to JAX module file"
    )

    parser.add_argument(
        "--pytorch-file",
        required=True,
        help="Path to PyTorch module file"
    )

    parser.add_argument(
        "--jax-class",
        required=True,
        help="Name of JAX module class to test"
    )

    parser.add_argument(
        "--pytorch-class",
        required=True,
        help="Name of PyTorch module class to test"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on"
    )

    parser.add_argument(
        "--output-dir",
        default="./parity_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Create agent
    agent = ParityAgent(verbose=args.verbose)

    # Run parity check
    try:
        results = agent.run_parity_check(
            jax_file="/home/deepan/neuraloperator/neuralop/layers/fno_block_jax.py",
            pytorch_file="/home/deepan/neuraloperator/neuralop/layers/fno_block.py",
            jax_class_name="FNOBlockJAX",
            pytorch_class_name="FNOBlock",
            device=args.device,
            output_dir=args.output_dir
        )

        # Exit with appropriate code
        if results.get("status") == "completed" and results.get("overall_parity"):
            print("\n✓ Parity check PASSED")
            return 0
        else:
            print("\n✗ Parity check FAILED")
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
