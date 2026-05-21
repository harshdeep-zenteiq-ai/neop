"""Example usage of Parity Checker framework."""

from agents.parity_agent import ParityAgent
from config import DEFAULT_TESTING_CONFIG


def example_basic_parity_check():
    """Example: Run basic parity check."""

    # Create agent
    agent = ParityAgent(verbose=True)

    # Run parity check
    results = agent.run_parity_check(
        jax_file="../neuralop/layers/fno_block_jax.py",
        pytorch_file="../neuralop/layers/fno_block.py",
        jax_class_name="FNOBlockJAX",
        pytorch_class_name="FNOBlock",
        device="cpu",
        output_dir="./fno_block_parity"
    )

    # Check results
    if results["status"] == "completed":
        print(f"\n{'='*60}")
        print("PARITY CHECK RESULTS")
        print(f"{'='*60}")
        print(f"Overall Parity: {results['overall_parity']}")
        print(f"Pass Rate: {results['summary']['pass_rate']*100:.1f}%")
        print(f"Max Difference: {results['summary']['max_difference_across_tests']:.2e}")

        if results['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")


def example_custom_test_config():
    """Example: Run parity check with custom test configuration."""

    from agents.test_case_generator import TestCaseGenerator
    from agents.module_analyzer import ModuleAnalyzer
    from utils.llm_client import QwenLLMClient

    # Create custom config
    custom_config = {
        "batch_sizes": [1, 4, 8],
        "input_shapes": [(3, 64, 64), (1, 128, 128)],
        "num_test_cases_per_config": 5,
        "tolerance": 1e-5,
        "relative_tolerance": 1e-4,
        "device": "cuda"
    }

    # Analyze modules with LLM
    llm = QwenLLMClient()
    analyzer = ModuleAnalyzer(llm)
    analysis, test_config = analyzer.generate_test_strategy(
        jax_file="../neuralop/layers/fno_block_jax.py",
        pytorch_file="../neuralop/layers/fno_block.py"
    )

    # Override with custom config
    test_config.update(custom_config)

    # Generate test cases
    test_generator = TestCaseGenerator(test_config)
    test_cases = test_generator.create_test_suite()
    test_generator.print_test_summary()


def example_programmatic_testing():
    """Example: Use components programmatically."""

    from agents.module_tester import ModuleTester
    from agents.test_case_generator import TestCaseGenerator
    import torch
    import jax.numpy as jnp
    import numpy as np

    # Create test config
    test_config = {
        "batch_sizes": [2],
        "input_shapes": [(3, 32, 32)],
        "num_test_cases_per_config": 2,
        "tolerance": 1e-4,
        "relative_tolerance": 1e-3,
        "device": "cpu"
    }

    # Generate test cases
    test_gen = TestCaseGenerator(test_config)
    test_cases = test_gen.create_test_suite()

    # Create tester
    tester = ModuleTester(
        tolerance=test_config["tolerance"],
        relative_tolerance=test_config["relative_tolerance"]
    )

    # Load modules
    pytorch_cls = tester.load_module(
        "../neuralop/layers/fno_block.py",
        "FNOBlock",
        framework="pytorch"
    )
    jax_cls = tester.load_module(
        "../neuralop/layers/fno_block_jax.py",
        "FNOBlockJAX",
        framework="jax"
    )

    if pytorch_cls and jax_cls:
        # Instantiate
        pytorch_module = pytorch_cls()
        jax_module = jax_cls()

        # Run tests
        results = tester.run_test_suite(
            pytorch_module,
            jax_module,
            test_cases,
            device="cpu"
        )

        # Print results
        tester.print_results()
        tester.save_results("./manual_test_results.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_name = sys.argv[1]

        if example_name == "basic":
            print("Running: Basic Parity Check")
            example_basic_parity_check()
        elif example_name == "custom_config":
            print("Running: Custom Test Configuration")
            example_custom_test_config()
        elif example_name == "programmatic":
            print("Running: Programmatic Testing")
            example_programmatic_testing()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: basic, custom_config, programmatic")
    else:
        print("Parity Checker - Example Usage")
        print("="*60)
        print("\nUsage: python example_usage.py <example_name>")
        print("\nAvailable examples:")
        print("  1. basic          - Run basic parity check")
        print("  2. custom_config  - Use custom test configuration")
        print("  3. programmatic   - Programmatic component usage")
        print("\nExample:")
        print("  python example_usage.py basic")
