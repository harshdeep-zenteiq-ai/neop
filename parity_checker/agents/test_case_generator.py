"""Test Case Generator - Creates test inputs based on module analysis."""

import numpy as np
import torch
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Union
from pathlib import Path
import json
import pickle


class TestCaseGenerator:
    """Generates test cases for parity checking."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize test case generator.

        Parameters
        ----------
        config : dict
            Testing configuration with batch_sizes, input_shapes, etc.
        """
        self.config = config
        self.test_cases = []

    def generate_random_input(
        self,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        seed: int = None
    ) -> Union[np.ndarray, torch.Tensor, jnp.ndarray]:
        """Generate random input tensor.

        Parameters
        ----------
        shape : tuple
            Tensor shape
        dtype : str
            Data type
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Random input tensor
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random data in reasonable range
        data = np.random.randn(*shape).astype(dtype)
        # Normalize to reasonable range
        return data / np.sqrt(np.prod(shape))

    def generate_edge_case_inputs(
        self,
        shape: Tuple[int, ...]
    ) -> List[np.ndarray]:
        """Generate edge case inputs.

        Parameters
        ----------
        shape : tuple
            Tensor shape

        Returns
        -------
        list
            List of edge case inputs
        """
        edge_cases = []

        # Zeros
        edge_cases.append(np.zeros(shape, dtype="float32"))

        # Ones
        edge_cases.append(np.ones(shape, dtype="float32"))

        # Small values
        edge_cases.append(np.full(shape, 1e-6, dtype="float32"))

        # Large values
        edge_cases.append(np.full(shape, 1e3, dtype="float32"))

        # Mixed positive/negative
        data = np.random.randn(*shape).astype("float32")
        edge_cases.append(data)

        return edge_cases

    def create_test_suite(
        self,
        batch_sizes: List[int] = None,
        input_shapes: List[Tuple] = None,
        num_cases_per_config: int = None
    ) -> List[Dict[str, Any]]:
        """Create comprehensive test suite.

        Parameters
        ----------
        batch_sizes : list, optional
            Batch sizes to test
        input_shapes : list, optional
            Input shapes to test
        num_cases_per_config : int, optional
            Number of test cases per configuration

        Returns
        -------
        list
            Test cases with format:
            [
                {
                    "id": "test_001",
                    "batch_size": 1,
                    "input_shape": (1, 3, 32, 32),
                    "input": np.ndarray,
                    "type": "random"
                },
                ...
            ]
        """
        batch_sizes = batch_sizes or self.config["batch_sizes"]
        input_shapes = input_shapes or self.config["input_shapes"]
        num_cases_per_config = num_cases_per_config or self.config["num_test_cases_per_config"]

        test_cases = []
        test_id = 0

        for batch_size in batch_sizes:
            for input_shape in input_shapes:
                # Create batched input shape
                if isinstance(input_shape, (list, tuple)):
                    full_shape = (batch_size,) + tuple(input_shape)
                else:
                    full_shape = (batch_size, input_shape)

                # Random cases
                for case_num in range(num_cases_per_config):
                    test_id += 1
                    input_data = self.generate_random_input(full_shape, seed=test_id)

                    test_cases.append({
                        "id": f"test_{test_id:03d}",
                        "batch_size": batch_size,
                        "input_shape": full_shape,
                        "input": input_data,
                        "type": "random",
                        "seed": test_id
                    })

                # Edge cases
                for edge_case_input in self.generate_edge_case_inputs(full_shape)[:2]:
                    test_id += 1
                    test_cases.append({
                        "id": f"test_{test_id:03d}",
                        "batch_size": batch_size,
                        "input_shape": full_shape,
                        "input": edge_case_input,
                        "type": "edge_case",
                        "seed": None
                    })

        self.test_cases = test_cases
        return test_cases

    def save_test_cases(self, output_dir: str = "./test_cases"):
        """Save test cases to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save test cases
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as pickle for efficient loading
        pickle_file = output_path / "test_cases.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.test_cases, f)

        # Save metadata as JSON
        metadata = {
            "total_cases": len(self.test_cases),
            "test_config": self.config,
            "cases": [
                {
                    "id": tc["id"],
                    "batch_size": tc["batch_size"],
                    "input_shape": tc["input_shape"],
                    "type": tc["type"]
                }
                for tc in self.test_cases
            ]
        }

        json_file = output_path / "test_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved {len(self.test_cases)} test cases to {output_dir}")

    def load_test_cases(self, input_dir: str = "./test_cases"):
        """Load test cases from disk.

        Parameters
        ----------
        input_dir : str
            Directory containing test cases

        Returns
        -------
        list
            Loaded test cases
        """
        pickle_file = Path(input_dir) / "test_cases.pkl"
        with open(pickle_file, 'rb') as f:
            self.test_cases = pickle.load(f)
        return self.test_cases

    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get current test cases.

        Returns
        -------
        list
            Test cases
        """
        return self.test_cases

    def print_test_summary(self):
        """Print summary of test cases."""
        if not self.test_cases:
            print("No test cases generated")
            return

        print(f"\nTest Suite Summary:")
        print(f"  Total test cases: {len(self.test_cases)}")

        # Group by type
        by_type = {}
        for tc in self.test_cases:
            test_type = tc["type"]
            by_type[test_type] = by_type.get(test_type, 0) + 1

        for test_type, count in by_type.items():
            print(f"  - {test_type}: {count}")

        # Show sample shapes
        print(f"\nSample input shapes:")
        for tc in self.test_cases[:3]:
            print(f"  - {tc['input_shape']} (batch_size={tc['batch_size']})")
