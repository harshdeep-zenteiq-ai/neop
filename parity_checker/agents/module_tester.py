"""Module Tester - Runs modules and compares outputs for parity."""

import sys
import torch
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import traceback


class ModuleTester:
    """Tests modules and compares outputs."""

    def __init__(self, tolerance: float = 1e-4, relative_tolerance: float = 1e-3):
        """Initialize module tester.

        Parameters
        ----------
        tolerance : float
            Absolute tolerance for output comparison
        relative_tolerance : float
            Relative tolerance for output comparison
        """
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.results = []

    def load_module(self, file_path: str, module_class_name: str, framework: str = "pytorch"):
        """Dynamically load a module class.

        Parameters
        ----------
        file_path : str
            Path to module file
        module_class_name : str
            Name of the class to load
        framework : str
            Framework type (pytorch or jax)

        Returns
        -------
        class
            Loaded module class
        """
        # Add file directory to path
        sys.path.insert(0, str(Path(file_path).parent))

        try:
            # Import module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"{framework}_{module_class_name}",
                file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get class
            cls = getattr(module, module_class_name)
            return cls
        except Exception as e:
            print(f"✗ Failed to load {module_class_name} from {file_path}: {e}")
            traceback.print_exc()
            return None

    def tensor_to_numpy(self, tensor) -> np.ndarray:
        """Convert tensor to numpy array.

        Parameters
        ----------
        tensor : torch.Tensor, jax.Array, or np.ndarray
            Tensor to convert

        Returns
        -------
        np.ndarray
            Numpy array
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, jnp.ndarray) or hasattr(tensor, '__array__'):
            return np.asarray(tensor)
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.asarray(tensor)

    def compare_outputs(
        self,
        pytorch_output,
        jax_output,
        test_id: str
    ) -> Dict[str, Any]:
        """Compare outputs from PyTorch and JAX implementations.

        Parameters
        ----------
        pytorch_output
            Output from PyTorch module
        jax_output
            Output from JAX module
        test_id : str
            Test case identifier

        Returns
        -------
        dict
            Comparison results with keys:
            - "test_id": test identifier
            - "pytorch_output": pytorch output shape
            - "jax_output": jax output shape
            - "shapes_match": whether shapes match
            - "max_difference": maximum absolute difference
            - "mean_difference": mean absolute difference
            - "within_tolerance": whether difference is within tolerance
            - "parity": boolean indicating parity
        """
        # Convert to numpy
        pt_np = self.tensor_to_numpy(pytorch_output)
        jax_np = self.tensor_to_numpy(jax_output)

        # Compare
        result = {
            "test_id": test_id,
            "pytorch_shape": str(pt_np.shape),
            "jax_shape": str(jax_np.shape),
            "shapes_match": pt_np.shape == jax_np.shape,
        }

        if pt_np.shape != jax_np.shape:
            result["parity"] = False
            result["error"] = f"Shape mismatch: PyTorch {pt_np.shape} vs JAX {jax_np.shape}"
            return result

        # Calculate differences
        abs_diff = np.abs(pt_np - jax_np)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))

        # Relative difference
        denominator = np.abs(pt_np) + 1e-8
        rel_diff = np.abs((pt_np - jax_np) / denominator)
        max_rel_diff = float(np.max(rel_diff))

        result.update({
            "max_absolute_difference": max_diff,
            "mean_absolute_difference": mean_diff,
            "max_relative_difference": max_rel_diff,
            "within_absolute_tolerance": max_diff < self.tolerance,
            "within_relative_tolerance": max_rel_diff < self.relative_tolerance,
            "parity": max_diff < self.tolerance and max_rel_diff < self.relative_tolerance,
        })

        return result

    def test_module(
        self,
        pytorch_module,
        jax_module,
        test_input: np.ndarray,
        test_id: str,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Test a single test case.

        Parameters
        ----------
        pytorch_module
            Instantiated PyTorch module
        jax_module
            Instantiated JAX module
        test_input : np.ndarray
            Input tensor
        test_id : str
            Test case identifier
        device : str
            Device to run on

        Returns
        -------
        dict
            Test result
        """
        result = {"test_id": test_id, "input_shape": str(test_input.shape)}

        try:
            # Run PyTorch
            with torch.no_grad():
                pt_input = torch.from_numpy(test_input).float()
                if device == "cuda":
                    pt_input = pt_input.cuda()
                pytorch_output = pytorch_module(pt_input)
        except Exception as e:
            result["pytorch_error"] = str(e)
            result["pytorch_traceback"] = traceback.format_exc()
            result["parity"] = False
            return result

        try:
            # Run JAX
            jax_input = jnp.array(test_input, dtype=jnp.float32)
            jax_output = jax_module(jax_input)
        except Exception as e:
            result["jax_error"] = str(e)
            result["jax_traceback"] = traceback.format_exc()
            result["parity"] = False
            return result

        # Compare outputs
        comparison = self.compare_outputs(pytorch_output, jax_output, test_id)
        result.update(comparison)

        return result

    def run_test_suite(
        self,
        pytorch_module,
        jax_module,
        test_cases: List[Dict[str, Any]],
        device: str = "cpu"
    ) -> List[Dict[str, Any]]:
        """Run full test suite.

        Parameters
        ----------
        pytorch_module
            Instantiated PyTorch module
        jax_module
            Instantiated JAX module
        test_cases : list
            List of test cases
        device : str
            Device to run on

        Returns
        -------
        list
            Test results
        """
        self.results = []

        print(f"\nRunning {len(test_cases)} test cases...")
        for i, test_case in enumerate(test_cases):
            test_id = test_case["id"]
            test_input = test_case["input"]

            # Run test
            result = self.test_module(
                pytorch_module,
                jax_module,
                test_input,
                test_id,
                device=device
            )

            self.results.append(result)

            # Print progress
            if "pytorch_error" in result or "jax_error" in result:
                status = "✗ ERROR"
            elif result.get("parity", False):
                status = "✓ PASS"
            else:
                status = "✗ FAIL"

            print(f"  [{i+1}/{len(test_cases)}] {test_id}: {status}")

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self.results:
            return {}

        passed = sum(1 for r in self.results if r.get("parity", False))
        errors = sum(1 for r in self.results if "pytorch_error" in r or "jax_error" in r)
        failed = len(self.results) - passed - errors

        max_diffs = [r.get("max_absolute_difference", float('inf')) for r in self.results if "max_absolute_difference" in r]

        return {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "max_difference_across_tests": max(max_diffs) if max_diffs else None,
            "mean_max_difference": sum(max_diffs) / len(max_diffs) if max_diffs else None,
        }

    def print_results(self):
        """Print test results."""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
        print(f"Max difference: {summary['max_difference_across_tests']:.2e}")
        print(f"Mean max difference: {summary['mean_max_difference']:.2e}")

        # Show failures
        failures = [r for r in self.results if not r.get("parity", False)]
        if failures:
            print(f"\nFailed tests:")
            for r in failures[:5]:
                print(f"  - {r['test_id']}: {r.get('error', 'Check differences')}")
                if "pytorch_error" in r:
                    print(f"    PyTorch: {r['pytorch_error']}")
                if "jax_error" in r:
                    print(f"    JAX: {r['jax_error']}")

    def save_results(self, output_file: str = "./reports/test_results.json"):
        """Save test results to file.

        Parameters
        ----------
        output_file : str
            Output file path
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "summary": self.get_summary(),
            "results": self.results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved results to {output_file}")
