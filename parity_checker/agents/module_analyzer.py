"""Module Analyzer Agent - Analyzes module code using LLM to determine test requirements."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

from utils.llm_client import QwenLLMClient
from config import PROMPTS, DEFAULT_TESTING_CONFIG


class ModuleAnalyzer:
    """Analyzes JAX and PyTorch modules to determine testing strategy."""

    def __init__(self, llm_client: QwenLLMClient = None):
        """Initialize the module analyzer.

        Parameters
        ----------
        llm_client : QwenLLMClient, optional
            LLM client instance (creates new one if not provided)
        """
        self.llm = llm_client or QwenLLMClient()

    def read_module_code(self, file_path: str) -> str:
        """Read module code from file.

        Parameters
        ----------
        file_path : str
            Path to the module file

        Returns
        -------
        str
            Module code content
        """
        with open(file_path, 'r') as f:
            return f.read()

    def analyze_module(self, file_path: str, language: str = "python") -> Dict[str, Any]:
        """Analyze a module using LLM to determine test requirements.

        Parameters
        ----------
        file_path : str
            Path to the module file
        language : str
            Programming language (python, jax, etc.)

        Returns
        -------
        dict
            Analysis results with keys:
            - "input_shapes": recommended input shapes
            - "output_shapes": expected output shapes
            - "critical_params": important parameters to vary
            - "edge_cases": edge cases to test
            - "test_strategy": testing strategy description
        """
        # Read module code
        code = self.read_module_code(file_path)

        # Create analysis prompt
        prompt_text = PROMPTS["module_analysis"].format(
            language=language,
            code=code[:3000]  # Limit code length to avoid token issues
        )

        # Call LLM
        print(f"Analyzing module: {file_path}")
        response = self.llm.call(
            messages=[{"role": "user", "content": prompt_text}],
            system_prompt="You are an expert in neural network architecture analysis. "
                         "Provide structured analysis in JSON format."
        )

        # Parse response
        try:
            analysis = self.llm.analyze_json_response(response)
            print(f"✓ Module analysis complete")
            return analysis
        except Exception as e:
            print(f"✗ Failed to parse analysis: {e}")
            print(f"Falling back to default config")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when LLM parsing fails.

        Returns
        -------
        dict
            Default analysis structure
        """
        return {
            "input_shapes": DEFAULT_TESTING_CONFIG["input_shapes"],
            "output_shapes": ["(batch, channels, height, width)"],
            "critical_params": ["batch_size", "input_shape"],
            "edge_cases": ["min_batch_size", "max_batch_size"],
            "test_strategy": "Test with multiple input shapes and batch sizes"
        }

    def compare_analyses(
        self,
        jax_analysis: Dict[str, Any],
        pytorch_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare analyses of JAX and PyTorch modules.

        Parameters
        ----------
        jax_analysis : dict
            Analysis of JAX module
        pytorch_analysis : dict
            Analysis of PyTorch module

        Returns
        -------
        dict
            Comparison results and recommended test config
        """
        # Find common input shapes
        jax_shapes = set(str(s) for s in jax_analysis.get("input_shapes", []))
        pytorch_shapes = set(str(s) for s in pytorch_analysis.get("input_shapes", []))
        common_shapes = jax_shapes & pytorch_shapes

        # Find critical params to test
        jax_params = set(jax_analysis.get("critical_params", []))
        pytorch_params = set(pytorch_analysis.get("critical_params", []))
        common_params = jax_params & pytorch_params

        return {
            "common_input_shapes": list(common_shapes),
            "common_critical_params": list(common_params),
            "jax_specifics": list(jax_params - pytorch_params),
            "pytorch_specifics": list(pytorch_params - jax_params),
            "recommended_test_config": {
                "batch_sizes": DEFAULT_TESTING_CONFIG["batch_sizes"],
                "input_shapes": list(common_shapes) if common_shapes else DEFAULT_TESTING_CONFIG["input_shapes"],
                "num_test_cases_per_config": DEFAULT_TESTING_CONFIG["num_test_cases_per_config"],
                "tolerance": DEFAULT_TESTING_CONFIG["tolerance"],
            }
        }

    def generate_test_strategy(
        self,
        jax_file: str,
        pytorch_file: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate comprehensive test strategy for parity checking.

        Parameters
        ----------
        jax_file : str
            Path to JAX module
        pytorch_file : str
            Path to PyTorch module

        Returns
        -------
        tuple
            (combined_analysis, test_config)
        """
        # Analyze both modules
        print("\n" + "="*60)
        print("ANALYZING MODULES")
        print("="*60)

        jax_analysis = self.analyze_module(jax_file, language="JAX")
        pytorch_analysis = self.analyze_module(pytorch_file, language="PyTorch")

        # Compare analyses
        print("\n" + "="*60)
        print("COMPARING ANALYSES")
        print("="*60)
        combined = self.compare_analyses(jax_analysis, pytorch_analysis)

        # Create final test config
        test_config = {
            **DEFAULT_TESTING_CONFIG,
            **combined["recommended_test_config"]
        }

        print(f"\nRecommended test config:")
        print(f"  - Input shapes: {test_config['input_shapes']}")
        print(f"  - Batch sizes: {test_config['batch_sizes']}")
        print(f"  - Test cases per config: {test_config['num_test_cases_per_config']}")
        print(f"  - Tolerance: {test_config['tolerance']}")

        return combined, test_config
