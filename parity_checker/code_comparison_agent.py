"""Single agent for code comparison between JAX and PyTorch implementations."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from utils.llm_client import QwenLLMClient
from config import PROMPTS


class CodeComparisonAgent:
    """Compares JAX and PyTorch module implementations without running tests."""

    def __init__(self, verbose: bool = True):
        """Initialize the comparison agent.

        Parameters
        ----------
        verbose : bool
            Whether to print detailed output
        """
        self.verbose = verbose
        self.llm = QwenLLMClient()
        self.jax_code = None
        self.pytorch_code = None
        self.jax_file = None
        self.pytorch_file = None

    def read_files(self, jax_file: str, pytorch_file: str) -> bool:
        """Read and validate module files.

        Parameters
        ----------
        jax_file : str
            Path to JAX module file
        pytorch_file : str
            Path to PyTorch module file

        Returns
        -------
        bool
            True if both files read successfully
        """
        jax_path = Path(jax_file)
        pytorch_path = Path(pytorch_file)

        if not jax_path.exists():
            print(f"✗ JAX file not found: {jax_file}")
            return False

        if not pytorch_path.exists():
            print(f"✗ PyTorch file not found: {pytorch_file}")
            return False

        try:
            with open(jax_path, 'r') as f:
                self.jax_code = f.read()
            with open(pytorch_path, 'r') as f:
                self.pytorch_code = f.read()

            self.jax_file = jax_file
            self.pytorch_file = pytorch_file

            if self.verbose:
                print(f"✓ JAX file loaded: {len(self.jax_code)} chars")
                print(f"✓ PyTorch file loaded: {len(self.pytorch_code)} chars")

            return True

        except Exception as e:
            print(f"✗ Error reading files: {e}")
            return False

    def compare_implementations(self) -> Dict[str, Any]:
        """Compare JAX and PyTorch implementations using LLM.

        Returns
        -------
        dict
            Comparison results including differences and assessment
        """
        if not self.jax_code or not self.pytorch_code:
            print("✗ Files not loaded. Call read_files() first.")
            return {}

        prompt = f"""Analyze these two neural network module implementations:

JAX Implementation:
```python
{self.jax_code}
```

PyTorch Implementation:
```python
{self.pytorch_code}
```

Provide a detailed comparison in JSON format with:
1. "overall_parity": true/false - Do they implement the same logic?
2. "key_differences": [list of structural differences]
3. "algorithm_differences": [differences in algorithm implementation]
4. "parameter_differences": [differences in parameters/initialization]
5. "likely_issues": [potential issues that could cause output differences]
6. "compatibility_score": 0-100 (how compatible are they?)
7. "assessment": "String with detailed analysis"
8. "recommendations": [list of things to fix for parity]

Be specific and technical in your analysis."""

        if self.verbose:
            print("\n📊 Analyzing implementations with LLM...")

        try:
            response = self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are an expert in comparing neural network implementations. Provide analysis in JSON format.",
                max_tokens=2000
            )

            # Parse JSON from response
            analysis = self.llm.analyze_json_response(response)

            if self.verbose:
                print("✓ Analysis complete")

            return analysis

        except Exception as e:
            print(f"✗ Error during LLM analysis: {e}")
            return {}

    def generate_report(self, analysis: Dict[str, Any], output_file: str = "comparison_report.json") -> bool:
        """Generate comparison report.

        Parameters
        ----------
        analysis : dict
            Analysis results from compare_implementations()
        output_file : str
            Path to save report

        Returns
        -------
        bool
            True if report saved successfully
        """
        report = {
            "jax_file": str(self.jax_file),
            "pytorch_file": str(self.pytorch_file),
            "analysis": analysis,
            "status": "completed"
        }

        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)

            if self.verbose:
                print(f"\n✓ Report saved: {output_file}")

            return True

        except Exception as e:
            print(f"✗ Error saving report: {e}")
            return False

    def print_assessment(self, analysis: Dict[str, Any]):
        """Print assessment in human-readable format.

        Parameters
        ----------
        analysis : dict
            Analysis results from compare_implementations()
        """
        print("\n" + "="*70)
        print("CODE COMPARISON ASSESSMENT")
        print("="*70)

        if "overall_parity" in analysis:
            parity = "✓ YES" if analysis["overall_parity"] else "✗ NO"
            print(f"\nOverall Parity: {parity}")

        if "compatibility_score" in analysis:
            score = analysis["compatibility_score"]
            print(f"Compatibility Score: {score}/100")

        if "assessment" in analysis:
            print(f"\nAssessment:\n{analysis['assessment']}")

        if "key_differences" in analysis and analysis["key_differences"]:
            print(f"\nKey Differences:")
            for i, diff in enumerate(analysis["key_differences"], 1):
                print(f"  {i}. {diff}")

        if "likely_issues" in analysis and analysis["likely_issues"]:
            print(f"\nLikely Issues:")
            for i, issue in enumerate(analysis["likely_issues"], 1):
                print(f"  {i}. {issue}")

        if "recommendations" in analysis and analysis["recommendations"]:
            print(f"\nRecommendations:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"  {i}. {rec}")

        print("\n" + "="*70)

    def run(self, jax_file: str, pytorch_file: str, output_dir: str = "./comparison_results") -> Dict[str, Any]:
        """Run complete comparison workflow.

        Parameters
        ----------
        jax_file : str
            Path to JAX module file
        pytorch_file : str
            Path to PyTorch module file
        output_dir : str
            Directory to save results

        Returns
        -------
        dict
            Complete analysis results
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Read files
        if not self.read_files(jax_file, pytorch_file):
            return {"status": "failed", "error": "Could not read files"}

        # Compare implementations
        analysis = self.compare_implementations()

        if not analysis:
            return {"status": "failed", "error": "Could not analyze implementations"}

        # Print assessment
        self.print_assessment(analysis)

        # Save report
        report_path = Path(output_dir) / "comparison_report.json"
        self.generate_report(analysis, str(report_path))

        return {
            "status": "completed",
            "analysis": analysis,
            "report_path": str(report_path)
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Code Comparison Agent - Compare JAX and PyTorch implementations"
    )

    parser.add_argument(
        "--jax-file",
        # required=True,
        default="/home/deepan/neuraloperator/neuralop/layers/spectral_convolution_jax.py",
        help="Path to JAX module file"
    )

    parser.add_argument(
        "--pytorch-file",
        # required=True,
        default="/home/deepan/neuraloperator/neuralop/layers/spectral_convolution.py",
        help="Path to PyTorch module file"
    )

    parser.add_argument(
        "--output-dir",
        default="./fno_block_comparison",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Create and run agent
    agent = CodeComparisonAgent(verbose=True)
    results = agent.run(
        jax_file=args.jax_file,
        pytorch_file=args.pytorch_file,
        output_dir=args.output_dir
    )

    # Exit with status
    import sys
    sys.exit(0 if results.get("status") == "completed" else 1)
