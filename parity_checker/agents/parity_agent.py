"""Parity Agent - Main orchestrator for the parity checking framework."""

from pathlib import Path
from typing import Dict, Any, Optional
import json

from utils.llm_client import QwenLLMClient
from config import PROMPTS, DEFAULT_TESTING_CONFIG
from agents.module_analyzer import ModuleAnalyzer
from agents.test_case_generator import TestCaseGenerator
from agents.module_tester import ModuleTester


class ParityAgent:
    """Main orchestrator for parity checking."""

    def __init__(self, verbose: bool = True):
        """Initialize parity agent.

        Parameters
        ----------
        verbose : bool
            Whether to print detailed output
        """
        self.verbose = verbose
        self.llm = QwenLLMClient()
        self.analyzer = ModuleAnalyzer(self.llm)
        self.test_generator = None
        self.tester = None
        self.results = None

    def validate_files(self, jax_file: str, pytorch_file: str) -> bool:
        """Validate that module files exist.

        Parameters
        ----------
        jax_file : str
            Path to JAX module
        pytorch_file : str
            Path to PyTorch module

        Returns
        -------
        bool
            True if both files exist
        """
        jax_path = Path(jax_file)
        pytorch_path = Path(pytorch_file)

        if not jax_path.exists():
            print(f"✗ JAX file not found: {jax_file}")
            return False
        if not pytorch_path.exists():
            print(f"✗ PyTorch file not found: {pytorch_file}")
            return False

        return True

    def run_parity_check(
        self,
        jax_file: str,
        pytorch_file: str,
        jax_class_name: str,
        pytorch_class_name: str,
        device: str = "cpu",
        output_dir: str = "./parity_results"
    ) -> Dict[str, Any]:
        """Run complete parity checking workflow.

        Parameters
        ----------
        jax_file : str
            Path to JAX module
        pytorch_file : str
            Path to PyTorch module
        jax_class_name : str
            Name of JAX module class
        pytorch_class_name : str
            Name of PyTorch module class
        device : str
            Device to run on (cpu or cuda)
        output_dir : str
            Directory to save results

        Returns
        -------
        dict
            Parity check results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("PARITY CHECKING FRAMEWORK")
        print("="*70)

        # Step 1: Validate files
        print("\n[STEP 1] Validating files...")
        if not self.validate_files(jax_file, pytorch_file):
            return {"status": "failed", "reason": "File validation failed"}

        # Step 2: Analyze modules
        print("\n[STEP 2] Analyzing modules with LLM...")
        analysis, test_config = self.analyzer.generate_test_strategy(jax_file, pytorch_file)

        # Save analysis
        analysis_file = output_path / "module_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"  ✓ Analysis saved to {analysis_file}")

        # Step 3: Generate test cases
        print("\n[STEP 3] Generating test cases...")
        self.test_generator = TestCaseGenerator(test_config)
        test_cases = self.test_generator.create_test_suite()
        self.test_generator.save_test_cases(str(output_path / "test_cases"))
        self.test_generator.print_test_summary()

        # Step 4: Load modules
        print("\n[STEP 4] Loading modules...")
        pytorch_cls = self.tester.load_module(pytorch_file, pytorch_class_name, framework="pytorch") if self.tester else None
        jax_cls = self.tester.load_module(jax_file, jax_class_name, framework="jax") if self.tester else None

        self.tester = ModuleTester(
            tolerance=test_config.get("tolerance", DEFAULT_TESTING_CONFIG["tolerance"]),
            relative_tolerance=test_config.get("relative_tolerance", DEFAULT_TESTING_CONFIG["relative_tolerance"])
        )

        pytorch_cls = self.tester.load_module(pytorch_file, pytorch_class_name, framework="pytorch")
        jax_cls = self.tester.load_module(jax_file, jax_class_name, framework="jax")

        if pytorch_cls is None or jax_cls is None:
            return {"status": "failed", "reason": "Module loading failed"}

        try:
            pytorch_module = pytorch_cls()
            jax_module = jax_cls()
        except Exception as e:
            print(f"✗ Failed to instantiate modules: {e}")
            return {"status": "failed", "reason": f"Module instantiation failed: {e}"}

        # Step 5: Run tests
        print("\n[STEP 5] Running parity tests...")
        self.results = self.tester.run_test_suite(
            pytorch_module,
            jax_module,
            test_cases,
            device=device
        )

        # Step 6: Analyze results with LLM
        print("\n[STEP 6] Analyzing results with LLM...")
        results_summary = self._format_results_for_llm()
        llm_analysis = self._get_llm_analysis(results_summary)

        # Step 7: Generate report
        print("\n[STEP 7] Generating report...")
        report = self._generate_report(
            analysis=analysis,
            test_config=test_config,
            test_results=self.results,
            llm_analysis=llm_analysis
        )

        # Save report
        report_file = output_path / "parity_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        self.tester.print_results()

        print("\n" + "="*70)
        print("PARITY CHECK COMPLETE")
        print("="*70)
        print(f"Results saved to: {output_dir}")

        return report

    def _format_results_for_llm(self) -> str:
        """Format test results for LLM analysis.

        Returns
        -------
        str
            Formatted results string
        """
        summary = self.tester.get_summary()
        results_text = f"""
Test Results Summary:
- Total tests: {summary['total_tests']}
- Passed: {summary['passed']}
- Failed: {summary['failed']}
- Errors: {summary['errors']}
- Pass rate: {summary['pass_rate']*100:.1f}%
- Max difference: {summary['max_difference_across_tests']:.2e}
- Mean max difference: {summary['mean_max_difference']:.2e}

Failed/Error tests:
"""
        for r in self.results:
            if not r.get("parity", False):
                results_text += f"\n- {r['test_id']}: "
                if "pytorch_error" in r:
                    results_text += f"PyTorch error - {r['pytorch_error'][:100]}"
                elif "jax_error" in r:
                    results_text += f"JAX error - {r['jax_error'][:100]}"
                else:
                    results_text += f"Max diff: {r.get('max_absolute_difference', 'N/A')}"

        return results_text

    def _get_llm_analysis(self, results_summary: str) -> Dict[str, Any]:
        """Get LLM analysis of test results.

        Parameters
        ----------
        results_summary : str
            Formatted test results

        Returns
        -------
        dict
            LLM analysis
        """
        prompt = PROMPTS["parity_analysis"].format(results=results_summary)

        response = self.llm.call(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are an expert in deep learning frameworks. "
                         "Analyze parity test results and provide insights."
        )

        try:
            return self.llm.analyze_json_response(response)
        except Exception as e:
            return {
                "error": f"Failed to parse LLM analysis: {e}",
                "raw_response": response
            }

    def _generate_report(
        self,
        analysis: Dict[str, Any],
        test_config: Dict[str, Any],
        test_results: list,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive parity report.

        Parameters
        ----------
        analysis : dict
            Module analysis results
        test_config : dict
            Test configuration
        test_results : list
            Test results
        llm_analysis : dict
            LLM analysis

        Returns
        -------
        dict
            Complete report
        """
        summary = self.tester.get_summary()

        return {
            "status": "completed",
            "summary": summary,
            "module_analysis": analysis,
            "test_configuration": test_config,
            "test_results": test_results,
            "llm_analysis": llm_analysis,
            "overall_parity": summary["pass_rate"] >= 0.95,
            "recommendations": self._generate_recommendations(summary, llm_analysis)
        }

    def _generate_recommendations(
        self,
        summary: Dict[str, Any],
        llm_analysis: Dict[str, Any]
    ) -> list:
        """Generate recommendations based on results.

        Parameters
        ----------
        summary : dict
            Test summary
        llm_analysis : dict
            LLM analysis

        Returns
        -------
        list
            List of recommendations
        """
        recommendations = []

        if summary["pass_rate"] < 0.95:
            recommendations.append("Investigate failed test cases for implementation differences")

        if summary["errors"] > 0:
            recommendations.append("Fix runtime errors in modules before proceeding")

        if summary["max_difference_across_tests"] and summary["max_difference_across_tests"] > 1e-3:
            recommendations.append("Numerical differences exceed expected tolerance - review algorithm implementations")

        if "recommendations" in llm_analysis:
            recommendations.extend(llm_analysis["recommendations"])

        return recommendations
