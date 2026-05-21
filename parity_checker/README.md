# Parity Checker - Agentic Framework

An intelligent agentic framework for checking parity between JAX and PyTorch neural network module implementations. Uses LLM (Qwen 3: 32B) for smart analysis and test generation.

## Overview

The framework automatically:
1. **Analyzes modules** using LLM to understand their structure and requirements
2. **Generates intelligent test cases** based on module analysis
3. **Runs comprehensive tests** with multiple input shapes and edge cases
4. **Compares outputs** with configurable tolerance levels
5. **Generates reports** with LLM-powered insights and recommendations

## Architecture

```
parity_checker/
├── config.py                 # Configuration and prompts
├── main.py                   # CLI entry point
├── agents/
│   ├── module_analyzer.py   # LLM-based module analysis
│   ├── test_case_generator.py # Test case generation
│   ├── module_tester.py      # Module testing and comparison
│   └── parity_agent.py       # Main orchestrator
├── utils/
│   └── llm_client.py         # Qwen API client
├── test_cases/               # Generated test cases
├── reports/                  # Test results and reports
└── README.md                 # This file
```

## Components

### 1. ModuleAnalyzer
- Reads JAX and PyTorch module code
- Uses LLM to understand:
  - Expected input/output shapes
  - Critical parameters
  - Edge cases
  - Testing strategy

### 2. TestCaseGenerator
- Generates random test inputs
- Creates edge case variations
- Saves test cases for reproducibility
- Supports batch variations and shape combinations

### 3. ModuleTester
- Dynamically loads and instantiates modules
- Runs test cases on both implementations
- Compares outputs with configurable tolerance
- Tracks detailed metrics:
  - Absolute differences
  - Relative differences
  - Pass/fail status

### 4. ParityAgent (Orchestrator)
- Coordinates all components
- Runs complete workflow
- Generates comprehensive reports
- Provides LLM-powered analysis and recommendations

## Installation

```bash
# Install dependencies
pip install torch jax jax[cuda]  # Optional: cuda support
pip install requests numpy

# Ensure config.py has correct LLM credentials
```

## Usage

### Command Line Interface

```bash
# Basic usage
python main.py \
  --jax-file path/to/module_jax.py \
  --pytorch-file path/to/module_pytorch.py \
  --jax-class MyModuleJAX \
  --pytorch-class MyModulePyTorch

# With options
python main.py \
  --jax-file path/to/module_jax.py \
  --pytorch-file path/to/module_pytorch.py \
  --jax-class MyModuleJAX \
  --pytorch-class MyModulePyTorch \
  --device cuda \
  --output-dir ./my_results \
  --verbose
```

### Programmatic Usage

```python
from agents.parity_agent import ParityAgent

# Create agent
agent = ParityAgent(verbose=True)

# Run parity check
results = agent.run_parity_check(
    jax_file="path/to/module_jax.py",
    pytorch_file="path/to/module_pytorch.py",
    jax_class_name="MyModuleJAX",
    pytorch_class_name="MyModulePyTorch",
    device="cpu",
    output_dir="./results"
)

# Check results
if results["overall_parity"]:
    print("✓ Parity achieved!")
else:
    print("✗ Parity issues detected")
    print("Recommendations:", results["recommendations"])
```

## Configuration

Edit `config.py` to customize:

### LLM Configuration
```python
LLM_CONFIG = {
    "base_url": "http://34.93.210.88:8080/v1",
    "model": "Qwen/Qwen3-32B",
    "api_key": "YOUR_API_KEY",
    "max_tokens": 1000,
    "temperature": 0.7,
}
```

### Testing Configuration
```python
DEFAULT_TESTING_CONFIG = {
    "batch_sizes": [1, 2, 4],                    # Batch sizes to test
    "input_shapes": [(3, 32, 32), (1, 64, 64)],  # Input shapes
    "num_test_cases_per_config": 3,              # Random cases per config
    "tolerance": 1e-4,                           # Absolute tolerance
    "relative_tolerance": 1e-3,                  # Relative tolerance
    "device": "cpu",                             # cpu or cuda
}
```

## Output

The framework generates several outputs in the results directory:

### 1. `module_analysis.json`
Module analysis from LLM:
```json
{
  "common_input_shapes": [...],
  "common_critical_params": [...],
  "recommended_test_config": {...}
}
```

### 2. `test_cases/`
Generated test cases:
- `test_cases.pkl` - Serialized test inputs
- `test_metadata.json` - Test configuration and metadata

### 3. `parity_report.json`
Comprehensive report:
```json
{
  "status": "completed",
  "summary": {
    "total_tests": 50,
    "passed": 48,
    "failed": 2,
    "errors": 0,
    "pass_rate": 0.96
  },
  "overall_parity": true,
  "llm_analysis": {...},
  "recommendations": [...]
}
```

## Workflow

```
┌─────────────────────────────────────┐
│ 1. Validate Input Files             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2. Analyze Modules with LLM         │
│    - Understand structure           │
│    - Identify critical params       │
│    - Plan test strategy             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 3. Generate Test Cases              │
│    - Random inputs                  │
│    - Edge cases                     │
│    - Multiple shapes/batches        │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 4. Load & Instantiate Modules       │
│    - Dynamically load classes       │
│    - Create instances               │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 5. Run Tests & Compare Outputs      │
│    - Forward pass on both           │
│    - Compare with tolerance         │
│    - Track differences              │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 6. LLM Analysis of Results          │
│    - Analyze differences            │
│    - Suggest fixes                  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 7. Generate Report & Recommendations│
│    - Summary statistics             │
│    - Detailed results               │
│    - Next steps                     │
└─────────────────────────────────────┘
```

## Example

### Testing FNO Blocks

```bash
python main.py \
  --jax-file ../neuralop/layers/fno_block_jax.py \
  --pytorch-file ../neuralop/layers/fno_block.py \
  --jax-class FNOBlockJAX \
  --pytorch-class FNOBlock \
  --output-dir ./fno_block_parity
```

### Result Example
```
==================== PARITY CHECK COMPLETE ====================
Results saved to: ./fno_block_parity

============================================================
TEST RESULTS SUMMARY
============================================================
Total tests: 45
Passed: 44
Failed: 1
Errors: 0
Pass rate: 97.8%
Max difference: 1.23e-05
Mean max difference: 2.45e-06
```

## Troubleshooting

### LLM Connection Issues
- Verify API credentials in `config.py`
- Test with: `agent.llm.test_connection()`
- Check network access to `34.93.210.88:8080`

### Module Loading Errors
- Ensure module classes are properly defined
- Check that imports within modules are resolvable
- Verify class names match exactly

### Test Failures
- Review `parity_report.json` for detailed differences
- Check `llm_analysis` section for recommendations
- Increase tolerance if differences are within acceptable range

## Next Steps

1. Run parity check on your modules
2. Review the generated report
3. Fix any issues identified by LLM analysis
4. Re-run to verify fixes

## License

Same as parent project (NeuralOperator)

## Support

For issues or questions, check:
1. `parity_report.json` - Detailed results
2. LLM recommendations in the report
3. Test case metadata in `test_cases/test_metadata.json`
