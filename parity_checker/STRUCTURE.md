# Parity Checker Framework - Structure & Components

## Directory Structure

```
parity_checker/
├── agents/                      # Agent modules
│   ├── __init__.py
│   ├── module_analyzer.py      # LLM-based module analysis
│   ├── test_case_generator.py  # Test case generation
│   ├── module_tester.py        # Module testing & comparison
│   └── parity_agent.py         # Main orchestrator
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── llm_client.py           # Qwen API client
├── config.py                    # Configuration & prompts
├── main.py                      # CLI entry point
├── example_usage.py             # Usage examples
├── __init__.py                  # Package init
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
├── QUICKSTART.md               # Quick start guide
└── STRUCTURE.md                # This file
```

## Component Responsibilities

### 1. **utils/llm_client.py** - LLM Integration
**Purpose**: Communicate with Qwen LLM API

**Key Classes**:
- `QwenLLMClient`: Main API client
  - `call()`: Make API requests
  - `analyze_json_response()`: Parse JSON from responses
  - `test_connection()`: Verify API connectivity

**Usage**:
```python
from utils.llm_client import QwenLLMClient

client = QwenLLMClient()
response = client.call(
    messages=[{"role": "user", "content": "Your prompt"}],
    system_prompt="You are..."
)
```

---

### 2. **agents/module_analyzer.py** - Smart Module Analysis
**Purpose**: Use LLM to understand module requirements

**Key Classes**:
- `ModuleAnalyzer`: Analyzes JAX and PyTorch modules
  - `analyze_module()`: Analyze single module
  - `compare_analyses()`: Compare two modules
  - `generate_test_strategy()`: Create comprehensive strategy

**Outputs**:
- Input shapes and dimensions
- Critical parameters to vary
- Edge cases to test
- Testing strategy

**Example**:
```python
from agents.module_analyzer import ModuleAnalyzer

analyzer = ModuleAnalyzer()
analysis, test_config = analyzer.generate_test_strategy(
    jax_file="module_jax.py",
    pytorch_file="module_pytorch.py"
)
```

---

### 3. **agents/test_case_generator.py** - Test Case Generation
**Purpose**: Create comprehensive test inputs

**Key Classes**:
- `TestCaseGenerator`: Generates test cases
  - `generate_random_input()`: Create random tensors
  - `generate_edge_case_inputs()`: Create edge cases
  - `create_test_suite()`: Full suite generation
  - `save_test_cases()`: Persist to disk
  - `load_test_cases()`: Load from disk

**Test Types Generated**:
- Random inputs with multiple seeds
- Edge cases (zeros, ones, small values, large values)
- Multiple batch sizes
- Multiple input shapes

**Example**:
```python
from agents.test_case_generator import TestCaseGenerator

gen = TestCaseGenerator(config)
test_cases = gen.create_test_suite(
    batch_sizes=[1, 2, 4],
    input_shapes=[(3, 32, 32), (1, 64, 64)],
    num_cases_per_config=3
)
gen.save_test_cases("./test_cases")
```

---

### 4. **agents/module_tester.py** - Testing & Comparison
**Purpose**: Run modules and compare outputs

**Key Classes**:
- `ModuleTester`: Tests modules and compares outputs
  - `load_module()`: Dynamically load modules
  - `test_module()`: Run single test
  - `run_test_suite()`: Run all tests
  - `compare_outputs()`: Compare with tolerance
  - `print_results()`: Display results
  - `save_results()`: Save to JSON

**Comparison Metrics**:
- Absolute difference
- Relative difference
- Shape matching
- Tolerance checking

**Example**:
```python
from agents.module_tester import ModuleTester

tester = ModuleTester(tolerance=1e-4, relative_tolerance=1e-3)

# Load modules
pytorch_cls = tester.load_module("module_pytorch.py", "MyModule", "pytorch")
jax_cls = tester.load_module("module_jax.py", "MyModule", "jax")

# Run tests
results = tester.run_test_suite(pytorch_module, jax_module, test_cases)
tester.print_results()
```

---

### 5. **agents/parity_agent.py** - Orchestrator
**Purpose**: Coordinate all components

**Key Classes**:
- `ParityAgent`: Main orchestrator
  - `run_parity_check()`: Full workflow
  - `validate_files()`: Check inputs
  - `_generate_report()`: Create report
  - `_get_llm_analysis()`: Get LLM insights

**Workflow**:
1. Validate input files
2. Analyze modules with LLM
3. Generate test cases
4. Load and instantiate modules
5. Run parity tests
6. Analyze results with LLM
7. Generate comprehensive report

**Example**:
```python
from agents.parity_agent import ParityAgent

agent = ParityAgent(verbose=True)
results = agent.run_parity_check(
    jax_file="module_jax.py",
    pytorch_file="module_pytorch.py",
    jax_class_name="MyModuleJAX",
    pytorch_class_name="MyModulePyTorch",
    output_dir="./results"
)
```

---

### 6. **config.py** - Configuration
**Purpose**: Centralized configuration

**Key Sections**:
- `LLM_CONFIG`: API credentials and settings
- `DEFAULT_TESTING_CONFIG`: Test parameters
- `FRAMEWORK_CONFIG`: Framework settings
- `PROMPTS`: LLM prompts for analysis

**Usage**:
```python
from config import LLM_CONFIG, DEFAULT_TESTING_CONFIG

# Customize as needed
test_config = DEFAULT_TESTING_CONFIG.copy()
test_config["tolerance"] = 1e-5
```

---

### 7. **main.py** - Command Line Interface
**Purpose**: User-friendly CLI

**Usage**:
```bash
python main.py \
  --jax-file module_jax.py \
  --pytorch-file module_pytorch.py \
  --jax-class MyModuleJAX \
  --pytorch-class MyModulePyTorch \
  [--device cuda] \
  [--output-dir ./results] \
  [--verbose]
```

---

## Data Flow

```
User Input
    ↓
main.py (CLI/Script)
    ↓
ParityAgent (Orchestrator)
    ├─→ ModuleAnalyzer
    │   └─→ QwenLLMClient (analyze modules)
    │
    ├─→ TestCaseGenerator
    │   └─→ Generate test inputs
    │
    ├─→ ModuleTester
    │   ├─→ Load modules dynamically
    │   ├─→ Run forward passes
    │   └─→ Compare outputs
    │
    └─→ Report Generation
        └─→ QwenLLMClient (analyze results)

Output: JSON reports + recommendations
```

---

## Output Files

### Generated During Execution

```
parity_results/
├── module_analysis.json         # LLM analysis of modules
├── test_cases/
│   ├── test_cases.pkl          # Serialized test inputs
│   └── test_metadata.json       # Test configuration
└── parity_report.json           # Final comprehensive report
```

### Report Structure

```json
{
  "status": "completed",
  "summary": {
    "total_tests": 50,
    "passed": 48,
    "pass_rate": 0.96,
    ...
  },
  "module_analysis": {...},
  "test_results": [...],
  "llm_analysis": {
    "parity_achieved": true,
    "recommendations": [...]
  },
  "recommendations": [...]
}
```

---

## Configuration Points

### In config.py

```python
# 1. LLM Settings
LLM_CONFIG = {
    "api_key": "YOUR_KEY",  # Update this
    "model": "Qwen/Qwen3-32B",
    "max_tokens": 1000,
    "temperature": 0.7,
}

# 2. Test Parameters
DEFAULT_TESTING_CONFIG = {
    "batch_sizes": [1, 2, 4],                    # Customize for your needs
    "input_shapes": [(3, 32, 32), (1, 64, 64)],
    "tolerance": 1e-4,                           # Adjust tolerance
}

# 3. Framework Settings
FRAMEWORK_CONFIG = {
    "verbose": True,
    "save_reports": True,
    "report_dir": "./reports",
}
```

---

## Extension Points

### Adding Custom Test Cases

```python
from agents.test_case_generator import TestCaseGenerator

gen = TestCaseGenerator(config)
# Generate default tests
test_cases = gen.create_test_suite()
# Add custom test
test_cases.append({
    "id": "custom_001",
    "input": your_custom_tensor,
    "type": "custom"
})
```

### Custom Module Loading

```python
from agents.module_tester import ModuleTester

tester = ModuleTester()
# Extend load_module for custom frameworks
pytorch_module = your_custom_load_function("module.py", "Class")
```

### Custom LLM Prompts

```python
from config import PROMPTS

# Add custom prompt
PROMPTS["my_analysis"] = "Your custom prompt..."

# Use in LLM calls
response = llm.call(
    messages=[{"role": "user", "content": PROMPTS["my_analysis"]}]
)
```

---

## Testing the Framework

```bash
# Test LLM connection
python -c "from utils.llm_client import QwenLLMClient; print(QwenLLMClient().test_connection())"

# Run example
python example_usage.py basic

# Run parity check
python main.py --jax-file module_jax.py --pytorch-file module_pytorch.py --jax-class MyJAX --pytorch-class MyPyTorch
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| LLM connection fails | Check API key in config.py |
| Module not found | Use absolute paths |
| Class not found | Verify class name matches exactly |
| High differences | Check data types, review implementations |
| Out of memory | Reduce batch_sizes in config |

---

## Key Design Decisions

1. **LLM-Powered**: Uses LLM for intelligent analysis, not just running tests
2. **Modular**: Each component is independent and reusable
3. **Reproducible**: Test cases saved for reproducibility
4. **Detailed Reports**: JSON outputs for programmatic access
5. **Extensible**: Easy to add custom tests, modules, prompts

---

## Performance Considerations

- **Default**: ~30 tests per module (varies by config)
- **Time**: 2-10 minutes depending on module complexity
- **Memory**: Minimal (batches are small)
- **LLM Calls**: ~3-4 API calls (analysis + results)

---

That's the complete framework! You can now:
- ✓ Analyze modules with LLM
- ✓ Generate intelligent test cases
- ✓ Run comprehensive parity checks
- ✓ Get LLM-powered recommendations
- ✓ Generate detailed reports

Happy testing! 🚀
