# Parity Checker - Quick Start Guide

Get started with parity checking in 5 minutes!

## Installation

```bash
# Navigate to parity_checker directory
cd parity_checker

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Update LLM credentials** in `config.py`:
   ```python
   LLM_CONFIG = {
       "base_url": "http://34.93.210.88:8080/v1",
       "model": "Qwen/Qwen3-32B",
       "api_key": "YOUR_API_KEY",  # ← Update this
       ...
   }
   ```

2. **Test LLM connection**:
   ```python
   from utils.llm_client import QwenLLMClient
   
   client = QwenLLMClient()
   if client.test_connection():
       print("✓ LLM connection successful!")
   ```

## Quick Examples

### 1. Basic Parity Check (5 minutes)

```bash
python main.py \
  --jax-file /path/to/module_jax.py \
  --pytorch-file /path/to/module_pytorch.py \
  --jax-class MyModuleJAX \
  --pytorch-class MyModulePyTorch
```

Output:
```
==================== PARITY CHECK COMPLETE ====================
Total tests: 30
Passed: 30
Pass rate: 100.0%
Max difference: 1.23e-05

✓ Parity check PASSED
```

### 2. With Custom Options

```bash
python main.py \
  --jax-file /path/to/module_jax.py \
  --pytorch-file /path/to/module_pytorch.py \
  --jax-class MyModuleJAX \
  --pytorch-class MyModulePyTorch \
  --device cuda \
  --output-dir ./my_results \
  --verbose
```

### 3. Python Script

```python
from agents.parity_agent import ParityAgent

# Create and run
agent = ParityAgent(verbose=True)
results = agent.run_parity_check(
    jax_file="module_jax.py",
    pytorch_file="module_pytorch.py",
    jax_class_name="MyModuleJAX",
    pytorch_class_name="MyModulePyTorch"
)

# Check results
if results["overall_parity"]:
    print("✓ Parity achieved!")
```

## Results

After running, check the output directory for:

1. **`parity_report.json`** - Main report
2. **`module_analysis.json`** - LLM analysis of modules
3. **`test_cases/`** - Generated test cases
   - `test_cases.pkl` - Serialized inputs
   - `test_metadata.json` - Configuration

## Understanding Results

### Pass Rate
- **90-100%**: Great! Minor differences acceptable
- **70-90%**: Investigate failing tests
- **<70%**: Significant differences detected

### Max Difference
- **< 1e-4**: Excellent parity
- **1e-4 to 1e-3**: Good, likely numerical precision
- **> 1e-3**: Investigate implementation

## Common Issues

### "Module not found"
- Ensure full paths are absolute
- Class names must match exactly
- Check module imports work standalone

### "LLM connection failed"
- Verify API key in config.py
- Check network connectivity
- Test with: `client.test_connection()`

### High differences
- Check data type conversions (float32 vs float64)
- Review algorithm implementations
- Verify padding and stride parameters

## Next Steps

1. Run your first parity check ✓
2. Review results in `parity_report.json`
3. Read LLM analysis and recommendations
4. Fix any issues identified
5. Re-run to verify fixes

## Tips

- **Start small**: Test with simple shapes first
- **Read recommendations**: LLM analysis is detailed
- **Check edge cases**: Parity checker tests zeros, ones, etc.
- **Review logs**: Add `--verbose` for detailed output
- **Iterate**: Fix issues and re-run tests

## Example: Testing FNO Blocks

```bash
python main.py \
  --jax-file ../neuralop/layers/fno_block_jax.py \
  --pytorch-file ../neuralop/layers/fno_block.py \
  --jax-class FNOBlockJAX \
  --pytorch-class FNOBlock \
  --output-dir ./fno_parity_results
```

## Help

```bash
# Show all options
python main.py --help

# See examples
python example_usage.py

# Check README for detailed documentation
cat README.md
```

---

**That's it!** You're ready to check parity between your JAX and PyTorch modules. 🚀
