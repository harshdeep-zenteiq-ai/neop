"""Configuration for the Parity Checker Agentic Framework."""

# LLM Configuration
LLM_CONFIG = {
    "base_url": "http://34.93.210.88:8080/v1",
    "model": "Qwen/Qwen3-32B",
    "api_key": "gw-idMKBlwAA6aR0dV5u-HP5M76--41UqAvZmA31EWPI8E",
    "max_tokens": 10000,
    "temperature": 0.1,
    "timeout": 120,
}

# Default Testing Configuration (will be adjusted by LLM based on module analysis)
DEFAULT_TESTING_CONFIG = {
    "batch_sizes": [1, 2, 4],
    "input_shapes": [(3, 32, 32), (1, 64, 64)],
    "num_test_cases_per_config": 3,
    "tolerance": 1e-4,  # Numerical tolerance for comparing outputs
    "relative_tolerance": 1e-3,  # Relative tolerance for normalized comparison
    "device": "cpu",
}

# Framework Configuration
FRAMEWORK_CONFIG = {
    "verbose": True,
    "save_reports": True,
    "report_dir": "./reports",
    "test_case_dir": "./test_cases",
    "max_output_size": 1000,  # Max size to print in reports
}

# LLM Prompts
PROMPTS = {
    "module_analysis": """Analyze the following {language} module code and determine:
1. Input tensor shapes and dimensions
2. Expected output shapes
3. Key parameters that affect computation
4. Edge cases to test (e.g., different kernel sizes, padding, stride)
5. Recommended test input shapes for comprehensive testing

Module code:
{code}

Provide a structured analysis in JSON format with keys:
- "input_shapes": list of recommended input shapes to test
- "output_shapes": expected output shapes
- "critical_params": list of important parameters to vary
- "edge_cases": list of edge cases to test
- "test_strategy": brief description of testing strategy""",

    "parity_analysis": """Given the following test results comparing JAX and PyTorch implementations:

Test Results:
{results}

Provide analysis on:
1. Whether the implementations have parity (same outputs)
2. Maximum difference found
3. Whether differences are within acceptable tolerance
4. Potential causes of any differences
5. Recommendations for fixing any issues

Format response as JSON with keys:
- "parity_achieved": boolean
- "max_difference": float
- "within_tolerance": boolean
- "analysis": string describing findings
- "recommendations": list of recommendations""",
}
