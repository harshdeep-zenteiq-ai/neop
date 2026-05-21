#!/bin/bash

echo "========================================================================"
echo "TIMING COMPARISON: JAX vs PyTorch"
echo "========================================================================"
echo ""

echo "Running JAX timing test..."
echo "========================================================================"
cd /home/deepan/neuraloperator
python test_timing_jax.py 2>&1 | tee /tmp/jax_timing.log

echo ""
echo ""
echo "Running PyTorch timing test..."
echo "========================================================================"
python test_timing_torch.py 2>&1 | tee /tmp/torch_timing.log

echo ""
echo ""
echo "========================================================================"
echo "COMPARISON SUMMARY"
echo "========================================================================"
echo ""
echo "JAX Results:"
grep -A 5 "JAX SUMMARY" /tmp/jax_timing.log
echo ""
echo "PyTorch Results:"
grep -A 5 "PyTorch SUMMARY" /tmp/torch_timing.log
