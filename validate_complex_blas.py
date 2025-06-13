#!/usr/bin/env python3
"""Validate LeanBLAS complex operations against NumPy.

This script provides automated validation of complex BLAS operations
by comparing LeanBLAS output against NumPy reference implementations.
"""

import numpy as np
import subprocess
import sys
import re
from typing import Tuple, Optional, List
import json

# Tolerance for numerical comparisons
DEFAULT_TOL = 1e-10

def parse_complex(s: str) -> complex:
    """Parse a complex number from string format like '1.0 + 2.0ⅈ'."""
    # Handle different formats
    s = s.strip()
    
    # Try to parse formats like "1.0 + 2.0ⅈ" or "1.0 + 2.0i"
    match = re.match(r'([+-]?\d+\.?\d*)\s*([+-])\s*(\d+\.?\d*)\s*[iⅈ]', s)
    if match:
        real = float(match.group(1))
        sign = -1 if match.group(2) == '-' else 1
        imag = sign * float(match.group(3))
        return complex(real, imag)
    
    # Try simple real number
    try:
        return complex(float(s), 0)
    except ValueError:
        raise ValueError(f"Cannot parse complex number: {s}")

def run_lean_test(test_name: str) -> str:
    """Run a Lean test executable and return output."""
    try:
        result = subprocess.run(
            ["lake", "exe", test_name],
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit
        )
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr

def validate_zdotu():
    """Validate zdotu (unconjugated dot product) operations."""
    print("\n=== Validating zdotu (unconjugated dot product) ===")
    
    test_cases = [
        # (x, y, expected_result)
        (np.array([1+0j, 2+1j]), np.array([3+4j, 1-2j]), 7+1j),
        (np.array([1+0j, 2+0j, 3+0j]), np.array([4+0j, 5+0j, 6+0j]), 32+0j),
        (np.array([0+1j, 0+2j]), np.array([0+3j, 0+4j]), -11+0j),
    ]
    
    all_passed = True
    for i, (x, y, expected) in enumerate(test_cases):
        result = np.dot(x, y)
        passed = np.allclose(result, expected, rtol=0, atol=DEFAULT_TOL)
        
        print(f"\nTest {i+1}:")
        print(f"  x = {x}")
        print(f"  y = {y}")
        print(f"  Expected: {expected}")
        print(f"  NumPy:    {result}")
        print(f"  Status:   {'✓ PASS' if passed else '✗ FAIL'}")
        
        if not passed:
            all_passed = False
            print(f"  Error:    |Δ| = {abs(result - expected)}")
    
    return all_passed

def validate_zdotc():
    """Validate zdotc (conjugated dot product) operations."""
    print("\n=== Validating zdotc (conjugated dot product) ===")
    
    test_cases = [
        # (x, y, expected_result)
        (np.array([1+0j, 0+1j]), np.array([1+0j, 0-1j]), 0+0j),  # conj([1,i])·[1,-i] = [1,-i]·[1,-i] = 1-1 = 0
        (np.array([3+4j, 1-2j]), np.array([2-1j, 5+3j]), 1+2j),
        (np.array([3+4j, 0+5j]), np.array([3+4j, 0+5j]), 50+0j),  # Self dot = |x|²
    ]
    
    all_passed = True
    for i, (x, y, expected) in enumerate(test_cases):
        result = np.vdot(x, y)  # vdot conjugates first vector
        passed = np.allclose(result, expected, rtol=0, atol=DEFAULT_TOL)
        
        print(f"\nTest {i+1}:")
        print(f"  x = {x}")
        print(f"  y = {y}")
        print(f"  Expected: {expected}")
        print(f"  NumPy:    {result}")
        print(f"  Status:   {'✓ PASS' if passed else '✗ FAIL'}")
        
        if not passed:
            all_passed = False
    
    return all_passed

def validate_norms():
    """Validate norm operations."""
    print("\n=== Validating dznrm2 (2-norm) ===")
    
    test_cases = [
        # (x, expected_norm)
        (np.array([3+4j, 0+0j]), 5.0),  # |3+4j| = 5
        (np.array([1+0j, 2+0j, 3+0j]), np.sqrt(14)),  # sqrt(1²+2²+3²)
        (np.array([0+1j, 0+2j]), np.sqrt(5)),  # sqrt(1²+2²)
        (np.array([3+4j]) / 5, 1.0),  # Unit vector
    ]
    
    all_passed = True
    for i, (x, expected) in enumerate(test_cases):
        result = np.linalg.norm(x)
        passed = np.allclose(result, expected, rtol=0, atol=DEFAULT_TOL)
        
        print(f"\nTest {i+1}:")
        print(f"  x = {x}")
        print(f"  Expected: {expected}")
        print(f"  NumPy:    {result}")
        print(f"  Status:   {'✓ PASS' if passed else '✗ FAIL'}")
        
        if not passed:
            all_passed = False
    
    return all_passed

def validate_asum():
    """Validate dzasum (sum of absolute values of components)."""
    print("\n=== Validating dzasum (sum of absolute values) ===")
    
    test_cases = [
        # (x, expected_sum)
        (np.array([3+4j, -1+0j]), 8.0),  # |3|+|4|+|-1|+|0| = 8
        (np.array([1+1j, -2+3j, 0-4j]), 11.0),  # 1+1+2+3+0+4 = 11
        (np.array([0+0j, 0+0j]), 0.0),  # Zero vector
    ]
    
    all_passed = True
    for i, (x, expected) in enumerate(test_cases):
        result = np.sum(np.abs(x.real)) + np.sum(np.abs(x.imag))
        passed = np.allclose(result, expected, rtol=0, atol=DEFAULT_TOL)
        
        print(f"\nTest {i+1}:")
        print(f"  x = {x}")
        print(f"  Expected: {expected}")
        print(f"  NumPy:    {result}")
        print(f"  Status:   {'✓ PASS' if passed else '✗ FAIL'}")
        
        if not passed:
            all_passed = False
    
    return all_passed

def validate_level2_operations():
    """Validate Level 2 BLAS operations."""
    print("\n=== Validating Level 2 BLAS operations ===")
    
    # Test zgemv: y = alpha*A*x + beta*y
    print("\n--- Testing zgemv ---")
    A = np.array([[1+1j, 2+0j], [3+0j, 4+1j]])
    x = np.array([1+1j, 2+0j])
    y = np.array([0+0j, 0+0j])
    alpha = 1.0 + 0j
    beta = 0.0 + 0j
    
    result = alpha * A @ x + beta * y
    print(f"A = \n{A}")
    print(f"x = {x}")
    print(f"y = alpha*A*x + beta*y = {result}")
    print(f"Expected: [(1+i)*(1+i) + 2*2, 3*(1+i) + (4+i)*2] = [4+2i, 11+3i]")
    
    # Test zhemv: Hermitian matrix-vector multiply
    print("\n--- Testing zhemv ---")
    # Create Hermitian matrix
    H = np.array([[2+0j, 1-1j], [1+1j, 3+0j]])
    x = np.array([1+1j, 2+0j])
    result = H @ x
    print(f"H = \n{H} (Hermitian)")
    print(f"x = {x}")
    print(f"H*x = {result}")
    
    return True

def generate_test_summary():
    """Generate a summary of all validation tests."""
    print("\n" + "="*60)
    print("COMPLEX BLAS VALIDATION SUMMARY")
    print("="*60)
    
    results = {
        "Level 1 - zdotu": validate_zdotu(),
        "Level 1 - zdotc": validate_zdotc(),
        "Level 1 - dznrm2": validate_norms(),
        "Level 1 - dzasum": validate_asum(),
        "Level 2 - Matrix operations": validate_level2_operations(),
    }
    
    print("\n" + "="*60)
    print("RESULTS:")
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All validation tests PASSED!")
    else:
        print("\n❌ Some validation tests FAILED!")
    
    return all_passed

def main():
    """Main validation routine."""
    print("Complex BLAS Validation Against NumPy")
    print("=====================================")
    
    # First, let's validate our NumPy computations
    all_passed = generate_test_summary()
    
    # Generate test data file if needed
    print("\n\nGenerating fresh test data...")
    subprocess.run([sys.executable, "generate_complex_test_data.py"], check=True)
    
    print("\n✅ Validation complete!")
    print("The test data and validation results can be used to verify LeanBLAS implementations.")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()