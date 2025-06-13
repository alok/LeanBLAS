#!/usr/bin/env python3
"""Generate test data for complex BLAS operations using NumPy/SciPy.

This script generates test vectors and expected results for complex BLAS
operations that can be used to validate LeanBLAS implementations.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any

def complex_to_dict(z: complex) -> Dict[str, float]:
    """Convert complex number to dictionary format."""
    return {"real": float(z.real), "imag": float(z.imag)}

def array_to_list(arr: np.ndarray) -> List[Dict[str, float]]:
    """Convert complex numpy array to list of dictionaries."""
    return [complex_to_dict(z) for z in arr.flatten()]

def matrix_to_list(mat: np.ndarray) -> List[List[Dict[str, float]]]:
    """Convert complex matrix to nested list format."""
    return [[complex_to_dict(z) for z in row] for row in mat]

class ComplexBLASTestGenerator:
    """Generate test cases for complex BLAS operations."""
    
    def __init__(self):
        self.test_data = {}
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def generate_level1_tests(self):
        """Generate test cases for Level 1 BLAS operations."""
        print("Generating Level 1 BLAS test cases...")
        
        # Test vectors of various sizes
        sizes = [2, 3, 5, 10]
        level1_tests = []
        
        for n in sizes:
            # Generate test vectors
            x = np.random.randn(n) + 1j * np.random.randn(n)
            y = np.random.randn(n) + 1j * np.random.randn(n)
            alpha = complex(2.5, -1.5)
            
            # zdotu - unconjugated dot product
            zdotu_result = np.dot(x, y)
            
            # zdotc - conjugated dot product
            zdotc_result = np.vdot(x, y)  # vdot conjugates first vector
            
            # dznrm2 - 2-norm
            norm2_result = np.linalg.norm(x)
            
            # dzasum - sum of absolute values of real and imaginary parts
            asum_result = np.sum(np.abs(x.real)) + np.sum(np.abs(x.imag))
            
            # zscal - scale by complex number
            scaled_x = alpha * x.copy()
            
            # zaxpy - y = alpha*x + y
            axpy_result = alpha * x + y
            
            # zcopy - just copy x to y
            copy_result = x.copy()
            
            # zswap - swap x and y
            x_swap, y_swap = y.copy(), x.copy()
            
            test_case = {
                "size": n,
                "x": array_to_list(x),
                "y": array_to_list(y),
                "alpha": complex_to_dict(alpha),
                "results": {
                    "zdotu": complex_to_dict(zdotu_result),
                    "zdotc": complex_to_dict(zdotc_result),
                    "dznrm2": float(norm2_result),
                    "dzasum": float(asum_result),
                    "zscal": array_to_list(scaled_x),
                    "zaxpy": array_to_list(axpy_result),
                    "zcopy": array_to_list(copy_result),
                    "zswap": {
                        "x_after": array_to_list(x_swap),
                        "y_after": array_to_list(y_swap)
                    }
                }
            }
            level1_tests.append(test_case)
        
        # Add special test cases
        special_tests = self.generate_special_cases_level1()
        level1_tests.extend(special_tests)
        
        self.test_data["level1"] = level1_tests
    
    def generate_special_cases_level1(self):
        """Generate special test cases for Level 1 operations."""
        special_tests = []
        
        # Test 1: Pure real numbers
        x = np.array([1.0, 2.0, 3.0], dtype=complex)
        y = np.array([4.0, 5.0, 6.0], dtype=complex)
        special_tests.append({
            "name": "pure_real",
            "size": 3,
            "x": array_to_list(x),
            "y": array_to_list(y),
            "results": {
                "zdotu": complex_to_dict(np.dot(x, y)),
                "zdotc": complex_to_dict(np.vdot(x, y)),
                "dznrm2": float(np.linalg.norm(x))
            }
        })
        
        # Test 2: Pure imaginary numbers
        x = np.array([1j, 2j, 3j])
        y = np.array([4j, 5j, 6j])
        special_tests.append({
            "name": "pure_imaginary",
            "size": 3,
            "x": array_to_list(x),
            "y": array_to_list(y),
            "results": {
                "zdotu": complex_to_dict(np.dot(x, y)),
                "zdotc": complex_to_dict(np.vdot(x, y)),
                "dznrm2": float(np.linalg.norm(x))
            }
        })
        
        # Test 3: Unit vector norm
        x = np.array([3+4j]) / 5  # |3+4j| = 5
        special_tests.append({
            "name": "unit_norm",
            "size": 1,
            "x": array_to_list(x),
            "results": {
                "dznrm2": float(np.linalg.norm(x))  # Should be 1.0
            }
        })
        
        return special_tests
    
    def generate_level2_tests(self):
        """Generate test cases for Level 2 BLAS operations."""
        print("Generating Level 2 BLAS test cases...")
        
        level2_tests = []
        
        # Test different matrix sizes
        sizes = [(2, 2), (3, 3), (4, 3), (3, 4)]
        
        for m, n in sizes:
            # Generate test data
            A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
            x = np.random.randn(n) + 1j * np.random.randn(n)
            y = np.random.randn(m) + 1j * np.random.randn(m)
            alpha = complex(1.5, 0.5)
            beta = complex(0.5, -0.5)
            
            # zgemv - general matrix-vector multiply: y = alpha*A*x + beta*y
            gemv_result = alpha * A @ x + beta * y
            
            test_case = {
                "m": m,
                "n": n,
                "A": matrix_to_list(A),
                "x": array_to_list(x),
                "y": array_to_list(y),
                "alpha": complex_to_dict(alpha),
                "beta": complex_to_dict(beta),
                "results": {
                    "zgemv": array_to_list(gemv_result)
                }
            }
            
            # For square matrices, add more tests
            if m == n:
                # Create Hermitian matrix
                H = (A + A.conj().T) / 2
                zhemv_result = alpha * H @ x + beta * y
                test_case["H"] = matrix_to_list(H)
                test_case["results"]["zhemv"] = array_to_list(zhemv_result)
                
                # Create upper triangular matrix
                U = np.triu(A)
                ztrmv_result = U @ x
                test_case["U"] = matrix_to_list(U)
                test_case["results"]["ztrmv"] = array_to_list(ztrmv_result)
                
                # Triangular solve: solve U*x = b for x
                b = np.random.randn(n) + 1j * np.random.randn(n)
                ztrsv_result = np.linalg.solve(U, b)
                test_case["b"] = array_to_list(b)
                test_case["results"]["ztrsv"] = array_to_list(ztrsv_result)
            
            level2_tests.append(test_case)
        
        # Add rank-1 update tests
        rank1_tests = self.generate_rank1_update_tests()
        level2_tests.extend(rank1_tests)
        
        self.test_data["level2"] = level2_tests
    
    def generate_rank1_update_tests(self):
        """Generate test cases for rank-1 updates."""
        tests = []
        
        # Test rank-1 updates
        m, n = 3, 3
        A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        x = np.random.randn(m) + 1j * np.random.randn(m)
        y = np.random.randn(n) + 1j * np.random.randn(n)
        alpha = complex(2.0, 1.0)
        
        # zgerc - A = A + alpha * x * conj(y)^T
        gerc_result = A + alpha * np.outer(x, y.conj())
        
        # zgeru - A = A + alpha * x * y^T (no conjugation)
        geru_result = A + alpha * np.outer(x, y)
        
        # zher - Hermitian rank-1 update: A = A + alpha * x * conj(x)^T
        H = (A + A.conj().T) / 2  # Make Hermitian
        her_result = H + alpha.real * np.outer(x, x.conj())  # alpha must be real
        
        test = {
            "name": "rank1_updates",
            "m": m,
            "n": n,
            "A": matrix_to_list(A),
            "H": matrix_to_list(H),
            "x": array_to_list(x),
            "y": array_to_list(y),
            "alpha": complex_to_dict(alpha),
            "alpha_real": float(alpha.real),
            "results": {
                "zgerc": matrix_to_list(gerc_result),
                "zgeru": matrix_to_list(geru_result),
                "zher": matrix_to_list(her_result)
            }
        }
        
        tests.append(test)
        return tests
    
    def generate_level3_tests(self):
        """Generate test cases for Level 3 BLAS operations."""
        print("Generating Level 3 BLAS test cases...")
        
        level3_tests = []
        
        # Test different matrix sizes
        sizes = [(2, 3, 4), (3, 3, 3), (4, 2, 3)]
        
        for m, k, n in sizes:
            # Generate test matrices
            A = np.random.randn(m, k) + 1j * np.random.randn(m, k)
            B = np.random.randn(k, n) + 1j * np.random.randn(k, n)
            C = np.random.randn(m, n) + 1j * np.random.randn(m, n)
            alpha = complex(2.0, -1.0)
            beta = complex(1.0, 0.5)
            
            # zgemm - general matrix multiply: C = alpha*A*B + beta*C
            gemm_result = alpha * A @ B + beta * C
            
            test_case = {
                "m": m,
                "k": k,
                "n": n,
                "A": matrix_to_list(A),
                "B": matrix_to_list(B),
                "C": matrix_to_list(C),
                "alpha": complex_to_dict(alpha),
                "beta": complex_to_dict(beta),
                "results": {
                    "zgemm": matrix_to_list(gemm_result)
                }
            }
            
            # For square matrices, add Hermitian tests
            if m == n == k:
                # Create Hermitian matrices
                H = (A + A.conj().T) / 2
                # zhemm - Hermitian matrix multiply
                zhemm_result = alpha * H @ B + beta * C
                test_case["H"] = matrix_to_list(H)
                test_case["results"]["zhemm"] = matrix_to_list(zhemm_result)
                
                # zherk - Hermitian rank-k update: C = alpha*A*A^H + beta*C
                C_herm = (C + C.conj().T) / 2
                zherk_result = alpha.real * A @ A.conj().T + beta.real * C_herm
                test_case["C_herm"] = matrix_to_list(C_herm)
                test_case["results"]["zherk"] = matrix_to_list(zherk_result)
            
            level3_tests.append(test_case)
        
        self.test_data["level3"] = level3_tests
    
    def save_test_data(self, filename="complex_blas_test_data.json"):
        """Save test data to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.test_data, f, indent=2)
        print(f"Test data saved to {filename}")
    
    def generate_all_tests(self):
        """Generate all test cases."""
        self.generate_level1_tests()
        self.generate_level2_tests()
        self.generate_level3_tests()
        self.save_test_data()

def main():
    generator = ComplexBLASTestGenerator()
    generator.generate_all_tests()
    
    # Also generate a simple example for documentation
    print("\nExample test case (zdotu):")
    x = np.array([1+0j, 2+1j])
    y = np.array([3+4j, 1-2j])
    result = np.dot(x, y)
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"zdotu(x, y) = {result}")
    print(f"Expected: (1+0j)*(3+4j) + (2+1j)*(1-2j) = 3+4j + 4-3j = 7+1j")

if __name__ == "__main__":
    main()