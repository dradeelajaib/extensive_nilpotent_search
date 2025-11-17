import numpy as np
from itertools import product
import time
from collections import defaultdict

def verify_nilpotent(M):
    """Check if M^2 = 0."""
    M_squared = M @ M
    return np.allclose(M_squared, 0, atol=1e-10)

def is_antisymmetric(M):
    """Check if M is antisymmetric (M^T = -M)."""
    return np.allclose(M.T, -M, atol=1e-10)

def check_anticommutator(M):
    """
    Check if {M, M†} = M M† + M† M is proportional to identity.
    Returns (is_proportional, constant_c)
    """
    M_dag = M.conj().T  # Hermitian conjugate
    anticommutator = M @ M_dag + M_dag @ M
    
    # Check if it's diagonal
    off_diag = anticommutator - np.diag(np.diag(anticommutator))
    if not np.allclose(off_diag, 0, atol=1e-10):
        return False, None
    
    # Check if diagonal elements are all equal
    diag_elements = np.diag(anticommutator)
    if not np.allclose(diag_elements, diag_elements[0], atol=1e-10):
        return False, None
    
    c = diag_elements[0]
    return True, c

def matrix_signature(M):
    """Create a canonical signature for a matrix to identify equivalent structures."""
    # Round to remove numerical noise
    M_rounded = np.round(M, decimals=10)
    return tuple(M_rounded.flatten())

def exhaustive_search_antisymmetric(values=[0, 1, -1, 1j, -1j]):
    """
    Exhaustive search for antisymmetric nilpotent matrices.
    For antisymmetric: M^T = -M, which means:
    - Diagonal must be zero
    - M[i,j] = -M[j,i]
    - Only need 6 free parameters (upper triangle off-diagonal)
    """
    print("="*70)
    print("COMPREHENSIVE SEARCH FOR ANTISYMMETRIC AJAIB MATRICES")
    print("="*70)
    print(f"Constraints:")
    print(f"  1. Antisymmetric: M^T = -M")
    print(f"  2. Nilpotent: M² = 0")
    print(f"  3. Anticommutator: {{M, M†}} = c·I")
    print(f"\nSearching entries from: {values}")
    print(f"Total matrices: 5^6 = {5**6:,}")
    print(f"(6 free parameters in upper triangle)")
    print("="*70)
    
    results = []
    seen_signatures = set()
    count = 0
    checkpoint = 5**6 // 100
    
    start_time = time.time()
    
    # For antisymmetric matrix, only need upper triangle (6 elements)
    # Matrix structure:
    # [[  0,  a,  b,  c],
    #  [ -a,  0,  d,  e],
    #  [ -b, -d,  0,  f],
    #  [ -c, -e, -f,  0]]
    
    for vals in product(values, repeat=6):
        # Construct antisymmetric matrix
        a, b, c, d, e, f = vals
        M = np.array([
            [0,  a,  b,  c],
            [-a, 0,  d,  e],
            [-b,-d,  0,  f],
            [-c,-e, -f, 0]
        ], dtype=complex)
        
        count += 1
        
        if count % checkpoint == 0:
            elapsed = time.time() - start_time
            percent = (count / 5**6) * 100
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (5**6 - count) / rate if rate > 0 else 0
            print(f"Progress: {percent:.0f}% | Found: {len(results)} | ETA: {remaining:.1f}s", end='\r')
        
        # Verify antisymmetric (should always be true by construction, but check)
        if not is_antisymmetric(M):
            continue
        
        # Check nilpotent first (faster filter)
        if not verify_nilpotent(M):
            continue
            
        # Check anticommutator
        is_anticom, c = check_anticommutator(M)
        if not is_anticom:
            continue
        
        # Avoid duplicates
        sig = matrix_signature(M)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        
        results.append((M.copy(), c))
    
    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Checked: {count:,} matrices")
    print(f"Found: {len(results)} unique solutions")
    
    return results

def analyze_solutions(results):
    """Detailed analysis of found solutions."""
    print("\n" + "="*70)
    print("DETAILED ANALYSIS OF SOLUTIONS")
    print("="*70)
    
    # Group by value of c
    by_c = defaultdict(list)
    for M, c in results:
        c_rounded = round(c.real, 10) + 1j*round(c.imag, 10)
        by_c[c_rounded].append(M)
    
    # Group by rank
    by_rank = defaultdict(list)
    for M, c in results:
        rank = np.linalg.matrix_rank(M, tol=1e-10)
        by_rank[rank].append((M, c))
    
    print(f"\n📊 STATISTICS")
    print(f"{'='*70}")
    print(f"Total unique solutions: {len(results)}")
    print(f"Distinct c values: {len(by_c)}")
    print(f"\nBy rank:")
    for rank in sorted(by_rank.keys()):
        print(f"  Rank {rank}: {len(by_rank[rank])} matrices")
    
    print(f"\n📈 SOLUTIONS BY ANTICOMMUTATOR VALUE (c)")
    print(f"{'='*70}")
    for c_val in sorted(by_c.keys(), key=lambda x: (x.real, x.imag)):
        matrices = by_c[c_val]
        print(f"\nc = {c_val} ({len(matrices)} matrices)")
        
        # Show a few examples for each c value
        for i, M in enumerate(matrices[:3], 1):
            rank = np.linalg.matrix_rank(M, tol=1e-10)
            print(f"\n  Example {i} (Rank {rank}):")
            print("  " + str(M).replace("\n", "\n  "))
            
            # Check structure
            is_real = np.allclose(M.imag, 0)
            has_block_structure = (np.allclose(M[0:2, 0:2], 0) and 
                                  np.allclose(M[2:4, 2:4], 0))
            
            if is_real:
                print("  Structure: Real entries")
            else:
                print("  Structure: Complex entries")
            
            if has_block_structure:
                print("  Block form: Zero diagonal 2x2 blocks")
        
        if len(matrices) > 3:
            print(f"\n  ... and {len(matrices) - 3} more matrices with c = {c_val}")
    
    return by_c, by_rank

def identify_patterns(results):
    """Identify common patterns in solutions."""
    print("\n" + "="*70)
    print("PATTERN IDENTIFICATION")
    print("="*70)
    
    patterns = {
        'block_form': [],
        'diagonal_blocks': [],
        'antidiagonal_blocks': [],
        'mixed': []
    }
    
    for M, c in results:
        if np.allclose(M, 0):
            continue  # Skip zero matrix
        
        # Check for block form (zero diagonal 2x2 blocks)
        has_zero_diag_blocks = (np.allclose(M[0:2, 0:2], 0) and 
                               np.allclose(M[2:4, 2:4], 0))
        
        # Check for diagonal 2x2 structure
        B = M[0:2, 2:4]
        C = M[2:4, 0:2]
        is_diagonal_blocks = (np.allclose(B, np.diag(np.diag(B))) and
                             np.allclose(C, np.diag(np.diag(C))))
        
        # Check for antidiagonal 2x2 structure
        is_antidiag_blocks = (np.allclose(np.diag(B), 0) and
                             np.allclose(np.diag(C), 0))
        
        if has_zero_diag_blocks:
            if is_diagonal_blocks:
                patterns['diagonal_blocks'].append((M, c))
            elif is_antidiag_blocks:
                patterns['antidiagonal_blocks'].append((M, c))
            else:
                patterns['block_form'].append((M, c))
        else:
            patterns['mixed'].append((M, c))
    
    for pattern_name, pattern_list in patterns.items():
        if pattern_list:
            print(f"\n{pattern_name.upper()}: {len(pattern_list)} matrices")
            if len(pattern_list) <= 3:
                for M, c in pattern_list:
                    print(f"\nc = {c}:")
                    print(M)

def save_results_mathematica(results, filename="antisymmetric_matrices.m"):
    """Save results in Mathematica format."""
    with open(filename, 'w') as f:
        f.write("(* " + "="*70 + " *)\n")
        f.write("(* ANTISYMMETRIC NILPOTENT MATRICES *)\n")
        f.write("(* WITH ANTICOMMUTATOR CONSTRAINT *)\n")
        f.write("(* " + "="*70 + " *)\n")
        f.write("(* Format: {{matrix, c_value}, ...} *)\n")
        f.write("(* Each matrix is 4x4 with entries in {0, 1, -1, I, -I} *)\n")
        f.write(f"(* Total matrices: {len(results)} *)\n")
        f.write("(* " + "="*70 + " *)\n\n")
        
        f.write("antisymmetricMatrices = {\n")
        
        for i, (M, c) in enumerate(results):
            # Convert matrix to Mathematica format
            f.write("  {")
            for row_idx in range(4):
                if row_idx == 0:
                    f.write("{")
                else:
                    f.write("   {")
                
                for col_idx in range(4):
                    val = M[row_idx, col_idx]
                    # Convert to Mathematica format
                    if np.abs(val) < 1e-10:
                        m_val = "0"
                    elif np.allclose(val, 1):
                        m_val = "1"
                    elif np.allclose(val, -1):
                        m_val = "-1"
                    elif np.allclose(val, 1j):
                        m_val = "I"
                    elif np.allclose(val, -1j):
                        m_val = "-I"
                    else:
                        # Shouldn't happen with our values
                        m_val = str(val)
                    
                    if col_idx < 3:
                        f.write(f"{m_val},")
                    else:
                        f.write(f"{m_val}")
                
                if row_idx < 3:
                    f.write("},\n")
                else:
                    f.write("}")
            
            c_int = int(round(c.real))
            if i < len(results) - 1:
                f.write(f",{c_int}}},\n")
            else:
                f.write(f",{c_int}}}\n")
        
        f.write("};\n\n")
        
        # Add utility functions
        f.write("(* Utility functions *)\n")
        f.write("getAntisymmetricMatrix[n_] := antisymmetricMatrices[[n, 1]];\n")
        f.write("getAntisymmetricCValue[n_] := antisymmetricMatrices[[n, 2]];\n")
        f.write("countAntisymmetricMatrices := Length[antisymmetricMatrices];\n\n")
        
        f.write("(* Verification functions *)\n")
        f.write("verifyNilpotent[M_] := MatrixPower[M, 2] == ConstantArray[0, {4, 4}];\n")
        f.write("verifyAnticommutator[M_] := Module[{Mdag, anticomm, diag},\n")
        f.write("  Mdag = ConjugateTranspose[M];\n")
        f.write("  anticomm = M.Mdag + Mdag.M;\n")
        f.write("  diag = Diagonal[anticomm];\n")
        f.write("  (anticomm == DiagonalMatrix[diag]) && (Length[Union[diag]] == 1)\n")
        f.write("];\n\n")
        
        f.write("(* Example usage: *)\n")
        f.write("(* M1 = getAntisymmetricMatrix[1]; *)\n")
        f.write("(* c1 = getAntisymmetricCValue[1]; *)\n")
        f.write("(* Print[MatrixForm[M1]]; *)\n")
        f.write("(* Print[\"c = \", c1]; *)\n")
        f.write("(* Print[\"Nilpotent: \", verifyNilpotent[M1]]; *)\n")
        f.write("(* Print[\"Anticommutator: \", verifyAnticommutator[M1]]; *)\n")
    
    print(f"\n💾 Results saved to {filename} in Mathematica format")

# Main execution
if __name__ == "__main__":
    print("\n🔬 Starting search for ANTISYMMETRIC matrices...")
    print("This should take less than 1 minute (5^6 = 15,625 matrices).\n")
    
    # Run comprehensive search
    results = exhaustive_search_antisymmetric()
    
    # Analyze results
    by_c, by_rank = analyze_solutions(results)
    
    # Identify patterns
    identify_patterns(results)
    
    # Save to Mathematica format
    save_results_mathematica(results)
    
    print("\n" + "="*70)
    print("🎉 SEARCH COMPLETE!")
    print("="*70)
    print(f"\nKey findings:")
    print(f"  • Total unique solutions: {len(results)}")
    print(f"  • Distinct c values: {len(by_c)}")
    print(f"  • Non-zero solutions: {len([r for r in results if not np.allclose(r[0], 0)])}")
    print("\nComparison with symmetric case:")
    print("  • Antisymmetric: Much more constrained (diagonal = 0)")
    print("  • Expected to have fewer solutions than symmetric case")
    print("  • All antisymmetric matrices in your previous data had c = 4")
