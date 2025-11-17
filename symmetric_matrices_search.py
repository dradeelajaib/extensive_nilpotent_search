import numpy as np
from itertools import product
import time
from collections import defaultdict

def verify_nilpotent(M):
    """Check if M^2 = 0."""
    M_squared = M @ M
    return np.allclose(M_squared, 0, atol=1e-10)

def is_symmetric(M):
    """Check if M is symmetric (M^T = M)."""
    return np.allclose(M, M.T, atol=1e-10)

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

def exhaustive_search_optimized(values=[0, 1, -1, 1j, -1j]):
    """
    Optimized exhaustive search for all solutions.
    """
    print("="*70)
    print("COMPREHENSIVE SEARCH FOR AJAIB REPRESENTATION MATRICES")
    print("="*70)
    print(f"Constraints:")
    print(f"  1. Symmetric: M^T = M")
    print(f"  2. Nilpotent: M² = 0")
    print(f"  3. Anticommutator: {{M, M†}} = c·I")
    print(f"\nSearching entries from: {values}")
    print(f"Total matrices: 5^10 = {5**10:,}")
    print("="*70)
    
    results = []
    seen_signatures = set()
    count = 0
    checkpoint = 5**10 // 100
    
    start_time = time.time()
    
    for vals in product(values, repeat=10):
        # Construct symmetric matrix: 4 diagonal + 6 upper triangle
        M = np.array([
            [vals[0], vals[4], vals[5], vals[6]],
            [vals[4], vals[1], vals[7], vals[8]],
            [vals[5], vals[7], vals[2], vals[9]],
            [vals[6], vals[8], vals[9], vals[3]]
        ], dtype=complex)
        
        count += 1
        
        if count % checkpoint == 0:
            elapsed = time.time() - start_time
            percent = (count / 5**10) * 100
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (5**10 - count) / rate if rate > 0 else 0
            print(f"Progress: {percent:.0f}% | Found: {len(results)} | ETA: {remaining/60:.1f}min", end='\r')
        
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
    print(f"Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
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
            is_imaginary_offdiag = np.allclose(np.diag(M).imag, 0) and not np.allclose(M.imag, 0)
            
            if is_real:
                print("  Structure: Real entries")
            elif is_imaginary_offdiag:
                print("  Structure: Real diagonal, imaginary off-diagonal")
            else:
                print("  Structure: Mixed real/imaginary")
        
        if len(matrices) > 3:
            print(f"\n  ... and {len(matrices) - 3} more matrices with c = {c_val}")
    
    return by_c, by_rank

def identify_patterns(results):
    """Identify common patterns in solutions."""
    print("\n" + "="*70)
    print("PATTERN IDENTIFICATION")
    print("="*70)
    
    patterns = {
        'block_diagonal': [],
        'single_block': [],
        'sparse': [],
        'full': []
    }
    
    for M, c in results:
        if np.allclose(M, 0):
            continue  # Skip zero matrix
        
        nonzero_count = np.count_nonzero(np.abs(M) > 1e-10)
        
        # Check if block diagonal (2x2 blocks)
        is_block_diag = (np.allclose(M[0:2, 2:4], 0) and 
                        np.allclose(M[2:4, 0:2], 0))
        
        if is_block_diag:
            patterns['block_diagonal'].append((M, c))
        elif nonzero_count <= 4:
            patterns['single_block'].append((M, c))
        elif nonzero_count <= 8:
            patterns['sparse'].append((M, c))
        else:
            patterns['full'].append((M, c))
    
    for pattern_name, pattern_list in patterns.items():
        if pattern_list:
            print(f"\n{pattern_name.upper()}: {len(pattern_list)} matrices")
            if len(pattern_list) <= 2:
                for M, c in pattern_list:
                    print(f"\nc = {c}:")
                    print(M)

def save_results(results, filename="ajaib_matrices.txt"):
    """Save results to file."""
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SYMMETRIC NILPOTENT MATRICES WITH ANTICOMMUTATOR ∝ I\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total solutions found: {len(results)}\n\n")
        
        for i, (M, c) in enumerate(results, 1):
            f.write(f"\n{'='*70}\n")
            f.write(f"Solution #{i}\n")
            f.write(f"c = {c}\n")
            f.write(f"Rank = {np.linalg.matrix_rank(M, tol=1e-10)}\n")
            f.write(f"{'='*70}\n")
            f.write(str(M) + "\n")
            f.write(f"\nVerification:\n")
            f.write(f"M² =\n{M @ M}\n")
            M_dag = M.conj().T
            f.write(f"\n{{M, M†}} =\n{M @ M_dag + M_dag @ M}\n")
    
    print(f"\n💾 Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    print("\n🔬 Starting search for matrices satisfying Ajaib representation constraints...")
    print("This will take approximately 5-10 minutes.\n")
    
    # Run comprehensive search
    results = exhaustive_search_optimized()
    
    # Analyze results
    by_c, by_rank = analyze_solutions(results)
    
    # Identify patterns
    identify_patterns(results)
    
    # Save to file
    save_results(results)
    
    print("\n" + "="*70)
    print("🎉 SEARCH COMPLETE!")
    print("="*70)
    print(f"\nKey findings:")
    print(f"  • Total unique solutions: {len(results)}")
    print(f"  • Distinct c values: {len(by_c)}")
    print(f"  • Non-zero solutions: {len([r for r in results if not np.allclose(r[0], 0)])}")
    print("\nThese matrices may be useful for your representation-dependent")
    print("quantum mechanics framework, particularly for boundary conditions")
    print("and spin-flip phenomena.")
