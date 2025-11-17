# Nilpotent Matrix Generation Codes

## Overview

These Python scripts systematically generate all 4×4 nilpotent matrices satisfying:
1. **Nilpotency**: M² = 0
2. **Anticommutator constraint**: {M, M†} = MM† + M†M = c·I (proportional to identity)
3. **Entry constraint**: All entries ∈ {0, 1, -1, I, -I}

## Files Provided

### 1. Antisymmetric Matrices (FASTEST)
**Script**: `find_antisymmetric_matrices.py`  
**Runtime**: < 1 second  
**Search space**: 5^6 = 15,625 matrices  
**Results**: 49 matrices (1 zero + 48 with c=4)

**Structure**: M^T = -M (diagonal must be zero)
- Only 6 free parameters (upper triangle off-diagonal)
- All non-zero solutions have c = 4
- All have rank 2
- All have zero diagonal 2×2 blocks

**Usage**:
```bash
python find_antisymmetric_matrices.py
```

---

### 2. Symmetric Matrices (MODERATE)
**Script**: `find_symmetric_matrices.py`  
**Runtime**: 5-10 minutes  
**Search space**: 5^10 = 9,765,625 matrices  
**Results**: 433 matrices

**Structure**: M^T = M
- 10 free parameters (4 diagonal + 6 upper triangle)
- c-values: 0, 4, 8
- Various rank structures

**Usage**:
```bash
python find_symmetric_matrices.py
```

---

## Comparison Table

| Matrix Type | Free Params | Search Space | Runtime | Results | c-values |
|-------------|-------------|--------------|---------|---------|----------|
| Antisymmetric | 6 | 15,625 | < 1 sec | 49 | 0, 4 |
| Symmetric | 10 | 9.7M | 5-10 min | 433 | 0, 4, 8 |


---

## Output Format

All scripts generate Mathematica-compatible `.m` files with format:
```mathematica
matrices = {
  {{{matrix_row1}, {matrix_row2}, {matrix_row3}, {matrix_row4}}, c_value},
  ...
};
```

Plus utility functions:
- `getMatrix[n]` - Get nth matrix
- `getCValue[n]` - Get c-value for nth matrix
- `countMatrices` - Total count
- `verifyNilpotent[M]` - Verify M² = 0
- `verifyAnticommutator[M]` - Verify anticommutator constraint

---

## Physical Interpretation


### Generation Structure
- **Gen 1** (Rank-1, mixed): 16 unpaired states
- **Gen 2** (Rank-2, diagonal): 8 pairs
- **Gen 3** (Rank-2, antidiagonal): 8 pairs

### Pairing
- Generation 1: All unpaired
- Generations 2 & 3: Perfect M₂ = M₁* pairing (particle/antiparticle)

### 12+4 Structure
Within Generations 1 and 2:
- **12 states**: Tr(B) real or zero
- **4 states**: Tr(B) imaginary

---

## Requirements

```python
numpy
itertools (standard library)
time (standard library)
pickle (standard library, for asymmetric checkpoints)
```

Install:
```bash
pip install numpy
```

---

## Recommendations

1. **Start with antisymmetric**: Fast and gives you the 48 fermion states
2. **Run symmetric**: Moderate time, gives right-handed fermions
3. **Avoid asymmetric exhaustive**: Unless you have a week to spare
4. **Use asymmetric fast**: Good compromise for exploring general structures

---

## Notes on Asymmetric Search

The asymmetric exhaustive search faces a combinatorial explosion:
- 5^16 = 152,587,890,625 matrices to check
- At 1M matrices/second: ~42 hours just to iterate
- With checks: 5-10 days realistically

**Alternatives**:
1. Use the fast version (special structures)
2. Use random sampling (option 2 in full script)
3. Run exhaustive search on a cluster/cloud
4. Focus on physically motivated structures only

---

## Citation

If you use these codes, please cite:
- Ajaib, M. A., "Representation-dependent quantum mechanics and boundary phenomena"
- Related papers on nilpotent representations and fermion structure

---

## Contact

Dr. Muhammad Adeel Ajaib  
Penn State Abington  
adeelajaib.com
