# lau-numerical-linear-algebra

Numerical linear algebra for large-scale computations — iterative solvers, Krylov subspace methods, eigenvalue algorithms, sparse solvers, preconditioners, SVD, and least squares.

Designed for large-scale agent simulations where you need to solve network equations efficiently without pulling in a heavyweight BLAS dependency.

---

## What This Does

This crate gives you a **from-scratch numerical linear algebra toolkit** in pure Rust:

| Category | What you get |
|---|---|
| **Iterative solvers** | Jacobi, Gauss-Seidel, SOR |
| **Krylov subspace** | Conjugate Gradient (CG), GMRES (restarted), BiCGSTAB |
| **Eigenvalue algorithms** | Power iteration, inverse iteration, QR algorithm, Lanczos |
| **Sparse solvers** | COO/CSR storage, sparse CG, sparse triangular solves |
| **Preconditioners** | Jacobi (diagonal), Incomplete Cholesky IC(0), preconditioned CG |
| **SVD** | Full SVD, truncated SVD, randomized SVD |
| **Least squares** | QR-based, SVD-based, normal equations |
| **Condition estimation** | κ(A) from eigenvalues or singular values |

57 unit tests cover correctness, convergence, and edge cases across every module.

---

## Key Idea

All algorithms operate on plain `Vec<Vec<f64>>` (dense) or `CsrMatrix`/`CooMatrix` (sparse). No unsafe code, no external solver libraries — just clean iterative numerics with clear convergence tracking.

The sparse infrastructure (COO → CSR with duplicate merging, CSR matrix-vector products, triangular solves) feeds directly into the preconditioned Krylov solvers, making this practical for systems with thousands of variables where dense methods would choke.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-numerical-linear-algebra = "0.1.0"
```

Or use it as a git dependency:

```toml
[dependencies]
lau-numerical-linear-algebra = { git = "https://github.com/SuperInstance/lau-numerical-linear-algebra" }
```

Requires **Rust 2021 edition**.

### Dependencies

| Crate | Why |
|---|---|
| `nalgebra` | `DVector` conversion helpers in `types` |
| `serde` | `Serialize`/`Deserialize` on result types |
| `num-traits` | Numeric trait utilities |

---

## Quick Start

### Solve a dense system with Conjugate Gradient

```rust
use lau_numerical_linear_algebra::{cg, ConvergenceCriteria};

let a = vec![
    vec![4.0, 1.0, 0.0],
    vec![1.0, 3.0, 1.0],
    vec![0.0, 1.0, 4.0],
];
let b = vec![5.0, 5.0, 5.0];

let result = cg(&a, &b, &ConvergenceCriteria::new(100, 1e-10));
assert!(result.converged);
println!("x = {:?}", result.x);       // [1.0, 1.0, 1.0]
println!("residual = {}", result.residual_norm);
println!("iterations = {}", result.iterations);
```

### Build a sparse system and solve it

```rust
use lau_numerical_linear_algebra::sparse::{CooMatrix, sparse_cg};

let mut coo = CooMatrix::new(3, 3);
coo.add_entry(0, 0, 4.0);
coo.add_entry(0, 1, 1.0);
coo.add_entry(1, 0, 1.0);
coo.add_entry(1, 1, 3.0);
coo.add_entry(1, 2, 1.0);
coo.add_entry(2, 1, 1.0);
coo.add_entry(2, 2, 4.0);

let csr = coo.to_csr();  // COO → CSR with duplicate merging
let b = vec![5.0, 5.0, 5.0];

let result = sparse_cg(&csr, &b, 100, 1e-10);
assert!(result.converged);
```

### Eigenvalues and SVD

```rust
use lau_numerical_linear_algebra::eigen::{power_iteration, qr_algorithm};

let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];

// Dominant eigenvalue
let (lambda, v) = power_iteration(&a, 200, 1e-12);

// All eigenvalues
let eigs = qr_algorithm(&a, 200, 1e-12);
```

```rust
use lau_numerical_linear_algebra::svd::{full_svd, truncated_svd};

let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

let svd = full_svd(&a);
println!("singular values = {:?}", svd.singular_values);

let rank1 = truncated_svd(&a, 1);  // keep only the largest
```

### Least squares (overdetermined systems)

```rust
use lau_numerical_linear_algebra::least_squares::qr_least_squares;

// Fit y = mx + c through 3 points
let a = vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![3.0, 1.0]];
let b = vec![2.0, 4.0, 6.0];

let result = qr_least_squares(&a, &b);
println!("m = {}, c = {}", result.x[0], result.x[1]);
```

### Preconditioned sparse CG

```rust
use lau_numerical_linear_algebra::sparse::CooMatrix;
use lau_numerical_linear_algebra::preconditioner::{
    JacobiPreconditioner, preconditioned_cg,
};

let mut coo = CooMatrix::new(3, 3);
// ... fill matrix ...
let csr = coo.to_csr();
let precond = JacobiPreconditioner::from_csr(&csr);

let result = preconditioned_cg(&csr, &b, |r| precond.apply(r), 100, 1e-10);
```

---

## API Reference

### `iterative` — Classic Iterative Solvers

| Function | Signature | Notes |
|---|---|---|
| `jacobi` | `(a, b, criteria) → SolverResult` | Diagonal-dominant systems |
| `gauss_seidel` | `(a, b, criteria) → SolverResult` | Faster convergence than Jacobi |
| `sor` | `(a, b, omega, criteria) → SolverResult` | ω ∈ (1, 2) for over-relaxation |

### `krylov` — Krylov Subspace Methods

| Function | Signature | Notes |
|---|---|---|
| `cg` | `(a, b, criteria) → SolverResult` | SPD systems only |
| `gmres` | `(a, b, m, criteria) → SolverResult` | General nonsymmetric; `m` = restart |
| `bicgstab` | `(a, b, criteria) → SolverResult` | General nonsymmetric; no restart param |

### `eigen` — Eigenvalue Algorithms

| Function | Signature | Notes |
|---|---|---|
| `power_iteration` | `(a, max_iter, tol) → (λ, v)` | Dominant eigenvalue |
| `inverse_iteration` | `(a, shift, max_iter, tol) → (λ, v)` | Eigenvalue nearest to shift |
| `qr_algorithm` | `(a, max_iter, tol) → Vec<f64>` | All eigenvalues |
| `lanczos` | `(a, k) → (T, Q)` | Tridiagonal reduction for symmetric A |
| `lanczos_eigenvalues` | `(a, k) → Vec<f64>` | Eigenvalues via Lanczos |

### `sparse` — Sparse Matrix Infrastructure

| Type/Function | Notes |
|---|---|
| `CooMatrix` | Coordinate format; build with `add_entry`, convert to CSR |
| `CsrMatrix` | Compressed Sparse Row; `mat_vec`, `diagonal`, `to_dense`, `nnz` |
| `sparse_cg` | Conjugate Gradient on CSR matrix |
| `solve_lower_triangular` | Forward substitution on sparse L |
| `solve_upper_triangular` | Back substitution on sparse U |

### `preconditioner` — Preconditioners

| Type/Function | Notes |
|---|---|
| `JacobiPreconditioner` | Diagonal scaling; `from_csr`, `apply` |
| `IncompleteCholesky` | IC(0) for SPD; `from_csr`, `apply` |
| `preconditioned_cg` | PCG with arbitrary preconditioner closure |

### `svd` — Singular Value Decomposition

| Function | Signature | Notes |
|---|---|---|
| `full_svd` | `(a) → SvdResult` | Via eigendecomposition of A^TA |
| `truncated_svd` | `(a, k) → SvdResult` | Keep top-k singular values |
| `randomized_svd` | `(a, k, oversampling, power_iter) → SvdResult` | Randomized projection approximation |

### `least_squares` — Least Squares Solvers

| Function | Notes |
|---|---|
| `qr_least_squares` | Householder QR → back-substitute; numerically stable |
| `svd_least_squares` | SVD-based; best for ill-conditioned problems |
| `normal_equations` | (A^TA)x = A^Tb via Gaussian elimination; simplest but least stable |

### `condition` — Condition Number Estimation

| Function | Notes |
|---|---|
| `condition_number` | κ(A) = σ_max / σ_min via eigenvalues of A^TA |
| `condition_number_from_singular_values` | κ from precomputed singular values |

### `types` — Shared Types

| Type | Fields |
|---|---|
| `SolverResult` | `x`, `iterations`, `residual_norm`, `converged` |
| `EigenResult` | `eigenvalues`, `eigenvectors`, `iterations` |
| `SvdResult` | `singular_values`, `u`, `vt` |
| `LeastSquaresResult` | `x`, `residual_norm` |
| `ConvergenceCriteria` | `max_iterations`, `tolerance` (default: 1000, 1e-10) |

---

## How It Works

### Iterative Solvers (`iterative.rs`)

All three methods decompose the matrix row-by-row:

- **Jacobi**: updates all entries simultaneously using previous-iteration values → `x_new[i] = (b[i] - Σ_{j≠i} A[i][j]·x[j]) / A[i][i]`
- **Gauss-Seidel**: uses updated values immediately as they become available → faster convergence
- **SOR**: adds a relaxation parameter ω that interpolates between old and new values → `x[i] = (1-ω)·x[i] + ω·(b[i] - Σ) / A[i][i]`. When ω=1, this reduces to Gauss-Seidel.

Convergence is checked by the residual norm `‖Ax - b‖₂` against the user-supplied tolerance.

### Krylov Methods (`krylov.rs`)

These methods search for solutions in the Krylov subspace `𝒦ₖ(A, b) = span{b, Ab, A²b, …, A^{k-1}b}`:

- **CG**: For SPD matrices only. Maintains conjugate search directions via short recurrences. Optimal in the sense that it minimizes the A-norm of the error over the Krylov subspace.
- **GMRES**: For general nonsymmetric systems. Builds an orthonormal basis via Arnoldi iteration and solves a least-squares problem in that subspace. Uses Givens rotations for the Hessenberg system. Restarts every `m` iterations to limit memory.
- **BiCGSTAB**: For general systems. Uses a shadow residual to produce a stabilized biconjugate gradient iteration. No restart needed, but convergence can be irregular.

### Eigenvalue Algorithms (`eigen.rs`)

- **Power iteration**: Repeatedly multiply by A and normalize → converges to the eigenvector of the dominant eigenvalue. Rate of convergence depends on the ratio |λ₂/λ₁|.
- **Inverse iteration**: Apply power iteration to (A - σI)⁻¹ → converges to the eigenvalue closest to the shift σ. The internal solve uses Gaussian elimination with partial pivoting.
- **QR algorithm**: Repeatedly factor A = QR and replace A ← RQ → converges to (quasi-)upper triangular Schur form with eigenvalues on the diagonal. Uses Householder reflections for the QR decomposition.
- **Lanczos**: For symmetric matrices, builds an orthonormal basis of the Krylov subspace with a three-term recurrence → produces a tridiagonal matrix T whose eigenvalues approximate those of A.

### Sparse Infrastructure (`sparse.rs`)

- **COO** (Coordinate): Insert entries in any order with `add_entry(row, col, val)`. Duplicates are summed during conversion to CSR.
- **CSR** (Compressed Sparse Row): `row_ptr[i]` to `row_ptr[i+1]` indexes the column indices and values for row i. This enables O(nnz) matrix-vector products.
- **Sparse CG**: Same algorithm as dense CG but uses `CsrMatrix::mat_vec` for O(nnz) per iteration instead of O(n²).
- **Triangular solves**: Forward/backward substitution exploiting the CSR structure.

### Preconditioners (`preconditioner.rs`)

Preconditioning transforms the system M⁻¹Ax = M⁻¹b so that M⁻¹A has better spectral properties:

- **Jacobi**: M = diag(A). Trivial to apply (element-wise division). Effective when the diagonal dominates.
- **Incomplete Cholesky IC(0)**: Computes an approximate Cholesky factor L ≈ chol(A) but only fills in positions where A has nonzeros. Solves LL^Tx = r via forward + backward substitution.
- **Preconditioned CG**: The closure-based design lets you swap preconditioners without changing the solver.

### SVD (`svd.rs`)

- **Full SVD**: Computes A^TA, finds its eigenvalues via QR algorithm, extracts singular values as √λᵢ, and computes singular vectors via inverse iteration (right vectors from A^TA, left vectors from Avᵢ/σᵢ).
- **Truncated SVD**: Computes full SVD then keeps only the top-k components → useful for low-rank approximation.
- **Randomized SVD**: Projects A onto a random Gaussian subspace of dimension k+p, uses power iteration to refine the subspace, then computes SVD of the small projected matrix. Complexity is O(mn·k) instead of O(mn·min(m,n)).

### Least Squares (`least_squares.rs`)

Three approaches to solving min ‖Ax - b‖₂ for overdetermined systems (m > n):

- **QR**: Factors A = QR, solves R̂x = Q̂^Tb where Q̂ is the first n columns. Householder reflections ensure numerical stability.
- **SVD**: x = VΣ⁻¹U^Tb. Most stable — handles rank-deficient and ill-conditioned problems gracefully.
- **Normal equations**: Solves (A^TA)x = A^Tb directly. Squares the condition number (κ(A^TA) = κ(A)²), so only suitable for well-conditioned problems.

### Condition Numbers (`condition.rs`)

κ(A) = σ_max / σ_min estimates how sensitive the solution is to perturbations in the data. Computed either from eigenvalues of A^TA or directly from precomputed singular values. Returns infinity for singular or near-singular matrices.

---

## The Math

### Conjugate Gradient

For SPD matrices, CG minimizes φ(x) = ½x^TAx - b^Tx over expanding Krylov subspaces. Search directions pₖ are A-conjugate (pᵢ^TApⱼ = 0 for i≠j). Each iteration requires one matrix-vector multiply and two inner products:

```
αₖ = (rₖ^Trₖ) / (pₖ^TApₖ)
xₖ₊₁ = xₖ + αₖpₖ
rₖ₊₁ = rₖ - αₖApₖ
βₖ = (rₖ₊₁^Trₖ₊₁) / (rₖ^Trₖ)
pₖ₊₁ = rₖ₊₁ + βₖpₖ
```

Converges in at most n iterations in exact arithmetic; often much fewer if eigenvalues are clustered.

### GMRES

At iteration k, builds orthonormal basis {v₁, ..., vₖ} of 𝒦ₖ(A, r₀) via Arnoldi. The Hessenberg matrix Hₖ satisfies AVₖ = Vₖ₊₁H̄ₖ. Minimizes ‖βe₁ - H̄ₖyₖ‖ over yₖ using Givens rotations (O(k²) per step). Restarted every m iterations to limit basis size.

### Lanczos

For symmetric A, the Arnoldi process simplifies to a three-term recurrence producing tridiagonal T:

```
βₖ₊₁vₖ₊₁ = Avₖ - αₖvₖ - βₖvₖ₋₁
```

The eigenvalues of T converge rapidly to the extreme eigenvalues of A. After k ≪ n steps, you get good approximations to the largest and smallest eigenvalues.

### Incomplete Cholesky

Standard Cholesky computes L where A = LL^T but may create fill-in (nonzeros in positions where A is zero). IC(0) discards fill-in, producing L̃ where L̃L̃^T ≈ A. The preconditioner M = L̃L̃^T is cheap to invert (two triangular solves) and captures the dominant coupling in A.

### Randomized SVD

Key insight: if A has rank k, then with high probability, the column space of Y = AΩ (where Ω is n×(k+p) random Gaussian) approximates the range of A. Power iteration (alternating Y ← A(A^TY)) sharpens this approximation. After projecting onto this subspace, you solve a small (k+p)×n SVD instead of the full m×n problem.

---

## License

MIT
