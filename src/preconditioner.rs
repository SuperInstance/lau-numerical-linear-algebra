//! Preconditioners: Jacobi, Incomplete Cholesky.

use crate::sparse::CsrMatrix;
use crate::sparse::CooMatrix;

/// Jacobi (diagonal) preconditioner.
/// M = diag(A), so M^{-1} * r = r ./ diag(A).
pub struct JacobiPreconditioner {
    diag_inv: Vec<f64>,
}

impl JacobiPreconditioner {
    pub fn from_csr(a: &CsrMatrix) -> Self {
        let diag = a.diagonal();
        let diag_inv: Vec<f64> = diag.iter().map(|d| {
            if d.abs() > 1e-15 { 1.0 / d } else { 0.0 }
        }).collect();
        Self { diag_inv }
    }

    /// Apply M^{-1} to vector r.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        r.iter().zip(self.diag_inv.iter()).map(|(ri, di)| ri * di).collect()
    }
}

/// Incomplete Cholesky preconditioner for SPD matrices.
/// Computes an approximate Cholesky factorization with zero fill-in (IC(0)).
pub struct IncompleteCholesky {
    /// Lower triangular factor L (CSR format).
    pub l: CsrMatrix,
}

impl IncompleteCholesky {
    pub fn from_csr(a: &CsrMatrix) -> Self {
        let n = a.nrows;
        let dense = a.to_dense();

        // Incomplete Cholesky on dense (IC(0) - only fill where A has nonzeros)
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i][k] * l[i][k];
            }
            let diag = dense[i][i] - sum;
            if diag > 0.0 {
                l[i][i] = diag.sqrt();
            } else {
                l[i][i] = 1e-10; // fallback for stability
            }

            for j in (i + 1)..n {
                // Only compute if A has a nonzero at (j, i)
                if dense[j][i].abs() < 1e-15 && dense[i][j].abs() < 1e-15 {
                    continue;
                }
                let mut sum = 0.0;
                for k in 0..i {
                    sum += l[j][k] * l[i][k];
                }
                let a_ji = if dense[j][i].abs() > 1e-15 { dense[j][i] } else { dense[i][j] };
                if l[i][i].abs() > 1e-30 {
                    l[j][i] = (a_ji - sum) / l[i][i];
                }
            }
        }

        // Convert L to CSR
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            for j in 0..=i {
                if l[i][j].abs() > 1e-30 {
                    coo.add_entry(i, j, l[i][j]);
                }
            }
        }

        Self { l: coo.to_csr() }
    }

    /// Solve L * L^T * x = r (forward then backward substitution).
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = self.l.nrows;

        // Forward solve: L * y = r
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = r[i];
            let start = self.l.row_ptr[i];
            let end = self.l.row_ptr[i + 1];
            for k in start..end {
                let j = self.l.col_ind[k];
                if j < i {
                    sum -= self.l.values[k] * y[j];
                }
            }
            // Find diagonal
            let diag: f64 = {
                let mut d = 1e-10;
                for k in start..end {
                    if self.l.col_ind[k] == i {
                        d = self.l.values[k];
                        break;
                    }
                }
                d
            };
            y[i] = sum / diag;
        }

        // Backward solve: L^T * x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            // L^T has column i of L as row i -> need to find all rows j > i where L[j][i] != 0
            // Since L is CSR, we need to scan all rows
            for j in (i + 1)..n {
                let start = self.l.row_ptr[j];
                let end = self.l.row_ptr[j + 1];
                for k in start..end {
                    if self.l.col_ind[k] == i {
                        sum -= self.l.values[k] * x[j];
                        break;
                    }
                }
            }
            let diag: f64 = {
                let start = self.l.row_ptr[i];
                let end = self.l.row_ptr[i + 1];
                let mut d = 1e-10;
                for k in start..end {
                    if self.l.col_ind[k] == i {
                        d = self.l.values[k];
                        break;
                    }
                }
                d
            };
            x[i] = sum / diag;
        }

        x
    }
}

/// Preconditioned CG with a general preconditioner.
pub fn preconditioned_cg<F>(
    a: &CsrMatrix,
    b: &[f64],
    precond: F,
    max_iterations: usize,
    tolerance: f64,
) -> crate::types::SolverResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let z = precond(&r);
    let mut p = z.clone();
    let mut rz_old: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();

    let mut converged = false;
    let mut iter = 0;

    for k in 0..max_iterations {
        iter = k + 1;
        let ap = a.mat_vec(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
        if pap.abs() < 1e-30 { break; }
        let alpha = rz_old / pap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rnorm: f64 = r.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        if rnorm < tolerance {
            converged = true;
            break;
        }

        let z = precond(&r);
        let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    let ax = a.mat_vec(&x);
    let res_norm: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();

    crate::types::SolverResult { x, iterations: iter, residual_norm: res_norm, converged }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn spd_coo() -> CooMatrix {
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 4.0);
        coo.add_entry(0, 1, 1.0);
        coo.add_entry(1, 0, 1.0);
        coo.add_entry(1, 1, 3.0);
        coo.add_entry(1, 2, 1.0);
        coo.add_entry(2, 1, 1.0);
        coo.add_entry(2, 2, 4.0);
        coo
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let coo = spd_coo();
        let csr = coo.to_csr();
        let precond = JacobiPreconditioner::from_csr(&csr);
        let r = vec![4.0, 3.0, 4.0];
        let z = precond.apply(&r);
        assert_relative_eq!(z[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(z[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(z[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_incomplete_cholesky() {
        let coo = spd_coo();
        let csr = coo.to_csr();
        let ic = IncompleteCholesky::from_csr(&csr);
        // L should exist and be lower triangular
        assert!(ic.l.nnz() > 0);
    }

    #[test]
    fn test_preconditioned_cg_jacobi() {
        let coo = spd_coo();
        let csr = coo.to_csr();
        let precond = JacobiPreconditioner::from_csr(&csr);
        let b = vec![5.0, 5.0, 5.0];
        let result = preconditioned_cg(&csr, &b, |r| precond.apply(r), 100, 1e-10);
        assert!(result.converged);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_preconditioned_cg_icholesky() {
        let coo = spd_coo();
        let csr = coo.to_csr();
        let ic = IncompleteCholesky::from_csr(&csr);
        let b = vec![5.0, 5.0, 5.0];
        let result = preconditioned_cg(&csr, &b, |r| ic.apply(r), 100, 1e-10);
        assert!(result.converged);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_jacobi_larger_preconditioned() {
        let n = 30;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.add_entry(i, i, (n + 2) as f64);
            if i > 0 {
                coo.add_entry(i, i - 1, 1.0);
                coo.add_entry(i - 1, i, 1.0);
            }
        }
        let csr = coo.to_csr();
        let precond = JacobiPreconditioner::from_csr(&csr);
        let b = vec![1.0; n];
        let result = preconditioned_cg(&csr, &b, |r| precond.apply(r), 500, 1e-10);
        assert!(result.converged);
        assert!(result.residual_norm < 1e-6);
    }
}
