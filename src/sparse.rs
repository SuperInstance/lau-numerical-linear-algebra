//! Sparse linear systems: COO and CSR storage formats, sparse direct solver.

use serde::{Deserialize, Serialize};
use crate::types::SolverResult;

/// Sparse matrix in COOrdinate format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
}

impl CooMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, rows: Vec::new(), cols: Vec::new(), vals: Vec::new() }
    }

    pub fn add_entry(&mut self, row: usize, col: usize, val: f64) {
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrMatrix {
        let nnz = self.vals.len();
        let mut entries: Vec<(usize, usize, f64)> = self.rows.iter()
            .zip(self.cols.iter())
            .zip(self.vals.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();

        // Sort by row, then column; sum duplicates
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Merge duplicates
        let mut merged: Vec<(usize, usize, f64)> = Vec::new();
        for e in entries {
            if let Some(last) = merged.last_mut() {
                if last.0 == e.0 && last.1 == e.1 {
                    last.2 += e.2;
                    continue;
                }
            }
            merged.push(e);
        }

        let mut row_ptr = vec![0usize; self.nrows + 1];
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        for (r, c, v) in merged {
            col_ind.push(c);
            values.push(v);
            row_ptr[r + 1] += 1;
        }

        // Cumulative sum
        for i in 1..=self.nrows {
            row_ptr[i] += row_ptr[i - 1];
        }

        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr,
            col_ind,
            values,
        }
    }

    /// Dense matrix-vector product (for testing/comparison).
    pub fn mat_vec(&self, x: &[f64]) -> Vec<f64> {
        let csr = self.to_csr();
        csr.mat_vec(x)
    }
}

/// Sparse matrix in Compressed Sparse Row format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_ind: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Create a new CSR matrix from raw data.
    pub fn new(nrows: usize, ncols: usize, row_ptr: Vec<usize>, col_ind: Vec<usize>, values: Vec<f64>) -> Self {
        Self { nrows, ncols, row_ptr, col_ind, values }
    }

    /// Sparse matrix-vector product.
    pub fn mat_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                y[i] += self.values[k] * x[self.col_ind[k]];
            }
        }
        y
    }

    /// Number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Extract diagonal entries.
    pub fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.nrows.min(self.ncols)];
        for i in 0..diag.len() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                if self.col_ind[k] == i {
                    diag[i] = self.values[k];
                    break;
                }
            }
        }
        diag
    }

    /// Convert to dense (for testing).
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                dense[i][self.col_ind[k]] = self.values[k];
            }
        }
        dense
    }
}

/// Solve sparse triangular system Lx = b (lower triangular).
pub fn solve_lower_triangular(l: &CsrMatrix, b: &[f64]) -> Vec<f64> {
    let n = l.nrows;
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        let start = l.row_ptr[i];
        let end = l.row_ptr[i + 1];
        for k in start..end {
            let j = l.col_ind[k];
            if j < i {
                sum -= l.values[k] * x[j];
            } else if j == i {
                // diagonal
                if l.values[k].abs() > 1e-30 {
                    x[i] = sum / l.values[k];
                }
            }
        }
    }
    x
}

/// Solve sparse triangular system Ux = b (upper triangular).
pub fn solve_upper_triangular(u: &CsrMatrix, b: &[f64]) -> Vec<f64> {
    let n = u.nrows;
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        let start = u.row_ptr[i];
        let end = u.row_ptr[i + 1];
        for k in start..end {
            let j = u.col_ind[k];
            if j > i {
                sum -= u.values[k] * x[j];
            } else if j == i {
                if u.values[k].abs() > 1e-30 {
                    x[i] = sum / u.values[k];
                }
            }
        }
    }
    x
}

/// Sparse Conjugate Gradient using CSR matrix.
pub fn sparse_cg(a: &CsrMatrix, b: &[f64], max_iterations: usize, tolerance: f64) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|ri| ri * ri).sum();

    let mut converged = false;
    let mut iter = 0;

    for k in 0..max_iterations {
        iter = k + 1;
        let ap = a.mat_vec(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
        if pap.abs() < 1e-30 { break; }
        let alpha = rs_old / pap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new: f64 = r.iter().map(|ri| ri * ri).sum();
        if rs_new.sqrt() < tolerance {
            converged = true;
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    let ax = a.mat_vec(&x);
    let res_norm: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();

    SolverResult { x, iterations: iter, residual_norm: res_norm, converged }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_coo_to_csr() {
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 4.0);
        coo.add_entry(0, 1, 1.0);
        coo.add_entry(1, 0, 1.0);
        coo.add_entry(1, 1, 3.0);
        coo.add_entry(1, 2, 1.0);
        coo.add_entry(2, 1, 1.0);
        coo.add_entry(2, 2, 4.0);

        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 7);
        assert_eq!(csr.row_ptr.len(), 4);
    }

    #[test]
    fn test_sparse_mat_vec() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 2.0);
        coo.add_entry(1, 1, 3.0);

        let y = coo.mat_vec(&[1.0, 1.0]);
        assert_relative_eq!(y[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparse_cg() {
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 4.0);
        coo.add_entry(0, 1, 1.0);
        coo.add_entry(1, 0, 1.0);
        coo.add_entry(1, 1, 3.0);
        coo.add_entry(1, 2, 1.0);
        coo.add_entry(2, 1, 1.0);
        coo.add_entry(2, 2, 4.0);

        let csr = coo.to_csr();
        let b = vec![5.0, 5.0, 5.0];
        let result = sparse_cg(&csr, &b, 100, 1e-10);
        assert!(result.converged);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_csr_diagonal() {
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 4.0);
        coo.add_entry(1, 1, 3.0);
        coo.add_entry(2, 2, 4.0);

        let csr = coo.to_csr();
        let diag = csr.diagonal();
        assert_relative_eq!(diag[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(diag[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(diag[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lower_triangular_solve() {
        // L = [[2, 0, 0], [1, 3, 0], [0, 1, 4]]
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 2.0);
        coo.add_entry(1, 0, 1.0);
        coo.add_entry(1, 1, 3.0);
        coo.add_entry(2, 1, 1.0);
        coo.add_entry(2, 2, 4.0);

        let csr = coo.to_csr();
        let b = vec![2.0, 4.0, 5.0];
        let x = solve_lower_triangular(&csr, &b);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_upper_triangular_solve() {
        // U = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]  (diagonal upper triangular)
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 2.0);
        coo.add_entry(1, 1, 3.0);
        coo.add_entry(2, 2, 4.0);

        let csr = coo.to_csr();
        let b = vec![2.0, 3.0, 4.0];
        let x = solve_upper_triangular(&csr, &b);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparse_cg_larger() {
        let n = 50;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.add_entry(i, i, (n + 2) as f64);
            if i > 0 {
                coo.add_entry(i, i - 1, -1.0);
                coo.add_entry(i - 1, i, -1.0);
            }
        }
        let csr = coo.to_csr();
        let b = vec![1.0; n];
        let result = sparse_cg(&csr, &b, 500, 1e-10);
        assert!(result.converged);
        assert!(result.residual_norm < 1e-6);
    }

    #[test]
    fn test_coo_duplicate_entries() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0);
        coo.add_entry(0, 0, 2.0); // duplicate
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 1);
        assert_relative_eq!(csr.values[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_csr_to_dense() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0);
        coo.add_entry(0, 1, 2.0);
        coo.add_entry(1, 0, 3.0);
        coo.add_entry(1, 1, 4.0);
        let csr = coo.to_csr();
        let d = csr.to_dense();
        assert_relative_eq!(d[0][0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(d[1][1], 4.0, epsilon = 1e-10);
    }
}
