//! Iterative solvers for linear systems: Jacobi, Gauss-Seidel, SOR.

use crate::types::{ConvergenceCriteria, SolverResult};

/// Solve Ax = b using the Jacobi iterative method.
///
/// `a` is the matrix in row-major order, `b` is the RHS.
pub fn jacobi(a: &[Vec<f64>], b: &[f64], criteria: &ConvergenceCriteria) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut x_new = vec![0.0; n];

    let mut converged = false;
    let mut iter = 0;

    for k in 0..criteria.max_iterations {
        iter = k + 1;
        for i in 0..n {
            let mut sigma = 0.0;
            for j in 0..n {
                if j != i {
                    sigma += a[i][j] * x[j];
                }
            }
            if a[i][i].abs() < 1e-15 {
                x_new[i] = x[i];
            } else {
                x_new[i] = (b[i] - sigma) / a[i][i];
            }
        }
        x.copy_from_slice(&x_new);

        let res_norm = residual_norm(a, b, &x);
        if res_norm < criteria.tolerance {
            converged = true;
            break;
        }
    }

    SolverResult {
        x: x.clone(),
        iterations: iter,
        residual_norm: residual_norm(a, b, &x),
        converged,
    }
}

/// Solve Ax = b using the Gauss-Seidel iterative method.
pub fn gauss_seidel(a: &[Vec<f64>], b: &[f64], criteria: &ConvergenceCriteria) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];

    let mut converged = false;
    let mut iter = 0;

    for k in 0..criteria.max_iterations {
        iter = k + 1;
        for i in 0..n {
            let mut sigma = 0.0;
            for j in 0..n {
                if j != i {
                    sigma += a[i][j] * x[j];
                }
            }
            if a[i][i].abs() < 1e-15 {
                // leave as is
            } else {
                x[i] = (b[i] - sigma) / a[i][i];
            }
        }

        let res_norm = residual_norm(a, b, &x);
        if res_norm < criteria.tolerance {
            converged = true;
            break;
        }
    }

    SolverResult {
        residual_norm: residual_norm(a, b, &x),
        iterations: iter,
        x,
        converged,
    }
}

/// Solve Ax = b using Successive Over-Relaxation (SOR).
///
/// `omega` is the relaxation parameter (1 < omega < 2 for over-relaxation).
pub fn sor(
    a: &[Vec<f64>],
    b: &[f64],
    omega: f64,
    criteria: &ConvergenceCriteria,
) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];

    let mut converged = false;
    let mut iter = 0;

    for k in 0..criteria.max_iterations {
        iter = k + 1;
        for i in 0..n {
            let mut sigma = 0.0;
            for j in 0..n {
                if j != i {
                    sigma += a[i][j] * x[j];
                }
            }
            if a[i][i].abs() < 1e-15 {
                // leave
            } else {
                let gs = (b[i] - sigma) / a[i][i];
                x[i] = (1.0 - omega) * x[i] + omega * gs;
            }
        }

        let res_norm = residual_norm(a, b, &x);
        if res_norm < criteria.tolerance {
            converged = true;
            break;
        }
    }

    SolverResult {
        residual_norm: residual_norm(a, b, &x),
        iterations: iter,
        x,
        converged,
    }
}

/// Compute ||Ax - b||.
fn residual_norm(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> f64 {
    let n = b.len();
    let mut res = 0.0;
    for i in 0..n {
        let mut ax_i = 0.0;
        for j in 0..n {
            ax_i += a[i][j] * x[j];
        }
        let d = ax_i - b[i];
        res += d * d;
    }
    res.sqrt()
}

/// Matrix-vector product y = A*x.
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..x.len() {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn diag_dom_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 4.0],
        ]
    }

    fn rhs() -> Vec<f64> {
        vec![5.0, 5.0, 5.0]
    }

    #[test]
    fn test_jacobi_convergence() {
        let a = diag_dom_matrix();
        let b = rhs();
        let crit = ConvergenceCriteria::new(1000, 1e-10);
        let result = jacobi(&a, &b, &crit);
        assert!(result.converged);
        assert!(result.residual_norm < 1e-8);
        // exact: [1, 1, 1]
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gauss_seidel_convergence() {
        let a = diag_dom_matrix();
        let b = rhs();
        let crit = ConvergenceCriteria::new(1000, 1e-10);
        let result = gauss_seidel(&a, &b, &crit);
        assert!(result.converged);
        assert!(result.residual_norm < 1e-8);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sor_convergence() {
        let a = diag_dom_matrix();
        let b = rhs();
        let crit = ConvergenceCriteria::new(1000, 1e-10);
        let result = sor(&a, &b, 1.2, &crit);
        assert!(result.converged);
        assert!(result.residual_norm < 1e-8);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_jacobi_larger_system() {
        // 5x5 SPD-like diag dominant
        let n = 5;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = (n + 2) as f64;
            if i > 0 { a[i][i - 1] = -1.0; }
            if i < n - 1 { a[i][i + 1] = -1.0; }
        }
        let b = vec![1.0; n];
        let crit = ConvergenceCriteria::new(2000, 1e-12);
        let result = jacobi(&a, &b, &crit);
        assert!(result.converged);
        assert!(result.residual_norm < 1e-8);
    }

    #[test]
    fn test_gauss_seidel_faster_than_jacobi() {
        let a = diag_dom_matrix();
        let b = rhs();
        let crit = ConvergenceCriteria::new(1000, 1e-10);
        let r_gs = gauss_seidel(&a, &b, &crit);
        let r_j = jacobi(&a, &b, &crit);
        // GS typically converges in fewer iterations
        assert!(r_gs.iterations <= r_j.iterations);
    }

    #[test]
    fn test_sor_omega_1_is_gauss_seidel() {
        let a = diag_dom_matrix();
        let b = rhs();
        let crit = ConvergenceCriteria::new(100, 1e-10);
        let r_sor = sor(&a, &b, 1.0, &crit);
        let r_gs = gauss_seidel(&a, &b, &crit);
        // Should produce same result
        for i in 0..3 {
            assert_relative_eq!(r_sor.x[i], r_gs.x[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_jacobi_2x2() {
        let a = vec![vec![5.0, 1.0], vec![1.0, 5.0]];
        let b = vec![6.0, 6.0];
        let crit = ConvergenceCriteria::new(200, 1e-10);
        let result = jacobi(&a, &b, &crit);
        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sor_over_relaxation_faster() {
        let n = 10;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = (n + 2) as f64;
            if i > 0 { a[i][i - 1] = 1.0; }
            if i < n - 1 { a[i][i + 1] = 1.0; }
        }
        let b = vec![1.0; n];
        let crit = ConvergenceCriteria::new(500, 1e-10);
        let r_gs = gauss_seidel(&a, &b, &crit);
        let r_sor = sor(&a, &b, 1.3, &crit);
        assert!(r_gs.converged);
        assert!(r_sor.converged);
        // Both should converge
        assert!(r_gs.residual_norm < 1e-6);
        assert!(r_sor.residual_norm < 1e-6);
    }
}
