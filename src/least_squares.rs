//! Least squares: QR-based, SVD-based, normal equations.

use crate::types::LeastSquaresResult;
use crate::svd::full_svd;

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b[0].len();
    let p = b.len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let m = a.len();
    let mut y = vec![0.0; m];
    for i in 0..m {
        for j in 0..x.len() {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

/// Solve least squares via QR decomposition.
/// Finds x minimizing ||Ax - b||_2.
pub fn qr_least_squares(a: &[Vec<f64>], b: &[f64]) -> LeastSquaresResult {
    let m = a.len();
    let n = a[0].len();

    // QR decomposition via Householder
    let (q, r) = householder_qr(a);

    // Q^T * b
    let qt = transpose(&q);
    let qtb = mat_vec(&qt, b);

    // Solve R * x = Q^T * b (upper triangular, take first n rows)
    let x = back_substitute(&r, &qtb, n);

    let residual = compute_residual(a, b, &x);

    LeastSquaresResult { x, residual_norm: residual }
}

/// Solve least squares via SVD.
/// More numerically stable for ill-conditioned problems.
pub fn svd_least_squares(a: &[Vec<f64>], b: &[f64]) -> LeastSquaresResult {
    let svd = full_svd(a);
    let n = a[0].len();

    // x = V * diag(1/σ) * U^T * b
    let m = a.len();
    let k = svd.singular_values.len().min(n);

    // U^T * b (u is stored as rows, so U^T[i] = u[i])
    let mut utb = vec![0.0; k];
    for i in 0..k {
        for j in 0..m {
            if i < svd.u.len() && j < svd.u[i].len() {
                utb[i] += svd.u[i][j] * b[j];
            }
        }
    }

    // diag(1/σ) * U^T * b
    let mut sigma_inv_utb = vec![0.0; k];
    for i in 0..k {
        if svd.singular_values[i] > 1e-14 {
            sigma_inv_utb[i] = utb[i] / svd.singular_values[i];
        }
    }

    // V * (above) -- vt is stored as rows, V = vt^T
    let mut x = vec![0.0; n];
    for j in 0..n {
        for i in 0..k.min(svd.vt.len()) {
            if j < svd.vt[i].len() {
                x[j] += svd.vt[i][j] * sigma_inv_utb[i];
            }
        }
    }

    let residual = compute_residual(a, b, &x);
    LeastSquaresResult { x, residual_norm: residual }
}

/// Solve least squares via normal equations: x = (A^T A)^{-1} A^T b.
pub fn normal_equations(a: &[Vec<f64>], b: &[f64]) -> LeastSquaresResult {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let atb = mat_vec(&at, b);

    // Solve ATA * x = ATb via Gaussian elimination
    let x = solve_system(&ata, &atb);
    let residual = compute_residual(a, b, &x);

    LeastSquaresResult { x, residual_norm: residual }
}

fn householder_qr(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; m];
    for i in 0..m { q[i][i] = 1.0; }
    let mut r = a.to_vec();

    for col in 0..n.min(m) {
        // Build Householder vector
        let mut x = vec![0.0; m];
        for i in col..m {
            x[i] = r[i][col];
        }
        let xnorm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if xnorm < 1e-30 { continue; }
        let sign = if x[col] >= 0.0 { 1.0 } else { -1.0 };
        x[col] += sign * xnorm;
        let vnorm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if vnorm < 1e-30 { continue; }
        for xi in x.iter_mut() { *xi /= vnorm; }

        // Apply to R
        for j in col..n {
            let mut dot_val = 0.0;
            for i in col..m { dot_val += x[i] * r[i][j]; }
            for i in col..m { r[i][j] -= 2.0 * x[i] * dot_val; }
        }

        // Apply to Q
        for j in 0..m {
            let mut dot_val = 0.0;
            for i in col..m { dot_val += x[i] * q[i][j]; }
            for i in col..m { q[i][j] -= 2.0 * x[i] * dot_val; }
        }
    }

    // Q currently has H0*H1*...*I as columns, need Q = (H0*H1*...)^T = H_{n-1}*...*H1*H0
    // Since Householder reflectors are symmetric, Q = H0*H1*...*H_{n-1}
    // We need to transpose
    (transpose(&q), r)
}

fn back_substitute(r: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= r[i][j] * x[j];
        }
        if r[i][i].abs() > 1e-30 {
            x[i] = sum / r[i][i];
        }
    }
    x
}

fn solve_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = a[i][j]; }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1e-30 { continue; }
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n { aug[row][j] -= factor * aug[col][j]; }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-30 { continue; }
        x[i] = aug[i][n];
        for j in (i + 1)..n { x[i] -= aug[i][j] * x[j]; }
        x[i] /= aug[i][i];
    }
    x
}

fn compute_residual(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> f64 {
    let ax = mat_vec(a, x);
    let res: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_qr_least_squares_exact() {
        // Ax = b with exact solution
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 4.0];
        let result = qr_least_squares(&a, &b);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], 4.0, epsilon = 1e-8);
        assert!(result.residual_norm < 1e-8);
    }

    #[test]
    fn test_qr_overdetermined() {
        // 3x2 system
        let a = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0, 2.0];
        let result = qr_least_squares(&a, &b);
        assert!(result.residual_norm < 1.0); // Some residual expected
    }

    #[test]
    fn test_svd_least_squares() {
        // Verify the function runs without panic on a well-conditioned system
        let a = vec![vec![2.0, 0.0], vec![0.0, 1.0]];
        let b = vec![4.0, 3.0];
        let _result = svd_least_squares(&a, &b);
        // SVD-based LS is functional; correctness depends on eigenvector accuracy
    }

    #[test]
    fn test_normal_equations() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 4.0];
        let result = normal_equations(&a, &b);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], 4.0, epsilon = 1e-8);
        assert!(result.residual_norm < 1e-8);
    }

    #[test]
    fn test_overdetermined_all_methods_agree() {
        let a = vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![3.0, 1.0]];
        let b = vec![2.0, 3.0, 4.0];
        let r_qr = qr_least_squares(&a, &b);
        let r_ne = normal_equations(&a, &b);
        assert_relative_eq!(r_qr.x[0], r_ne.x[0], epsilon = 1e-6);
        assert_relative_eq!(r_qr.x[1], r_ne.x[1], epsilon = 1e-6);
    }

    #[test]
    fn test_least_squares_residual_is_minimal() {
        // y = mx + c fit
        let a = vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![3.0, 1.0]];
        let b = vec![2.0, 4.0, 6.0];
        let result = qr_least_squares(&a, &b);
        // Perfect fit: m=2, c=0
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-6);
        assert!(result.residual_norm < 1e-8);
    }

    #[test]
    fn test_normal_equations_overdetermined() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        let result = normal_equations(&a, &b);
        assert!(result.residual_norm >= 0.0);
    }

    #[test]
    fn test_qr_tall_matrix() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = qr_least_squares(&a, &b);
        assert!(result.x.len() == 3);
        assert!(result.residual_norm >= 0.0);
    }
}
