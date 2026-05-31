//! Eigenvalue algorithms: power iteration, inverse iteration, QR algorithm, Lanczos.

use crate::iterative::mat_vec;
use crate::types::EigenResult;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Power iteration to find the dominant eigenvalue and eigenvector.
pub fn power_iteration(
    a: &[Vec<f64>],
    max_iterations: usize,
    tolerance: f64,
) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let vnorm = norm(&v);
    for i in 0..n {
        v[i] /= vnorm;
    }

    let mut eigenvalue = 0.0;
    for _ in 0..max_iterations {
        let w = mat_vec(a, &v);
        let new_eigenvalue = dot(&v, &w);
        let wnorm = norm(&w);
        if wnorm < 1e-30 {
            break;
        }
        let mut new_v: Vec<f64> = w.iter().map(|wi| wi / wnorm).collect();

        // Check convergence
        let diff: f64 = new_v.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = new_v;
        eigenvalue = new_eigenvalue;
        if diff < tolerance {
            break;
        }
    }

    (eigenvalue, v)
}

/// Inverse iteration to find the eigenvalue closest to a given shift.
/// Requires A to be invertible.
pub fn inverse_iteration(
    a: &[Vec<f64>],
    shift: f64,
    max_iterations: usize,
    tolerance: f64,
) -> (f64, Vec<f64>) {
    let n = a.len();
    // Form (A - shift*I)
    let mut a_shifted = a.to_vec();
    for i in 0..n {
        a_shifted[i][i] -= shift;
    }

    let mut v = vec![1.0; n];
    let vnorm = norm(&v);
    for i in 0..n {
        v[i] /= vnorm;
    }

    for _ in 0..max_iterations {
        // Solve (A - shift*I) * w = v using Gaussian elimination
        let w = solve_small_system(&a_shifted, &v);
        let wnorm = norm(&w);
        if wnorm < 1e-30 {
            break;
        }
        let new_v: Vec<f64> = w.iter().map(|wi| wi / wnorm).collect();

        let diff: f64 = new_v.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = new_v;
        if diff < tolerance {
            break;
        }
    }

    // Rayleigh quotient for eigenvalue
    let av = mat_vec(a, &v);
    let eigenvalue = dot(&v, &av);

    (eigenvalue, v)
}

/// QR algorithm for all eigenvalues of a matrix.
/// Returns eigenvalues sorted by absolute value descending.
pub fn qr_algorithm(a: &[Vec<f64>], max_iterations: usize, tolerance: f64) -> Vec<f64> {
    let n = a.len();
    let mut t = a.to_vec();

    for _ in 0..max_iterations {
        // QR decomposition via Householder
        let (q, r) = qr_decompose(&t);
        // T = R * Q
        t = mat_mul(&r, &q);

        // Check for convergence: off-diagonal elements
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_diag += t[i][j] * t[i][j];
                }
            }
        }
        if off_diag.sqrt() < tolerance {
            break;
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| t[i][i]).collect();
    eigenvalues.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    eigenvalues
}

/// Lanczos algorithm for symmetric matrices.
/// Returns the tridiagonal matrix T and the Lanczos vectors.
pub fn lanczos(a: &[Vec<f64>], k: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let k = k.min(n);

    let mut q = Vec::new();
    let mut alpha = vec![0.0; k];
    let mut beta = vec![0.0; k];

    // q0 = random / normalized
    let mut q0 = vec![1.0; n];
    let qn = norm(&q0);
    for i in 0..n { q0[i] /= qn; }
    q.push(q0);

    let mut r = mat_vec(a, &q[0]);
    alpha[0] = dot(&q[0], &r);
    for i in 0..n {
        r[i] -= alpha[0] * q[0][i];
    }

    for j in 1..k {
        let rnorm = norm(&r);
        if rnorm < 1e-14 {
            break;
        }
        let qj: Vec<f64> = r.iter().map(|ri| ri / rnorm).collect();
        beta[j] = rnorm;
        q.push(qj.clone());

        r = mat_vec(a, &qj);
        alpha[j] = dot(&qj, &r);
        for i in 0..n {
            r[i] -= alpha[j] * qj[i] + beta[j] * q[j - 1][i];
        }
    }

    // Build tridiagonal matrix
    let actual_k = q.len();
    let mut t = vec![vec![0.0; actual_k]; actual_k];
    for i in 0..actual_k {
        t[i][i] = alpha[i];
        if i > 0 {
            t[i][i - 1] = beta[i];
            t[i - 1][i] = beta[i];
        }
    }

    (t, q)
}

/// Get eigenvalues from Lanczos tridiagonal matrix.
pub fn lanczos_eigenvalues(a: &[Vec<f64>], k: usize) -> Vec<f64> {
    let (t, _) = lanczos(a, k);
    qr_algorithm(&t, 200, 1e-12)
}

// ---- helpers ----

fn qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n { q[i][i] = 1.0; }
    let mut r = a.to_vec();

    for col in 0..n {
        let mut x = vec![0.0; n];
        for i in col..n {
            x[i] = r[i][col];
        }
        let xnorm = norm(&x);
        if xnorm < 1e-30 { continue; }
        let sign = if x[col] >= 0.0 { 1.0 } else { -1.0 };
        x[col] += sign * xnorm;
        let vnorm = norm(&x);
        if vnorm < 1e-30 { continue; }
        for xi in x.iter_mut() { *xi /= vnorm; }

        for j in 0..n {
            let mut dot_r = 0.0;
            for i in col..n { dot_r += x[i] * r[i][j]; }
            for i in col..n { r[i][j] -= 2.0 * x[i] * dot_r; }

            let mut dot_q = 0.0;
            for i in col..n { dot_q += x[i] * q[i][j]; }
            for i in col..n { q[i][j] -= 2.0 * x[i] * dot_q; }
        }
    }

    // Q currently has H0*H1*...*I as columns. Since H are symmetric,
    // Q = H_{n-1}*...*H1*H0 = transpose of H0*H1*...*H_{n-1}
    (transpose(&q), r)
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut t = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..p {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Gaussian elimination with partial pivoting
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
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-30 { continue; }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_power_iteration_dominant() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let (eigenvalue, _v) = power_iteration(&a, 200, 1e-12);
        // Eigenvalues are approx 4.618 and 2.382
        assert_relative_eq!(eigenvalue, 4.618, epsilon = 0.01);
    }

    #[test]
    fn test_inverse_iteration() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let (eigenvalue, _v) = inverse_iteration(&a, 2.3, 200, 1e-12);
        assert_relative_eq!(eigenvalue, 2.382, epsilon = 0.05);
    }

    #[test]
    fn test_qr_algorithm() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let eigenvalues = qr_algorithm(&a, 200, 1e-12);
        assert_eq!(eigenvalues.len(), 2);
        assert_relative_eq!(eigenvalues[0], 4.618, epsilon = 0.01);
        assert_relative_eq!(eigenvalues[1], 2.382, epsilon = 0.01);
    }

    #[test]
    fn test_lanczos_symmetric() {
        let a = vec![vec![4.0, 1.0, 0.0], vec![1.0, 3.0, 1.0], vec![0.0, 1.0, 4.0]];
        let eigs = lanczos_eigenvalues(&a, 3);
        // Approx eigenvalues: 5.0, 4.0, 2.0
        assert!(eigs.len() >= 1);
        // Check the largest eigenvalue is close to 5
        assert_relative_eq!(eigs[0].abs(), 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_power_iteration_diagonal() {
        let a = vec![vec![5.0, 0.0], vec![0.0, 2.0]];
        let (ev, v) = power_iteration(&a, 100, 1e-12);
        assert_relative_eq!(ev, 5.0, epsilon = 1e-8);
        // v should be approx [1, 0] or [-1, 0]
        assert!(v[0].abs() > 0.9);
        assert!(v[1].abs() < 0.1);
    }

    #[test]
    fn test_qr_algorithm_3x3() {
        let a = vec![vec![6.0, 2.0, 1.0], vec![2.0, 3.0, 1.0], vec![1.0, 1.0, 1.0]];
        let eigs = qr_algorithm(&a, 500, 1e-12);
        // Trace = 10
        let sum: f64 = eigs.iter().sum();
        assert_relative_eq!(sum, 10.0, epsilon = 0.1);
    }

    #[test]
    fn test_power_iteration_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (ev, v) = power_iteration(&a, 100, 1e-10);
        assert_relative_eq!(ev, 1.0, epsilon = 1e-6);
        assert!(v.iter().map(|x| x * x).sum::<f64>().sqrt() > 0.99);
    }

    #[test]
    fn test_inverse_iteration_diagonal() {
        let a = vec![vec![5.0, 0.0], vec![0.0, 2.0]];
        let (ev, _) = inverse_iteration(&a, 1.8, 200, 1e-10);
        assert_relative_eq!(ev, 2.0, epsilon = 0.1);
    }
}
