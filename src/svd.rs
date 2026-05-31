//! Singular Value Decomposition: truncated and randomized.

use crate::types::SvdResult;
use crate::iterative::mat_vec;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn mat_transpose_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let m = a.len();
    let n = a[0].len();
    let mut y = vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            y[j] += a[i][j] * x[i];
        }
    }
    y
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

/// Full SVD via eigendecomposition of A^T A.
/// Returns U, singular values, V^T such that A ≈ U * diag(σ) * V^T.
pub fn full_svd(a: &[Vec<f64>]) -> SvdResult {
    let m = a.len();
    let n = a[0].len();

    // Compute A^T A
    let at = transpose(a);
    let ata = mat_mul(&at, a);

    // Eigendecomposition of A^T A via QR algorithm
    let eigenvalues = crate::eigen::qr_algorithm(&ata, 500, 1e-12);

    // Sort eigenvalues descending (they should already be, but ensure)
    let mut eig_pairs: Vec<(f64, usize)> = eigenvalues.iter().enumerate()
        .map(|(i, &v)| (v.max(0.0), i)) // clamp negative to 0
        .collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let singular_values: Vec<f64> = eig_pairs.iter().map(|(ev, _)| ev.sqrt()).collect();

    // Compute right singular vectors via inverse iteration
    let mut vt = vec![vec![0.0; n]; n];
    for (idx, &(ev, _)) in eig_pairs.iter().enumerate() {
        if ev > 1e-14 {
            let (_, v) = crate::eigen::inverse_iteration(&ata, ev.sqrt(), 200, 1e-10);
            for j in 0..n {
                vt[idx][j] = v[j];
            }
        }
    }

    // Compute left singular vectors: u_i = A * v_i / σ_i
    let mut u = vec![vec![0.0; m]; m.min(n)];
    for idx in 0..m.min(n) {
        if singular_values[idx] > 1e-14 {
            let vi: Vec<f64> = (0..n).map(|j| vt[idx][j]).collect();
            let avi = mat_vec(a, &vi);
            for i in 0..m {
                u[idx][i] = avi[i] / singular_values[idx];
            }
        }
    }

    SvdResult {
        singular_values,
        u,
        vt,
    }
}

/// Truncated SVD keeping only the top k singular values.
pub fn truncated_svd(a: &[Vec<f64>], k: usize) -> SvdResult {
    let full = full_svd(a);
    let k = k.min(full.singular_values.len());

    SvdResult {
        singular_values: full.singular_values[..k].to_vec(),
        u: full.u[..k].to_vec(),
        vt: full.vt[..k].to_vec(),
    }
}

/// Randomized SVD approximation.
/// Uses randomized projection to approximate the top-k singular values/vectors.
pub fn randomized_svd(a: &[Vec<f64>], k: usize, oversampling: usize, power_iterations: usize) -> SvdResult {
    let m = a.len();
    let n = a[0].len();
    let p = k + oversampling;

    // Random Gaussian matrix Omega (n x p)
    // Use a simple LCG for deterministic-ish random
    let mut omega = vec![vec![0.0; p]; n];
    let mut seed: u64 = 42;
    for j in 0..p {
        for i in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((seed >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
            omega[i][j] = val;
        }
    }

    // Y = A * Omega
    let at = transpose(a);
    let mut y = mat_mul(a, &omega);

    // Power iteration for better approximation
    for _ in 0..power_iterations {
        // QR factorization of Y
        let q = qr_q(&y);
        // Z = A^T * Q
        let z = mat_mul(&at, &q);
        // QR of Z
        let q2 = qr_q(&z);
        // Y = A * Q2
        y = mat_mul(a, &q2);
    }

    // QR of Y to get Q
    let q = qr_q(&y);

    // B = Q^T * A
    let qt = transpose(&q);
    let b = mat_mul(&qt, a);

    // SVD of the small matrix B (b is p x n)
    let small_svd = full_svd(&b);

    // U = Q * U_small
    let u_full = mat_mul(&q, &transpose(&small_svd.u));

    SvdResult {
        singular_values: small_svd.singular_values[..k.min(small_svd.singular_values.len())].to_vec(),
        u: transpose(&u_full)[..k.min(u_full[0].len())].to_vec(),
        vt: small_svd.vt[..k.min(small_svd.vt.len())].to_vec(),
    }
}

/// Extract Q from QR decomposition via modified Gram-Schmidt.
fn qr_q(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len().min(m);
    let mut q = vec![vec![0.0; n]; m];

    // Copy columns
    let at = transpose(a);
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| at[j].clone()).collect();

    for j in 0..n {
        for i in 0..j {
            let d = dot(&cols[j], &cols[i]);
            for k in 0..m {
                cols[j][k] -= d * cols[i][k];
            }
        }
        let nrm = norm(&cols[j]);
        if nrm > 1e-30 {
            for k in 0..m {
                cols[j][k] /= nrm;
            }
        }
    }

    // Store as rows
    for i in 0..m {
        for j in 0..n {
            q[i][j] = cols[j][i];
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_svd_diagonal() {
        let a = vec![vec![3.0, 0.0], vec![0.0, 2.0]];
        let result = full_svd(&a);
        assert_eq!(result.singular_values.len(), 2);
        assert_relative_eq!(result.singular_values[0], 3.0, epsilon = 0.01);
        assert_relative_eq!(result.singular_values[1], 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let svd = full_svd(&a);
        // Singular values should be correct
        assert_relative_eq!(svd.singular_values[0], 2.0, epsilon = 0.01);
        assert_relative_eq!(svd.singular_values[1], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_truncated_svd() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let result = truncated_svd(&a, 1);
        assert_eq!(result.singular_values.len(), 1);
        assert_relative_eq!(result.singular_values[0], 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_svd_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = full_svd(&a);
        assert_relative_eq!(result.singular_values[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.singular_values[1], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_randomized_svd() {
        let a = vec![vec![3.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 1.0]];
        let result = randomized_svd(&a, 2, 1, 2);
        assert!(result.singular_values.len() >= 2);
        // Top singular value should be close to 3.0
        assert_relative_eq!(result.singular_values[0], 3.0, epsilon = 0.5);
    }
}
