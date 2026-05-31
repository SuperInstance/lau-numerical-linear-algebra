//! Condition number estimation.

use crate::eigen::qr_algorithm;

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

/// Estimate the condition number κ(A) = σ_max / σ_min using eigenvalues of A^T A.
/// Returns infinity for rank-deficient matrices.
pub fn condition_number(a: &[Vec<f64>]) -> f64 {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let eigenvalues = qr_algorithm(&ata, 500, 1e-12);

    // Check if matrix is rank-deficient
    let max_eig = eigenvalues.iter().cloned().fold(0.0f64, |a, b| a.max(b));
    let min_eig = eigenvalues.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));

    if min_eig < 1e-30 {
        return f64::INFINITY;
    }

    if max_eig < 1e-30 {
        return f64::INFINITY;
    }

    (max_eig / min_eig).sqrt()
}

/// Estimate condition number from a vector of singular values.
pub fn condition_number_from_singular_values(sv: &[f64]) -> f64 {
    let non_zero: Vec<f64> = sv.iter().filter(|&&s| s > 1e-14).cloned().collect();
    if non_zero.is_empty() {
        return f64::INFINITY;
    }
    let max_s = non_zero.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_s = non_zero.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_s < 1e-30 {
        return f64::INFINITY;
    }
    max_s / min_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_condition_number_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let kappa = condition_number(&a);
        assert_relative_eq!(kappa, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_condition_number_diagonal() {
        let a = vec![vec![10.0, 0.0], vec![0.0, 1.0]];
        let kappa = condition_number(&a);
        assert_relative_eq!(kappa, 10.0, epsilon = 0.01);
    }

    #[test]
    fn test_condition_number_singular() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let kappa = condition_number(&a);
        assert!(kappa > 1e5 || kappa.is_infinite());
    }

    #[test]
    fn test_condition_number_from_sv() {
        let sv = vec![5.0, 2.0, 1.0];
        let kappa = condition_number_from_singular_values(&sv);
        assert_relative_eq!(kappa, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_condition_number_spd() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let kappa = condition_number(&a);
        // Eigenvalues are ~4.618 and ~2.382, so singular values are sqrt of those
        // κ = sqrt(4.618/2.382) ≈ 1.393
        assert!(kappa > 1.0 && kappa < 3.0);
    }
}
