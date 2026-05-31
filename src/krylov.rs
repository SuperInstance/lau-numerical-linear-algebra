//! Krylov subspace methods: CG, GMRES, BiCGSTAB.

use crate::types::{ConvergenceCriteria, SolverResult};
use crate::iterative::mat_vec;

/// Conjugate Gradient method for symmetric positive definite systems Ax = b.
pub fn cg(a: &[Vec<f64>], b: &[f64], criteria: &ConvergenceCriteria) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];

    // r = b - A*x (with x=0, r=b)
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    let mut converged = false;
    let mut iter = 0;

    for k in 0..criteria.max_iterations {
        iter = k + 1;
        let ap = mat_vec(a, &p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            converged = true;
            break;
        }
        let alpha = rs_old / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(&r, &r);
        if rs_new.sqrt() < criteria.tolerance {
            converged = true;
            break;
        }

        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    let res = residual(&r);
    SolverResult { x, iterations: iter, residual_norm: res, converged }
}

/// GMRES (Generalized Minimal Residual) for general nonsymmetric systems.
///
/// Uses restarted GMRES with the given restart parameter `m`.
pub fn gmres(a: &[Vec<f64>], b: &[f64], m: usize, criteria: &ConvergenceCriteria) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut converged = false;
    let mut total_iter = 0;

    for _outer in 0..criteria.max_iterations {
        let ax = mat_vec(a, &x);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
        let r_norm = norm(&r);
        if r_norm < criteria.tolerance {
            converged = true;
            break;
        }

        let mut v: Vec<Vec<f64>> = Vec::new();
        let mut v0 = r.iter().map(|ri| ri / r_norm).collect();
        v.push(v0);

        let mut beta = vec![0.0; m + 1];
        beta[0] = r_norm;

        // Hessenberg matrix stored as rows
        let mut h: Vec<Vec<f64>> = vec![vec![0.0; m]; m + 1];
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];

        let mut j = 0;
        for j_inner in 0..m {
            total_iter += 1;
            j = j_inner;

            // Arnoldi step
            let w = mat_vec(a, &v[j]);
            let mut w = w;

            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                for k in 0..n {
                    w[k] -= h[i][j] * v[i][k];
                }
            }
            h[j + 1][j] = norm(&w);

            if h[j + 1][j].abs() < 1e-14 {
                converged = true;
                break;
            }

            let mut vnew = w.iter().map(|wi| wi / h[j + 1][j]).collect();
            v.push(vnew);

            // Apply previous rotations
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }

            // Compute rotation
            let hr = h[j][j].hypot(h[j + 1][j]);
            if hr.abs() < 1e-30 {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = h[j][j] / hr;
                sn[j] = h[j + 1][j] / hr;
            }

            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            h[j + 1][j] = 0.0;

            beta[j + 1] = -sn[j] * beta[j];
            beta[j] = cs[j] * beta[j];

            if beta[j + 1].abs() < criteria.tolerance {
                converged = true;
                break;
            }
        }

        // Solve upper triangular system H*y = beta
        let mut y = vec![0.0; j + 1];
        for i in (0..=j).rev() {
            let mut s = beta[i];
            for k in (i + 1)..=j {
                s -= h[i][k] * y[k];
            }
            if h[i][i].abs() > 1e-30 {
                y[i] = s / h[i][i];
            }
        }

        // Update x
        for i in 0..n {
            for k in 0..=j {
                x[i] += v[k][i] * y[k];
            }
        }

        if converged {
            break;
        }

        let ax = mat_vec(a, &x);
        let rn: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();
        if rn < criteria.tolerance {
            converged = true;
            break;
        }
    }

    let ax = mat_vec(a, &x);
    let rn: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();

    SolverResult { x, iterations: total_iter, residual_norm: rn, converged }
}

/// BiCGSTAB (Biconjugate Gradient Stabilized) for general nonsymmetric systems.
pub fn bicgstab(a: &[Vec<f64>], b: &[f64], criteria: &ConvergenceCriteria) -> SolverResult {
    let n = b.len();
    let mut x = vec![0.0; n];

    let mut r = b.to_vec();
    let r0h = r.clone(); // shadow residual

    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];

    let mut converged = false;
    let mut iter = 0;

    for k in 0..criteria.max_iterations {
        iter = k + 1;
        let rho_new = dot(&r0h, &r);
        if rho_new.abs() < 1e-30 {
            break;
        }

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        v = mat_vec(a, &p);
        alpha = rho / dot(&r0h, &v);
        let mut s = vec![0.0; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }

        let t = mat_vec(a, &s);
        let omega_num = dot(&t, &s);
        let omega_den = dot(&t, &t);
        if omega_den.abs() < 1e-30 {
            omega = 0.0;
        } else {
            omega = omega_num / omega_den;
        }

        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        if norm(&r) < criteria.tolerance {
            converged = true;
            break;
        }
    }

    let ax = mat_vec(a, &x);
    let rn: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();
    SolverResult { x, iterations: iter, residual_norm: rn, converged }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn residual(r: &[f64]) -> f64 {
    norm(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn spd_matrix() -> Vec<Vec<f64>> {
        // Symmetric positive definite
        vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 4.0],
        ]
    }

    fn nonsym_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 5.0, 2.0],
            vec![0.0, 1.0, 6.0],
        ]
    }

    #[test]
    fn test_cg_spd() {
        let a = spd_matrix();
        let b = vec![5.0, 5.0, 5.0];
        let crit = ConvergenceCriteria::new(100, 1e-12);
        let r = cg(&a, &b, &crit);
        assert!(r.converged);
        for i in 0..3 {
            assert_relative_eq!(r.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cg_larger() {
        let n = 20;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = (n + 2) as f64;
            if i > 0 { a[i][i - 1] = 1.0; a[i - 1][i] = 1.0; }
        }
        let b = vec![1.0; n];
        let crit = ConvergenceCriteria::new(500, 1e-12);
        let r = cg(&a, &b, &crit);
        assert!(r.converged);
        assert!(r.residual_norm < 1e-8);
    }

    #[test]
    fn test_gmres_nonsymmetric() {
        let a = nonsym_matrix();
        let b = vec![5.0, 8.0, 7.0];
        let crit = ConvergenceCriteria::new(100, 1e-12);
        let r = gmres(&a, &b, 10, &crit);
        assert!(r.converged);
        assert!(r.residual_norm < 1e-8);
    }

    #[test]
    fn test_bicgstab_nonsymmetric() {
        let a = nonsym_matrix();
        let b = vec![5.0, 8.0, 7.0];
        let crit = ConvergenceCriteria::new(100, 1e-10);
        let r = bicgstab(&a, &b, &crit);
        assert!(r.converged);
        assert!(r.residual_norm < 1e-6);
    }

    #[test]
    fn test_cg_matches_exact_solution() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let b = vec![4.0, 9.0];
        let crit = ConvergenceCriteria::new(100, 1e-12);
        let r = cg(&a, &b, &crit);
        assert!(r.converged);
        assert_relative_eq!(r.x[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(r.x[1], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_gmres_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 4.0];
        let crit = ConvergenceCriteria::new(100, 1e-12);
        let r = gmres(&a, &b, 5, &crit);
        assert!(r.converged);
        assert_relative_eq!(r.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(r.x[1], 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_cg_diagonal() {
        let a = vec![vec![4.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 1.0]];
        let b = vec![8.0, 4.0, 3.0];
        let crit = ConvergenceCriteria::new(100, 1e-12);
        let r = cg(&a, &b, &crit);
        assert!(r.converged);
        assert_relative_eq!(r.x[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(r.x[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(r.x[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bicgstab_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![5.0, 7.0];
        let crit = ConvergenceCriteria::new(100, 1e-10);
        let r = bicgstab(&a, &b, &crit);
        assert!(r.converged);
        assert_relative_eq!(r.x[0], 5.0, epsilon = 1e-6);
        assert_relative_eq!(r.x[1], 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gmres_convergence() {
        let n = 5;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = (n + 2) as f64;
            if i > 0 { a[i][i - 1] = 1.0; }
            if i < n - 1 { a[i][i + 1] = 1.0; }
        }
        let b = vec![1.0; n];
        let crit = ConvergenceCriteria::new(500, 1e-6);
        let r = gmres(&a, &b, 5, &crit);
        assert!(r.converged);
        assert!(r.residual_norm < 0.01);
    }
}
