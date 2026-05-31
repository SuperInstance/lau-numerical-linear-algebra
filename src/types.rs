//! Common types and result structures.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Result of an iterative solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResult {
    /// The solution vector.
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

/// Result of an eigenvalue computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenResult {
    /// Computed eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors (columns).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of iterations.
    pub iterations: usize,
}

/// Result of SVD computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvdResult {
    /// Singular values (descending).
    pub singular_values: Vec<f64>,
    /// Left singular vectors (columns).
    pub u: Vec<Vec<f64>>,
    /// Right singular vectors (columns).
    pub vt: Vec<Vec<f64>>,
}

/// Result of least squares solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeastSquaresResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Residual norm ||Ax - b||.
    pub residual_norm: f64,
}

/// Convergence criteria for iterative methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance for residual norm.
    pub tolerance: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl ConvergenceCriteria {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self { max_iterations, tolerance }
    }
}

/// Convert a slice to a nalgebra DVector.
pub fn to_dvector(v: &[f64]) -> DVector<f64> {
    DVector::from_vec(v.to_vec())
}

/// Convert a nalgebra DVector to a Vec.
pub fn from_dvector(v: &DVector<f64>) -> Vec<f64> {
    v.iter().copied().collect()
}
