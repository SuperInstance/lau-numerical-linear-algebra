//! # lau-numerical-linear-algebra
//!
//! Numerical linear algebra for large-scale computations.
//! Iterative solvers, Krylov subspace methods, eigenvalue algorithms,
//! sparse solvers, preconditioners, SVD, and least squares.
//!
//! Designed for large-scale agent simulations — solving agent network
//! equations efficiently.

pub mod iterative;
pub mod krylov;
pub mod eigen;
pub mod sparse;
pub mod preconditioner;
pub mod svd;
pub mod least_squares;
pub mod condition;
pub mod types;

pub use types::*;
