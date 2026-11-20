//! Integration tests: hand-made instances, Moving AI parsing, property tests that compare
//! every solver with an independent joint-space search, and cross-checks between the
//! solvers on larger grids.

mod cases;
mod cross_check;
#[cfg(feature = "multi-objective")]
mod mo_cbs;
mod movingai;
mod oracle;
#[cfg(feature = "multi-objective")]
mod pareto_oracle;
mod properties;
mod solvers;
