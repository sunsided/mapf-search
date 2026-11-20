//! # Multi-agent pathfinding
//!
//! Multi-agent pathfinding (MAPF): one path per agent on a shared graph, without
//! collisions, minimizing a joint objective.
//!
//! An [`Instance`] is a [`Graph`] (such as a [`Grid`] or an [`AdjacencyList`]) and a list
//! of [`Agent`]s with distinct starts and goals. Time is discrete; each step an agent moves
//! along an edge or waits, at cost 1. After its path ends, an agent stays at its goal and
//! keeps blocking it. Two agents [`Conflict`] if they share a vertex at the same time or
//! swap vertices along an edge.
//!
//! - [`cbs()`]: conflict-based search, optimal for [`sum_of_costs`].
//! - [`space_time_astar`]: the single-agent low level under [`Constraint`]s, built on the
//!   `heuristic-search` crate's A*.
//! - [`first_conflict`] and [`conflicts`]: conflict detection on any set of paths.
//!
//! [`Budget`], [`Limits`] and [`Stats`] are re-exported from `heuristic-search`; they bound
//! and count CBS's constraint-tree expansions.
//!
//! The crate is `no_std` and needs only `alloc`.
//!
//! ## Example
//!
//! ```
//! use mapf_search::{Agent, Grid, Instance, Limits, cbs, first_conflict};
//!
//! // Two agents swap the ends of a corridor that has one side pocket.
//! let grid = Grid::parse(&["...", "@.@"]);
//! let agents = vec![
//!     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
//!     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
//! ];
//! let instance = Instance::new(grid, agents)?;
//!
//! let result = cbs(&instance, Limits::expansions(1_000));
//! let solution = result.solution().expect("the pocket lets them pass");
//!
//! assert_eq!(first_conflict(&solution.paths), None);
//! assert_eq!(solution.cost, 7);
//! # Ok::<(), mapf_search::InstanceError>(())
//! ```
//!
//! ## Planned scope
//!
//! - Conflict-based: ECBS, CBS-TA, and CBS improvements (prioritized conflicts,
//!   heuristics, disjoint splitting)
//! - Cost-vector search: ICTS
//! - Priority-based: PBS
//! - Subdimensional: M*
//! - Local repair: MAPF-LNS2
//! - Partial expansion: EPEA*
//! - Moving AI `.map` and `.scen` benchmarks
//!
//! See `docs/design.md` in the repository for the design and milestones.
//!
//! ## Status
//!
//! Pre-release. The API will change.
//!
//! Repository: <https://github.com/sunsided/mapf-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod cbs;
mod conflict;
mod constraint;
mod graph;
mod instance;
mod low_level;
mod path;

pub use cbs::{CbsResult, Outcome, Solution, cbs};
pub use conflict::{Conflict, conflicts, first_conflict};
pub use constraint::{Constraint, ConstraintSet};
pub use graph::{AdjacencyList, Graph, Grid, Vertex};
pub use heuristic_search::{Budget, Limits, Stats};
pub use instance::{Agent, AgentId, Instance, InstanceError};
pub use low_level::space_time_astar;
pub use path::{Path, Time, makespan, path_cost, position_at, sum_of_costs};
