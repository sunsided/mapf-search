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
//! - [`cbs()`]: conflict-based search, optimal for [`sum_of_costs`] or [`makespan`]
//!   ([`Objective`]), with prioritized conflicts, bypass, the CG, DG or WDG heuristic
//!   ([`CbsHeuristic`]), a conflict
//!   avoidance table and disjoint [`Splitting`]; [`cbs_with`] and [`CbsConfig`] switch
//!   them individually.
//! - [`ecbs`]: enhanced CBS, bounded-suboptimal within a [`Weight`] `w`, with focal search
//!   at both levels.
//! - [`icts`]: increasing cost tree search with pairwise pruning, optimal.
//! - [`mstar`]: recursive M* (subdimensional expansion), optimal, and able to prove
//!   unsolvability; [`mstar_with`] and [`MStarConfig`] select basic M*.
//! - [`pbs()`]: priority-based search, fast but neither optimal nor complete.
//!
//! Every solver takes an [`Objective`]: sum of costs or makespan.
//! - [`space_time_astar`]: the single-agent low level under [`Constraint`]s, built on the
//!   `heuristic-search` crate's A*.
//! - [`first_conflict`] and [`conflicts`]: conflict detection on any set of paths.
//! - [`parse_map`], [`parse_scenario`] and [`scenario_instance`]: Moving AI benchmarks.
//!
//! Every solver returns a [`MapfResult`]: an [`Outcome`] (with a [`Solution`] if solved)
//! and high-level [`Stats`]. [`Budget`], [`Limits`] and [`Stats`] are re-exported from
//! `heuristic-search`; they bound and count the high-level expansions. [`Weight`] is too.
//!
//! The crate is `no_std` and needs only `alloc`.
//!
//! ## Example
//!
//! ```
//! use mapf_search::{Agent, Grid, Instance, Limits, Objective, cbs, first_conflict};
//!
//! // Two agents swap the ends of a corridor that has one side pocket.
//! let grid = Grid::parse(&["...", "@.@"]);
//! let agents = vec![
//!     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
//!     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
//! ];
//! let instance = Instance::new(grid, agents)?;
//!
//! let result = cbs(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
//! let solution = result.solution().expect("the pocket lets them pass");
//!
//! assert_eq!(first_conflict(&solution.paths), None);
//! assert_eq!(solution.sum_of_costs, 7);
//! # Ok::<(), mapf_search::InstanceError>(())
//! ```
//!
//! ## Planned scope
//!
//! - Conflict-based: CBS-TA
//! - `ODrM*`: operator decomposition for recursive M*
//! - Local repair: MAPF-LNS2
//! - Partial expansion: EPEA*
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
mod conflict_table;
mod constraint;
mod ecbs;
mod focal;
mod graph;
mod icts;
mod instance;
mod low_level;
mod mdd;
mod movingai;
mod mstar;
mod path;
mod pbs;
mod solution;

pub use cbs::{CbsConfig, CbsHeuristic, Splitting, cbs, cbs_with};
pub use conflict::{Conflict, conflicts, first_conflict};
pub use constraint::{Constraint, ConstraintSet};
pub use ecbs::ecbs;
pub use graph::{AdjacencyList, Graph, Grid, Vertex};
pub use heuristic_search::{Budget, Limits, Stats, Weight};
pub use icts::icts;
pub use instance::{Agent, AgentId, Instance, InstanceError};
pub use low_level::space_time_astar;
pub use movingai::{
    ParseError, ParseErrorKind, ScenarioEntry, ScenarioError, parse_map, parse_scenario,
    scenario_instance,
};
pub use mstar::{MStarConfig, mstar, mstar_with};
pub use path::{Objective, Path, Time, makespan, path_cost, position_at, sum_of_costs};
pub use pbs::pbs;
pub use solution::{MapfResult, Outcome, Solution};
