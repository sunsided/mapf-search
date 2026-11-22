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
//! - [`cbs_ta()`]: CBS with task assignment over a [`TaskInstance`], whose agents each have
//!   candidate goals; optimal over all assignments, with every [`CbsConfig`] improvement
//!   ([`cbs_ta_with`]).
//! - [`ecbs()`]: enhanced CBS, bounded-suboptimal within a [`Weight`] `w`, with focal search
//!   at both levels.
//! - [`icts()`]: increasing cost tree search with pairwise pruning, optimal.
//! - [`mstar()`]: recursive M* with operator decomposition (subdimensional expansion),
//!   optimal, and able to prove unsolvability; [`mstar_with`] and [`MStarConfig`] select
//!   basic M*, recursive M* without operator decomposition, and OD-M*.
//! - [`epea()`]: enhanced partial expansion A* in the joint space, optimal, and able to
//!   prove unsolvability; it generates only the children at a node's stored `f`.
//! - [`lns2()`]: MAPF-LNS2, large neighborhood search that repairs a plan with collisions
//!   a few agents at a time; seeded ([`Lns2Config`]), neither optimal nor complete.
//! - [`pibt()`]: priority inheritance with backtracking, one time step for all agents at a
//!   time; very fast, neither optimal nor complete.
//! - [`pibt_turns()`]: PIBT*_T, PIBT for agents that face a heading, move only forward and
//!   need time to turn, on any [`OrientedGraph`] such as a [`Grid`] ([`TurnResult`]).
//! - [`pbs()`]: priority-based search, fast but neither optimal nor complete.
//! - `mo_cbs`, `mo_cbs_with` and `mo_cbs_per_agent`, with the `multi-objective` feature:
//!   multi-objective CBS (MO-CBS and MO-CBS-t, over BOA* or NAMOA*dr), every
//!   Pareto-optimal plan when moves and waits cost vectors, shared or per agent, with an
//!   opt-in feasibility check, conflict prioritization, bypass and disjoint splitting.
//!
//! Every solver but [`lns2()`], [`pibt()`] and [`pibt_turns()`], which only seek a feasible
//! plan, takes an [`Objective`]: sum of costs or makespan.
//! - [`space_time_astar`]: the single-agent low level under [`Constraint`]s, built on the
//!   `heuristic-search` crate's A*.
//! - [`first_conflict`] and [`conflicts`]: conflict detection on any set of paths.
//! - [`parse_map`], [`parse_scenario`] and [`scenario_instance`]: Moving AI benchmarks.
//!
//! Every solver returns a [`MapfResult`] ([`pibt_turns()`] inside a [`TurnResult`] with the
//! headings): an [`Outcome`] (with a [`Solution`] if solved)
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
//! Repository: <https://github.com/sunsided/mapf-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

mod assignment;
mod cbs;
mod conflict;
mod conflict_table;
mod constraint;
mod ecbs;
mod epea;
mod focal;
mod graph;
mod icts;
mod instance;
mod lns2;
mod low_level;
mod mdd;
#[cfg(feature = "multi-objective")]
mod mo_cbs;
mod movingai;
mod mstar;
mod path;
mod pbs;
mod pibt;
mod pibt_turns;
mod random;
mod solution;

pub use cbs::{CbsConfig, CbsHeuristic, Splitting, cbs, cbs_ta, cbs_ta_with, cbs_with};
pub use conflict::{Conflict, conflicts, first_conflict};
pub use constraint::{Constraint, ConstraintSet};
pub use ecbs::ecbs;
pub use epea::epea;
pub use graph::{AdjacencyList, Graph, Grid, Heading, OrientedGraph, Vertex};
pub use heuristic_search::{Budget, Limits, Stats, Weight};
pub use icts::icts;
pub use instance::{Agent, AgentId, Instance, InstanceError, TaskAgent, TaskInstance};
pub use lns2::{Lns2Config, lns2};
pub use low_level::space_time_astar;
#[cfg(feature = "multi-objective")]
pub use mo_cbs::{
    LowLevel, MoCbsConfig, ParetoOutcome, ParetoResult, ParetoSolution, mo_cbs, mo_cbs_per_agent,
    mo_cbs_with,
};
pub use movingai::{
    ParseError, ParseErrorKind, ScenarioEntry, ScenarioError, parse_map, parse_scenario,
    scenario_instance,
};
pub use mstar::{MStarConfig, mstar, mstar_with};
pub use path::{Objective, Path, Time, makespan, path_cost, position_at, sum_of_costs};
pub use pbs::pbs;
pub use pibt::pibt;
pub use pibt_turns::{TurnResult, pibt_turns};
pub use solution::{MapfResult, Outcome, Solution};
