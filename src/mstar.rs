//! M* ([`mstar`], Wagner and Choset 2015): A* in the joint space of all agents with
//! subdimensional expansion, optimal for sum of costs or makespan ([`Objective`]), in its
//! basic and its recursive form ([`MStarConfig`]).
//!
//! Each agent follows its individual optimal policy (a fixed shortest path to its goal)
//! until it collides. Colliding agents join the *collision set* of the state where the
//! collision was found, and that set is passed back to every predecessor. Only agents in a
//! state's collision set leave their policy; the others keep following it. So the search
//! stays low-dimensional where agents do not interact.
//!
//! - **Basic M\*** keeps one collision set per state, and its agents branch over all their
//!   moves.
//! - **Recursive M\* (rM\*)** keeps the collision set as disjoint groups: two collisions
//!   that share no agent stay apart. A group that holds every agent of the search branches
//!   over all moves, as in basic M\*. A smaller group moves along its own optimal joint
//!   path, which a separate M\* search over just that group computes, recursively, and
//!   caches. Only when a group's path collides with another agent do the two merge. So
//!   two pairs of interacting agents far apart cost two small searches, not one over all
//!   four.
//!
//! Goal-stay pricing, the same as an exact joint search: a joint state is every agent's
//! vertex plus a *finished* flag. An unfinished agent at its goal may finish; from then on
//! it stays there for free. Every other unfinished agent pays 1 per step, move or wait. So
//! waiting at the goal before leaving it again is paid for. The heuristic, the sum of the
//! unfinished agents' distances to their goals, is consistent. For makespan a joint step
//! costs 1 if any agent pays, and the heuristic is the largest of those distances, which is
//! consistent too.
//!
//! ```mermaid
//! flowchart TD
//!     pop[pop the state with the lowest f] -->|all agents finished| solved[Solved]
//!     pop -->|open list empty| unsolvable[Unsolvable]
//!     pop --> succ[successors: a group of all agents branches, a smaller group follows its own optimal path, the rest follow their policy]
//!     succ -.->|rM*: a smaller group's path is not cached| sub[M* over that group alone]
//!     sub -.-> succ
//!     succ -->|transition collides| back[merge the colliders into the collision set, pass it back to predecessors, reopen them]
//!     succ -->|collision-free| relax[record predecessor, pass back its collision set, relax g]
//!     back --> pop
//!     relax --> pop
//! ```

mod collision;
mod planner;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use self::planner::{Context, JointState, LimitReached, Planner};
use crate::graph::{Graph, Vertex};
use crate::instance::Instance;
use crate::low_level::distance_tables;
use crate::path::{Objective, Path};
use crate::solution::{MapfResult, Outcome, Solution};

/// What [`mstar_with`] optimizes and which form of M* it runs.
///
/// Both forms return a plan with the optimal value of the objective; they differ in speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MStarConfig {
    /// What the search minimizes.
    pub objective: Objective,
    /// Recursive M*: plan disjoint collision groups as separate subproblems. `false` runs
    /// basic M*, with one collision set per state.
    pub recursive: bool,
}

impl MStarConfig {
    /// Basic M*: one collision set per state.
    #[must_use]
    pub const fn basic(objective: Objective) -> Self {
        Self {
            objective,
            recursive: false,
        }
    }

    /// Recursive M*; what [`mstar`] uses.
    #[must_use]
    pub const fn recursive(objective: Objective) -> Self {
        Self {
            objective,
            recursive: true,
        }
    }
}

/// Finds a conflict-free plan with the optimal `objective` value with recursive M*
/// ([`MStarConfig::recursive`]).
///
/// See [`mstar_with`] for the budget and what is counted.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Objective, Outcome, mstar};
///
/// // Two agents swap the ends of a corridor with one side pocket.
/// let grid = Grid::parse(&["...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid.clone(), agents)?;
/// let result = mstar(&instance, Objective::SumOfCosts, Limits::NONE);
/// assert_eq!(result.solution().expect("solvable").sum_of_costs, 7);
///
/// // Without the pocket they cannot pass, and M* proves it.
/// let corridor = mapf_search::Grid::parse(&["..."]);
/// let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
/// let instance = Instance::new(corridor, agents)?;
/// let result = mstar(&instance, Objective::Makespan, Limits::NONE);
/// assert_eq!(result.outcome, Outcome::Unsolvable);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn mstar<G: Graph>(
    instance: &Instance<G>,
    objective: Objective,
    budget: impl Budget<Stats>,
) -> MapfResult {
    mstar_with(instance, MStarConfig::recursive(objective), budget)
}

/// Finds a conflict-free plan with the optimal `config.objective` value with the form of M*
/// that `config` selects.
///
/// The joint space is finite, so M* proves unsolvability by exhausting it
/// ([`Outcome::Unsolvable`]), also where CBS and ICTS only stop at their budget, such as
/// two agents swapping the ends of a corridor. `budget` is asked before every expansion, in
/// the top-level search and in every subgroup search of recursive M*;
/// [`MapfResult::stats`] counts expansions (re-expansions included) in
/// [`Stats::expanded`], generated joint states in [`Stats::generated`] and expansions after
/// a collision set grew in [`Stats::reopened`], all summed over every search.
///
/// For [`Objective::Makespan`] the plan's sum of costs is not guaranteed minimal among the
/// makespan-optimal plans. Makespan searches are much slower than sum-of-costs ones: an
/// agent with slack may wander without raising `f`, so the plateaus of equal `f` are wide.
/// Open states are therefore ordered by an estimate of their sum of costs within equal `f`.
/// On the benchmark's 8-agent instance recursive M* takes about 60 ms for sum of costs and
/// over a minute for makespan.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, MStarConfig, Objective, mstar_with};
///
/// // Two pairs of agents swap places in two rooms, one sooner than the other.
/// let grid = Grid::parse(&["...@.....", "@.@@@@@.@"]);
/// let at = |x, y| grid.vertex(x, y);
/// let agents = vec![
///     Agent { start: at(0, 0), goal: at(2, 0) },
///     Agent { start: at(2, 0), goal: at(0, 0) },
///     Agent { start: at(4, 0), goal: at(8, 0) },
///     Agent { start: at(8, 0), goal: at(4, 0) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let soc = Objective::SumOfCosts;
/// let basic = mstar_with(&instance, MStarConfig::basic(soc), Limits::NONE);
/// let recursive = mstar_with(&instance, MStarConfig::recursive(soc), Limits::NONE);
/// let cost = |result: &mapf_search::MapfResult| result.solution().map(|s| s.sum_of_costs);
/// assert_eq!(cost(&basic), Some(18));
/// assert_eq!(cost(&recursive), Some(18));
/// assert!(recursive.stats.expanded < basic.stats.expanded);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn mstar_with<G: Graph>(
    instance: &Instance<G>,
    config: MStarConfig,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    let distances = distance_tables(instance.graph(), instance.agents());
    let mut cx = Context {
        policy: policies(instance.graph(), &distances),
        instance,
        objective: config.objective,
        recursive: config.recursive,
        distances,
        planners: BTreeMap::new(),
        stats: Stats::default(),
    };
    let outcome = run(&mut cx, &mut budget);

    MapfResult {
        outcome,
        stats: cx.stats,
    }
}

/// Runs the top-level search over every agent to its end.
fn run<G: Graph>(cx: &mut Context<'_, G>, budget: &mut impl Budget<Stats>) -> Outcome {
    let agents = cx.instance.agents();
    let start: JointState = agents.iter().map(|a| (a.start, false)).collect();
    let mut planner = Planner::new((0..agents.len()).collect());

    match planner.search(cx, start, budget) {
        Ok(Some(states)) => Outcome::Solved(Solution::new(paths(&states))),
        Ok(None) => Outcome::Unsolvable,
        Err(LimitReached) => Outcome::LimitReached,
    }
}

/// Each agent's path along the joint `states`, trimmed of trailing waits.
fn paths(states: &[JointState]) -> Vec<Path> {
    let agents = states.first().map_or(0, Vec::len);

    (0..agents)
        .map(|agent| {
            let mut path: Path = states.iter().map(|state| state[agent].0).collect();
            while path.len() > 1 && path[path.len() - 2] == path[path.len() - 1] {
                path.pop();
            }

            path
        })
        .collect()
}

/// Each agent's individual optimal policy: from every vertex, the smallest-numbered
/// neighbour one step closer to its goal.
fn policies<G: Graph>(graph: &G, distances: &[Vec<Option<u32>>]) -> Vec<Vec<Option<Vertex>>> {
    distances
        .iter()
        .map(|distances| {
            (0..graph.vertex_count())
                .map(|vertex| {
                    let closer = distances[vertex]?.checked_sub(1)?;
                    let mut best = None;
                    graph.neighbors(vertex, |next| {
                        if distances[next] == Some(closer) {
                            best = Some(best.map_or(next, |b: Vertex| b.min(next)));
                        }
                    });

                    best
                })
                .collect()
        })
        .collect()
}
