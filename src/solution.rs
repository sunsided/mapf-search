//! What every solver returns: a [`Solution`], how the run ended ([`Outcome`]) and its
//! counters ([`MapfResult`]).

use alloc::vec::Vec;

use heuristic_search::Stats;

use crate::path::{Objective, Path, makespan, sum_of_costs};

/// A conflict-free plan: one path per agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    /// The path of each agent, indexed by [`AgentId`](crate::AgentId). Each starts at the
    /// agent's start and ends at its goal, where the agent stays.
    pub paths: Vec<Path>,
    /// The sum of costs of [`Solution::paths`] (see [`sum_of_costs`]).
    pub sum_of_costs: u32,
    /// The makespan of [`Solution::paths`] (see [`makespan`](crate::makespan())).
    pub makespan: u32,
}

impl Solution {
    /// A solution of `paths`, with both objectives computed.
    pub(crate) fn new(paths: Vec<Path>) -> Self {
        Self {
            sum_of_costs: sum_of_costs(&paths),
            makespan: makespan(&paths),
            paths,
        }
    }

    /// The value of `objective` for this solution.
    #[must_use]
    pub fn cost(&self, objective: Objective) -> u32 {
        match objective {
            Objective::SumOfCosts => self.sum_of_costs,
            Objective::Makespan => self.makespan,
        }
    }
}

/// How a solver run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A conflict-free solution. Its quality is what the solver guarantees: optimal for
    /// CBS, ICTS and M*, within the suboptimality factor for ECBS, none for PBS and LNS2.
    Solved(Solution),
    /// Proven unsolvable: an agent cannot reach its goal even alone, or a complete solver
    /// ran out of options.
    Unsolvable,
    /// The [`Budget`](crate::Budget) stopped the search. The instance may be unsolvable:
    /// for some unsolvable instances, such as two agents swapping the ends of a corridor,
    /// the search space is infinite and complete solvers only ever stop this way.
    LimitReached,
    /// An incomplete solver ([`pbs`](crate::pbs())) ran out of options before its budget.
    /// This proves nothing: the instance may still be solvable.
    Exhausted,
}

/// The result of a solver run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapfResult {
    /// How the run ended.
    pub outcome: Outcome,
    /// High-level counters; each solver documents what it counts.
    pub stats: Stats,
}

impl MapfResult {
    /// The solution, if one was found.
    #[must_use]
    pub fn solution(&self) -> Option<&Solution> {
        match &self.outcome {
            Outcome::Solved(solution) => Some(solution),
            Outcome::Unsolvable | Outcome::LimitReached | Outcome::Exhausted => None,
        }
    }
}
