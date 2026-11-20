//! Increasing cost tree search ([`icts`], Sharon et al. 2013): optimal for sum of costs or
//! makespan ([`Objective`]).
//!
//! The high level searches cost vectors `(c_1, ..., c_k)`, one cost per agent, in order of
//! their sum, starting at every agent's individual optimum; a vector's children raise one
//! agent's cost by 1. For each vector, the low level builds the MDD of every agent at its
//! cost and looks for one conflict-free combination of MDD paths. Simple pairwise pruning
//! first checks every pair of agents on its own, which rules out most vectors cheaply.
//!
//! For makespan the tree is a chain: the vector `(m, ..., m)` for `m` from the largest
//! individual optimum on. An MDD of depth `m` holds every path that arrives by `m` and waits
//! at the goal, so the first `m` with a conflict-free combination is the optimal makespan.
//!
//! ```mermaid
//! flowchart TD
//!     root[individual optimal costs] -->|an agent cannot reach its goal| unsolvable[Unsolvable]
//!     root --> pop[pop the cheapest unvisited vector]
//!     pop --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| pairs{every pair has a joint path?}
//!     pairs -->|no| grow[queue the vectors with one cost raised]
//!     pairs -->|yes| all{all agents have one?}
//!     all -->|yes| solved[Solved]
//!     all -->|no| grow
//!     grow --> pop
//! ```

mod joint;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use self::joint::joint_paths;
use crate::constraint::ConstraintSet;
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::low_level::distance_tables;
use crate::mdd::Mdd;
use crate::path::{Objective, Path, Time};
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds a conflict-free plan with the optimal `objective` value by increasing cost tree
/// search.
///
/// `budget` is asked before evaluating each cost vector; [`MapfResult::stats`] counts
/// evaluated vectors in [`Stats::expanded`] and queued vectors (the root included) in
/// [`Stats::generated`]. ICTS proves unsolvability only when an agent cannot reach its
/// goal even alone; on other unsolvable instances the tree is infinite and the run ends at
/// the budget.
///
/// For [`Objective::Makespan`] the plan's sum of costs is low but not guaranteed minimal
/// among the makespan-optimal plans: the joint MDD search returns the first combination it
/// finds. The search is exponential in the number of agents whose costs must rise, so ICTS
/// suits instances where few agents interact strongly.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Objective, icts};
///
/// // Two agents swap the ends of a corridor with one side pocket.
/// let grid = Grid::parse(&["...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = icts(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
/// assert_eq!(result.solution().expect("solvable").sum_of_costs, 7);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn icts<G: Graph>(
    instance: &Instance<G>,
    objective: Objective,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    let mut search = Search {
        instance,
        objective,
        distances: distance_tables(instance.graph(), instance.agents()),
        mdds: BTreeMap::new(),
        stats: Stats::default(),
    };
    let outcome = search.run(&mut budget);

    MapfResult {
        outcome,
        stats: search.stats,
    }
}

/// The state of one ICTS run.
struct Search<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// What the search minimizes; decides the children of a cost vector.
    objective: Objective,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// The MDD of each `(agent, cost)` built so far.
    mdds: BTreeMap<(AgentId, Time), Rc<Mdd>>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Search<'_, G> {
    /// Runs the high-level search to its end.
    fn run(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let Some(root) = self.root() else {
            return Outcome::Unsolvable;
        };

        let mut open = BTreeSet::from([(root.iter().sum::<u32>(), root.clone())]);
        let mut seen = BTreeSet::from([root]);
        self.stats.generated += 1;
        while let Some((_, costs)) = open.pop_first() {
            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.stats.expanded += 1;
            if let Some(paths) = self.evaluate(&costs) {
                return Outcome::Solved(Solution::new(paths));
            }

            for child in self.children(&costs) {
                if seen.insert(child.clone()) {
                    self.stats.generated += 1;
                    open.insert((child.iter().sum(), child));
                }
            }
        }

        unreachable!("the cost tree is infinite")
    }

    /// The children of the cost vector `costs`: one agent's cost raised by 1 for sum of
    /// costs, every cost raised by 1 for makespan.
    fn children(&self, costs: &[Time]) -> Vec<Vec<Time>> {
        match self.objective {
            Objective::SumOfCosts => (0..costs.len())
                .map(|agent| {
                    let mut child = costs.to_vec();
                    child[agent] += 1;
                    child
                })
                .collect(),
            Objective::Makespan => Vec::from([costs.iter().map(|cost| cost + 1).collect()]),
        }
    }

    /// The first cost vector: every agent's cost alone for sum of costs, their maximum for
    /// every agent for makespan. `None` if an agent cannot reach its goal.
    fn root(&self) -> Option<Vec<Time>> {
        let alone = self.individual_costs()?;

        Some(match self.objective {
            Objective::SumOfCosts => alone,
            Objective::Makespan => {
                let makespan = alone.iter().copied().max().unwrap_or(0);
                alloc::vec![makespan; alone.len()]
            }
        })
    }

    /// Every agent's cost alone; `None` if one cannot reach its goal.
    fn individual_costs(&self) -> Option<Vec<Time>> {
        self.instance
            .agents()
            .iter()
            .zip(&self.distances)
            .map(|(agent, distances)| distances[agent.start])
            .collect()
    }

    /// A conflict-free plan in which every agent arrives at exactly its cost in `costs`
    /// (or earlier, then waits), or `None` if there is none.
    fn evaluate(&mut self, costs: &[Time]) -> Option<Vec<Path>> {
        let mdds: Vec<Rc<Mdd>> = (0..costs.len())
            .map(|agent| self.mdd(agent, costs[agent]))
            .collect();

        for (i, a) in mdds.iter().enumerate() {
            for b in &mdds[i + 1..] {
                joint_paths(&[a, b])?;
            }
        }

        let all: Vec<&Mdd> = mdds.iter().map(Rc::as_ref).collect();

        joint_paths(&all)
    }

    /// The MDD of `agent` at `cost`, built once.
    fn mdd(&mut self, agent: AgentId, cost: Time) -> Rc<Mdd> {
        let (instance, distances) = (self.instance, &self.distances);

        Rc::clone(self.mdds.entry((agent, cost)).or_insert_with(|| {
            let mdd = Mdd::build(
                instance.graph(),
                instance.agents()[agent],
                &ConstraintSet::new(),
                &distances[agent],
                cost,
            )
            .expect("every cost from the individual optimum on has paths");

            Rc::new(mdd)
        }))
    }
}
