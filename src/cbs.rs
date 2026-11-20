//! Conflict-based search ([`cbs`], Sharon et al. 2015): optimal for sum of costs or for
//! makespan ([`Objective`]).
//!
//! The high level is a best-first search over a constraint tree. Each node holds
//! constraints and one path per agent that is optimal under them; the node with the lowest
//! objective value is expanded by splitting its earliest conflict into two children, each of
//! which forbids the conflict to one of the two agents and replans only that agent.
//!
//! ```mermaid
//! flowchart TD
//!     root[plan every agent alone] -->|an agent has no path| unsolvable[Unsolvable]
//!     root --> pop[pop cheapest node]
//!     pop -->|open list empty| unsolvable
//!     pop --> conflict{earliest conflict?}
//!     conflict -->|none| solved[Solved]
//!     conflict --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| split[two children: constrain one agent each, replan it]
//!     split --> pop
//! ```

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use heuristic_search::{Budget, Stats};

use crate::conflict::{Conflict, conflicts};
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::low_level::{distance_tables, plan};
use crate::path::{Objective, Path, makespan, sum_of_costs};

/// A conflict-free plan: one path per agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    /// The path of each agent, indexed by [`AgentId`]. Each starts at the agent's start and
    /// ends at its goal, where the agent stays.
    pub paths: Vec<Path>,
    /// The sum of costs of [`Solution::paths`] (see [`sum_of_costs`]).
    pub sum_of_costs: u32,
    /// The makespan of [`Solution::paths`] (see [`makespan`](crate::makespan())).
    pub makespan: u32,
}

impl Solution {
    /// The value of `objective` for this solution.
    #[must_use]
    pub fn cost(&self, objective: Objective) -> u32 {
        match objective {
            Objective::SumOfCosts => self.sum_of_costs,
            Objective::Makespan => self.makespan,
        }
    }
}

/// How a run of [`cbs`] ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A solution that is optimal for the requested [`Objective`].
    Solved(Solution),
    /// Proven unsolvable: an agent cannot reach its goal even alone, or every branch of the
    /// constraint tree was closed.
    Unsolvable,
    /// The [`Budget`] stopped the search. The instance may be unsolvable: for some
    /// unsolvable instances, such as two agents swapping the ends of a corridor, the
    /// constraint tree is infinite and CBS only ever stops this way.
    LimitReached,
}

/// The result of a [`cbs`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CbsResult {
    /// How the run ended.
    pub outcome: Outcome,
    /// Constraint-tree counters: [`Stats::expanded`] counts split nodes,
    /// [`Stats::generated`] counts children with a path (the root included).
    pub stats: Stats,
}

impl CbsResult {
    /// The solution, if one was found.
    #[must_use]
    pub fn solution(&self) -> Option<&Solution> {
        match &self.outcome {
            Outcome::Solved(solution) => Some(solution),
            Outcome::Unsolvable | Outcome::LimitReached => None,
        }
    }
}

/// Finds a conflict-free plan that is optimal for `objective` with conflict-based search.
///
/// Constraint-tree nodes are ordered by `objective`, then by the other objective, then by
/// the number of conflicts. For [`Objective::Makespan`], the solution has the optimal
/// makespan; its sum of costs is low but not guaranteed minimal among all makespan-optimal
/// plans, because each agent's path is still the cheapest one under its constraints.
///
/// `budget` is asked before every constraint-tree expansion, with the counters described at
/// [`CbsResult::stats`]; pass [`Limits::expansions`](crate::Limits::expansions) to bound the run.
/// The low level runs to completion on every call.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Objective, cbs};
///
/// // Two agents cross on a plus-shaped map; one has to wait.
/// let grid = Grid::parse(&["@.@", "...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 1), goal: grid.vertex(2, 1) },
///     Agent { start: grid.vertex(1, 0), goal: grid.vertex(1, 2) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = cbs(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
/// let solution = result.solution().expect("solvable");
/// assert_eq!(solution.sum_of_costs, 5);
/// assert_eq!(solution.makespan, 3);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn cbs<G: Graph>(
    instance: &Instance<G>,
    objective: Objective,
    mut budget: impl Budget,
) -> CbsResult {
    let mut search = Search::new(instance, objective);
    let outcome = search.run(&mut budget);

    CbsResult {
        outcome,
        stats: search.stats,
    }
}

/// One constraint-tree node.
struct Node {
    /// All constraints on the way from the root, on any agent.
    constraints: Vec<Constraint>,
    /// One path per agent, optimal under its constraints.
    paths: Vec<Path>,
    /// Sum of costs of [`Node::paths`].
    sum_of_costs: u32,
    /// Makespan of [`Node::paths`].
    makespan: u32,
    /// The earliest conflict among [`Node::paths`]; `None` if they are a solution.
    conflict: Option<Conflict>,
    /// The number of conflicts among [`Node::paths`]; a tie-breaker.
    conflict_count: usize,
}

/// An open-list entry: a node and its order key.
struct Entry {
    /// The order key: the objective, the other objective, the conflict count, then
    /// creation order. Smallest first.
    key: (u32, u32, usize, u64),
    /// The node.
    node: Node,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    /// Reversed, so that [`BinaryHeap`] pops the smallest key.
    fn cmp(&self, other: &Self) -> Ordering {
        Reverse(self.key).cmp(&Reverse(other.key))
    }
}

/// The state of one CBS run.
struct Search<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// What the search minimizes.
    objective: Objective,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// The open nodes.
    open: BinaryHeap<Entry>,
    /// Counters reported in [`CbsResult::stats`].
    stats: Stats,
}

impl<'a, G: Graph> Search<'a, G> {
    /// Prepares a run: computes the distance tables.
    fn new(instance: &'a Instance<G>, objective: Objective) -> Self {
        Self {
            instance,
            objective,
            distances: distance_tables(instance.graph(), instance.agents()),
            open: BinaryHeap::new(),
            stats: Stats::default(),
        }
    }

    /// Runs the high-level search to its end.
    fn run(&mut self, budget: &mut impl Budget) -> Outcome {
        let Some(root) = self.root() else {
            return Outcome::Unsolvable;
        };

        self.push(root);
        while let Some(Entry { node, .. }) = self.open.pop() {
            let Some(conflict) = node.conflict else {
                return Outcome::Solved(Solution {
                    paths: node.paths,
                    sum_of_costs: node.sum_of_costs,
                    makespan: node.makespan,
                });
            };

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.expand(&node, conflict);
        }

        Outcome::Unsolvable
    }

    /// The root node: every agent planned alone. `None` if one of them has no path.
    fn root(&self) -> Option<Node> {
        let paths = (0..self.instance.agents().len())
            .map(|agent| self.replan(agent, &[]))
            .collect::<Option<Vec<_>>>()?;

        Some(node(Vec::new(), paths))
    }

    /// Splits `conflict` of `parent` into two children and opens those that have a path.
    fn expand(&mut self, parent: &Node, conflict: Conflict) {
        self.stats.expanded += 1;

        for constraint in split(conflict) {
            if let Some(child) = self.child(parent, constraint) {
                self.push(child);
            }
        }
    }

    /// The child of `parent` with `constraint` added, or `None` if the constrained agent
    /// then has no path.
    fn child(&self, parent: &Node, constraint: Constraint) -> Option<Node> {
        let mut constraints = parent.constraints.clone();
        constraints.push(constraint);

        let agent = constraint.agent();
        let path = self.replan(agent, &constraints)?;
        let mut paths = parent.paths.clone();
        paths[agent] = path;

        Some(node(constraints, paths))
    }

    /// The cheapest path of `agent` under those of `constraints` that name it.
    fn replan(&self, agent: AgentId, constraints: &[Constraint]) -> Option<Path> {
        let mut set = ConstraintSet::new();
        constraints
            .iter()
            .filter(|constraint| constraint.agent() == agent)
            .for_each(|constraint| set.insert(constraint));

        let graph = self.instance.graph();

        plan(
            graph,
            self.instance.agents()[agent],
            &set,
            &self.distances[agent],
        )
    }

    /// Opens `node`.
    fn push(&mut self, node: Node) {
        let (soc, makespan) = (node.sum_of_costs, node.makespan);
        let (first, second) = match self.objective {
            Objective::SumOfCosts => (soc, makespan),
            Objective::Makespan => (makespan, soc),
        };
        let key = (first, second, node.conflict_count, self.stats.generated);

        self.stats.generated += 1;
        self.open.push(Entry { key, node });
    }
}

/// A node with `constraints` and `paths`, its cost and conflicts filled in.
fn node(constraints: Vec<Constraint>, paths: Vec<Path>) -> Node {
    let found = conflicts(&paths);

    Node {
        constraints,
        sum_of_costs: sum_of_costs(&paths),
        makespan: makespan(&paths),
        conflict: found.first().copied(),
        conflict_count: found.len(),
        paths,
    }
}

/// The two constraints that resolve `conflict`, one per agent.
fn split(conflict: Conflict) -> [Constraint; 2] {
    match conflict {
        Conflict::Vertex {
            agents: (a, b),
            vertex,
            time,
        } => [a, b].map(|agent| Constraint::Vertex {
            agent,
            vertex,
            time,
        }),
        Conflict::Edge {
            agents: (a, b),
            from,
            to,
            time,
        } => [
            Constraint::Edge {
                agent: a,
                from,
                to,
                time,
            },
            Constraint::Edge {
                agent: b,
                from: to,
                to: from,
                time,
            },
        ],
    }
}
