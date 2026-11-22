//! Priority-based search ([`pbs`], Ma et al. 2019): a depth-first search over partial
//! priority orders between agents.
//!
//! Each node holds a partial order (as its transitive closure) and one path per agent
//! that is cheapest given the paths of all agents above it, which are moving obstacles.
//! For the earliest conflict between agents `a` and `b`, the node gets two children, "`a`
//! above `b`" and "`b` above `a`". A child replans the newly lower agent, then, from the top
//! down, every agent below it that now conflicts with an agent above it. The child that is
//! cheaper for the [`Objective`] is explored first.
//!
//! ```mermaid
//! flowchart TD
//!     root[plan every agent alone] -->|an agent has no path| unsolvable[Unsolvable]
//!     root --> pop[pop the top of the stack]
//!     pop -->|stack empty| exhausted[Exhausted]
//!     pop --> conflict{earliest conflict?}
//!     conflict -->|none| solved[Solved]
//!     conflict --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| split[two children: one agent above the other, replan below]
//!     split -->|push the dearer, then the cheaper| pop
//! ```

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use crate::conflict::{Conflict, first_conflict};
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::low_level::{distance_tables, plan};
use crate::path::{Objective, Path, Reservation, makespan, reservations, sum_of_costs};
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds a conflict-free plan with priority-based search, aiming at a low `objective`
/// value.
///
/// The objective decides which child is explored first: the one with the lower objective
/// value, ties broken by the other objective. Each agent's path is still the cheapest one
/// given the agents above it.
///
/// PBS is fast but neither optimal nor complete: it can return a plan dearer than the
/// optimum, and it can give up ([`Outcome::Exhausted`]) on a solvable instance. It proves
/// unsolvability ([`Outcome::Unsolvable`]) only when an agent cannot reach its goal even
/// alone.
///
/// `budget` is asked before every expansion; [`MapfResult::stats`] counts expanded nodes in
/// [`Stats::expanded`] and nodes whose replanning succeeded (the root included) in
/// [`Stats::generated`].
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Objective, pbs, first_conflict};
///
/// let grid = Grid::parse(&["@.@", "...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 1), goal: grid.vertex(2, 1) },
///     Agent { start: grid.vertex(1, 0), goal: grid.vertex(1, 2) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = pbs(&instance, Objective::SumOfCosts, Limits::expansions(100));
/// let solution = result.solution().expect("one agent goes first");
/// assert_eq!(first_conflict(&solution.paths), None);
/// assert_eq!(solution.sum_of_costs, 5);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn pbs<G: Graph>(
    instance: &Instance<G>,
    objective: Objective,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    let mut search = Search {
        instance,
        objective,
        distances: distance_tables(instance.graph(), instance.agents()),
        stats: Stats::default(),
    };
    let outcome = search.run(&mut budget);

    MapfResult {
        outcome,
        stats: search.stats,
    }
}

/// One node of the priority tree.
#[derive(Debug, Clone)]
struct Node {
    /// `higher[a]` is every agent with priority over `a`, transitively. Acyclic: no agent
    /// is above itself.
    higher: Vec<BTreeSet<AgentId>>,
    /// One path per agent, cheapest given the paths of the agents above it.
    paths: Vec<Path>,
    /// The objective value of [`Node::paths`], then the other objective's: the order in
    /// which children are explored.
    cost: (u32, u32),
}

/// The state of one PBS run.
struct Search<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// What orders the children.
    objective: Objective,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Search<'_, G> {
    /// Runs the depth-first search to its end.
    fn run(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let Some(root) = self.root() else {
            return Outcome::Unsolvable;
        };

        self.stats.generated += 1;
        let mut stack = Vec::from([root]);
        while let Some(node) = stack.pop() {
            let Some(conflict) = first_conflict(&node.paths) else {
                return Outcome::Solved(Solution::new(node.paths));
            };

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.stats.expanded += 1;
            let mut children = self.children(&node, conflict);
            self.stats.generated += children.len() as u64;
            children.sort_by_key(|child| core::cmp::Reverse(child.cost));
            stack.extend(children);
        }

        Outcome::Exhausted
    }

    /// The root: every agent planned alone, no priorities. `None` if an agent has no path.
    fn root(&self) -> Option<Node> {
        let count = self.instance.agents().len();
        let paths = (0..count)
            .map(|agent| self.replan(agent, &[]))
            .collect::<Option<Vec<_>>>()?;

        Some(Node {
            higher: vec![BTreeSet::new(); count],
            cost: self.cost(&paths),
            paths,
        })
    }

    /// The children of `node` that resolve `conflict` by ordering its two agents and
    /// whose replanning succeeds.
    fn children(&self, node: &Node, conflict: Conflict) -> Vec<Node> {
        let (a, b) = conflict.agents();

        [(a, b), (b, a)]
            .into_iter()
            .filter_map(|(high, low)| self.child(node, high, low))
            .collect()
    }

    /// `node` with `high` above `low`, replanned; `None` if that contradicts the order or
    /// an agent has no path.
    fn child(&self, node: &Node, high: AgentId, low: AgentId) -> Option<Node> {
        if node.higher[high].contains(&low) {
            return None;
        }

        let mut child = node.clone();
        let above: Vec<AgentId> = node.higher[high].iter().copied().chain([high]).collect();
        let affected = below_or_equal(&node.higher, low);
        for &agent in &affected {
            child.higher[agent].extend(above.iter().copied());
        }

        self.replan_below(&mut child, low, affected)?;
        child.cost = self.cost(&child.paths);

        Some(child)
    }

    /// Replans `low`, then, from the top down, each of `affected` (the agents at or below
    /// `low`) whose path conflicts with an agent above it. `None` if one has no path.
    fn replan_below(
        &self,
        node: &mut Node,
        low: AgentId,
        mut affected: Vec<AgentId>,
    ) -> Option<()> {
        // In a transitive closure, an agent above another has strictly fewer agents above
        // it, so this is a topological order.
        affected.sort_by_key(|&agent| node.higher[agent].len());

        for agent in affected {
            let higher: Vec<&Path> = node.higher[agent].iter().map(|&h| &node.paths[h]).collect();
            let collides = higher
                .iter()
                .any(|path| first_conflict(&[path.as_slice(), &node.paths[agent]]).is_some());
            if agent == low || collides {
                node.paths[agent] = self.replan(agent, &higher)?;
            }
        }

        Some(())
    }

    /// The objective value of `paths`, then the other objective's.
    fn cost(&self, paths: &[Path]) -> (u32, u32) {
        let (soc, makespan) = (sum_of_costs(paths), makespan(paths));

        match self.objective {
            Objective::SumOfCosts => (soc, makespan),
            Objective::Makespan => (makespan, soc),
        }
    }

    /// The cheapest path of `agent` that avoids the paths of `higher`, including where
    /// they stay at their goals for good.
    fn replan(&self, agent: AgentId, higher: &[&Path]) -> Option<Path> {
        let mut constraints = ConstraintSet::new();
        for path in higher {
            avoid(&mut constraints, agent, path);
        }

        plan(
            self.instance.graph(),
            self.instance.agents()[agent],
            &constraints,
            &self.distances[agent],
        )
    }
}

/// `low` and every agent below it in `higher`.
fn below_or_equal(higher: &[BTreeSet<AgentId>], low: AgentId) -> Vec<AgentId> {
    (0..higher.len())
        .filter(|&agent| agent == low || higher[agent].contains(&low))
        .collect()
}

/// Adds to `constraints` (for `agent`) what keeps it out of the way of `path`: its vertex
/// at every time, the reverse of each of its moves, and its last vertex from its arrival on.
fn avoid(constraints: &mut ConstraintSet, agent: AgentId, path: &[Vertex]) {
    for reservation in reservations(path) {
        match reservation {
            Reservation::Visit { vertex, time } => {
                constraints.insert(&Constraint::Vertex {
                    agent,
                    vertex,
                    time,
                });
            }
            Reservation::Move { from, to, time } => constraints.insert(&Constraint::Edge {
                agent,
                from: to,
                to: from,
                time,
            }),
            Reservation::Park { vertex, time } => constraints.block_from(vertex, time),
        }
    }
}
