//! Enhanced CBS ([`ecbs`], Barer et al. 2014): bounded-suboptimal for sum of costs, with
//! focal search at both levels.
//!
//! - **Low level:** each agent's path costs at most `w` times its cheapest path under its
//!   constraints, preferring few conflicts with the other agents' current paths. The search
//!   also reports `f_min`, a lower bound on that cheapest cost.
//! - **High level:** constraint-tree nodes are ordered by their lower bound, the sum of the
//!   agents' bounds. The focal sublist is every node whose sum of costs is at most `w`
//!   times the smallest lower bound, ordered by the number of conflicting agent pairs.
//!
//! The returned solution therefore costs at most `w` times the optimal sum of costs. With
//! `w = 1` ECBS is optimal.

mod low_level;

use alloc::vec::Vec;

use heuristic_search::{Budget, Stats, Weight};

use self::low_level::{ConflictTable, FocalPath, Problem, focal_plan};
use crate::conflict::{Conflict, conflicts};
use crate::constraint::{Constraint, ConstraintSet};
use crate::focal::FocalList;
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::low_level::distance_tables;
use crate::path::{Path, path_cost, sum_of_costs};
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds a conflict-free plan whose sum of costs is at most `weight` times the optimum, with
/// enhanced conflict-based search.
///
/// `budget` is asked before every constraint-tree expansion; [`MapfResult::stats`] counts
/// expanded nodes in [`Stats::expanded`] and nodes with paths (the root included) in
/// [`Stats::generated`]. Low-level searches run to completion. Like CBS, ECBS can only
/// prove unsolvability when an agent has no path at all or the tree runs empty.
///
/// Not supported: makespan, and goal-stay conflicts in the low level's conflict count (an
/// agent that finishes where others pass later is not penalized; the high level still
/// resolves such conflicts).
///
/// # Panics
///
/// Panics if `weight` is below `1`.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Weight, ecbs, first_conflict};
///
/// let grid = Grid::parse(&["@.@", "...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 1), goal: grid.vertex(2, 1) },
///     Agent { start: grid.vertex(1, 0), goal: grid.vertex(1, 2) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = ecbs(&instance, Weight::new(3, 2), Limits::expansions(1_000));
/// let solution = result.solution().expect("solvable");
/// assert_eq!(first_conflict(&solution.paths), None);
/// assert!(2 * solution.sum_of_costs <= 3 * 5, "the optimum is 5");
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn ecbs<G: Graph>(
    instance: &Instance<G>,
    weight: Weight,
    mut budget: impl Budget,
) -> MapfResult {
    assert!(
        weight.numerator() >= weight.denominator(),
        "the suboptimality factor must be at least 1"
    );

    let mut search = Search {
        instance,
        weight,
        distances: distance_tables(instance.graph(), instance.agents()),
        open: FocalList::new(weight),
        stats: Stats::default(),
    };
    let outcome = search.run(&mut budget);

    MapfResult {
        outcome,
        stats: search.stats,
    }
}

/// One constraint-tree node.
struct Node {
    /// All constraints on the way from the root, on any agent.
    constraints: Vec<Constraint>,
    /// One path per agent, within `w` of its bound under its constraints.
    paths: Vec<Path>,
    /// A lower bound per agent on its cheapest path under its constraints.
    lower_bounds: Vec<u32>,
    /// The earliest conflict among [`Node::paths`]; `None` if they are a solution.
    conflict: Option<Conflict>,
    /// The number of agent pairs in conflict; the focal key.
    conflicting_pairs: usize,
}

impl Node {
    /// A node with `constraints`, `paths` and their bounds; finds its conflicts.
    fn new(constraints: Vec<Constraint>, paths: Vec<Path>, lower_bounds: Vec<u32>) -> Self {
        let found = conflicts(&paths);
        let mut pairs: Vec<(AgentId, AgentId)> = found.iter().map(Conflict::agents).collect();
        pairs.sort_unstable();
        pairs.dedup();

        Self {
            constraints,
            paths,
            lower_bounds,
            conflict: found.first().copied(),
            conflicting_pairs: pairs.len(),
        }
    }
}

/// The state of one ECBS run.
struct Search<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// The suboptimality factor of both levels.
    weight: Weight,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// The open nodes: lower bound as `f`, sum of costs as value; focal key: conflicting
    /// pairs, then sum of costs.
    open: FocalList<(usize, u32), Node>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Search<'_, G> {
    /// Runs the high-level search to its end.
    fn run(&mut self, budget: &mut impl Budget) -> Outcome {
        let Some(root) = self.root() else {
            return Outcome::Unsolvable;
        };

        self.push(root);
        while let Some((node, _)) = self.open.pop() {
            let Some(conflict) = node.conflict else {
                return Outcome::Solved(Solution::new(node.paths));
            };

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.stats.expanded += 1;
            for constraint in Constraint::resolving(conflict) {
                if let Some(child) = self.child(&node, constraint) {
                    self.push(child);
                }
            }
        }

        Outcome::Unsolvable
    }

    /// The root: agents planned one after another, each avoiding the earlier ones' paths.
    /// `None` if one of them has no path.
    fn root(&self) -> Option<Node> {
        let count = self.instance.agents().len();
        let mut paths: Vec<Path> = Vec::with_capacity(count);
        let mut lower_bounds = Vec::with_capacity(count);
        for agent in 0..count {
            let found = self.replan(agent, &[], &paths)?;
            paths.push(found.path);
            lower_bounds.push(found.lower_bound);
        }

        Some(Node::new(Vec::new(), paths, lower_bounds))
    }

    /// The child of `parent` with `constraint` added, or `None` if the constrained agent
    /// then has no path.
    fn child(&self, parent: &Node, constraint: Constraint) -> Option<Node> {
        let mut constraints = parent.constraints.clone();
        constraints.push(constraint);

        let agent = constraint.agent();
        let found = self.replan(agent, &constraints, &parent.paths)?;

        let mut paths = parent.paths.clone();
        paths[agent] = found.path;
        // The parent's bound stays valid under more constraints and may be the tighter one.
        let mut lower_bounds = parent.lower_bounds.clone();
        lower_bounds[agent] = lower_bounds[agent].max(found.lower_bound);

        Some(Node::new(constraints, paths, lower_bounds))
    }

    /// A focal path of `agent` under those of `constraints` that name it, avoiding the paths
    /// of `others` (indexed by agent; may be a prefix, and may include `agent`) where cheap.
    fn replan(
        &self,
        agent: AgentId,
        constraints: &[Constraint],
        others: &[Path],
    ) -> Option<FocalPath> {
        let set = ConstraintSet::for_agent(agent, constraints);
        let table = ConflictTable::without(others, agent);
        let problem = Problem {
            graph: self.instance.graph(),
            agent: self.instance.agents()[agent],
            constraints: &set,
            distances: &self.distances[agent],
            table: &table,
        };

        focal_plan(&problem, self.weight)
    }

    /// Opens `node`.
    fn push(&mut self, node: Node) {
        let lower_bound = node.lower_bounds.iter().sum();
        let cost = sum_of_costs(&node.paths);
        debug_assert!(
            node.paths
                .iter()
                .zip(&node.lower_bounds)
                .all(|(p, &lb)| path_cost(p) >= lb),
            "bounds are below path costs"
        );

        self.stats.generated += 1;
        self.open
            .push(lower_bound, cost, (node.conflicting_pairs, cost), node);
    }
}
