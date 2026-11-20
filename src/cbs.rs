//! Conflict-based search ([`cbs`], Sharon et al. 2015): optimal for sum of costs or for
//! makespan ([`Objective`]), with the ICBS improvements (Boyarski et al. 2015) and the CG
//! heuristic (Felner et al. 2018), each switchable in [`CbsConfig`].
//!
//! The high level is a best-first search over a constraint tree. Each node holds
//! constraints and one path per agent that is optimal under them; the node with the lowest
//! objective value (plus heuristic) is expanded by splitting one of its conflicts into two
//! children, each of which forbids the conflict to one of the two agents and replans only
//! that agent.
//!
//! ```mermaid
//! flowchart TD
//!     root[plan every agent alone] -->|an agent has no path| unsolvable[Unsolvable]
//!     root --> pop[pop cheapest node]
//!     pop -->|open list empty| unsolvable
//!     pop --> conflict{chosen conflict?}
//!     conflict -->|none| solved[Solved]
//!     conflict --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| split[two children: constrain one agent each, replan it]
//!     split -->|bypass: a child is as cheap, with fewer conflicts| adopt[adopt its paths, reopen the node]
//!     split --> push[open both children]
//!     adopt --> pop
//!     push --> pop
//! ```

mod cardinal;

use alloc::collections::BinaryHeap;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use heuristic_search::{Budget, Stats};

use self::cardinal::{Cardinality, cardinality, vertex_cover_bound};
use crate::conflict::{Conflict, conflicts};
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::low_level::{distance_tables, plan};
use crate::mdd::Mdd;
use crate::path::{Objective, Path, makespan, path_cost, sum_of_costs};
use crate::solution::{MapfResult, Outcome, Solution};

/// What [`cbs_with`] optimizes and which improvements it uses.
///
/// None of the improvements changes the objective value of the result; they only reduce
/// the number of constraint-tree nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CbsConfig {
    /// What the search minimizes.
    pub objective: Objective,
    /// Split cardinal conflicts first, then semi-cardinal ones, then the rest (earliest
    /// first within a class). Needs an MDD per conflicting agent and node.
    pub prioritize_conflicts: bool,
    /// Bypass (Boyarski et al. 2015): when a child is as cheap for its replanned agent as
    /// the parent and has fewer conflicts, take over its paths instead of splitting.
    pub bypass: bool,
    /// The CG heuristic: a lower bound on the minimum vertex cover of the graph of agents
    /// with cardinal conflicts. Admissible for [`Objective::SumOfCosts`] only; ignored for
    /// [`Objective::Makespan`]. Needs the same MDDs as
    /// [`prioritize_conflicts`](Self::prioritize_conflicts).
    pub cardinal_heuristic: bool,
}

impl CbsConfig {
    /// Plain CBS: earliest conflict first, no bypass, no heuristic.
    #[must_use]
    pub const fn plain(objective: Objective) -> Self {
        Self {
            objective,
            prioritize_conflicts: false,
            bypass: false,
            cardinal_heuristic: false,
        }
    }

    /// Every improvement switched on; what [`cbs`] uses.
    #[must_use]
    pub const fn improved(objective: Objective) -> Self {
        Self {
            objective,
            prioritize_conflicts: true,
            bypass: true,
            cardinal_heuristic: true,
        }
    }

    /// Returns `true` if nodes need the MDDs of their conflicting agents.
    fn needs_mdds(self) -> bool {
        self.prioritize_conflicts || self.uses_heuristic()
    }

    /// Returns `true` if the CG heuristic is on and admissible for the objective.
    fn uses_heuristic(self) -> bool {
        self.cardinal_heuristic && self.objective == Objective::SumOfCosts
    }
}

/// Finds a conflict-free plan that is optimal for `objective` with conflict-based search,
/// with every improvement of [`CbsConfig::improved`].
///
/// See [`cbs_with`] for the budget and the tie-breaking.
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
    budget: impl Budget,
) -> MapfResult {
    cbs_with(instance, CbsConfig::improved(objective), budget)
}

/// Finds a conflict-free plan that is optimal for `config.objective` with conflict-based
/// search and the improvements `config` selects.
///
/// Constraint-tree nodes are ordered by the objective (plus the CG heuristic, if on), then
/// by the other objective, then by the number of conflicts. For [`Objective::Makespan`],
/// the solution has the optimal makespan; its sum of costs is low but not guaranteed
/// minimal among all makespan-optimal plans, because each agent's path is still the
/// cheapest one under its constraints.
///
/// `budget` is asked before every constraint-tree expansion; [`MapfResult::stats`] counts
/// expanded nodes (bypasses included) in [`Stats::expanded`] and nodes with a path (the
/// root included) in [`Stats::generated`]. Pass [`Limits::expansions`](crate::Limits::expansions)
/// to bound the run. The low level runs to completion on every call.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, CbsConfig, Grid, Instance, Limits, Objective, cbs_with};
///
/// let grid = Grid::parse(&["...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let plain = cbs_with(&instance, CbsConfig::plain(Objective::Makespan), Limits::NONE);
/// assert_eq!(plain.solution().expect("solvable").makespan, 4);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn cbs_with<G: Graph>(
    instance: &Instance<G>,
    config: CbsConfig,
    mut budget: impl Budget,
) -> MapfResult {
    let mut search = Search::new(instance, config);
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
    /// One path per agent, optimal under its constraints.
    paths: Vec<Path>,
    /// The MDD of each agent at its current cost under its constraints, where computed:
    /// only for conflicting agents, and only if the configuration needs them. Shared with
    /// the parent where the agent was not replanned.
    mdds: Vec<Option<Rc<Mdd>>>,
    /// Sum of costs of [`Node::paths`].
    sum_of_costs: u32,
    /// Makespan of [`Node::paths`].
    makespan: u32,
    /// The conflict to split next; `None` if the paths are a solution.
    conflict: Option<Conflict>,
    /// The number of conflicts among [`Node::paths`]; a tie-breaker.
    conflict_count: usize,
    /// The CG heuristic value; `0` if off.
    heuristic: u32,
}

/// An open-list entry: a node and its order key.
struct Entry {
    /// The order key: the objective plus heuristic, the other objective, the conflict
    /// count, then creation order. Smallest first.
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
    /// What the search minimizes and how.
    config: CbsConfig,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// The open nodes.
    open: BinaryHeap<Entry>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<'a, G: Graph> Search<'a, G> {
    /// Prepares a run: computes the distance tables.
    fn new(instance: &'a Instance<G>, config: CbsConfig) -> Self {
        Self {
            instance,
            config,
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
                return Outcome::Solved(Solution::new(node.paths));
            };

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.expand(node, conflict);
        }

        Outcome::Unsolvable
    }

    /// The root node: every agent planned alone. `None` if one of them has no path.
    fn root(&self) -> Option<Node> {
        let agents = self.instance.agents().len();
        let paths = (0..agents)
            .map(|agent| self.replan(agent, &[]))
            .collect::<Option<Vec<_>>>()?;

        Some(self.node(Vec::new(), paths, vec![None; agents]))
    }

    /// Splits `conflict` of `parent`: bypasses it if a child allows, else opens the
    /// children that have a path.
    fn expand(&mut self, parent: Node, conflict: Conflict) {
        self.stats.expanded += 1;

        let children: Vec<(AgentId, Node)> = Constraint::resolving(conflict)
            .into_iter()
            .filter_map(|constraint| Some((constraint.agent(), self.child(&parent, constraint)?)))
            .collect();

        if self.config.bypass {
            let bypass = children
                .iter()
                .find(|(agent, child)| is_bypass(&parent, child, *agent));
            if let Some((agent, child)) = bypass {
                let adopted = self.adopt(parent, *agent, &child.paths[*agent]);
                self.push(adopted);
                return;
            }
        }

        for (_, child) in children {
            self.push(child);
        }
    }

    /// `parent` with the path of `agent` replaced by `path`, which satisfies the parent's
    /// constraints at the same cost, so the agent's MDD stays valid.
    fn adopt(&self, parent: Node, agent: AgentId, path: &[Vertex]) -> Node {
        let mut paths = parent.paths;
        paths[agent] = path.to_vec();

        self.node(parent.constraints, paths, parent.mdds)
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
        let mut mdds = parent.mdds.clone();
        mdds[agent] = None;

        Some(self.node(constraints, paths, mdds))
    }

    /// The cheapest path of `agent` under those of `constraints` that name it.
    fn replan(&self, agent: AgentId, constraints: &[Constraint]) -> Option<Path> {
        let set = ConstraintSet::for_agent(agent, constraints);

        plan(
            self.instance.graph(),
            self.instance.agents()[agent],
            &set,
            &self.distances[agent],
        )
    }

    /// A node with `constraints`, `paths` and the known `mdds`: computes its costs, its
    /// conflicts, the missing MDDs it needs, the conflict to split and the heuristic.
    fn node(
        &self,
        constraints: Vec<Constraint>,
        paths: Vec<Path>,
        mut mdds: Vec<Option<Rc<Mdd>>>,
    ) -> Node {
        let found = conflicts(&paths);
        let (conflict, heuristic) = if self.config.needs_mdds() {
            self.fill_mdds(&constraints, &paths, &mut mdds, &found);
            self.classify(&found, &mdds)
        } else {
            (found.first().copied(), 0)
        };

        Node {
            sum_of_costs: sum_of_costs(&paths),
            makespan: makespan(&paths),
            conflict,
            conflict_count: found.len(),
            heuristic,
            constraints,
            paths,
            mdds,
        }
    }

    /// Builds the missing MDDs of the agents in `found`.
    fn fill_mdds(
        &self,
        constraints: &[Constraint],
        paths: &[Path],
        mdds: &mut [Option<Rc<Mdd>>],
        found: &[Conflict],
    ) {
        for conflict in found {
            let (a, b) = conflict.agents();
            for agent in [a, b] {
                if mdds[agent].is_some() {
                    continue;
                }

                let set = ConstraintSet::for_agent(agent, constraints);
                let mdd = Mdd::build(
                    self.instance.graph(),
                    self.instance.agents()[agent],
                    &set,
                    &self.distances[agent],
                    path_cost(&paths[agent]),
                )
                .expect("the agent's own path is in its MDD");
                mdds[agent] = Some(Rc::new(mdd));
            }
        }
    }

    /// The conflict to split (the earliest of the most cardinal class, if prioritizing,
    /// else the earliest) and the CG heuristic value.
    fn classify(&self, found: &[Conflict], mdds: &[Option<Rc<Mdd>>]) -> (Option<Conflict>, u32) {
        let classes: Vec<Cardinality> = found.iter().map(|c| cardinality(c, mdds)).collect();

        let conflict = if self.config.prioritize_conflicts {
            let best = classes.iter().max();
            best.and_then(|best| found.iter().zip(&classes).find(|(_, class)| *class == best))
                .map(|(conflict, _)| *conflict)
        } else {
            found.first().copied()
        };

        let heuristic = if self.config.uses_heuristic() {
            let mut edges: Vec<(AgentId, AgentId)> = found
                .iter()
                .zip(&classes)
                .filter(|(_, class)| **class == Cardinality::Cardinal)
                .map(|(conflict, _)| conflict.agents())
                .collect();
            edges.sort_unstable();
            edges.dedup();

            vertex_cover_bound(&edges)
        } else {
            0
        };

        (conflict, heuristic)
    }

    /// Opens `node`.
    fn push(&mut self, node: Node) {
        let (soc, makespan) = (node.sum_of_costs, node.makespan);
        let (first, second) = match self.config.objective {
            Objective::SumOfCosts => (soc + node.heuristic, makespan),
            Objective::Makespan => (makespan, soc),
        };
        let key = (first, second, node.conflict_count, self.stats.generated);

        self.stats.generated += 1;
        self.open.push(Entry { key, node });
    }
}

/// Returns `true` if `child`, which replanned `agent`, may replace the paths of `parent`:
/// the agent's path costs the same, so it is still optimal under the parent's constraints,
/// and there are strictly fewer conflicts.
fn is_bypass(parent: &Node, child: &Node, agent: AgentId) -> bool {
    path_cost(&child.paths[agent]) == path_cost(&parent.paths[agent])
        && child.conflict_count < parent.conflict_count
}
