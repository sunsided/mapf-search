//! Multi-objective conflict-based search ([`mo_cbs`], Ren, Rathinam and Choset 2021,
//! arXiv:2101.03805): every cost-unique Pareto-optimal joint plan when each move and wait
//! costs a vector of `N` objectives.
//!
//! MO-CBS generalizes [`cbs`](crate::cbs()). The low level ([`low_level`]) returns all
//! cost-unique Pareto-optimal paths of one agent under its constraints, not one cheapest
//! path. The roots are every combination of the agents' unconstrained Pareto-optimal paths,
//! so the constraint tree becomes a forest. The high level pops nodes in lexicographic cost
//! order, which pops a node that no other open node dominates, and keeps the set `C` of
//! solution costs found so far. A node whose cost some element of `C` weakly dominates is
//! filtered. A conflict-free node adds its cost to `C`, which then drops what the new cost
//! dominates. The search ends only when the open list is empty; `C` is then the
//! Pareto-optimal front.
//!
//! ```mermaid
//! flowchart TD
//!     init[Pareto paths of every agent alone] -->|an agent has none| unsolvable[Unsolvable]
//!     init --> roots[root nodes: all combinations, or one tree at a time]
//!     roots --> pop[pop the lexicographically cheapest node]
//!     pop -->|open list empty, next tree| roots
//!     pop -->|weakly dominated by a solution| pop
//!     pop --> conflict{first conflict?}
//!     conflict -->|none| record[add to the front, drop what it dominates]
//!     record --> pop
//!     conflict --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| split[two constraints; one child per Pareto path of the replanned agent]
//!     split --> pop
//!     pop -->|all trees done| complete[Complete]
//! ```
//!
//! The paper's variants map onto [`MoCbsConfig`]: [`MoCbsConfig::tree_by_tree`] is the
//! `-t` variant, and [`LowLevel`] picks BOA*-st (`-b`, two objectives) or NAMOA*dr-st
//! (`-n`, any number).
//!
//! Costs follow goal-stay, as everywhere in the crate: an agent pays for every move and
//! wait until its final arrival at its goal and nothing afterwards. With one objective and
//! unit costs, the front is the single optimal sum of costs.

mod low_level;

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};
use multiobjective_search::{Dominates, add, weakly_dominates};

use self::low_level::{Heuristic, pareto_paths};
use crate::conflict::{Conflict, first_conflict};
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::{Graph, Vertex};
use crate::instance::Instance;
use crate::path::Path;

/// Which multi-objective search the low level runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LowLevel {
    /// BOA*-st for two objectives (the paper's MO-CBS-b), NAMOA*dr-st otherwise.
    #[default]
    Auto,
    /// NAMOA*dr-st for any number of objectives (the paper's MO-CBS-n).
    NamoaDr,
}

/// How [`mo_cbs_with`] orders the high-level search and which low level it uses.
///
/// Every configuration returns the same front; they differ in effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MoCbsConfig {
    /// Tree-by-tree expansion (MO-CBS-t): generate one root at a time and exhaust its tree
    /// before the next. Plain MO-CBS generates every root up front, which is the product
    /// of the agents' Pareto set sizes, and expands across all trees.
    pub tree_by_tree: bool,
    /// The single-agent multi-objective planner.
    pub low_level: LowLevel,
}

impl MoCbsConfig {
    /// Plain MO-CBS: all roots up front, BOA* for two objectives.
    #[must_use]
    pub const fn plain() -> Self {
        Self {
            tree_by_tree: false,
            low_level: LowLevel::Auto,
        }
    }

    /// MO-CBS-t: one tree at a time, BOA* for two objectives. What [`mo_cbs`] uses.
    #[must_use]
    pub const fn tree_by_tree() -> Self {
        Self {
            tree_by_tree: true,
            low_level: LowLevel::Auto,
        }
    }
}

/// One Pareto-optimal joint plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoSolution<const N: usize> {
    /// The conflict-free path of each agent, indexed by [`AgentId`](crate::AgentId), from
    /// its start to its goal, where it stays.
    pub paths: Vec<Path>,
    /// The sum over agents of the cost vectors of their moves and waits up to their final
    /// arrival.
    pub cost: [u32; N],
}

/// How an [`mo_cbs`] run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParetoOutcome {
    /// The front is the complete Pareto-optimal front: one plan per Pareto-optimal cost
    /// vector.
    Complete,
    /// Proven unsolvable: an agent cannot reach its goal even alone, or the search ran out
    /// of nodes without a solution.
    Unsolvable,
    /// The [`Budget`] stopped the search. The front holds the conflict-free plans found so
    /// far, mutually non-dominated, but a later plan might have dominated some of them, and
    /// cost vectors may be missing. As with CBS, some unsolvable instances only ever end
    /// this way.
    LimitReached,
}

/// The result of an [`mo_cbs`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoResult<const N: usize> {
    /// How the run ended.
    pub outcome: ParetoOutcome,
    /// The plans found, sorted lexicographically by cost, at most one per cost vector.
    /// Empty unless a solution was found.
    pub front: Vec<ParetoSolution<N>>,
    /// High-level counters: expanded (split) nodes in [`Stats::expanded`], created nodes
    /// (roots and children, filtered or not) in [`Stats::generated`].
    pub stats: Stats,
}

/// [`mo_cbs_with`] with [`MoCbsConfig::tree_by_tree`]: every Pareto-optimal plan of
/// `instance`.
///
/// `costs(from, to)` is the cost vector of a move along the edge `from -> to`, and
/// `costs(v, v)` the cost of waiting at `v`. See [`mo_cbs_with`] for the requirements.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, ParetoOutcome, first_conflict, mo_cbs};
///
/// // Time and risk: every step takes 1 time unit; entering the middle row is risky.
/// let grid = Grid::new(3, 3);
/// let risk = |to: usize| if grid.cell(to).1 == 1 { 5 } else { 1 };
/// let costs = |_from: usize, to: usize| [1, risk(to)];
/// let agents = vec![Agent { start: grid.vertex(0, 1), goal: grid.vertex(2, 1) }];
/// let instance = Instance::new(grid.clone(), agents)?;
///
/// let result = mo_cbs(&instance, &costs, Limits::expansions(1_000));
///
/// assert_eq!(result.outcome, ParetoOutcome::Complete);
/// let costs: Vec<_> = result.front.iter().map(|plan| plan.cost).collect();
/// assert_eq!(costs, [[2, 10], [4, 8]]); // straight through, or around the risky row
/// assert!(result.front.iter().all(|plan| first_conflict(&plan.paths).is_none()));
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn mo_cbs<G: Graph, const N: usize>(
    instance: &Instance<G>,
    costs: &impl Fn(Vertex, Vertex) -> [u32; N],
    budget: impl Budget<Stats>,
) -> ParetoResult<N> {
    mo_cbs_with(instance, costs, MoCbsConfig::tree_by_tree(), budget)
}

/// Multi-objective CBS: every cost-unique Pareto-optimal plan of `instance` under the
/// vector costs `costs` (see [`mo_cbs`]), searched as `config` says.
///
/// Every cost component should be positive, waits included, as the paper requires: the
/// search then terminates on every solvable instance. With zero components a returned
/// [`ParetoOutcome::Complete`] front is still exact, but the search may not end on its
/// own. Sums must fit in `u32`.
///
/// `budget` is asked before every high-level expansion and before every root of plain
/// MO-CBS; the low level runs to completion on every call.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, MoCbsConfig, cbs, mo_cbs_with, Objective};
///
/// // With one objective and unit costs MO-CBS is CBS.
/// let grid = Grid::parse(&["...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let unit = |_: usize, _: usize| [1];
/// let front = mo_cbs_with(&instance, &unit, MoCbsConfig::plain(), Limits::NONE).front;
/// let optimum = cbs(&instance, Objective::SumOfCosts, Limits::NONE);
///
/// assert_eq!(front.len(), 1);
/// assert_eq!(front[0].cost, [optimum.solution().expect("solvable").sum_of_costs]);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn mo_cbs_with<G: Graph, const N: usize>(
    instance: &Instance<G>,
    costs: &impl Fn(Vertex, Vertex) -> [u32; N],
    config: MoCbsConfig,
    mut budget: impl Budget<Stats>,
) -> ParetoResult<N> {
    let mut search = Search::new(instance, costs, config);
    let outcome = search.run(&mut budget);
    let mut front = search.front;
    front.sort_by_key(|plan| plan.cost);

    ParetoResult {
        outcome,
        front,
        stats: search.stats,
    }
}

/// A shared path with its cost vector.
type PricedPath<const N: usize> = (Rc<[Vertex]>, [u32; N]);

/// One high-level node.
#[derive(Clone)]
struct Node<const N: usize> {
    /// All constraints on the way from the root, on any agent.
    constraints: Vec<Constraint>,
    /// One path per agent, Pareto-optimal under its constraints. Shared with the parent
    /// where the agent was not replanned.
    paths: Vec<Rc<[Vertex]>>,
    /// The cost vector of each path.
    costs: Vec<[u32; N]>,
    /// The sum of [`Node::costs`].
    cost: [u32; N],
}

impl<const N: usize> Node<N> {
    /// A node of `paths` with their `costs` under `constraints`.
    fn new(constraints: Vec<Constraint>, paths: Vec<Rc<[Vertex]>>, costs: Vec<[u32; N]>) -> Self {
        let cost = costs.iter().fold([0; N], |sum, &c| add(sum, c));

        Self {
            constraints,
            paths,
            costs,
            cost,
        }
    }
}

/// The state of one MO-CBS run.
struct Search<'a, G, F, const N: usize> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// The cost vector of each move and wait.
    costs: &'a F,
    /// Expansion order and low level.
    config: MoCbsConfig,
    /// The low-level heuristic of each agent.
    heuristics: Vec<Heuristic<N>>,
    /// The open nodes by cost, then creation order: lexicographically smallest first.
    open: BTreeMap<([u32; N], u64), Node<N>>,
    /// The solutions found so far (the paper's `C`), mutually non-dominated, at most one
    /// per cost vector.
    front: Vec<ParetoSolution<N>>,
    /// Counters reported in [`ParetoResult::stats`].
    stats: Stats,
}

impl<'a, G: Graph, F: Fn(Vertex, Vertex) -> [u32; N], const N: usize> Search<'a, G, F, N> {
    /// Prepares a run: computes the heuristic tables.
    fn new(instance: &'a Instance<G>, costs: &'a F, config: MoCbsConfig) -> Self {
        let graph = instance.graph();
        let heuristics = instance
            .agents()
            .iter()
            .map(|agent| Heuristic::new(graph, costs, agent.goal))
            .collect();

        Self {
            instance,
            costs,
            config,
            heuristics,
            open: BTreeMap::new(),
            front: Vec::new(),
            stats: Stats::default(),
        }
    }

    /// Runs the search to its end (Alg. 1 of the paper).
    fn run(&mut self, budget: &mut impl Budget<Stats>) -> ParetoOutcome {
        let Some(mut roots) = self.roots() else {
            return ParetoOutcome::Unsolvable;
        };

        if !self.config.tree_by_tree {
            for root in roots.by_ref() {
                if budget.exhausted(&self.stats) {
                    return ParetoOutcome::LimitReached;
                }

                self.push(root);
            }
        }

        loop {
            if !self.run_open(budget) {
                return ParetoOutcome::LimitReached;
            }

            let Some(root) = roots.next() else { break };
            self.push(root);
        }

        if self.front.is_empty() {
            ParetoOutcome::Unsolvable
        } else {
            ParetoOutcome::Complete
        }
    }

    /// The root nodes, or `None` if an agent has no path even alone.
    fn roots(&self) -> Option<Roots<N>> {
        let choices: Vec<Vec<PricedPath<N>>> = (0..self.instance.agents().len())
            .map(|agent| {
                self.replan(agent, &[])
                    .into_iter()
                    .map(|(path, cost)| (Rc::from(path), cost))
                    .collect::<Vec<_>>()
            })
            .collect();

        if choices.iter().any(Vec::is_empty) {
            return None;
        }

        Some(Roots::new(choices))
    }

    /// Pops and handles open nodes until the open list is empty. Returns `false` if the
    /// budget stopped it.
    fn run_open(&mut self, budget: &mut impl Budget<Stats>) -> bool {
        while let Some((_, node)) = self.open.pop_first() {
            if self.is_filtered(&node.cost) {
                continue;
            }

            let Some(conflict) = first_conflict(&node.paths) else {
                self.record(&node);
                continue;
            };

            if budget.exhausted(&self.stats) {
                return false;
            }

            self.expand(&node, conflict);
        }

        true
    }

    /// Returns `true` if a solution found so far weakly dominates `cost` (the paper's
    /// Filter): such a node cannot lead to a new Pareto-optimal cost vector.
    fn is_filtered(&self, cost: &[u32; N]) -> bool {
        self.front
            .iter()
            .any(|plan| weakly_dominates(&plan.cost, cost))
    }

    /// Adds the conflict-free `node` to the front and drops the solutions its cost
    /// dominates (the paper's Update).
    fn record(&mut self, node: &Node<N>) {
        self.front.retain(|plan| !node.cost.dominates(&plan.cost));
        self.front.push(ParetoSolution {
            paths: node.paths.iter().map(|path| path.to_vec()).collect(),
            cost: node.cost,
        });
    }

    /// Splits `conflict` of `node` into one constraint per agent and opens one child per
    /// Pareto-optimal path of the constrained agent.
    fn expand(&mut self, node: &Node<N>, conflict: Conflict) {
        self.stats.expanded += 1;

        for constraint in Constraint::resolving(conflict) {
            let agent = constraint.agent();
            let mut constraints = node.constraints.clone();
            constraints.push(constraint);

            for (path, cost) in self.replan(agent, &constraints) {
                let mut paths = node.paths.clone();
                let mut costs = node.costs.clone();
                paths[agent] = Rc::from(path);
                costs[agent] = cost;

                self.push(Node::new(constraints.clone(), paths, costs));
            }
        }
    }

    /// The Pareto-optimal paths of `agent` under `constraints`.
    fn replan(&self, agent: usize, constraints: &[Constraint]) -> Vec<(Path, [u32; N])> {
        pareto_paths(
            self.instance.graph(),
            self.costs,
            self.instance.agents()[agent],
            &ConstraintSet::for_agent(agent, constraints),
            &self.heuristics[agent],
            self.config.low_level,
        )
    }

    /// Opens `node` unless it is filtered.
    fn push(&mut self, node: Node<N>) {
        let order = self.stats.generated;
        self.stats.generated += 1;

        if !self.is_filtered(&node.cost) {
            self.open.insert((node.cost, order), node);
        }
    }
}

/// The root nodes: every combination of one unconstrained Pareto-optimal path per agent,
/// generated on demand in mixed-radix order (the last agent's choice varies fastest).
struct Roots<const N: usize> {
    /// The Pareto-optimal paths of each agent with their costs. None is empty.
    choices: Vec<Vec<PricedPath<N>>>,
    /// The choice index of each agent for the next root; `None` once all are generated.
    next: Option<Vec<usize>>,
}

impl<const N: usize> Roots<N> {
    /// The roots over `choices`, starting with the first path of every agent.
    fn new(choices: Vec<Vec<PricedPath<N>>>) -> Self {
        let next = Some(vec![0; choices.len()]);

        Self { choices, next }
    }

    /// Advances `indices` to the next combination; `false` after the last one.
    fn advance(&self, indices: &mut [usize]) -> bool {
        for (agent, index) in indices.iter_mut().enumerate().rev() {
            *index += 1;
            if *index < self.choices[agent].len() {
                return true;
            }

            *index = 0;
        }

        false
    }
}

impl<const N: usize> Iterator for Roots<N> {
    type Item = Node<N>;

    fn next(&mut self) -> Option<Node<N>> {
        let mut indices = self.next.take()?;
        let (paths, costs) = indices
            .iter()
            .zip(&self.choices)
            .map(|(&index, choices)| choices[index].clone())
            .unzip();

        if self.advance(&mut indices) {
            self.next = Some(indices);
        }

        Some(Node::new(Vec::new(), paths, costs))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the root enumeration.

    use super::*;

    /// Two agents with two and three paths give six roots, each combination once.
    #[test]
    fn roots_enumerate_the_product() {
        let path = |v: Vertex| -> Rc<[Vertex]> { Rc::from(vec![v]) };
        let roots = Roots::new(vec![
            vec![(path(0), [1]), (path(1), [2])],
            vec![(path(5), [10]), (path(6), [20]), (path(7), [30])],
        ]);

        let costs: Vec<[u32; 1]> = roots.map(|node| node.cost).collect();

        assert_eq!(costs, [[11], [21], [31], [12], [22], [32]]);
    }
}
