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
//!     check{feasibility check proves no plan?} -->|yes| unsolvable[Unsolvable]
//!     check -->|no, or off| init[Pareto paths of every agent alone]
//!     init -->|an agent has none| unsolvable
//!     init --> roots[root nodes: all combinations, or one tree at a time]
//!     roots --> pop[pop the lexicographically cheapest node]
//!     pop -->|open list empty, next tree| roots
//!     pop -->|weakly dominated by a solution| pop
//!     pop --> conflict{chosen conflict?}
//!     conflict -->|none| record[add to the front, drop what it dominates]
//!     record --> pop
//!     conflict --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| split[two constraints; one child per combination of Pareto paths]
//!     split -->|bypass: same costs, fewer conflicts| pop
//!     split --> pop
//!     pop -->|all trees done| complete[Complete]
//! ```
//!
//! The paper's variants map onto [`MoCbsConfig`]: [`MoCbsConfig::tree_by_tree`] is the
//! `-t` variant, and [`LowLevel`] picks BOA*-st (`-b`, two objectives) or NAMOA*dr-st
//! (`-n`, any number). [`MoCbsConfig::improved`] adds the paper's feasibility check and,
//! beyond the paper, ICBS-style conflict prioritization, bypass and disjoint splitting
//! ([`split`]); none of them changes the front.
//!
//! Costs follow goal-stay, as everywhere in the crate: an agent pays for every move and
//! wait until its final arrival at its goal and nothing afterwards. With one objective and
//! unit costs, the front is the single optimal sum of costs.

mod low_level;
mod node;
mod split;

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use heuristic_search::{Budget, Stats};
use multiobjective_search::{Dominates, weakly_dominates};

use self::low_level::{Heuristic, Problem, pareto_paths};
use self::node::{Combinations, Node, ParetoSet};
use crate::cbs::Splitting;
use crate::conflict::Conflict;
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::mstar::{MStarConfig, mstar_with};
use crate::path::{Objective, Path};
use crate::solution::Outcome;

/// Which multi-objective search the low level runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LowLevel {
    /// BOA*-st for two objectives (the paper's MO-CBS-b), NAMOA*dr-st otherwise.
    #[default]
    Auto,
    /// NAMOA*dr-st for any number of objectives (the paper's MO-CBS-n).
    NamoaDr,
}

/// How [`mo_cbs_with`] orders the high-level search, which low level it uses and which
/// improvements it applies.
///
/// Every configuration returns the same front; they differ in effort. The default is
/// [`MoCbsConfig::improved`], what [`mo_cbs`] uses.
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, MoCbsConfig, ParetoOutcome, mo_cbs_with};
///
/// // Two agents swap the ends of a corridor: no plan exists.
/// let grid = Grid::new(3, 1);
/// let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
/// let instance = Instance::new(grid, agents)?;
/// let costs = |_: usize, _: usize| [1, 1];
///
/// // The paper's MO-CBS only stops at its budget; the feasibility check proves it.
/// let plain = mo_cbs_with(&instance, &costs, MoCbsConfig::plain(), Limits::expansions(100));
/// let improved = mo_cbs_with(&instance, &costs, MoCbsConfig::improved(), Limits::NONE);
///
/// assert_eq!(plain.outcome, ParetoOutcome::LimitReached);
/// assert_eq!(improved.outcome, ParetoOutcome::Unsolvable);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent on-off switches, each documented, as in CbsConfig"
)]
pub struct MoCbsConfig {
    /// Tree-by-tree expansion (MO-CBS-t): generate one root at a time and exhaust its tree
    /// before the next. Plain MO-CBS generates every root up front, which is the product
    /// of the agents' Pareto set sizes, and expands across all trees.
    pub tree_by_tree: bool,
    /// The single-agent multi-objective planner.
    pub low_level: LowLevel,
    /// Before the search, prove infeasible instances [`ParetoOutcome::Unsolvable`] (paper,
    /// Sec. VII-B): feasibility does not depend on the costs, so recursive M* with unit
    /// costs decides it, under the same budget with M*'s counters. If that budget runs out,
    /// the search goes on without the proof.
    pub feasibility_check: bool,
    /// Split cardinal conflicts first, then semi-cardinal ones, then the rest (earliest
    /// first within a class). A conflict is cardinal for an agent if every path of the
    /// agent's current Pareto set takes part in it.
    pub prioritize_conflicts: bool,
    /// Bypass: when a child's replanned agents each have a path of the parent's cost that
    /// leaves fewer conflicts, take those paths over instead of splitting.
    pub bypass: bool,
    /// How a conflict is split into children. [`Splitting::Disjoint`] constrains the agent
    /// for which the conflict is cardinal if only one is, else the conflict's first agent.
    pub splitting: Splitting,
}

impl MoCbsConfig {
    /// Plain MO-CBS, Alg. 1 of the paper: all roots up front, BOA* for two objectives, no
    /// improvements.
    #[must_use]
    pub const fn plain() -> Self {
        Self {
            tree_by_tree: false,
            low_level: LowLevel::Auto,
            feasibility_check: false,
            prioritize_conflicts: false,
            bypass: false,
            splitting: Splitting::Standard,
        }
    }

    /// MO-CBS-t, as in the paper: one tree at a time, BOA* for two objectives, no
    /// improvements.
    #[must_use]
    pub const fn tree_by_tree() -> Self {
        Self {
            tree_by_tree: true,
            ..Self::plain()
        }
    }

    /// MO-CBS-t with the feasibility check, conflict prioritization, bypass and disjoint
    /// splitting. What [`mo_cbs`] uses.
    #[must_use]
    pub const fn improved() -> Self {
        Self {
            tree_by_tree: true,
            low_level: LowLevel::Auto,
            feasibility_check: true,
            prioritize_conflicts: true,
            bypass: true,
            splitting: Splitting::Disjoint,
        }
    }
}

impl Default for MoCbsConfig {
    fn default() -> Self {
        Self::improved()
    }
}

/// One Pareto-optimal joint plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoSolution<const N: usize> {
    /// The conflict-free path of each agent, indexed by [`AgentId`], from its start to its
    /// goal, where it stays.
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
    /// Proven unsolvable: by the feasibility check, because an agent cannot reach its goal
    /// even alone, or because the search ran out of nodes without a solution.
    Unsolvable,
    /// The [`Budget`] stopped the search. The front holds the conflict-free plans found so
    /// far, mutually non-dominated, but a later plan might have dominated some of them, and
    /// cost vectors may be missing. Without the feasibility check, some unsolvable instances
    /// only ever end this way.
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
    /// High-level counters: expanded (split) nodes, bypasses included, in
    /// [`Stats::expanded`], and created nodes (roots and children, filtered or not) in
    /// [`Stats::generated`]. The feasibility check's counters are not included.
    pub stats: Stats,
}

/// [`mo_cbs_with`] with [`MoCbsConfig::improved`]: every Pareto-optimal plan of
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
    mo_cbs_with(instance, costs, MoCbsConfig::improved(), budget)
}

/// Multi-objective CBS: every cost-unique Pareto-optimal plan of `instance` under the
/// vector costs `costs`, shared by all agents (see [`mo_cbs`]), searched as `config` says.
///
/// Every cost component should be positive, waits included, as the paper requires: the
/// search then terminates on every solvable instance. With zero components a returned
/// [`ParetoOutcome::Complete`] front is still exact, but the search may not end on its
/// own. Sums must fit in `u32`.
///
/// `budget` is asked before every high-level expansion, before every root of plain
/// MO-CBS, and inside the low level, before its first expansion and every 256 after; a
/// stop there ends the run with [`ParetoOutcome::LimitReached`]. The low level sees the
/// high-level counters, so a [`Limits`](crate::Limits) budget stops it only when those
/// are exhausted, while a deadline or a cancellation flag in a closure stops it at once.
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
    budget: impl Budget<Stats>,
) -> ParetoResult<N> {
    mo_cbs_per_agent(instance, &|_, from, to| costs(from, to), config, budget)
}

/// [`mo_cbs_with`] with one cost function per agent: `costs(agent, from, to)` is what the
/// move (or, if `from == to`, the wait) costs `agent`.
///
/// # Examples
///
/// ```
/// use mapf_search::{AdjacencyList, Agent, Instance, Limits, MoCbsConfig, mo_cbs_per_agent};
///
/// // A diamond with an upper and a lower route: agent 0 finds the upper one cheap, any
/// // other agent the lower one.
/// let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 3), (0, 2), (2, 3)]);
/// let agents = vec![Agent { start: 0, goal: 3 }];
/// let instance = Instance::new(graph, agents)?;
/// let costs = |agent: usize, from: usize, to: usize| {
///     let upper = from == 1 || to == 1;
///     if upper == (agent == 0) { [1, 1] } else { [2, 2] }
/// };
///
/// let result = mo_cbs_per_agent(&instance, &costs, MoCbsConfig::improved(), Limits::NONE);
/// assert_eq!(result.front[0].paths[0], [0, 1, 3]);
/// assert_eq!(result.front[0].cost, [2, 2]);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn mo_cbs_per_agent<G: Graph, const N: usize>(
    instance: &Instance<G>,
    costs: &impl Fn(AgentId, Vertex, Vertex) -> [u32; N],
    config: MoCbsConfig,
    mut budget: impl Budget<Stats>,
) -> ParetoResult<N> {
    let mut search = Search::new(instance, costs, config, &mut budget);
    let outcome = search.run();
    let mut front = search.front;
    front.sort_by_key(|plan| plan.cost);

    ParetoResult {
        outcome,
        front,
        stats: search.stats,
    }
}

/// The low level was interrupted by the budget.
struct Interrupted;

/// A child of a split, before its nodes are built: its constraints and the new Pareto set
/// of each agent it replans.
struct Child<const N: usize> {
    /// The parent's constraints plus the new one.
    constraints: Vec<Constraint>,
    /// The replanned agents with their Pareto sets under [`Child::constraints`]; none
    /// empty.
    replanned: Vec<(AgentId, ParetoSet<N>)>,
}

/// The state of one MO-CBS run.
struct Search<'a, G, F, const N: usize> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// The cost vector of each move and wait, per agent.
    costs: &'a F,
    /// Expansion order, low level and improvements.
    config: MoCbsConfig,
    /// The low-level heuristic of each agent.
    heuristics: Vec<Heuristic<N>>,
    /// The caller's budget, shared with the low level's interrupt hook.
    budget: RefCell<&'a mut dyn Budget<Stats>>,
    /// The open nodes by cost, then creation order: lexicographically smallest first.
    open: BTreeMap<([u32; N], u64), Node<N>>,
    /// The solutions found so far (the paper's `C`), mutually non-dominated, at most one
    /// per cost vector.
    front: Vec<ParetoSolution<N>>,
    /// Counters reported in [`ParetoResult::stats`].
    stats: Stats,
}

impl<'a, G: Graph, F: Fn(AgentId, Vertex, Vertex) -> [u32; N], const N: usize> Search<'a, G, F, N> {
    /// Prepares a run: computes the heuristic tables.
    fn new(
        instance: &'a Instance<G>,
        costs: &'a F,
        config: MoCbsConfig,
        budget: &'a mut dyn Budget<Stats>,
    ) -> Self {
        let graph = instance.graph();
        let heuristics = instance
            .agents()
            .iter()
            .enumerate()
            .map(|(id, agent)| Heuristic::new(graph, &|u, v| costs(id, u, v), agent.goal))
            .collect();

        Self {
            instance,
            costs,
            config,
            heuristics,
            budget: RefCell::new(budget),
            open: BTreeMap::new(),
            front: Vec::new(),
            stats: Stats::default(),
        }
    }

    /// Runs the search to its end (Alg. 1 of the paper).
    fn run(&mut self) -> ParetoOutcome {
        if self.config.feasibility_check && self.is_proven_infeasible() {
            return ParetoOutcome::Unsolvable;
        }

        let mut roots = match self.roots() {
            Ok(Some(roots)) => roots,
            Ok(None) => return ParetoOutcome::Unsolvable,
            Err(Interrupted) => return ParetoOutcome::LimitReached,
        };

        if !self.config.tree_by_tree {
            for root in roots.by_ref() {
                if self.is_exhausted() {
                    return ParetoOutcome::LimitReached;
                }

                self.push(root);
            }
        }

        loop {
            if self.run_open().is_err() {
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

    /// Returns `true` if recursive M* proves that no conflict-free plan exists. M* gets
    /// the budget with its own counters.
    fn is_proven_infeasible(&self) -> bool {
        let config = MStarConfig::recursive(Objective::SumOfCosts);
        let budget = |stats: &Stats| self.budget.borrow_mut().exhausted(stats);

        mstar_with(self.instance, config, budget).outcome == Outcome::Unsolvable
    }

    /// Returns `true` if the budget says stop, given the high-level counters.
    fn is_exhausted(&self) -> bool {
        self.budget.borrow_mut().exhausted(&self.stats)
    }

    /// The root nodes; `None` if an agent has no path even alone.
    fn roots(&self) -> Result<Option<Combinations<N>>, Interrupted> {
        let mut sets = Vec::new();
        for agent in 0..self.instance.agents().len() {
            let set = self.replan(agent, &[])?;
            if set.is_empty() {
                return Ok(None);
            }

            sets.push(set);
        }

        Ok(Some(Combinations::roots(sets)))
    }

    /// Pops and handles open nodes until the open list is empty.
    fn run_open(&mut self) -> Result<(), Interrupted> {
        while let Some((_, node)) = self.open.pop_first() {
            if self.is_filtered(&node.cost) {
                continue;
            }

            let Some(conflict) = split::choose(&node, self.config.prioritize_conflicts) else {
                self.record(&node);
                continue;
            };

            if self.is_exhausted() {
                return Err(Interrupted);
            }

            self.expand(&node, conflict)?;
        }

        Ok(())
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

    /// Splits `conflict` of `node` and opens the children: one node per combination of
    /// the replanned agents' Pareto paths, or, with bypass, `node` with better paths.
    fn expand(&mut self, node: &Node<N>, conflict: Conflict) -> Result<(), Interrupted> {
        self.stats.expanded += 1;

        let mut children = Vec::new();
        for constraint in split::constraints(node, conflict, self.config.splitting) {
            if let Some(child) = self.child(node, constraint)? {
                children.push(child);
            }
        }

        if self.config.bypass {
            if let Some(adopted) = children.iter().find_map(|child| split::bypass(node, child)) {
                self.push(adopted);
                return Ok(());
            }
        }

        for child in children {
            for grandchild in Combinations::children(node, child) {
                self.push(grandchild);
            }
        }

        Ok(())
    }

    /// The child of `node` with `constraint` added; `None` if an agent it replans has no
    /// path.
    ///
    /// A negative constraint replans its agent. A positive one keeps its agent's path,
    /// which takes part in the conflict and so meets it, and replans every other agent
    /// whose path breaks what it implies for them.
    fn child(
        &self,
        node: &Node<N>,
        constraint: Constraint,
    ) -> Result<Option<Child<N>>, Interrupted> {
        let mut constraints = node.constraints.clone();
        constraints.push(constraint);

        let mut replanned = Vec::new();
        for agent in split::affected(constraint, &node.paths) {
            let set = self.replan(agent, &constraints)?;
            if set.is_empty() {
                return Ok(None);
            }

            replanned.push((agent, set));
        }

        Ok(Some(Child {
            constraints,
            replanned,
        }))
    }

    /// The Pareto-optimal paths of `agent` under `constraints`.
    fn replan(
        &self,
        agent: AgentId,
        constraints: &[Constraint],
    ) -> Result<ParetoSet<N>, Interrupted> {
        let costs = |from, to| (self.costs)(agent, from, to);
        let problem = Problem {
            graph: self.instance.graph(),
            costs: &costs,
            agent: self.instance.agents()[agent],
            heuristic: &self.heuristics[agent],
            low_level: self.config.low_level,
        };
        let set = ConstraintSet::for_agent(agent, constraints);
        let paths = pareto_paths(&problem, &set, &mut || self.is_exhausted()).ok_or(Interrupted)?;

        Ok(paths
            .into_iter()
            .map(|(path, cost)| (Rc::from(path), cost))
            .collect())
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

#[cfg(test)]
mod tests {
    //! Unit tests for the front's Filter and Update.

    use super::*;
    use crate::graph::AdjacencyList;
    use crate::instance::Agent;

    /// Update drops a recorded solution that a later one dominates, and Filter then drops
    /// nodes the new solution weakly dominates. The random properties rarely reach this:
    /// in lexicographic pop order a dominating solution is usually found first.
    #[test]
    fn record_drops_dominated_solutions() {
        let graph = AdjacencyList::undirected(2, &[(0, 1)]);
        let instance =
            Instance::new(graph, alloc::vec![Agent { start: 0, goal: 1 }]).expect("valid");
        let costs = |_: AgentId, _: Vertex, _: Vertex| [1, 1];
        let mut budget = |_: &Stats| false;
        let mut search = Search::new(&instance, &costs, MoCbsConfig::default(), &mut budget);
        let node = |cost: [u32; 2]| {
            let path: Rc<[Vertex]> = Rc::from(alloc::vec![0, 1]);
            Node::new(Vec::new(), alloc::vec![path], alloc::vec![cost], Vec::new())
        };

        search.record(&node([3, 3]));
        search.record(&node([1, 5]));
        search.record(&node([2, 2]));

        let front: Vec<_> = search.front.iter().map(|plan| plan.cost).collect();
        assert_eq!(front, [[1, 5], [2, 2]]);
        assert!(search.is_filtered(&[2, 2]));
        assert!(search.is_filtered(&[3, 4]));
        assert!(!search.is_filtered(&[1, 4]));
    }
}
