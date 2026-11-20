//! Conflict-based search ([`cbs`], Sharon et al. 2015): optimal for sum of costs or for
//! makespan ([`Objective`]), with the ICBS improvements (Boyarski et al. 2015), the CG
//! heuristic (Felner et al. 2018) and a conflict avoidance table (Sharon et al. 2015), each
//! switchable in [`CbsConfig`].
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

use self::cardinal::{Cardinality, cardinality, forced, vertex_cover_bound};
use crate::conflict::{Conflict, conflicts};
use crate::conflict_table::ConflictTable;
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::low_level::{distance_tables, plan, plan_avoiding};
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
    /// How a conflict is split into two children.
    pub splitting: Splitting,
    /// The admissible high-level heuristic added to a node's sum of costs. Admissible for
    /// [`Objective::SumOfCosts`] only; ignored for [`Objective::Makespan`].
    pub heuristic: CbsHeuristic,
    /// The conflict avoidance table (Sharon et al. 2015): among an agent's cheapest paths,
    /// the low level returns one with the fewest conflicts with the other agents' current
    /// paths. The root plans the agents one after another, each avoiding the earlier ones.
    pub conflict_avoidance: bool,
}

impl CbsConfig {
    /// Plain CBS: earliest conflict first, no bypass, no heuristic.
    #[must_use]
    pub const fn plain(objective: Objective) -> Self {
        Self {
            objective,
            prioritize_conflicts: false,
            bypass: false,
            splitting: Splitting::Standard,
            heuristic: CbsHeuristic::None,
            conflict_avoidance: false,
        }
    }

    /// Every improvement switched on; what [`cbs`] uses.
    #[must_use]
    pub const fn improved(objective: Objective) -> Self {
        Self {
            objective,
            prioritize_conflicts: true,
            bypass: true,
            splitting: Splitting::Disjoint,
            heuristic: CbsHeuristic::Cg,
            conflict_avoidance: true,
        }
    }

    /// Returns `true` if nodes need the MDDs of their conflicting agents.
    fn needs_mdds(self) -> bool {
        self.prioritize_conflicts || self.uses_heuristic()
    }

    /// The heuristic in use: [`CbsConfig::heuristic`] for sum of costs, none for makespan.
    fn effective_heuristic(self) -> CbsHeuristic {
        match self.objective {
            Objective::SumOfCosts => self.heuristic,
            Objective::Makespan => CbsHeuristic::None,
        }
    }

    /// Returns `true` if a heuristic is on and admissible for the objective.
    fn uses_heuristic(self) -> bool {
        self.effective_heuristic() != CbsHeuristic::None
    }
}

/// How CBS splits a conflict between agents `a` and `b` ([`CbsConfig::splitting`]).
///
/// Both keep every solution in exactly one child or in both, so both are optimal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Splitting {
    /// One child forbids the conflict to `a`, the other to `b` (Sharon et al. 2015). A
    /// solution that avoids the conflict both ways lies below both children.
    #[default]
    Standard,
    /// Disjoint splitting (Li et al. 2019): one child forbids the conflict to one agent,
    /// the other requires it of that agent with a positive constraint, which forbids it to
    /// every other agent. The two subtrees share no solution.
    ///
    /// The split agent is the one for whom the conflict is forced (every cheapest path takes
    /// part in it) if only one is and its MDD is known, else the smaller id. Forcing it keeps
    /// its cost in the positive child and raises it in the negative one.
    Disjoint,
}

/// An admissible estimate of how much the sum of costs of a constraint-tree node must still
/// rise ([`CbsConfig::heuristic`]).
///
/// Each one needs the MDDs of the node's conflicting agents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CbsHeuristic {
    /// No heuristic: nodes are ordered by their sum of costs.
    #[default]
    None,
    /// CG (Felner et al. 2018): a lower bound on the minimum vertex cover of the graph of
    /// agents with cardinal conflicts.
    Cg,
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
    budget: impl Budget<Stats>,
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
    mut budget: impl Budget<Stats>,
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
    fn run(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
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

    /// The root node: every agent planned without constraints, avoiding the earlier
    /// agents' paths if [`CbsConfig::conflict_avoidance`] is on. `None` if one of them has
    /// no path.
    fn root(&self) -> Option<Node> {
        let agents = self.instance.agents().len();
        let mut paths = Vec::with_capacity(agents);
        for agent in 0..agents {
            let path = self.replan(agent, &[], &paths)?;
            paths.push(path);
        }

        Some(self.node(Vec::new(), paths, vec![None; agents]))
    }

    /// Splits `conflict` of `parent`: bypasses it if a child allows, else opens the
    /// children that have a path.
    fn expand(&mut self, parent: Node, conflict: Conflict) {
        self.stats.expanded += 1;

        let children: Vec<Child> = self
            .split(&parent, conflict)
            .into_iter()
            .filter_map(|constraint| self.child(&parent, constraint))
            .collect();

        if self.config.bypass {
            if let Some(child) = children.iter().find(|child| is_bypass(&parent, child)) {
                let adopted = self.adopt(parent, child);
                self.push(adopted);
                return;
            }
        }

        for child in children {
            self.push(child.node);
        }
    }

    /// The constraints of the two children that split `conflict` of `parent`.
    fn split(&self, parent: &Node, conflict: Conflict) -> [Constraint; 2] {
        match self.config.splitting {
            Splitting::Standard => Constraint::resolving(conflict),
            Splitting::Disjoint => Constraint::disjoint(conflict, split_side(parent, &conflict)),
        }
    }

    /// `parent` with the paths of the agents `child` replanned, which satisfy the parent's
    /// constraints at the same costs, so their MDDs stay valid.
    fn adopt(&self, parent: Node, child: &Child) -> Node {
        let mut paths = parent.paths;
        for &agent in &child.replanned {
            paths[agent].clone_from(&child.node.paths[agent]);
        }

        self.node(parent.constraints, paths, parent.mdds)
    }

    /// The child of `parent` with `constraint` added, or `None` if an agent it replans has
    /// no path.
    ///
    /// A negative constraint replans its agent. A positive one keeps its agent's path,
    /// which takes part in the conflict and so meets it, and replans every other agent whose
    /// path breaks what it implies for them.
    fn child(&self, parent: &Node, constraint: Constraint) -> Option<Child> {
        let mut constraints = parent.constraints.clone();
        constraints.push(constraint);

        let replanned = affected(constraint, &parent.paths);
        let mut paths = parent.paths.clone();
        let mut mdds = parent.mdds.clone();
        for &agent in &replanned {
            paths[agent] = self.replan(agent, &constraints, &paths)?;
            mdds[agent] = None;
        }

        mdds[constraint.agent()] = None;
        let node = self.node(constraints, paths, mdds);

        Some(Child { node, replanned })
    }

    /// The cheapest path of `agent` under those of `constraints` that name it; with
    /// [`CbsConfig::conflict_avoidance`], one with the fewest conflicts with `paths`
    /// (indexed by agent; may be a prefix, and may include `agent`) among the cheapest.
    fn replan(&self, agent: AgentId, constraints: &[Constraint], paths: &[Path]) -> Option<Path> {
        let set = ConstraintSet::for_agent(agent, constraints);
        let (graph, distances) = (self.instance.graph(), &self.distances[agent]);
        let start_and_goal = self.instance.agents()[agent];

        if self.config.conflict_avoidance {
            let table = ConflictTable::without(paths, agent);
            plan_avoiding(graph, start_and_goal, &set, distances, &table)
        } else {
            plan(graph, start_and_goal, &set, distances)
        }
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

/// A child node and the agents it replanned.
struct Child {
    /// The node.
    node: Node,
    /// The agents whose paths were replanned for the child's new constraint.
    replanned: Vec<AgentId>,
}

/// Returns `true` if `child` may lend its replanned paths to `parent`: each costs the same
/// as the parent's, so it is still optimal under the parent's constraints, and there are
/// strictly fewer conflicts.
fn is_bypass(parent: &Node, child: &Child) -> bool {
    let same_cost = child
        .replanned
        .iter()
        .all(|&agent| path_cost(&child.node.paths[agent]) == path_cost(&parent.paths[agent]));

    same_cost && child.node.conflict_count < parent.conflict_count
}

/// The agents whose paths in `paths` must be replanned once `constraint` is added: its own
/// agent for a negative constraint, every other agent whose path breaks what a positive
/// one implies for it.
fn affected(constraint: Constraint, paths: &[Path]) -> Vec<AgentId> {
    if !constraint.is_positive() {
        return Vec::from([constraint.agent()]);
    }

    (0..paths.len())
        .filter(|&agent| agent != constraint.agent())
        .filter(|&agent| {
            constraint
                .implied_for(agent)
                .iter()
                .any(|implied| implied.is_violated_by(&paths[agent]))
        })
        .collect()
}

/// Which agent of `conflict` disjoint splitting constrains: `1` (the second) if only that
/// one is forced through the conflict by its MDD, else `0`. Without both MDDs, `0`.
fn split_side(parent: &Node, conflict: &Conflict) -> usize {
    let (a, b) = conflict.agents();
    if parent.mdds[a].is_none() || parent.mdds[b].is_none() {
        return 0;
    }

    match forced(conflict, &parent.mdds) {
        [false, true] => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the CBS improvements on single nodes.

    use super::*;
    use crate::graph::Grid;
    use crate::instance::Agent;

    /// An agent's start and goal cell, as `(x, y)`.
    type Route = ((usize, usize), (usize, usize));

    /// An instance on `rows` with one agent per route.
    fn instance(rows: &[&str], agents: &[Route]) -> Instance<Grid> {
        let grid = Grid::parse(rows);
        let agents = agents
            .iter()
            .map(|&((sx, sy), (gx, gy))| Agent {
                start: grid.vertex(sx, sy),
                goal: grid.vertex(gx, gy),
            })
            .collect();

        Instance::new(grid, agents).expect("valid")
    }

    /// On a plus-shaped map both agents must cross the centre at time 1: one cardinal
    /// conflict, so the CG heuristic is 1, and the node ranks one above its cost.
    #[test]
    fn crossing_on_a_plus_is_cardinal() {
        let plus = instance(
            &["@.@", "...", "@.@"],
            &[((0, 1), (2, 1)), ((1, 0), (1, 2))],
        );
        let search = Search::new(&plus, CbsConfig::improved(Objective::SumOfCosts));
        let root = search.root().expect("both reach their goals");

        assert_eq!(root.heuristic, 1);
        assert_eq!(root.sum_of_costs, 4);

        let plain = Search::new(&plus, CbsConfig::plain(Objective::SumOfCosts));
        assert_eq!(plain.root().expect("reachable").heuristic, 0);
    }

    /// Prioritizing picks a later cardinal conflict over an earlier non-cardinal one;
    /// without it, the earliest conflict wins.
    #[test]
    fn prioritizing_prefers_cardinal_conflicts() {
        // On a free 3x2 grid (`0 1 2` over `3 4 5`): agents 0 and 1 have two ways through
        // time 1, agents 2 and 3 must be at vertex 1 at time 1.
        let grid = Grid::new(3, 2);
        let mdd = |start, goal, depth| {
            let distances = crate::graph::distances_to(&grid, goal);
            let agent = Agent { start, goal };
            Mdd::build(&grid, agent, &ConstraintSet::new(), &distances, depth).map(Rc::new)
        };
        let mdds = [mdd(0, 5, 3), mdd(2, 3, 3), mdd(4, 1, 1), mdd(0, 2, 2)];
        let at_one = |agents| Conflict::Vertex {
            agents,
            vertex: 1,
            time: 1,
        };
        let (wide, forced) = (at_one((0, 1)), at_one((2, 3)));

        let any = instance(&["..."], &[((0, 0), (2, 0))]);
        let improved = Search::new(&any, CbsConfig::improved(Objective::SumOfCosts));
        let plain = Search::new(&any, CbsConfig::plain(Objective::SumOfCosts));

        assert_eq!(improved.classify(&[wide, forced], &mdds), (Some(forced), 1));
        assert_eq!(plain.classify(&[wide, forced], &mdds), (Some(wide), 0));
    }

    /// Bypass: on a free 3x3 grid, two agents crossing diagonally meet at a vertex both
    /// can avoid at no cost; the expansion takes over a child's paths instead of splitting.
    #[test]
    fn bypass_adopts_an_equally_cheap_child() {
        let open = instance(
            &["...", "...", "..."],
            &[((0, 0), (2, 2)), ((2, 0), (0, 2))],
        );
        let mut config = CbsConfig::plain(Objective::SumOfCosts);
        config.bypass = true;
        let mut search = Search::new(&open, config);
        let root = search.root().expect("reachable");
        let conflict = root.conflict.expect("the diagonal paths meet");

        search.expand(root, conflict);
        let Entry { node, .. } = search.open.pop().expect("the adopted node");

        assert!(search.open.is_empty(), "no children were opened");
        assert!(
            node.constraints.is_empty(),
            "the parent keeps its constraints"
        );
        assert_eq!((node.sum_of_costs, node.conflict_count), (8, 0));
    }
}
