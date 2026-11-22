//! The admissible high-level heuristics of CBS ([`CbsHeuristic`]): CG (Felner et al.
//! 2018), DG and WDG (Li et al. 2019).
//!
//! Each builds a graph over the agents of a node's conflicting pairs and bounds a vertex
//! cover of it from below:
//!
//! - **CG**: an edge per pair with a cardinal conflict.
//! - **DG**: an edge per *dependent* pair, whose MDDs have no conflict-free combination.
//!   A cardinal conflict makes a pair dependent, so the check is skipped then.
//! - **WDG**: the DG edges, weighted by how much the pair's sum of costs must rise: the
//!   optimum of the two agents alone under their constraints minus their current costs, at
//!   least 1. The bound is an edge-weighted vertex cover.
//!
//! Pairs without a conflict at the node are independent there: their current paths are a
//! conflict-free combination. Pair weights depend only on the two agents' constraints,
//! so they are cached by them for the whole run.

use alloc::rc::Rc;
use alloc::vec::Vec;

use super::cardinal::Cardinality;
use super::cover::{vertex_cover_bound, weighted_cover_bound};
use super::pairs::{Pair, is_dependent, pair_cost};
use super::{CbsHeuristic, Search};
use crate::conflict::Conflict;
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::Graph;
use crate::instance::AgentId;
use crate::mdd::Mdd;
use crate::path::{Path, path_cost};

/// The key of a cached pair weight: the two agents (smaller first), their assigned goals
/// and their constraint sets, implied constraints included.
pub(super) type PairKey = (AgentId, AgentId, [usize; 2], ConstraintSet, ConstraintSet);

impl<G: Graph> Search<'_, G> {
    /// The heuristic value of a node of `tree` with `constraints`, `paths` and the MDDs of its
    /// conflicting agents, whose conflicts `found` have the cardinalities `classes`; `None`
    /// if WDG proved a pair to have no joint plan, so the node has no solution.
    pub(super) fn heuristic(
        &self,
        tree: usize,
        constraints: &[Constraint],
        paths: &[Path],
        found: &[Conflict],
        classes: &[Cardinality],
        mdds: &[Option<Rc<Mdd>>],
    ) -> Option<u32> {
        let pairs = conflicting_pairs(found, classes);

        match self.config.effective_heuristic() {
            CbsHeuristic::None => Some(0),
            CbsHeuristic::Cg => {
                let cardinal: Vec<_> = pairs
                    .iter()
                    .filter(|&&(_, cardinal)| cardinal)
                    .map(|&(pair, _)| pair)
                    .collect();
                Some(vertex_cover_bound(&cardinal))
            }
            CbsHeuristic::Dg => {
                let dependent: Vec<_> = pairs
                    .iter()
                    .filter(|&&(pair, cardinal)| cardinal || dependent(pair, mdds))
                    .map(|&(pair, _)| pair)
                    .collect();
                Some(vertex_cover_bound(&dependent))
            }
            CbsHeuristic::Wdg => {
                let weighted = pairs
                    .iter()
                    .map(|&(pair, cardinal)| {
                        let weight =
                            self.pair_weight(tree, pair, cardinal, constraints, paths, mdds)?;
                        Some((pair.0, pair.1, weight))
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(weighted_cover_bound(&weighted))
            }
        }
    }

    /// The WDG weight of `pair` in `tree`: `0` if independent, else the rise of the pair's optimal
    /// sum of costs over the agents' current costs (at least 1), cached by constraints.
    /// `None` if the pair has no joint plan.
    fn pair_weight(
        &self,
        tree: usize,
        pair: (AgentId, AgentId),
        cardinal: bool,
        constraints: &[Constraint],
        paths: &[Path],
        mdds: &[Option<Rc<Mdd>>],
    ) -> Option<u32> {
        let (a, b) = pair;
        let sets = [a, b].map(|agent| ConstraintSet::for_agent(agent, constraints));
        let goals = [a, b].map(|agent| self.trees[tree][agent]);
        let key: PairKey = (a, b, goals, sets[0].clone(), sets[1].clone());
        if let Some(&weight) = self.pair_weights.borrow().get(&key) {
            return weight;
        }

        let weight = if cardinal || dependent(pair, mdds) {
            let current = path_cost(&paths[a]) + path_cost(&paths[b]);
            let problem = Pair {
                graph: self.graph,
                agents: [a, b].map(|agent| self.agent(tree, agent)),
                distances: [a, b].map(|agent| self.distances(tree, agent)),
                constraints: sets,
            };
            pair_cost(&problem).map(|optimum| optimum.saturating_sub(current).max(1))
        } else {
            Some(0)
        };

        self.pair_weights.borrow_mut().insert(key, weight);

        weight
    }
}

/// The distinct agent pairs of `found` (smaller id first), each with whether one of its
/// conflicts is cardinal.
fn conflicting_pairs(
    found: &[Conflict],
    classes: &[Cardinality],
) -> Vec<((AgentId, AgentId), bool)> {
    let mut pairs: Vec<((AgentId, AgentId), bool)> = found
        .iter()
        .zip(classes)
        .map(|(conflict, &class)| (conflict.agents(), class == Cardinality::Cardinal))
        .collect();
    // Cardinal first within a pair, so dedup keeps the cardinal flag.
    pairs.sort_unstable_by_key(|&(pair, cardinal)| (pair, !cardinal));
    pairs.dedup_by_key(|&mut (pair, _)| pair);

    pairs
}

/// Returns `true` if the agents of `pair` are dependent by their MDDs.
fn dependent((a, b): (AgentId, AgentId), mdds: &[Option<Rc<Mdd>>]) -> bool {
    let mdd = |agent: AgentId| mdds[agent].as_deref().expect("MDDs of conflicting agents");

    is_dependent(mdd(a), mdd(b))
}

#[cfg(test)]
mod tests {
    //! Admissibility of CG, DG and WDG at the nodes plain CBS generates, against plain
    //! CBS's optimum below each node.

    use super::*;
    use alloc::vec;

    use heuristic_search::{Limits, Stats};
    use proptest::prelude::*;

    use crate::cbs::cardinal::cardinality;
    use crate::cbs::{CbsConfig, Entry, Node, Splitting};
    use crate::conflict::conflicts;
    use crate::graph::{Grid, Vertex};

    use crate::instance::{Agent, Instance};
    use crate::path::Objective;
    use crate::solution::Outcome;

    /// Constraint-tree nodes examined per instance.
    const NODES: usize = 12;

    /// A random 3x3 grid with about one cell in eight blocked and 2-3 agents.
    fn instance() -> impl Strategy<Value = Instance<Grid>> {
        (
            prop::collection::vec(prop::bool::weighted(0.12), 9),
            2..=3_usize,
        )
            .prop_filter("enough free cells", |(blocked, agents)| {
                blocked.iter().filter(|&&b| !b).count() >= *agents
            })
            .prop_flat_map(|(blocked, agents)| {
                let free: Vec<Vertex> = (0..9).filter(|&v| !blocked[v]).collect();
                let starts = Just(free.clone()).prop_shuffle();
                let goals = Just(free).prop_shuffle();
                (Just(blocked), Just(agents), starts, goals)
            })
            .prop_map(|(blocked, count, starts, goals)| {
                let mut grid = Grid::new(3, 3);
                for vertex in (0..9).filter(|&v| blocked[v]) {
                    let (x, y) = grid.cell(vertex);
                    grid.block(x, y);
                }

                let agents = starts
                    .iter()
                    .zip(&goals)
                    .take(count)
                    .map(|(&start, &goal)| Agent { start, goal })
                    .collect();

                Instance::new(grid, agents).expect("distinct starts and goals")
            })
    }

    /// Up to [`NODES`] nodes that plain CBS with `splitting` generates, in expansion order.
    fn tree(instance: &Instance<Grid>, splitting: Splitting) -> Vec<Node> {
        let config = CbsConfig {
            splitting,
            ..CbsConfig::plain(Objective::SumOfCosts)
        };
        let mut search = Search::new(instance, config);
        let Some(root) = search.root(0) else {
            return Vec::new();
        };

        search.push(root);
        let mut nodes = Vec::new();
        while let Some(Entry { node, .. }) = search.open.pop() {
            if nodes.len() == NODES {
                break;
            }

            nodes.push(node.clone());
            if let Some(conflict) = node.conflict {
                search.expand(node, conflict);
            }
        }

        nodes
    }

    /// How plain CBS from `node` ends: solved with the optimum below it, proven
    /// unsolvable, or at its budget.
    fn below(instance: &Instance<Grid>, node: &Node) -> Outcome {
        let mut search = Search::new(instance, CbsConfig::plain(Objective::SumOfCosts));
        search.push(node.clone());
        search.stats = Stats::default();

        search.run_open(&mut Limits::expansions(1_000))
    }

    /// The value of `heuristic` at `node`.
    fn value(instance: &Instance<Grid>, heuristic: CbsHeuristic, node: &Node) -> Option<u32> {
        let config = CbsConfig {
            heuristic,
            ..CbsConfig::plain(Objective::SumOfCosts)
        };
        let search = Search::new(instance, config);
        let found = conflicts(&node.paths);
        let mut mdds = vec![None; node.paths.len()];
        search.fill_mdds(0, &node.constraints, &node.paths, &mut mdds, &found);
        let classes: Vec<Cardinality> = found.iter().map(|c| cardinality(c, &mdds)).collect();

        search.heuristic(0, &node.constraints, &node.paths, &found, &classes, &mdds)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

        /// At every examined node, `cost + h` is at most the optimum below it, CG <= DG <=
        /// WDG, and WDG proves a node dead only if CBS finds nothing below it. With
        /// standard and with disjoint splitting (positive constraints). Each value is
        /// computed from MDDs built fresh for the node and an empty pair-weight cache;
        /// inherited MDDs and cached weights are covered by the oracle properties of the
        /// integration tests.
        #[test]
        fn heuristics_are_admissible(instance in instance(), disjoint in any::<bool>()) {
            let splitting = if disjoint { Splitting::Disjoint } else { Splitting::Standard };
            for node in tree(&instance, splitting) {
                let [cg, dg, wdg] = [CbsHeuristic::Cg, CbsHeuristic::Dg, CbsHeuristic::Wdg]
                    .map(|heuristic| value(&instance, heuristic, &node));
                let below = below(&instance, &node);

                let Some(wdg) = wdg else {
                    prop_assert!(!matches!(below, Outcome::Solved(_)), "WDG killed a solvable node");
                    continue;
                };

                let (cg, dg) = (cg.expect("CG never kills"), dg.expect("DG never kills"));
                prop_assert!(cg <= dg && dg <= wdg, "CG {cg}, DG {dg}, WDG {wdg}");
                if let Outcome::Solved(solution) = below {
                    let optimum = solution.sum_of_costs;
                    prop_assert!(
                        node.sum_of_costs + wdg <= optimum,
                        "cost {} + WDG {wdg} over the optimum {optimum}",
                        node.sum_of_costs
                    );
                }
            }
        }
    }
}
