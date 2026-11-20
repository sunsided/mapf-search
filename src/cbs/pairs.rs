//! Pairwise dependency for the DG and WDG heuristics (Li et al. 2019).
//!
//! Two agents are *dependent* at a node if no pair of their cheapest paths under their
//! constraints is conflict-free: the product of their MDDs has no joint path
//! ([`is_dependent`]). Then their costs must rise by at least 1 together. WDG weighs the
//! pair by how much: the optimal sum of costs of the two agents alone under their
//! constraints, found by a small CBS ([`pair_cost`]), minus their current costs.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::conflict::first_conflict;
use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::Graph;
use crate::instance::Agent;
use crate::low_level::plan;
use crate::mdd::{Mdd, joint_paths};
use crate::path::{Path, sum_of_costs};

/// Expansions after which [`pair_cost`] stops and returns its lower bound so far.
///
/// The pair search can be infinite: two agents swapping the ends of a corridor have no
/// joint plan, yet each one alone does. Li et al. cap it too.
const PAIR_EXPANSION_LIMIT: usize = 64;

/// Returns `true` if no combination of one path from each MDD is conflict-free.
pub(super) fn is_dependent(first: &Mdd, second: &Mdd) -> bool {
    joint_paths(&[first, second]).is_none()
}

/// The two agents of a pair subproblem, with their constraints.
pub(super) struct Pair<'a, G> {
    /// The graph.
    pub(super) graph: &'a G,
    /// The two agents.
    pub(super) agents: [Agent; 2],
    /// Their distance tables to their goals.
    pub(super) distances: [&'a [Option<u32>]; 2],
    /// Their constraints at the node, implied ones included.
    pub(super) constraints: [ConstraintSet; 2],
}

/// One node of the pair search.
struct Node {
    /// The constraints of both agents.
    constraints: [ConstraintSet; 2],
    /// One path per agent, cheapest under its constraints.
    paths: [Path; 2],
}

/// A lower bound on the optimal sum of costs of `pair` alone, exact if the search ends
/// within [`PAIR_EXPANSION_LIMIT`] expansions; `None` if the pair provably has no joint
/// plan, so neither has the node it came from.
///
/// A best-first CBS over the two agents with standard splitting. On the limit, the cost of
/// the cheapest open node bounds every joint plan from below.
pub(super) fn pair_cost<G: Graph>(pair: &Pair<'_, G>) -> Option<u32> {
    let paths = [0, 1].map(|k| replan(pair, k, &pair.constraints[k]));
    let [Some(first), Some(second)] = paths else {
        return None;
    };

    let root = Node {
        constraints: pair.constraints.clone(),
        paths: [first, second],
    };
    let mut open = BinaryHeap::from([Reverse((cost(&root), 0_usize))]);
    let mut nodes = Vec::from([Some(root)]);
    let mut expansions = 0;
    while let Some(Reverse((cost_so_far, id))) = open.pop() {
        let node = nodes[id].take().expect("each node is popped once");
        let Some(conflict) = first_conflict(&node.paths) else {
            return Some(cost_so_far);
        };

        if expansions == PAIR_EXPANSION_LIMIT {
            return Some(cost_so_far);
        }

        expansions += 1;
        for constraint in Constraint::resolving(conflict) {
            if let Some(child) = child(pair, &node, constraint) {
                open.push(Reverse((cost(&child), nodes.len())));
                nodes.push(Some(child));
            }
        }
    }

    None
}

/// `node` with `constraint` (on agent `0` or `1` of the pair) added and that agent
/// replanned; `None` if it then has no path.
fn child<G: Graph>(pair: &Pair<'_, G>, node: &Node, constraint: Constraint) -> Option<Node> {
    let agent = constraint.agent();
    let mut constraints = node.constraints.clone();
    constraints[agent].insert(&constraint);

    let mut paths = node.paths.clone();
    paths[agent] = replan(pair, agent, &constraints[agent])?;

    Some(Node { constraints, paths })
}

/// The cheapest path of the pair's agent `k` under `constraints`.
fn replan<G: Graph>(pair: &Pair<'_, G>, k: usize, constraints: &ConstraintSet) -> Option<Path> {
    plan(pair.graph, pair.agents[k], constraints, pair.distances[k])
}

/// The sum of costs of a pair node.
fn cost(node: &Node) -> u32 {
    sum_of_costs(&node.paths)
}

#[cfg(test)]
mod tests {
    //! Unit tests for pair dependency and the pair search.

    use super::*;
    use crate::graph::{Grid, distances_to};

    /// Two agents crossing a plus-shaped map must both pass the centre at time 1: they are
    /// dependent, and the pair optimum is one more than their sum alone.
    #[test]
    fn crossing_agents_are_dependent() {
        let grid = Grid::parse(&["@.@", "...", "@.@"]);
        let agents = [Agent { start: 3, goal: 5 }, Agent { start: 1, goal: 7 }];
        let distances = agents.map(|agent| distances_to(&grid, agent.goal));
        let mdd = |k: usize| {
            let agent = agents[k];
            Mdd::build(&grid, agent, &ConstraintSet::new(), &distances[k], 2).expect("reachable")
        };

        assert!(is_dependent(&mdd(0), &mdd(1)));

        let pair = Pair {
            graph: &grid,
            agents,
            distances: [&distances[0], &distances[1]],
            constraints: [ConstraintSet::new(), ConstraintSet::new()],
        };
        assert_eq!(pair_cost(&pair), Some(5));
    }

    /// Two agents in parallel lanes are independent.
    #[test]
    fn parallel_agents_are_independent() {
        let grid = Grid::new(3, 2);
        let agents = [Agent { start: 0, goal: 2 }, Agent { start: 3, goal: 5 }];
        let mdds = agents.map(|agent| {
            let distances = distances_to(&grid, agent.goal);
            Mdd::build(&grid, agent, &ConstraintSet::new(), &distances, 2).expect("reachable")
        });

        assert!(!is_dependent(&mdds[0], &mdds[1]));
    }

    /// The corridor swap never ends; the pair search stops at its limit with a bound.
    #[test]
    fn corridor_swap_stops_at_the_limit() {
        let grid = Grid::new(3, 1);
        let agents = [Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
        let distances = agents.map(|agent| distances_to(&grid, agent.goal));
        let pair = Pair {
            graph: &grid,
            agents,
            distances: [&distances[0], &distances[1]],
            constraints: [ConstraintSet::new(), ConstraintSet::new()],
        };

        assert!(pair_cost(&pair).is_some_and(|bound| bound > 4));
    }
}
