//! How MO-CBS splits a node: which conflict ([`choose`]), into which constraints
//! ([`constraints`]), and bypass ([`bypass`]). Which agents a constraint replans is
//! [`Constraint::affected`], shared with CBS.
//!
//! None of these changes the front. Any conflict may be split (the paper's Lemma 2 holds
//! for every detected conflict). Disjoint splitting partitions the solutions a node permits
//! (Li et al. 2019), and its positive child keeps, for every replanned agent, a path that
//! weakly dominates that agent's path in each permitted solution, so the child is a P-node
//! in the paper's sense. Bypass keeps the node's constraints and every agent's cost
//! vector, the two things P-nodes are defined by.

use alloc::rc::Rc;

use super::Child;
use super::node::{Node, ParetoSet};
use crate::cbs::Splitting;
use crate::conflict::{Conflict, conflicts, first_conflict};
use crate::constraint::Constraint;
use crate::graph::Vertex;

/// The conflict of `node` to split, or `None` if its paths are conflict-free: with
/// `prioritize`, the earliest of the most cardinal class ([`class`]), else the earliest.
pub(super) fn choose<const N: usize>(node: &Node<N>, prioritize: bool) -> Option<Conflict> {
    if !prioritize {
        return first_conflict(&node.paths);
    }

    conflicts(&node.paths)
        .into_iter()
        .enumerate()
        .min_by_key(|&(index, conflict)| (class(node, conflict), index))
        .map(|(_, conflict)| conflict)
}

/// `0` if `conflict` is cardinal for both agents, `1` for one, `2` for neither.
fn class<const N: usize>(node: &Node<N>, conflict: Conflict) -> usize {
    let [a, b] = cardinal(node, conflict);

    2 - usize::from(a) - usize::from(b)
}

/// For each of the conflict's two agents, whether the conflict is cardinal for it: every
/// path in its Pareto set takes part, so whichever Pareto path it keeps, a child that
/// forbids the conflict to it must replace it. `false` without recorded sets.
fn cardinal<const N: usize>(node: &Node<N>, conflict: Conflict) -> [bool; 2] {
    let (a, b) = conflict.agents();
    let negatives = Constraint::resolving(conflict);

    [a, b].map(|agent| {
        let constraint = if agent == a {
            negatives[0]
        } else {
            negatives[1]
        };

        node.pareto.get(agent).is_some_and(|set: &ParetoSet<N>| {
            set.iter().all(|(path, _)| constraint.is_violated_by(path))
        })
    })
}

/// The two constraints that split `conflict` of `node`. Disjoint splitting constrains the
/// second agent if the conflict is cardinal only for it, else the first.
pub(super) fn constraints<const N: usize>(
    node: &Node<N>,
    conflict: Conflict,
    splitting: Splitting,
) -> [Constraint; 2] {
    match splitting {
        Splitting::Standard => Constraint::resolving(conflict),
        Splitting::Disjoint => {
            let side = usize::from(cardinal(node, conflict) == [false, true]);

            Constraint::disjoint(conflict, side)
        }
    }
}

/// `parent` with the paths `child` offers, if every replanned agent has a path of exactly
/// its cost in `parent` and the joint paths then have fewer conflicts; else `None`.
///
/// Those paths meet the parent's constraints (the child's are a superset), so the node
/// keeps its constraints and cost vectors. Each bypass removes a conflict, so bypasses
/// cannot repeat forever.
pub(super) fn bypass<const N: usize>(parent: &Node<N>, child: &Child<N>) -> Option<Node<N>> {
    let mut paths = parent.paths.clone();
    let mut pareto = parent.pareto.clone();
    for (agent, set) in &child.replanned {
        let cost = parent.costs[*agent];
        let (path, _) = set.iter().find(|(_, c)| *c == cost)?;
        paths[*agent] = Rc::clone(path);
        if let Some(slot) = pareto.get_mut(*agent) {
            *slot = with_representative(slot, path, cost);
        }
    }

    if conflicts(&paths).len() >= conflicts(&parent.paths).len() {
        return None;
    }

    Some(Node::new(
        parent.constraints.clone(),
        paths,
        parent.costs.clone(),
        pareto,
    ))
}

/// `set` with `path` as the representative of `cost`.
fn with_representative<const N: usize>(
    set: &ParetoSet<N>,
    path: &Rc<[Vertex]>,
    cost: [u32; N],
) -> ParetoSet<N> {
    set.iter()
        .map(|(p, c)| {
            if *c == cost {
                (Rc::clone(path), *c)
            } else {
                (Rc::clone(p), *c)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests for conflict choice and bypass.

    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A shared path.
    fn path(vertices: &[Vertex]) -> Rc<[Vertex]> {
        Rc::from(vertices)
    }

    /// The conflict both agents' only paths take part in is chosen over an earlier one
    /// that the first agent could avoid.
    #[test]
    fn prioritization_prefers_cardinal_conflicts() {
        // Agent 0 meets agent 1 at vertex 4 at time 1 on one of its two paths only; both
        // agents' every path is at vertex 2 at time 2.
        let a = [path(&[0, 4, 2]), path(&[0, 5, 2])];
        let b = path(&[4, 4, 2, 6]);
        let pareto: Vec<ParetoSet<1>> = vec![
            Rc::from([(Rc::clone(&a[0]), [2]), (Rc::clone(&a[1]), [3])]),
            Rc::from([(Rc::clone(&b), [3])]),
        ];
        let node = Node::new(
            Vec::new(),
            vec![Rc::clone(&a[0]), b],
            vec![[2], [3]],
            pareto,
        );

        let earliest = first_conflict(&node.paths).expect("they meet at 4");
        let chosen = choose(&node, true).expect("they conflict");

        assert_eq!(earliest.time(), 1);
        assert_eq!(
            chosen.time(),
            2,
            "vertex 2 at time 2 is on every path of both"
        );
    }

    /// Bypass takes over an equally cheap path with fewer conflicts and keeps the costs.
    #[test]
    fn bypass_adopts_an_equally_cheap_path() {
        let parent = Node::new(
            Vec::new(),
            vec![path(&[0, 1, 2]), path(&[5, 1])],
            vec![[2], [1]],
            Vec::new(),
        );
        let child = Child {
            constraints: Vec::new(),
            replanned: vec![(0, Rc::from([(path(&[0, 4, 2]), [2])]))],
        };

        let adopted = bypass(&parent, &child).expect("fewer conflicts at the same cost");

        assert_eq!(&*adopted.paths[0], [0, 4, 2]);
        assert_eq!(adopted.cost, parent.cost);
        assert_eq!(first_conflict(&adopted.paths), None);
    }
}
