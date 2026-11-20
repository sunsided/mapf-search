//! High-level nodes of MO-CBS ([`Node`]) and their generation as combinations of the
//! agents' Pareto-optimal paths ([`Combinations`]): the roots, and the children of a split.

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

use multiobjective_search::add;

use super::Child;
use crate::constraint::Constraint;
use crate::graph::Vertex;

/// A shared path with its cost vector.
pub(super) type PricedPath<const N: usize> = (Rc<[Vertex]>, [u32; N]);

/// The cost-unique Pareto-optimal paths of one agent under its constraints, sorted
/// lexicographically by cost.
pub(super) type ParetoSet<const N: usize> = Rc<[PricedPath<N>]>;

/// One high-level node.
#[derive(Clone)]
pub(super) struct Node<const N: usize> {
    /// All constraints on the way from the root, on any agent.
    pub(super) constraints: Vec<Constraint>,
    /// One path per agent, Pareto-optimal under its constraints. Shared with the parent
    /// where the agent was not replanned.
    pub(super) paths: Vec<Rc<[Vertex]>>,
    /// The cost vector of each path.
    pub(super) costs: Vec<[u32; N]>,
    /// The sum of [`Node::costs`].
    pub(super) cost: [u32; N],
    /// The Pareto set each agent's path was taken from, for conflict prioritization. An
    /// agent's set predates the negative constraints that other agents' later positive
    /// constraints imply for it, so it may hold paths those forbid; prioritization is only
    /// an ordering, so that does not matter. Empty in unit tests that do not need it.
    pub(super) pareto: Vec<ParetoSet<N>>,
}

impl<const N: usize> Node<N> {
    /// A node of `paths` with their `costs` under `constraints`, taken from `pareto`.
    pub(super) fn new(
        constraints: Vec<Constraint>,
        paths: Vec<Rc<[Vertex]>>,
        costs: Vec<[u32; N]>,
        pareto: Vec<ParetoSet<N>>,
    ) -> Self {
        let cost = costs.iter().fold([0; N], |sum, &c| add(sum, c));

        Self {
            constraints,
            paths,
            costs,
            cost,
            pareto,
        }
    }
}

/// Nodes for every combination of one path per agent from `choices`, generated on demand
/// in mixed-radix order (the last agent's choice varies fastest).
pub(super) struct Combinations<const N: usize> {
    /// The constraints of every node.
    constraints: Vec<Constraint>,
    /// The Pareto set of each agent, recorded in every node.
    pareto: Vec<ParetoSet<N>>,
    /// The paths to choose from, per agent. None is empty.
    choices: Vec<ParetoSet<N>>,
    /// The choice index of each agent for the next node; `None` once all are generated.
    next: Option<Vec<usize>>,
}

impl<const N: usize> Combinations<N> {
    /// The roots: every combination of the agents' unconstrained Pareto sets `sets`.
    pub(super) fn roots(sets: Vec<ParetoSet<N>>) -> Self {
        Self::new(Vec::new(), sets.clone(), sets)
    }

    /// The nodes of `child` of `parent`: each replanned agent takes each path of its new
    /// Pareto set, the others keep their paths.
    pub(super) fn children(parent: &Node<N>, child: Child<N>) -> Self {
        let mut pareto = parent.pareto.clone();
        let mut choices: Vec<ParetoSet<N>> = parent
            .paths
            .iter()
            .zip(&parent.costs)
            .map(|(path, &cost)| Rc::from([(Rc::clone(path), cost)]))
            .collect();
        for (agent, set) in child.replanned {
            if let Some(slot) = pareto.get_mut(agent) {
                slot.clone_from(&set);
            }

            choices[agent] = set;
        }

        Self::new(child.constraints, pareto, choices)
    }

    /// The combinations of `choices`, each node under `constraints`.
    fn new(
        constraints: Vec<Constraint>,
        pareto: Vec<ParetoSet<N>>,
        choices: Vec<ParetoSet<N>>,
    ) -> Self {
        let next = Some(vec![0; choices.len()]);

        Self {
            constraints,
            pareto,
            choices,
            next,
        }
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

impl<const N: usize> Iterator for Combinations<N> {
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

        Some(Node::new(
            self.constraints.clone(),
            paths,
            costs,
            self.pareto.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the combinations.

    use super::*;

    /// A one-vertex path at `vertex` with `cost`.
    fn priced(vertex: Vertex, cost: u32) -> PricedPath<1> {
        (Rc::from(vec![vertex]), [cost])
    }

    /// Two agents with two and three paths give six roots, each combination once.
    #[test]
    fn roots_enumerate_the_product() {
        let roots = Combinations::roots(vec![
            Rc::from([priced(0, 1), priced(1, 2)]),
            Rc::from([priced(5, 10), priced(6, 20), priced(7, 30)]),
        ]);

        let costs: Vec<[u32; 1]> = roots.map(|node| node.cost).collect();

        assert_eq!(costs, [[11], [21], [31], [12], [22], [32]]);
    }

    /// A child replanning one agent keeps the other agent's path in every node.
    #[test]
    fn children_vary_only_the_replanned_agents() {
        let sets: Vec<ParetoSet<1>> = vec![
            Rc::from([priced(0, 1), priced(1, 2)]),
            Rc::from([priced(5, 10), priced(6, 20)]),
        ];
        let parent = Combinations::roots(sets).nth(1).expect("four roots");
        let child = Child {
            constraints: Vec::new(),
            replanned: vec![(1, Rc::from([priced(8, 40), priced(9, 50)]))],
        };

        let costs: Vec<[u32; 1]> = Combinations::children(&parent, child)
            .map(|node| node.cost)
            .collect();

        assert_eq!(costs, [[41], [51]]);
    }
}
