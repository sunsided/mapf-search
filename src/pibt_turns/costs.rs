//! The turn-aware costs of PIBT*_T ([`Costs`]): how many turns lie between two headings,
//! each agent's exact remaining time from every `(vertex, heading)` state, the paper's
//! `Compute_action` ([`Costs::action`]) and its node selection cost `TC` ([`Costs::rank`]).

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::{Heading, OrientedGraph, Vertex};
use crate::instance::Agent;

/// The cost tables of one run.
pub(super) struct Costs<'a, G> {
    /// The graph.
    graph: &'a G,
    /// The number of headings.
    headings: usize,
    /// `turns[a * headings + b]`: the fewest turns from heading `a` to heading `b`.
    turns: Vec<u32>,
    /// For each agent and state `vertex * headings + heading`, the fewest steps (moves and
    /// turns) to its goal, facing any way; `None` if unreachable.
    remaining: Vec<Vec<Option<u32>>>,
}

impl<'a, G: OrientedGraph> Costs<'a, G> {
    /// The tables of `agents` on `graph`: one backward breadth-first search per agent over
    /// the `(vertex, heading)` states.
    pub(super) fn new(graph: &'a G, agents: &[Agent]) -> Self {
        let headings = graph.heading_count();
        let turns = turn_distances(graph, headings);
        let predecessors = Predecessors::new(graph, headings);
        let remaining = agents
            .iter()
            .map(|agent| predecessors.distances_to(agent.goal))
            .collect();

        Self {
            graph,
            headings,
            turns,
            remaining,
        }
    }

    /// The fewest steps for `agent` from `vertex` facing `heading` to its goal.
    pub(super) fn remaining(&self, agent: usize, vertex: Vertex, heading: Heading) -> Option<u32> {
        self.remaining[agent][vertex * self.headings + heading]
    }

    /// The fewest steps for `agent` from `vertex` facing its best heading to its goal; the
    /// heading-free distance the swap rules compare.
    pub(super) fn least(&self, agent: usize, vertex: Vertex) -> Option<u32> {
        (0..self.headings)
            .filter_map(|heading| self.remaining(agent, vertex, heading))
            .min()
    }

    /// The paper's `TC` for `agent` at `vertex` facing `heading` and requesting `target`: the
    /// steps to arrive there (turns plus one move, or one step for staying) plus the fewest
    /// remaining steps from the state it arrives in. `None` if the goal is out of reach.
    pub(super) fn rank(
        &self,
        agent: usize,
        vertex: Vertex,
        heading: Heading,
        target: Vertex,
    ) -> Option<u32> {
        if target == vertex {
            return self
                .remaining(agent, vertex, heading)
                .map(|steps| steps + 1);
        }

        self.facing(vertex, target)
            .filter_map(|facing| {
                let rest = self.remaining(agent, target, facing)?;
                Some(self.turns(heading, facing) + 1 + rest)
            })
            .min()
    }

    /// The paper's `Compute_action`: the vertex and heading after one step of an agent at
    /// `vertex` facing `heading` that wants to be at `target`, its own vertex or a neighbour.
    /// It moves if it faces `target`, else turns once along a shortest turn sequence towards
    /// a heading that does.
    pub(super) fn action(
        &self,
        vertex: Vertex,
        heading: Heading,
        target: Vertex,
    ) -> (Vertex, Heading) {
        if target == vertex {
            return (vertex, heading);
        }

        if self.graph.forward(vertex, heading) == Some(target) {
            return (target, heading);
        }

        let goal = self
            .facing(vertex, target)
            .min_by_key(|&facing| self.turns(heading, facing))
            .expect("the target is a neighbour");
        let mut first = None;
        self.graph.turns(heading, |next| {
            let closer = self.turns(next, goal) + 1 == self.turns(heading, goal);
            if first.is_none() && closer {
                first = Some(next);
            }
        });

        (vertex, first.expect("a turn leads closer to every heading"))
    }

    /// The headings at `vertex` that move forward to `target`.
    fn facing(&self, vertex: Vertex, target: Vertex) -> impl Iterator<Item = Heading> + '_ {
        (0..self.headings)
            .filter(move |&heading| self.graph.forward(vertex, heading) == Some(target))
    }

    /// The fewest turns from `from` to `to`.
    fn turns(&self, from: Heading, to: Heading) -> u32 {
        self.turns[from * self.headings + to]
    }
}

/// The fewest turns between every two headings, by breadth-first search from each.
///
/// # Panics
///
/// Panics if turning cannot reach every heading from every other.
fn turn_distances<G: OrientedGraph>(graph: &G, headings: usize) -> Vec<u32> {
    let mut table = vec![u32::MAX; headings * headings];
    for from in 0..headings {
        let row = &mut table[from * headings..(from + 1) * headings];
        row[from] = 0;
        let mut queue = VecDeque::from([from]);
        while let Some(heading) = queue.pop_front() {
            let next_cost = row[heading] + 1;
            graph.turns(heading, |next| {
                if row[next] == u32::MAX {
                    row[next] = next_cost;
                    queue.push_back(next);
                }
            });
        }
    }

    assert!(
        table.iter().all(|&turns| turns != u32::MAX),
        "turning reaches every heading"
    );

    table
}

/// The reverse edges of the `(vertex, heading)` state graph, shared by every agent's search.
struct Predecessors {
    /// The number of headings.
    headings: usize,
    /// For each state, the states one move behind it: `u * headings + h` with
    /// `forward(u, h) == vertex` and the same heading.
    moves: Vec<Vec<usize>>,
    /// For each heading, the headings one turn before it.
    turns: Vec<Vec<Heading>>,
}

impl Predecessors {
    /// The reverse edges of `graph`.
    fn new<G: OrientedGraph>(graph: &G, headings: usize) -> Self {
        let mut moves = vec![Vec::new(); graph.vertex_count() * headings];
        for vertex in 0..graph.vertex_count() {
            for heading in 0..headings {
                if let Some(next) = graph.forward(vertex, heading) {
                    moves[next * headings + heading].push(vertex * headings + heading);
                }
            }
        }

        let mut turns = vec![Vec::new(); headings];
        for heading in 0..headings {
            graph.turns(heading, |next| turns[next].push(heading));
        }

        Self {
            headings,
            moves,
            turns,
        }
    }

    /// The fewest steps from every state to `goal`, facing any way.
    fn distances_to(&self, goal: Vertex) -> Vec<Option<u32>> {
        let mut distances = vec![None; self.moves.len()];
        let mut queue = VecDeque::new();
        for heading in 0..self.headings {
            distances[goal * self.headings + heading] = Some(0);
            queue.push_back(goal * self.headings + heading);
        }

        while let Some(state) = queue.pop_front() {
            let next = distances[state].expect("queued states have a distance") + 1;
            let (vertex, heading) = (state / self.headings, state % self.headings);
            let behind = self.moves[state].iter().copied().chain(
                self.turns[heading]
                    .iter()
                    .map(|&before| vertex * self.headings + before),
            );
            for previous in behind {
                if distances[previous].is_none() {
                    distances[previous] = Some(next);
                    queue.push_back(previous);
                }
            }
        }

        distances
    }
}

#[cfg(test)]
mod tests {
    //! The cost tables and actions on a small grid.

    use super::*;

    use crate::graph::Grid;

    /// On a free 3x1 corridor, facing away from the goal costs two turns.
    #[test]
    fn counts_turns_and_moves() {
        let grid = Grid::new(3, 1);
        let costs = Costs::new(&grid, &[Agent { start: 0, goal: 2 }]);

        assert_eq!(costs.remaining(0, 0, Grid::EAST), Some(2));
        assert_eq!(costs.remaining(0, 0, Grid::NORTH), Some(3));
        assert_eq!(costs.remaining(0, 0, Grid::WEST), Some(4));
        assert_eq!(costs.least(0, 1), Some(1));
        assert_eq!(costs.rank(0, 0, Grid::WEST, 1), Some(4));
        assert_eq!(costs.rank(0, 0, Grid::WEST, 0), Some(5));
    }

    /// The action moves when facing the target, turns otherwise, counter-clockwise first
    /// for a half turn, and stays put for the own vertex.
    #[test]
    fn computes_actions() {
        let grid = Grid::new(3, 3);
        let costs = Costs::new(&grid, &[Agent { start: 4, goal: 4 }]);

        assert_eq!(costs.action(4, Grid::EAST, 5), (5, Grid::EAST));
        assert_eq!(costs.action(4, Grid::EAST, 1), (4, Grid::NORTH));
        assert_eq!(costs.action(4, Grid::EAST, 7), (4, Grid::SOUTH));
        assert_eq!(costs.action(4, Grid::EAST, 3), (4, Grid::NORTH));
        assert_eq!(costs.action(4, Grid::EAST, 4), (4, Grid::EAST));
    }
}
