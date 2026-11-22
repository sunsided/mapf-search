//! The operator selection function of EPEA* for MAPF ([`Operators`]): each agent's moves
//! grouped by how much they raise `f`, and the joint moves that raise it by exactly a
//! given amount.
//!
//! With the sum of distances as heuristic, a joint step raises `f` by the sum of what each
//! agent's own move raises it by: `1 + d(next) - d(here)` for a move or a wait, `0` for
//! finishing at the goal or staying finished. A step lowers the distance by at most one, so
//! a rise is never negative; on an undirected graph it is at most `2`, on a directed one it
//! can be larger (an edge into a long one-way detour). So the joint moves with a given rise are the combinations of agent
//! moves whose rises add up to it, found by a depth-first search that skips combinations
//! whose remaining agents cannot reach the target, and colliding ones.

use alloc::vec;
use alloc::vec::Vec;

use crate::graph::Vertex;

/// One agent's move in a joint step and the rise of `f` it causes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Move {
    /// The agent's position before the step.
    pub(super) from: Vertex,
    /// The agent's position after the step.
    pub(super) to: Vertex,
    /// Whether the agent is finished after the step.
    pub(super) finished: bool,
    /// What the step costs the agent: `0` if it is finished after it, else `1`.
    pub(super) cost: u32,
    /// How much the step raises `f`: the cost plus the change of the agent's distance.
    pub(super) rise: u32,
}

/// The moves of every agent of one joint state, each agent's sorted by rise.
pub(super) struct Operators {
    /// The moves of each agent, by rise ascending. Never empty.
    moves: Vec<Vec<Move>>,
    /// The smallest total rise of agents `i..`, indexed by `i`; one longer than
    /// [`Operators::moves`].
    least: Vec<u32>,
    /// The largest total rise of agents `i..`, like [`Operators::least`].
    most: Vec<u32>,
}

impl Operators {
    /// The operators of agents with the given `moves`, each agent's list non-empty.
    pub(super) fn new(mut moves: Vec<Vec<Move>>) -> Self {
        for list in &mut moves {
            list.sort_by_key(|step| step.rise);
        }

        let mut least = vec![0; moves.len() + 1];
        let mut most = vec![0; moves.len() + 1];
        for (agent, list) in moves.iter().enumerate().rev() {
            let rise = |step: Option<&Move>| step.map_or(0, |step| step.rise);
            least[agent] = least[agent + 1] + rise(list.first());
            most[agent] = most[agent + 1] + rise(list.last());
        }

        Self { moves, least, most }
    }

    /// The smallest total rise above `rise` that some joint move has, collisions ignored;
    /// `None` if there is none.
    pub(super) fn next_rise(&self, rise: u32) -> Option<u32> {
        let mut reachable = vec![true];
        for list in &self.moves {
            let widest = list.last().map_or(0, |step| step.rise as usize);
            let mut next = vec![false; reachable.len() + widest];
            for (sum, _) in reachable
                .iter()
                .enumerate()
                .filter(|(_, reached)| **reached)
            {
                for step in list {
                    next[sum + step.rise as usize] = true;
                }
            }

            reachable = next;
        }

        reachable
            .iter()
            .enumerate()
            .skip(rise as usize + 1)
            .find(|(_, reached)| **reached)
            .map(|(sum, _)| u32::try_from(sum).expect("rise sums fit in u32"))
    }

    /// Every collision-free joint move whose rises add up to `rise`, one move per agent.
    pub(super) fn with_rise(&self, rise: u32) -> Vec<Vec<Move>> {
        let mut found = Vec::new();
        self.extend(&mut Vec::with_capacity(self.moves.len()), rise, &mut found);

        found
    }

    /// Extends the collision-free `partial` joint move, which leaves `left` of the target
    /// rise to the remaining agents, by every move of the next agent that keeps the target
    /// reachable.
    fn extend(&self, partial: &mut Vec<Move>, left: u32, found: &mut Vec<Vec<Move>>) {
        let agent = partial.len();
        if agent == self.moves.len() {
            found.push(partial.clone());
            return;
        }

        for &step in &self.moves[agent] {
            let Some(rest) = left.checked_sub(step.rise) else {
                break;
            };

            let reachable = (self.least[agent + 1]..=self.most[agent + 1]).contains(&rest);
            if reachable && !partial.iter().any(|other| collide(*other, step)) {
                partial.push(step);
                self.extend(partial, rest, found);
                partial.pop();
            }
        }
    }
}

/// Returns `true` if two agents' moves collide: they end at the same vertex, or swap along
/// an edge.
fn collide(a: Move, b: Move) -> bool {
    let swap = a.from != a.to && a.from == b.to && a.to == b.from;

    a.to == b.to || swap
}

#[cfg(test)]
mod tests {
    //! The joint moves of small operator tables against brute force.

    use super::*;

    /// A wait-or-move of rise `rise` from `from` to `to`.
    fn step(from: Vertex, to: Vertex, rise: u32) -> Move {
        Move {
            from,
            to,
            finished: false,
            cost: 1,
            rise,
        }
    }

    /// Two agents on a corridor `0 1 2 3`, at 0 and 3: the joint moves by rise, without the
    /// one that would put both at the same vertex.
    #[test]
    fn combines_rises_and_skips_collisions() {
        let operators = Operators::new(vec![
            vec![step(0, 1, 0), step(0, 0, 1)],
            vec![step(3, 2, 0), step(3, 3, 1), step(3, 1, 0)],
        ]);

        let zero = operators.with_rise(0);
        let one = operators.with_rise(1);

        assert_eq!(zero.len(), 1, "0->1 with 3->1 collides");
        assert_eq!(zero[0][1].to, 2);
        assert_eq!(one.len(), 3);
        assert_eq!(operators.next_rise(0), Some(1));
        assert_eq!(operators.next_rise(1), Some(2));
        assert_eq!(operators.next_rise(2), None);
    }

    /// Rises above 2, as on a directed graph, are combined like any other.
    #[test]
    fn handles_rises_above_two() {
        let operators = Operators::new(vec![
            vec![step(0, 4, 0), step(0, 1, 3)],
            vec![step(2, 2, 1)],
        ]);

        assert_eq!(operators.next_rise(1), Some(4));
        assert_eq!(operators.with_rise(4).len(), 1);
        assert_eq!(operators.next_rise(4), None);
    }

    /// A swap along an edge collides.
    #[test]
    fn swaps_collide() {
        let operators = Operators::new(vec![vec![step(0, 1, 0)], vec![step(1, 0, 0)]]);

        assert_eq!(operators.with_rise(0), Vec::<Vec<Move>>::new());
        assert_eq!(operators.next_rise(0), None);
    }
}
