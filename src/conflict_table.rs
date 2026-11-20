//! A conflict avoidance table ([`ConflictTable`]): where the other agents are, so that one
//! agent's low-level search can count the conflicts its moves would cause.
//!
//! CBS uses it to break ties among an agent's cheapest paths (Sharon et al. 2015), ECBS as
//! the focal key of its low level (Barer et al. 2014). Both count with goal-stay
//! semantics: an agent sitting at its goal blocks it for good, and an agent that finishes
//! at its goal conflicts with everyone who passes there later ([`ConflictTable::after`]).

use alloc::collections::BTreeMap;

use crate::graph::Vertex;
use crate::instance::AgentId;
use crate::path::{Path, Time};

/// Where the other agents are, for counting the conflicts of one agent's moves.
#[derive(Debug, Clone, Default)]
pub(crate) struct ConflictTable {
    /// How many other agents occupy a vertex at a time before their final arrival, as
    /// `(vertex, time)`.
    visits: BTreeMap<(Vertex, Time), u32>,
    /// The vertices other agents stay at for good, with the time they arrive there.
    parked: BTreeMap<Vertex, Time>,
    /// How many other agents move `from -> to` between a time and the next, as
    /// `(time, from, to)`; waits are not recorded.
    moves: BTreeMap<(Time, Vertex, Vertex), u32>,
}

impl ConflictTable {
    /// The table of every path in `paths`, indexed by agent, except that of `agent`.
    /// `paths` may be a prefix of the agents.
    pub(crate) fn without(paths: &[Path], agent: AgentId) -> Self {
        let mut table = Self::default();
        for (_, path) in paths
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != agent)
        {
            table.add(path);
        }

        table
    }

    /// Records `path`, which must not be empty.
    fn add(&mut self, path: &[Vertex]) {
        let last = path.len() - 1;
        let time = |index: usize| Time::try_from(index).expect("path lengths fit in u32");

        for (index, &vertex) in path[..last].iter().enumerate() {
            *self.visits.entry((vertex, time(index))).or_default() += 1;
        }

        for (index, step) in path.windows(2).enumerate() {
            if step[0] != step[1] {
                *self
                    .moves
                    .entry((time(index), step[0], step[1]))
                    .or_default() += 1;
            }
        }

        self.parked.insert(path[last], time(last));
    }

    /// The number of conflicts of the move `from -> to` (a wait if equal) between `time`
    /// and `time + 1`: vertex conflicts at `to`, also with an agent parked there, and swaps.
    pub(crate) fn count(&self, from: Vertex, to: Vertex, time: Time) -> u32 {
        let next = time + 1;
        let visits = self.visits.get(&(to, next)).copied().unwrap_or(0);
        let parked = u32::from(self.parked.get(&to).is_some_and(|&since| since <= next));
        let swaps = if from == to {
            0
        } else {
            self.moves.get(&(time, to, from)).copied().unwrap_or(0)
        };

        visits + parked + swaps
    }

    /// The number of conflicts of an agent that stays at `vertex` for good from `time` on:
    /// every later visit of another agent, and another agent that parks there later.
    pub(crate) fn after(&self, vertex: Vertex, time: Time) -> u32 {
        let later = (vertex, time.saturating_add(1))..=(vertex, Time::MAX);
        let visits: u32 = self.visits.range(later).map(|(_, &count)| count).sum();
        let parked = u32::from(self.parked.get(&vertex).is_some_and(|&since| since > time));

        visits + parked
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for conflict counting.

    use super::*;
    use alloc::vec;

    /// Moves count vertex conflicts, parked agents and swaps.
    #[test]
    fn counts_moves() {
        // Agent 1 goes 2 -> 1 -> 0 and parks; agent 0 is left out.
        let table = ConflictTable::without(&[vec![9], vec![2, 1, 0]], 0);

        assert_eq!(table.count(0, 1, 0), 1, "meets agent 1 at 1");
        assert_eq!(table.count(1, 2, 0), 1, "swaps with agent 1");
        assert_eq!(table.count(3, 0, 1), 1, "agent 1 parks at 0 at time 2");
        assert_eq!(table.count(3, 3, 1), 0);
    }

    /// Finishing at a vertex conflicts with later visits and later parking only.
    #[test]
    fn counts_conflicts_after_finishing() {
        let table = ConflictTable::without(&[vec![5, 4, 3, 4, 7], vec![6, 6, 6, 3]], 9);

        assert_eq!(table.after(4, 0), 2, "visits at 1 and 3");
        assert_eq!(table.after(4, 1), 1);
        assert_eq!(table.after(4, 3), 0);
        assert_eq!(
            table.after(3, 1),
            2,
            "agent 0 visits at 2, agent 1 parks at 3"
        );
        assert_eq!(table.after(3, 3), 0);
    }
}
