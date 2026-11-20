//! Single-agent paths over discrete time and the joint objectives over them
//! ([`Objective`]): [`sum_of_costs`] and [`makespan`].
//!
//! A path lists the vertex an agent occupies at each time step, so `path[t]` is the
//! position at time `t`. After its last entry the agent stays at that vertex forever
//! (goal-stay semantics), see [`position_at`].

use alloc::vec::Vec;

use crate::graph::Vertex;

/// A time step. Time starts at `0`.
pub type Time = u32;

/// The positions of one agent at `t = 0, 1, ...`. Never empty in a solution.
pub type Path = Vec<Vertex>;

/// What a multi-agent search minimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Objective {
    /// The sum over agents of the final arrival time at the goal ([`sum_of_costs`]).
    SumOfCosts,
    /// The time the last agent arrives at its goal for good ([`makespan`]).
    Makespan,
}

/// The vertex `path` occupies at `time`, counting the stay at the last vertex.
///
/// # Panics
///
/// Panics if `path` is empty.
#[must_use]
pub fn position_at(path: &[Vertex], time: Time) -> Vertex {
    let index = usize::try_from(time).unwrap_or(usize::MAX);

    path.get(index)
        .or(path.last())
        .copied()
        .expect("a path has at least one position")
}

/// The cost of one path: the time of its final arrival at its last vertex.
///
/// Waits before the final arrival count, also waits at the goal before leaving it again.
/// Trailing waits at the last vertex do not. An empty path costs `0`.
///
/// # Panics
///
/// Panics if a path has more than `u32::MAX` positions.
#[must_use]
pub fn path_cost(path: &[Vertex]) -> u32 {
    let Some(&last) = path.last() else {
        return 0;
    };

    let trailing = path.iter().rev().take_while(|&&v| v == last).count();
    let arrival = path.len() - trailing;

    u32::try_from(arrival).expect("path lengths fit in u32 time steps")
}

/// The sum of the [`path_cost`]s of all paths.
///
/// # Panics
///
/// Panics if a path has more than `u32::MAX` positions.
#[must_use]
pub fn sum_of_costs<P: AsRef<[Vertex]>>(paths: &[P]) -> u32 {
    paths.iter().map(|path| path_cost(path.as_ref())).sum()
}

/// The largest [`path_cost`] of all paths: when the last agent arrives for good. `0` for
/// no paths.
///
/// # Panics
///
/// Panics if a path has more than `u32::MAX` positions.
#[must_use]
pub fn makespan<P: AsRef<[Vertex]>>(paths: &[P]) -> u32 {
    paths
        .iter()
        .map(|path| path_cost(path.as_ref()))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    //! Unit tests for path costs.

    use super::*;
    use alloc::vec;

    /// Trailing waits are free; waits before the final arrival are not.
    #[test]
    fn cost_counts_until_final_arrival() {
        assert_eq!(path_cost(&[3]), 0);
        assert_eq!(path_cost(&[0, 1, 2]), 2);
        assert_eq!(path_cost(&[0, 1, 1, 2, 2, 2]), 3);
        assert_eq!(path_cost(&[2, 1, 2]), 2);
        assert_eq!(sum_of_costs(&[vec![0, 1], vec![5, 5, 4]]), 3);
        assert_eq!(makespan(&[vec![0, 1], vec![5, 5, 4]]), 2);
    }

    /// The last position holds forever.
    #[test]
    fn position_stays_at_goal() {
        assert_eq!(position_at(&[0, 1, 2], 1), 1);
        assert_eq!(position_at(&[0, 1, 2], 10), 2);
    }
}
