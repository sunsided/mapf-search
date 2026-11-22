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

/// What a path holds under goal-stay semantics, for building obstacles from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reservation {
    /// The path is at `vertex` at `time`, before its final position.
    Visit {
        /// The occupied vertex.
        vertex: Vertex,
        /// The time step.
        time: Time,
    },
    /// The path moves `from -> to` between `time` and `time + 1`; `from != to`. Another
    /// agent moving `to -> from` then swaps with it.
    Move {
        /// The vertex the path leaves.
        from: Vertex,
        /// The vertex the path enters.
        to: Vertex,
        /// The time step the move starts at.
        time: Time,
    },
    /// The path ends at `vertex` at `time` and stays there for good.
    Park {
        /// The last vertex.
        vertex: Vertex,
        /// The time of the last position.
        time: Time,
    },
}

/// Every [`Reservation`] of `path`: its visits before the last position, its moves (not
/// its waits), and its park at the last position.
///
/// # Panics
///
/// Panics if `path` is empty or has more than `u32::MAX` positions.
pub(crate) fn reservations(path: &[Vertex]) -> impl Iterator<Item = Reservation> + '_ {
    let time = |index: usize| Time::try_from(index).expect("path lengths fit in u32 time steps");
    let last = path.len() - 1;

    let visits = path[..last]
        .iter()
        .enumerate()
        .map(move |(index, &vertex)| Reservation::Visit {
            vertex,
            time: time(index),
        });
    let moves = path
        .windows(2)
        .enumerate()
        .filter(|(_, step)| step[0] != step[1])
        .map(move |(index, step)| Reservation::Move {
            from: step[0],
            to: step[1],
            time: time(index),
        });
    let park = Reservation::Park {
        vertex: path[last],
        time: time(last),
    };

    visits.chain(moves).chain(core::iter::once(park))
}

/// Drops the waits at the end of `path`, keeping its last vertex: after that, goal-stay
/// semantics put the agent there anyway.
pub(crate) fn trim_trailing_waits(path: &mut Path) {
    while path.len() > 1 && path[path.len() - 2] == path[path.len() - 1] {
        path.pop();
    }
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

    /// A path reserves its visits, its moves without waits, and its park.
    #[test]
    fn reserves_visits_moves_and_park() {
        let reserved: Vec<Reservation> = reservations(&[3, 3, 4]).collect();

        assert_eq!(
            reserved,
            [
                Reservation::Visit { vertex: 3, time: 0 },
                Reservation::Visit { vertex: 3, time: 1 },
                Reservation::Move {
                    from: 3,
                    to: 4,
                    time: 1
                },
                Reservation::Park { vertex: 4, time: 2 },
            ]
        );
    }

    /// Only the waits at the end go; a single position stays.
    #[test]
    fn trims_only_trailing_waits() {
        let mut path = vec![1, 1, 2, 2, 2];
        trim_trailing_waits(&mut path);
        assert_eq!(path, [1, 1, 2]);

        let mut single = vec![4, 4];
        trim_trailing_waits(&mut single);
        assert_eq!(single, [4]);
    }

    /// The last position holds forever.
    #[test]
    fn position_stays_at_goal() {
        assert_eq!(position_at(&[0, 1, 2], 1), 1);
        assert_eq!(position_at(&[0, 1, 2], 10), 2);
    }
}
