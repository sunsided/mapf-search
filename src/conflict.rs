//! Conflicts between agents' paths ([`Conflict`]) and their detection
//! ([`first_conflict`], [`conflicts`]).
//!
//! Detection follows goal-stay semantics: a path that ended keeps occupying its last
//! vertex, so an agent waiting at its goal conflicts with every agent passing through it.

use alloc::vec::Vec;

use crate::graph::Vertex;
use crate::instance::AgentId;
use crate::path::{Time, position_at};

/// A conflict between the paths of two agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Conflict {
    /// Both agents occupy `vertex` at `time`.
    Vertex {
        /// The two agents, the smaller id first.
        agents: (AgentId, AgentId),
        /// The shared vertex.
        vertex: Vertex,
        /// The time step.
        time: Time,
    },
    /// The agents swap vertices: the first moves `from -> to` while the second moves
    /// `to -> from`, between `time` and `time + 1`. `from != to`.
    Edge {
        /// The two agents, the smaller id first.
        agents: (AgentId, AgentId),
        /// The vertex the first agent leaves and the second enters.
        from: Vertex,
        /// The vertex the first agent enters and the second leaves.
        to: Vertex,
        /// The time step at which the swap starts.
        time: Time,
    },
}

impl Conflict {
    /// The two agents involved, the smaller id first.
    #[must_use]
    pub fn agents(&self) -> (AgentId, AgentId) {
        match *self {
            Self::Vertex { agents, .. } | Self::Edge { agents, .. } => agents,
        }
    }

    /// The time step the conflict happens at (for swaps: the step it starts at).
    #[must_use]
    pub fn time(&self) -> Time {
        match *self {
            Self::Vertex { time, .. } | Self::Edge { time, .. } => time,
        }
    }
}

/// The earliest conflict among `paths`, indexed by agent: smallest time first, then the
/// smallest agent pair, and a vertex conflict before a swap starting at the same time.
///
/// Returns `None` if the paths are conflict-free.
///
/// # Panics
///
/// Panics if a path is empty.
#[must_use]
pub fn first_conflict<P: AsRef<[Vertex]>>(paths: &[P]) -> Option<Conflict> {
    let mut found = None;
    scan(paths, |conflict| {
        found = Some(conflict);
        false
    });

    found
}

/// All conflicts among `paths`, in the order of [`first_conflict`].
///
/// An agent that waits at its goal can conflict with another agent at many time steps;
/// each counts.
///
/// # Panics
///
/// Panics if a path is empty.
#[must_use]
pub fn conflicts<P: AsRef<[Vertex]>>(paths: &[P]) -> Vec<Conflict> {
    let mut found = Vec::new();
    scan(paths, |conflict| {
        found.push(conflict);
        true
    });

    found
}

/// `found`, the conflicts among an earlier version of `paths`, after the agents in
/// `changed` took new paths: the old conflicts between unchanged agents, and those of the
/// changed agents rescanned, in the order of [`first_conflict`].
///
/// Costs `O(k n T)` for `k` changed agents of `n` with paths of up to `T` positions,
/// where [`conflicts`] costs `O(n^2 T)`.
///
/// # Panics
///
/// Panics if a path is empty.
pub(crate) fn update_conflicts<P: AsRef<[Vertex]>>(
    found: &[Conflict],
    paths: &[P],
    changed: &[AgentId],
) -> Vec<Conflict> {
    let paths: Vec<&[Vertex]> = paths.iter().map(AsRef::as_ref).collect();
    let involved = |conflict: &Conflict| {
        let (a, b) = conflict.agents();
        changed.contains(&a) || changed.contains(&b)
    };

    let mut updated: Vec<Conflict> = found.iter().filter(|c| !involved(c)).copied().collect();
    for time in 0..horizon(&paths) {
        for &a in changed {
            // A pair of two changed agents is scanned once, from its smaller agent.
            let others = (0..paths.len()).filter(|&b| b != a && !(b < a && changed.contains(&b)));
            for b in others {
                let pair = (a.min(b), a.max(b));
                let found = conflicts_at(paths[pair.0], paths[pair.1], pair, time);
                updated.extend(found.into_iter().flatten());
            }
        }
    }

    updated.sort_unstable_by_key(scan_order);

    updated
}

/// The position of `conflict` in the order of [`first_conflict`]: by time, then agent
/// pair, then a vertex conflict before a swap.
fn scan_order(conflict: &Conflict) -> (Time, (AgentId, AgentId), bool) {
    (
        conflict.time(),
        conflict.agents(),
        matches!(conflict, Conflict::Edge { .. }),
    )
}

/// The number of time steps to scan `paths` for conflicts: the longest path's length.
fn horizon(paths: &[&[Vertex]]) -> Time {
    let longest = paths.iter().map(|path| path.len()).max().unwrap_or(0);

    Time::try_from(longest).expect("path lengths fit in u32 time steps")
}

/// Reports conflicts to `visit` in time order until it returns `false`.
fn scan<P: AsRef<[Vertex]>>(paths: &[P], mut visit: impl FnMut(Conflict) -> bool) {
    let paths: Vec<&[Vertex]> = paths.iter().map(AsRef::as_ref).collect();

    for time in 0..horizon(&paths) {
        if !scan_time(&paths, time, &mut visit) {
            return;
        }
    }
}

/// Reports the conflicts at `time` (and swaps starting at `time`) between every pair of
/// `paths` to `visit`. Returns `false` as soon as `visit` does.
fn scan_time(paths: &[&[Vertex]], time: Time, visit: &mut impl FnMut(Conflict) -> bool) -> bool {
    for (i, a) in paths.iter().enumerate() {
        for (j, b) in paths.iter().enumerate().skip(i + 1) {
            let pair = conflicts_at(a, b, (i, j), time);
            if pair.into_iter().flatten().any(|conflict| !visit(conflict)) {
                return false;
            }
        }
    }

    true
}

/// The vertex conflict at `time` and the swap from `time` to `time + 1` between paths `a`
/// and `b` of `agents`, if any.
fn conflicts_at(
    a: &[Vertex],
    b: &[Vertex],
    agents: (AgentId, AgentId),
    time: Time,
) -> [Option<Conflict>; 2] {
    let (a_now, b_now) = (position_at(a, time), position_at(b, time));
    let (a_next, b_next) = (position_at(a, time + 1), position_at(b, time + 1));

    let vertex = (a_now == b_now).then_some(Conflict::Vertex {
        agents,
        vertex: a_now,
        time,
    });
    let swap = (a_now != a_next && a_now == b_next && a_next == b_now).then_some(Conflict::Edge {
        agents,
        from: a_now,
        to: a_next,
        time,
    });

    [vertex, swap]
}

#[cfg(test)]
mod tests {
    //! Unit tests for incremental conflict detection.

    use super::*;
    use alloc::vec;

    /// Updating after some agents changed gives what a full scan gives, in the same order.
    #[test]
    fn update_matches_a_full_scan() {
        let before = [vec![0, 1, 2], vec![2, 1, 0], vec![5, 1, 6], vec![7, 6, 5]];
        let after = [
            vec![0, 1, 2],
            vec![2, 2, 1, 0],
            vec![5, 1, 6],
            vec![6, 6, 5],
        ];
        let found = conflicts(&before);

        for changed in [&[][..], &[1], &[3], &[1, 3], &[3, 1], &[0, 1, 2, 3]] {
            let paths: Vec<_> = (0..4)
                .map(|agent| {
                    if changed.contains(&agent) {
                        &after[agent]
                    } else {
                        &before[agent]
                    }
                })
                .collect();

            assert_eq!(
                update_conflicts(&found, &paths, changed),
                conflicts(&paths),
                "changed {changed:?}"
            );
        }
    }
}
