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

/// Reports conflicts to `visit` in time order until it returns `false`.
fn scan<P: AsRef<[Vertex]>>(paths: &[P], mut visit: impl FnMut(Conflict) -> bool) {
    let paths: Vec<&[Vertex]> = paths.iter().map(AsRef::as_ref).collect();
    let horizon = paths.iter().map(|path| path.len()).max().unwrap_or(0);
    let horizon = Time::try_from(horizon).expect("path lengths fit in u32 time steps");

    for time in 0..horizon {
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
