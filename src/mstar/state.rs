//! The states M* searches ([`Key`]) and one agent's move in a joint step ([`Choice`]).
//!
//! A *standard* state is every agent's vertex and finished flag at one time step. With
//! operator decomposition (Standley 2010; Ferner, Wagner and Choset 2013), a joint step is
//! taken one agent at a time, through *intermediate* states where the first agents of the
//! planner's order have moved already and the rest have not.

use alloc::vec::Vec;

use crate::graph::Vertex;
use crate::path::Time;

/// A joint state of a planner's agents, in the planner's agent order: every agent's vertex
/// and whether it has finished (stays there for good).
pub(super) type JointState = Vec<(Vertex, bool)>;

/// A search state of one planner: standard if [`Key::moved`] is empty, else intermediate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Key {
    /// The time of [`Key::base`]. Always `0` without a deadline, where time does not
    /// matter; counted only when a deadline prunes late states.
    pub(super) time: Time,
    /// Every agent's position at [`Key::time`].
    pub(super) base: JointState,
    /// The positions after the step of the first `moved.len()` agents; empty for a
    /// standard state. Never as long as [`Key::base`].
    pub(super) moved: JointState,
}

impl Key {
    /// The standard state `base` at `time`.
    pub(super) fn standard(time: Time, base: JointState) -> Self {
        Self {
            time,
            base,
            moved: Vec::new(),
        }
    }

    /// Returns `true` for a standard state.
    pub(super) fn is_standard(&self) -> bool {
        self.moved.is_empty()
    }

    /// Returns `true` for a standard state in which every agent has finished: a goal.
    pub(super) fn is_goal(&self) -> bool {
        self.is_standard() && self.base.iter().all(|&(_, finished)| finished)
    }

    /// Every agent's current position: the moved agents' new ones, the rest's old ones.
    pub(super) fn positions(&self) -> impl Iterator<Item = (Vertex, bool)> + '_ {
        self.moved
            .iter()
            .chain(&self.base[self.moved.len()..])
            .copied()
    }
}

/// One agent's choice for one joint step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Choice {
    /// Where the agent is after the step.
    pub(super) next: Vertex,
    /// Whether the agent is finished after the step.
    pub(super) finished: bool,
    /// What the step costs the agent: 1 unless it is finished after it.
    pub(super) cost: u32,
}

impl Choice {
    /// Finishing, or staying finished, at `vertex`; free.
    pub(super) fn finish(vertex: Vertex) -> Self {
        Self {
            next: vertex,
            finished: true,
            cost: 0,
        }
    }

    /// A move to `next`, or a wait if it is the current vertex; costs 1.
    pub(super) fn move_to(next: Vertex) -> Self {
        Self {
            next,
            finished: false,
            cost: 1,
        }
    }

    /// The step from `from` to `to`, finished or not.
    pub(super) fn between(from: (Vertex, bool), to: (Vertex, bool)) -> Self {
        debug_assert!(!from.1 || to == from, "finished agents stay finished");

        if to.1 {
            Self::finish(to.0)
        } else {
            Self::move_to(to.0)
        }
    }

    /// The position after the choice.
    pub(super) fn position(self) -> (Vertex, bool) {
        (self.next, self.finished)
    }
}

/// Returns `true` if two agents' moves collide: they end at the same vertex, or swap along
/// an edge. Each move is `(from, to)`.
pub(super) fn collide(a: (Vertex, Vertex), b: (Vertex, Vertex)) -> bool {
    let same = a.1 == b.1;
    let swap = a.0 != a.1 && a.0 == b.1 && a.1 == b.0;

    same || swap
}
