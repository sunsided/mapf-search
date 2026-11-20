//! Negative constraints on one agent ([`Constraint`]) and the per-agent lookup structure
//! the low-level search uses ([`ConstraintSet`]).

use alloc::collections::BTreeSet;

use crate::graph::Vertex;
use crate::instance::AgentId;
use crate::path::Time;

/// Forbids one agent one vertex or one move at one time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Constraint {
    /// `agent` must not occupy `vertex` at `time`.
    Vertex {
        /// The constrained agent.
        agent: AgentId,
        /// The forbidden vertex.
        vertex: Vertex,
        /// The time step at which it is forbidden.
        time: Time,
    },
    /// `agent` must not move `from -> to` between `time` and `time + 1`.
    Edge {
        /// The constrained agent.
        agent: AgentId,
        /// The vertex the forbidden move leaves.
        from: Vertex,
        /// The vertex the forbidden move enters.
        to: Vertex,
        /// The time step at which the forbidden move starts.
        time: Time,
    },
}

impl Constraint {
    /// The constrained agent.
    #[must_use]
    pub fn agent(&self) -> AgentId {
        match *self {
            Self::Vertex { agent, .. } | Self::Edge { agent, .. } => agent,
        }
    }
}

/// The constraints of one agent, indexed for the checks of a space-time search.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstraintSet {
    /// Forbidden `(time, vertex)` pairs.
    vertices: BTreeSet<(Time, Vertex)>,
    /// Forbidden `(time, from, to)` moves.
    edges: BTreeSet<(Time, Vertex, Vertex)>,
    /// The latest time step any constraint touches; an edge constraint at `t` touches
    /// `t + 1`. `0` while empty.
    latest_step: Time,
}

impl ConstraintSet {
    /// Creates an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The set of those of `constraints` that name `agent`.
    #[must_use]
    pub fn for_agent(agent: AgentId, constraints: &[Constraint]) -> Self {
        let mut set = Self::new();
        constraints
            .iter()
            .filter(|constraint| constraint.agent() == agent)
            .for_each(|constraint| set.insert(constraint));

        set
    }

    /// Adds `constraint`, ignoring which agent it names.
    pub fn insert(&mut self, constraint: &Constraint) {
        match *constraint {
            Constraint::Vertex { vertex, time, .. } => {
                self.vertices.insert((time, vertex));
                self.latest_step = self.latest_step.max(time);
            }
            Constraint::Edge { from, to, time, .. } => {
                self.edges.insert((time, from, to));
                self.latest_step = self.latest_step.max(time.saturating_add(1));
            }
        }
    }

    /// Returns `true` if the agent may move (or wait, when `from == to`) from `from` at
    /// `time` to `to` at `time + 1`.
    #[must_use]
    pub fn allows(&self, from: Vertex, to: Vertex, time: Time) -> bool {
        let next = time.saturating_add(1);

        !self.vertices.contains(&(next, to)) && !self.edges.contains(&(time, from, to))
    }

    /// Returns `true` if the agent may occupy `vertex` at `time`.
    #[must_use]
    pub fn allows_vertex(&self, vertex: Vertex, time: Time) -> bool {
        !self.vertices.contains(&(time, vertex))
    }

    /// The latest time step any constraint touches; `0` while empty.
    ///
    /// After this step the agent is unconstrained.
    #[must_use]
    pub fn latest_step(&self) -> Time {
        self.latest_step
    }

    /// The latest time `vertex` is forbidden, if ever.
    ///
    /// For the goal vertex, an agent must not finish at or before this time: goal-stay
    /// semantics would put it there.
    #[must_use]
    pub fn blocked_until(&self, vertex: Vertex) -> Option<Time> {
        self.vertices
            .iter()
            .rev()
            .find(|&&(_, v)| v == vertex)
            .map(|&(time, _)| time)
    }
}
