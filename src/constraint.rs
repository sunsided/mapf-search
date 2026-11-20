//! Negative constraints on one agent ([`Constraint`]) and the per-agent lookup structure
//! the low-level search uses ([`ConstraintSet`]).

use alloc::collections::{BTreeMap, BTreeSet};

use crate::conflict::Conflict;
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
    /// The two constraints that resolve `conflict`, one per agent.
    pub(crate) fn resolving(conflict: Conflict) -> [Self; 2] {
        match conflict {
            Conflict::Vertex {
                agents: (a, b),
                vertex,
                time,
            } => [a, b].map(|agent| Self::Vertex {
                agent,
                vertex,
                time,
            }),
            Conflict::Edge {
                agents: (a, b),
                from,
                to,
                time,
            } => [
                Self::Edge {
                    agent: a,
                    from,
                    to,
                    time,
                },
                Self::Edge {
                    agent: b,
                    from: to,
                    to: from,
                    time,
                },
            ],
        }
    }

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
    /// Vertices forbidden from a time on, forever: where another agent stays for good
    /// ([`ConstraintSet::block_from`]). The earliest such time per vertex.
    blocked_from: BTreeMap<Vertex, Time>,
    /// The latest time step any constraint touches; an edge constraint at `t` touches
    /// `t + 1`, a permanent block touches its first time. `0` while empty.
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

    /// Forbids `vertex` at `time` and at every later time, as for a vertex where another
    /// agent stays for good after arriving at `time`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mapf_search::ConstraintSet;
    ///
    /// let mut constraints = ConstraintSet::new();
    /// constraints.block_from(4, 2);
    ///
    /// assert!(constraints.allows_vertex(4, 1));
    /// assert!(!constraints.allows_vertex(4, 9));
    /// assert_eq!(constraints.blocked_until(4), Some(u32::MAX));
    /// ```
    pub fn block_from(&mut self, vertex: Vertex, time: Time) {
        let from = self.blocked_from.entry(vertex).or_insert(time);
        *from = (*from).min(time);
        self.latest_step = self.latest_step.max(time);
    }

    /// Returns `true` if the agent may move (or wait, when `from == to`) from `from` at
    /// `time` to `to` at `time + 1`.
    #[must_use]
    pub fn allows(&self, from: Vertex, to: Vertex, time: Time) -> bool {
        let next = time.saturating_add(1);

        self.allows_vertex(to, next) && !self.edges.contains(&(time, from, to))
    }

    /// Returns `true` if the agent may occupy `vertex` at `time`.
    #[must_use]
    pub fn allows_vertex(&self, vertex: Vertex, time: Time) -> bool {
        let blocked = self
            .blocked_from
            .get(&vertex)
            .is_some_and(|&from| from <= time);

        !blocked && !self.vertices.contains(&(time, vertex))
    }

    /// The latest time step any constraint touches; `0` while empty.
    ///
    /// After this step the constraints no longer change: the agent is free except for the
    /// vertices blocked forever ([`ConstraintSet::block_from`]).
    #[must_use]
    pub fn latest_step(&self) -> Time {
        self.latest_step
    }

    /// The latest time `vertex` is forbidden, if ever.
    ///
    /// For the goal vertex, an agent must not finish at or before this time: goal-stay
    /// semantics would put it there. [`Time::MAX`] if the vertex is blocked forever
    /// ([`ConstraintSet::block_from`]).
    #[must_use]
    pub fn blocked_until(&self, vertex: Vertex) -> Option<Time> {
        if self.blocked_from.contains_key(&vertex) {
            return Some(Time::MAX);
        }

        self.vertices
            .iter()
            .rev()
            .find(|&&(_, v)| v == vertex)
            .map(|&(time, _)| time)
    }
}
