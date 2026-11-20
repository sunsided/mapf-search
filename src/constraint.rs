//! Constraints on one agent ([`Constraint`]), negative (forbid a vertex or move) and
//! positive (require one, for disjoint splitting), and the per-agent lookup structure the
//! low-level search uses ([`ConstraintSet`]).
//!
//! A positive constraint on one agent implies negative ones on every other agent: if `a`
//! must be at `v` at `t`, nobody else may be. [`ConstraintSet::for_agent`] adds those.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::conflict::Conflict;
use crate::graph::Vertex;
use crate::instance::AgentId;
use crate::path::{Time, position_at};

/// Forbids or requires one agent one vertex or one move at one time.
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
    /// `agent` must occupy `vertex` at `time`; no other agent may (disjoint splitting, Li
    /// et al. 2019).
    PositiveVertex {
        /// The constrained agent.
        agent: AgentId,
        /// The required vertex.
        vertex: Vertex,
        /// The time step at which it is required.
        time: Time,
    },
    /// `agent` must move `from -> to` between `time` and `time + 1`, with `from != to`; no
    /// other agent may be at `from` at `time` or at `to` at `time + 1`, or move `to -> from`
    /// then.
    PositiveEdge {
        /// The constrained agent.
        agent: AgentId,
        /// The vertex the required move leaves.
        from: Vertex,
        /// The vertex the required move enters.
        to: Vertex,
        /// The time step at which the required move starts.
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

    /// The two children of disjoint splitting (Li et al. 2019) for `conflict`, split on
    /// the agent at `side` (`0` for the conflict's first agent, else its second): the
    /// negative constraint on it, then the positive one.
    pub(crate) fn disjoint(conflict: Conflict, side: usize) -> [Self; 2] {
        let negative = Self::resolving(conflict)[usize::from(side != 0)];

        [negative, negative.positive()]
    }

    /// The positive counterpart of a negative constraint; positive ones stay as they are.
    fn positive(self) -> Self {
        match self {
            Self::Vertex {
                agent,
                vertex,
                time,
            } => Self::PositiveVertex {
                agent,
                vertex,
                time,
            },
            Self::Edge {
                agent,
                from,
                to,
                time,
            } => Self::PositiveEdge {
                agent,
                from,
                to,
                time,
            },
            positive => positive,
        }
    }

    /// The constrained agent.
    #[must_use]
    pub fn agent(&self) -> AgentId {
        match *self {
            Self::Vertex { agent, .. }
            | Self::Edge { agent, .. }
            | Self::PositiveVertex { agent, .. }
            | Self::PositiveEdge { agent, .. } => agent,
        }
    }

    /// Returns `true` for [`Constraint::PositiveVertex`] and [`Constraint::PositiveEdge`].
    #[must_use]
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            Self::PositiveVertex { .. } | Self::PositiveEdge { .. }
        )
    }

    /// The negative constraints that this positive constraint implies for `other`, an
    /// agent other than the constrained one; nothing for a negative constraint.
    #[must_use]
    pub fn implied_for(&self, other: AgentId) -> Vec<Self> {
        match *self {
            Self::PositiveVertex { vertex, time, .. } => Vec::from([Self::Vertex {
                agent: other,
                vertex,
                time,
            }]),
            Self::PositiveEdge { from, to, time, .. } => Vec::from([
                Self::Vertex {
                    agent: other,
                    vertex: from,
                    time,
                },
                Self::Vertex {
                    agent: other,
                    vertex: to,
                    time: time.saturating_add(1),
                },
                Self::Edge {
                    agent: other,
                    from: to,
                    to: from,
                    time,
                },
            ]),
            Self::Vertex { .. } | Self::Edge { .. } => Vec::new(),
        }
    }

    /// Returns `true` if `path` (with goal-stay) breaks this constraint, whichever agent it
    /// names.
    #[must_use]
    pub fn is_violated_by(&self, path: &[Vertex]) -> bool {
        let at = |time: Time| position_at(path, time);

        match *self {
            Self::Vertex { vertex, time, .. } => at(time) == vertex,
            Self::Edge { from, to, time, .. } => at(time) == from && at(time + 1) == to,
            Self::PositiveVertex { vertex, time, .. } => at(time) != vertex,
            Self::PositiveEdge { from, to, time, .. } => at(time) != from || at(time + 1) != to,
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
    /// Required `(time, vertex)` pairs: at such a time, every other vertex is forbidden.
    /// Two different vertices at one time make the set unsatisfiable.
    required: BTreeSet<(Time, Vertex)>,
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

    /// The set of what `constraints` mean for `agent`: those that name it, and the
    /// negative constraints implied by the positive ones on other agents
    /// ([`Constraint::implied_for`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use mapf_search::{Constraint, ConstraintSet};
    ///
    /// // Agent 0 must be at vertex 4 at time 2, so agent 1 must not.
    /// let positive = Constraint::PositiveVertex { agent: 0, vertex: 4, time: 2 };
    /// let first = ConstraintSet::for_agent(0, &[positive]);
    /// let second = ConstraintSet::for_agent(1, &[positive]);
    ///
    /// assert!(!first.allows_vertex(3, 2));
    /// assert!(first.allows_vertex(4, 2));
    /// assert!(!second.allows_vertex(4, 2));
    /// assert!(second.allows_vertex(3, 2));
    /// ```
    #[must_use]
    pub fn for_agent(agent: AgentId, constraints: &[Constraint]) -> Self {
        let mut set = Self::new();
        for constraint in constraints {
            if constraint.agent() == agent {
                set.insert(constraint);
            } else {
                constraint
                    .implied_for(agent)
                    .iter()
                    .for_each(|implied| set.insert(implied));
            }
        }

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
            Constraint::PositiveVertex { vertex, time, .. } => self.require(vertex, time),
            Constraint::PositiveEdge { from, to, time, .. } => {
                self.require(from, time);
                self.require(to, time.saturating_add(1));
            }
        }
    }

    /// Requires `vertex` at `time`.
    fn require(&mut self, vertex: Vertex, time: Time) {
        self.required.insert((time, vertex));
        self.latest_step = self.latest_step.max(time);
    }

    /// The required `(time, vertex)` pairs ([`Constraint::PositiveVertex`] and both ends of
    /// a [`Constraint::PositiveEdge`]), sorted by time.
    pub(crate) fn landmarks(&self) -> impl Iterator<Item = (Time, Vertex)> + '_ {
        self.required.iter().copied()
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

        let elsewhere = self
            .required
            .range((time, 0)..=(time, Vertex::MAX))
            .any(|&(_, required)| required != vertex);

        !blocked && !elsewhere && !self.vertices.contains(&(time, vertex))
    }

    /// The latest time step any constraint touches; `0` while empty.
    ///
    /// After this step the constraints no longer change: the agent is free except for the
    /// vertices blocked forever ([`ConstraintSet::block_from`]).
    #[must_use]
    pub fn latest_step(&self) -> Time {
        self.latest_step
    }

    /// The latest time `vertex` is forbidden, if ever: by a negative constraint, or by a
    /// positive one that requires another vertex then.
    ///
    /// For the goal vertex, an agent must not finish at or before this time: goal-stay
    /// semantics would put it there. [`Time::MAX`] if the vertex is blocked forever
    /// ([`ConstraintSet::block_from`]).
    #[must_use]
    pub fn blocked_until(&self, vertex: Vertex) -> Option<Time> {
        if self.blocked_from.contains_key(&vertex) {
            return Some(Time::MAX);
        }

        let forbidden = self
            .vertices
            .iter()
            .rev()
            .find(|&&(_, v)| v == vertex)
            .map(|&(time, _)| time);
        let elsewhere = self
            .required
            .iter()
            .rev()
            .find(|&&(_, v)| v != vertex)
            .map(|&(time, _)| time);

        forbidden.max(elsewhere)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for positive constraints.

    use super::*;
    use alloc::vec;

    /// A positive edge requires both ends and implies three negatives for the others.
    #[test]
    fn positive_edges_require_both_ends() {
        let positive = Constraint::PositiveEdge {
            agent: 0,
            from: 1,
            to: 2,
            time: 3,
        };
        let own = ConstraintSet::for_agent(0, &[positive]);

        assert!(own.allows(1, 2, 3));
        assert!(!own.allows(1, 1, 3), "must leave 1");
        assert!(!own.allows_vertex(0, 3));
        assert_eq!(own.latest_step(), 4);
        assert_eq!(own.blocked_until(2), Some(3), "elsewhere at 3");
        assert_eq!(own.blocked_until(9), Some(4));

        let other = ConstraintSet::for_agent(1, &[positive]);
        assert!(!other.allows_vertex(1, 3));
        assert!(!other.allows_vertex(2, 4));
        assert!(!other.allows(2, 1, 3));
        assert!(other.allows(0, 0, 3));
    }

    /// Two different required vertices at one time forbid everything then.
    #[test]
    fn conflicting_requirements_forbid_everything() {
        let at = |vertex| Constraint::PositiveVertex {
            agent: 0,
            vertex,
            time: 2,
        };
        let set = ConstraintSet::for_agent(0, &[at(1), at(2)]);

        assert!((0..4).all(|vertex| !set.allows_vertex(vertex, 2)));
    }

    /// Violation checks follow goal-stay, for negative and positive constraints alike.
    #[test]
    fn violations_follow_goal_stay() {
        let path = vec![0, 1, 2];
        let vertex = |vertex, time| Constraint::Vertex {
            agent: 0,
            vertex,
            time,
        };

        assert!(vertex(2, 7).is_violated_by(&path));
        assert!(!vertex(1, 7).is_violated_by(&path));
        assert!(vertex(1, 7).positive().is_violated_by(&path));
        assert!(!vertex(2, 7).positive().is_violated_by(&path));

        let swap = Constraint::Edge {
            agent: 0,
            from: 1,
            to: 2,
            time: 1,
        };
        assert!(swap.is_violated_by(&path));
        assert!(!swap.positive().is_violated_by(&path));
    }

    /// Disjoint splitting yields the negative and the positive constraint on one agent.
    #[test]
    fn disjoint_split_constrains_one_agent() {
        let conflict = Conflict::Edge {
            agents: (3, 5),
            from: 1,
            to: 2,
            time: 4,
        };
        let [negative, positive] = Constraint::disjoint(conflict, 1);

        assert_eq!(
            negative,
            Constraint::Edge {
                agent: 5,
                from: 2,
                to: 1,
                time: 4
            }
        );
        assert_eq!(positive, negative.positive());
        assert!(positive.is_positive() && !negative.is_positive());
        assert_eq!(negative.implied_for(3), []);
    }
}
