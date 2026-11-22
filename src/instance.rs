//! A MAPF problem instance: a [`Graph`] and a list of [`Agent`]s, validated by
//! [`Instance::new`]; and its task-assignment variant, [`TaskInstance`], whose agents each
//! have a set of candidate goals.

use alloc::vec::Vec;
use core::fmt;

use crate::graph::{Graph, Vertex};

/// Index of an agent in [`Instance::agents`].
pub type AgentId = usize;

/// One agent: where it starts and where it must end up and stay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Agent {
    /// The vertex the agent occupies at time `0`.
    pub start: Vertex,
    /// The vertex the agent must reach and then occupy forever (goal-stay semantics).
    pub goal: Vertex,
}

/// A validated MAPF instance.
///
/// Starts are pairwise distinct, goals are pairwise distinct, and every start and goal is a
/// vertex of the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance<G> {
    /// The graph the agents move on.
    graph: G,
    /// The agents, indexed by [`AgentId`].
    agents: Vec<Agent>,
}

impl<G: Graph> Instance<G> {
    /// Creates an instance after checking the agents against the graph.
    ///
    /// # Errors
    ///
    /// - [`InstanceError::OutOfRange`] if a start or goal is not a vertex of `graph`;
    /// - [`InstanceError::SharedStart`] if two agents start at the same vertex;
    /// - [`InstanceError::SharedGoal`] if two agents have the same goal, which goal-stay
    ///   semantics makes unsolvable.
    pub fn new(graph: G, agents: Vec<Agent>) -> Result<Self, InstanceError> {
        let count = graph.vertex_count();
        for (agent, a) in agents.iter().enumerate() {
            if a.start >= count || a.goal >= count {
                return Err(InstanceError::OutOfRange { agent });
            }

            if let Some(other) = agents[..agent].iter().position(|b| b.start == a.start) {
                return Err(InstanceError::SharedStart(other, agent));
            }

            if let Some(other) = agents[..agent].iter().position(|b| b.goal == a.goal) {
                return Err(InstanceError::SharedGoal(other, agent));
            }
        }

        Ok(Self { graph, agents })
    }

    /// The graph the agents move on.
    #[must_use]
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// The agents, indexed by [`AgentId`].
    #[must_use]
    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }
}

/// One agent of a [`TaskInstance`]: where it starts and the goals it may be assigned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskAgent {
    /// The vertex the agent occupies at time `0`.
    pub start: Vertex,
    /// The vertices the agent may end up at, one of which a solution assigns to it and to
    /// no other agent. The order does not matter; an empty list makes the instance
    /// unsolvable.
    pub goals: Vec<Vertex>,
}

/// A validated task-assignment MAPF instance: each agent must end at one of its candidate
/// goals, and no two agents at the same one.
///
/// Starts are pairwise distinct, and every start and candidate goal is a vertex of the
/// graph. Agents may share candidate goals.
///
/// # Examples
///
/// ```
/// use mapf_search::{Grid, TaskAgent, TaskInstance};
///
/// let grid = Grid::new(3, 1);
/// let anywhere = vec![0, 1, 2];
/// let agents = vec![
///     TaskAgent { start: 0, goals: anywhere.clone() },
///     TaskAgent { start: 2, goals: anywhere },
/// ];
///
/// let instance = TaskInstance::new(grid, agents)?;
/// assert_eq!(instance.agents()[1].goals, [0, 1, 2]);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskInstance<G> {
    /// The graph the agents move on.
    graph: G,
    /// The agents, indexed by [`AgentId`].
    agents: Vec<TaskAgent>,
}

impl<G: Graph> TaskInstance<G> {
    /// Creates a task-assignment instance after checking the agents against the graph.
    ///
    /// # Errors
    ///
    /// - [`InstanceError::OutOfRange`] if a start or a candidate goal is not a vertex of
    ///   `graph`;
    /// - [`InstanceError::SharedStart`] if two agents start at the same vertex.
    ///
    /// Never [`InstanceError::SharedGoal`]: shared candidates are allowed, and the solver
    /// assigns distinct goals.
    pub fn new(graph: G, agents: Vec<TaskAgent>) -> Result<Self, InstanceError> {
        let count = graph.vertex_count();
        for (agent, a) in agents.iter().enumerate() {
            if a.start >= count || a.goals.iter().any(|&goal| goal >= count) {
                return Err(InstanceError::OutOfRange { agent });
            }

            if let Some(other) = agents[..agent].iter().position(|b| b.start == a.start) {
                return Err(InstanceError::SharedStart(other, agent));
            }
        }

        Ok(Self { graph, agents })
    }

    /// The graph the agents move on.
    #[must_use]
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// The agents, indexed by [`AgentId`].
    #[must_use]
    pub fn agents(&self) -> &[TaskAgent] {
        &self.agents
    }
}

/// Why [`Instance::new`] or [`TaskInstance::new`] rejected an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceError {
    /// The start or goal of `agent` is not a vertex of the graph.
    OutOfRange {
        /// The offending agent.
        agent: AgentId,
    },
    /// Two agents, the earlier one first, start at the same vertex.
    SharedStart(AgentId, AgentId),
    /// Two agents, the earlier one first, have the same goal.
    SharedGoal(AgentId, AgentId),
}

impl fmt::Display for InstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { agent } => {
                write!(f, "agent {agent} starts or ends outside the graph")
            }
            Self::SharedStart(a, b) => write!(f, "agents {a} and {b} share a start"),
            Self::SharedGoal(a, b) => write!(f, "agents {a} and {b} share a goal"),
        }
    }
}

impl core::error::Error for InstanceError {}
