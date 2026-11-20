//! A MAPF problem instance: a [`Graph`] and a list of [`Agent`]s, validated by
//! [`Instance::new`].

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

/// Why [`Instance::new`] rejected an instance.
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
