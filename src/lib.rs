//! # Multi-agent pathfinding
//!
//! Multi-agent pathfinding: CBS, ECBS, CBS-TA, ICTS, PBS, M* and MAPF-LNS2.
//!
//! ## Planned scope
//!
//! - Conflict-based: CBS, ECBS, CBS-TA
//! - Cost-vector search: ICTS
//! - Priority-based: PBS
//! - Subdimensional: M*
//! - Local repair: MAPF-LNS2
//! - Partial expansion: EPEA*
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/mapf-search>

#![no_std]

/// A conflict between the paths of two agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Conflict<V> {
    /// Both agents occupy `vertex` at `time`.
    Vertex {
        /// The two agents.
        agents: (usize, usize),
        /// The shared vertex.
        vertex: V,
        /// The time step.
        time: u32,
    },
    /// The agents swap `from` and `to` between `time` and `time + 1`.
    Edge {
        /// The two agents.
        agents: (usize, usize),
        /// The vertex the first agent leaves.
        from: V,
        /// The vertex the first agent enters.
        to: V,
        /// The time step at which the swap starts.
        time: u32,
    },
}
