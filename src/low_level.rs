//! The single-agent low level: space-time A* under a [`ConstraintSet`]
//! ([`space_time_astar`]), run by `heuristic_search::astar` over `(vertex, time)` states.
//!
//! The time-expanded graph is infinite, so successors are cut at a horizon: after the
//! constraints' [`latest_step`](ConstraintSet::latest_step) the agent is free, and from any
//! vertex that can reach the goal it does so within `vertex_count - 1` more steps. Every
//! path that exists also exists within the horizon, so an exhausted search proves that the
//! agent has no path.

use alloc::vec::Vec;

use heuristic_search::{Limits, Successors, astar};

use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex, distances_to};
use crate::instance::Agent;
use crate::path::{Path, Time};

/// A state of the time-expanded graph: a vertex and the time the agent is there.
type State = (Vertex, Time);

/// The cheapest path of `agent` on `graph` that satisfies `constraints` and ends at the
/// goal for good (goal-stay: no later constraint forbids the goal).
///
/// Every step, move or wait, costs 1. Returns `None` if no such path exists. Computes a
/// distance table to the goal on every call; [`cbs`](crate::cbs) computes it once per agent.
///
/// # Panics
///
/// Panics if the start or goal is not a vertex of `graph`.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, ConstraintSet, Constraint, Grid, space_time_astar};
///
/// let grid = Grid::new(3, 1);
/// let agent = Agent { start: 0, goal: 2 };
/// let mut constraints = ConstraintSet::new();
/// constraints.insert(&Constraint::Vertex { agent: 0, vertex: 1, time: 1 });
///
/// let path = space_time_astar(&grid, agent, &constraints).expect("waiting helps");
/// assert_eq!(path, [0, 0, 1, 2]);
/// ```
#[must_use]
pub fn space_time_astar<G: Graph>(
    graph: &G,
    agent: Agent,
    constraints: &ConstraintSet,
) -> Option<Path> {
    let distances = distances_to(graph, agent.goal);

    plan(graph, agent, constraints, &distances)
}

/// [`space_time_astar`] with a precomputed table of distances to the agent's goal, as
/// returned by [`distances_to`].
pub(crate) fn plan<G: Graph>(
    graph: &G,
    agent: Agent,
    constraints: &ConstraintSet,
    distances: &[Option<u32>],
) -> Option<Path> {
    if !constraints.allows_vertex(agent.start, 0) {
        return None;
    }

    let space = SpaceTime::new(graph, constraints);
    let earliest_finish = constraints.blocked_until(agent.goal).map_or(0, |t| t + 1);
    let goal = |&(vertex, time): &State| vertex == agent.goal && time >= earliest_finish;
    let heuristic = |&(vertex, time): &State| {
        let distance = distances[vertex]?;

        Some(distance.max(earliest_finish.saturating_sub(time)))
    };

    let result = astar(&space, (agent.start, 0), &goal, &heuristic, Limits::NONE);
    let solution = result.into_solution()?;

    Some(
        solution
            .states
            .into_iter()
            .map(|(vertex, _)| vertex)
            .collect(),
    )
}

/// The time-expanded graph of one agent: `(vertex, time)` states, waits and moves at cost
/// 1, minus what the constraints forbid, cut at a horizon.
struct SpaceTime<'a, G> {
    /// The graph the agent moves on.
    graph: &'a G,
    /// The agent's constraints.
    constraints: &'a ConstraintSet,
    /// No successor is generated at a later time than this.
    horizon: Time,
}

impl<'a, G: Graph> SpaceTime<'a, G> {
    /// Wraps `graph` with `constraints`, with the horizon described in the module docs.
    fn new(graph: &'a G, constraints: &'a ConstraintSet) -> Self {
        let vertices = Time::try_from(graph.vertex_count()).unwrap_or(Time::MAX);
        let horizon = constraints.latest_step().saturating_add(vertices);

        Self {
            graph,
            constraints,
            horizon,
        }
    }
}

impl<G: Graph> Successors for SpaceTime<'_, G> {
    type State = State;
    type Action = ();
    type Cost = u32;

    fn successors(&self, &(vertex, time): &State, mut emit: impl FnMut((), State, u32)) {
        if time >= self.horizon {
            return;
        }

        let mut step = |next: Vertex| {
            if self.constraints.allows(vertex, next, time) {
                emit((), (next, time + 1), 1);
            }
        };

        step(vertex);
        self.graph.neighbors(vertex, step);
    }
}

/// Distance tables to each agent's goal, indexed by agent.
pub(crate) fn distance_tables<G: Graph>(graph: &G, agents: &[Agent]) -> Vec<Vec<Option<u32>>> {
    agents
        .iter()
        .map(|agent| distances_to(graph, agent.goal))
        .collect()
}
