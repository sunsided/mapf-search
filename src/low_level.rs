//! The single-agent low level: space-time A* under a [`ConstraintSet`]
//! ([`space_time_astar`]), run by `heuristic_search::astar` over `(vertex, time)` states.
//!
//! The time-expanded graph is infinite, so successors are cut at a horizon: after the
//! constraints' [`latest_step`](ConstraintSet::latest_step) the agent moves on a fixed
//! graph (minus the vertices blocked forever), and from any vertex that can reach the goal
//! there it does so within `vertex_count - 1` more steps. Every path that exists also exists
//! within the horizon, so an exhausted search proves that the agent has no path.
//!
//! Positive constraints (disjoint splitting) are [`Landmarks`]: the heuristic also bounds
//! the cost through the next required vertex, and prunes states too far from it.
//!
//! With a [`ConflictTable`] ([`plan_avoiding`]), the cost is [`Lexicographic`]: time steps
//! first, then conflicts with the other agents' paths, so the result is a cheapest path with
//! the fewest conflicts among the cheapest (the conflict avoidance table of Sharon et al.
//! 2015). The conflicts an agent causes by staying at its goal after it arrived
//! ([`ConflictTable::after`]) are not counted: every cheapest path arrives at the same
//! goal at the same time, so they would add the same number to each. (ECBS's focal low
//! level, which may take longer paths, does count them.)

mod landmarks;
mod lexicographic;

use alloc::vec::Vec;

use heuristic_search::{Cost, Limits, Successors, astar, heuristic_fn};

use self::landmarks::Landmarks;
use self::lexicographic::Lexicographic;
use crate::conflict_table::ConflictTable;
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
/// distance table to the goal on every call; [`cbs`](crate::cbs()) computes it once per agent.
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
    search(graph, agent, constraints, distances, Plain)
}

/// [`plan`], breaking ties among the cheapest paths towards the fewest conflicts with the
/// paths in `table`.
pub(crate) fn plan_avoiding<G: Graph>(
    graph: &G,
    agent: Agent,
    constraints: &ConstraintSet,
    distances: &[Option<u32>],
    table: &ConflictTable,
) -> Option<Path> {
    search(graph, agent, constraints, distances, Avoiding(table))
}

/// How [`SpaceTime`] prices a step: plainly by time, or by time and then conflicts.
trait Pricing: Copy {
    /// The path cost type.
    type Cost: Cost;

    /// The cost of moving (or waiting, if equal) `from -> to` between `time` and
    /// `time + 1`.
    fn step(self, from: Vertex, to: Vertex, time: Time) -> Self::Cost;

    /// The heuristic for `steps` remaining time steps.
    fn estimate(steps: u32) -> Self::Cost;
}

/// Every step costs 1: the `u32` cost of plain CBS and PBS, the hot path.
#[derive(Debug, Clone, Copy)]
struct Plain;

impl Pricing for Plain {
    type Cost = u32;

    fn step(self, _: Vertex, _: Vertex, _: Time) -> u32 {
        1
    }

    fn estimate(steps: u32) -> u32 {
        steps
    }
}

/// Every step costs 1 time step and its conflicts with the table.
#[derive(Debug, Clone, Copy)]
struct Avoiding<'a>(&'a ConflictTable);

impl Pricing for Avoiding<'_> {
    type Cost = Lexicographic;

    fn step(self, from: Vertex, to: Vertex, time: Time) -> Lexicographic {
        Lexicographic {
            primary: 1,
            secondary: self.0.count(from, to, time),
        }
    }

    fn estimate(steps: u32) -> Lexicographic {
        Lexicographic {
            primary: steps,
            secondary: 0,
        }
    }
}

/// The A* run behind [`plan`] and [`plan_avoiding`].
fn search<G: Graph, P: Pricing>(
    graph: &G,
    agent: Agent,
    constraints: &ConstraintSet,
    distances: &[Option<u32>],
    pricing: P,
) -> Option<Path> {
    if !constraints.allows_vertex(agent.start, 0) {
        return None;
    }

    let space = SpaceTime::new(graph, agent, constraints, pricing);
    let earliest_finish = space.earliest_finish;
    let landmarks = Landmarks::new(graph, constraints, distances, earliest_finish)?;
    let goal = |&(vertex, time): &State| vertex == agent.goal && time >= earliest_finish;
    let heuristic = heuristic_fn(|&(vertex, time): &State| {
        let direct = distances[vertex]?.max(earliest_finish.saturating_sub(time));
        let via = landmarks.estimate(vertex, time)?;

        Some(P::estimate(direct.max(via)))
    });

    let result = astar(&space, (agent.start, 0), &goal, heuristic, Limits::NONE);
    let solution = result.into_solution()?;

    Some(
        solution
            .states
            .into_iter()
            .map(|(vertex, _)| vertex)
            .collect(),
    )
}

/// The time-expanded graph of one agent: `(vertex, time)` states, waits and moves priced by
/// `P`, minus what the constraints forbid, cut at a horizon.
struct SpaceTime<'a, G, P> {
    /// The graph the agent moves on.
    graph: &'a G,
    /// The agent's constraints.
    constraints: &'a ConstraintSet,
    /// What a step costs.
    pricing: P,
    /// The agent may finish at its goal from this time on.
    earliest_finish: Time,
    /// No successor is generated at a later time than this.
    horizon: Time,
}

impl<'a, G: Graph, P> SpaceTime<'a, G, P> {
    /// The graph of `agent` on `graph` under `constraints`, with the horizon described in
    /// the module docs.
    fn new(graph: &'a G, agent: Agent, constraints: &'a ConstraintSet, pricing: P) -> Self {
        let vertices = Time::try_from(graph.vertex_count()).unwrap_or(Time::MAX);

        Self {
            graph,
            constraints,
            pricing,
            earliest_finish: constraints
                .blocked_until(agent.goal)
                .map_or(0, |t| t.saturating_add(1)),
            horizon: constraints.latest_step().saturating_add(vertices),
        }
    }
}

impl<G: Graph, P: Pricing> Successors for SpaceTime<'_, G, P> {
    type State = State;
    type Action = ();
    type Cost = P::Cost;

    fn successors(&self, &(vertex, time): &State, mut emit: impl FnMut((), State, P::Cost)) {
        if time >= self.horizon {
            return;
        }

        let mut step = |next: Vertex| {
            if self.constraints.allows(vertex, next, time) {
                emit((), (next, time + 1), self.pricing.step(vertex, next, time));
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

#[cfg(test)]
mod tests {
    //! Unit tests for the conflict-avoiding low level.

    use super::*;
    use alloc::vec;

    use crate::conflict::conflicts;
    use crate::graph::Grid;

    /// Among the cheapest paths across a free 3x3 grid, the one that avoids another agent
    /// sitting in the way is chosen; the cost stays optimal.
    #[test]
    fn avoids_conflicts_among_the_cheapest_paths() {
        let grid = Grid::new(3, 3);
        let agent = Agent { start: 0, goal: 8 };
        let distances = distances_to(&grid, agent.goal);
        // The other agent sits at the centre and at the top right corner's neighbour.
        let others = [vec![4, 4, 4, 4, 2]];
        let table = ConflictTable::without(&others, 9);

        let path = plan_avoiding(&grid, agent, &ConstraintSet::new(), &distances, &table)
            .expect("reachable");

        assert_eq!(path.len(), 5, "still a cheapest path: {path:?}");
        assert_eq!(conflicts(&[path, others[0].clone()]), []);
    }
}
