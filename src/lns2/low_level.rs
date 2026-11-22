//! The test oracle of SIPPS ([`fewest_collisions`]): a path for one agent with exactly the
//! fewest collisions with a fixed set of other paths, then the earliest arrival. MAPF-LNS2
//! used it as its low level before SIPPS; it is several times slower per call on crowded
//! grids, but its collision count is exact.
//!
//! This is the problem the paper calls pathfinding with mixed dynamic obstacles, with soft
//! obstacles only. It is solved by space-time A* with search-core's [`Lexicographic`]
//! cost, collisions first, then time steps, the variant the paper compares SIPPS against.
//! Collisions count with goal-stay semantics: entering a vertex another agent occupies or
//! has parked at, swapping with another agent, and, once the agent finishes at its goal,
//! every later visit of another agent there ([`ConflictTable::after`]).
//!
//! The time-expanded graph is cut at a horizon: once every other path has ended, nothing
//! moves any more, and a simple path reaches the goal within `vertex_count - 1` more steps
//! with no more collisions than any longer one.

use heuristic_search::{Limits, Successors, a_star, heuristic_fn};
use search_core::Lexicographic;

use crate::conflict_table::ConflictTable;
use crate::graph::{Graph, Vertex};
use crate::instance::Agent;
use crate::path::{Path, Time, trim_trailing_waits};

/// A search state: a vertex, the time the agent is there, and whether it has finished at
/// its goal for good.
type State = (Vertex, Time, bool);

/// The cost of a path: collisions first, then time steps.
type Cost = Lexicographic<u32, u32>;

/// The path of `agent` on `graph` with the fewest collisions with the paths in `table`, the
/// earliest arrival among those; `None` if the goal is unreachable. `distances` holds the
/// distances to the agent's goal, and `last` is the time after which no path of the table
/// moves any more (the length of the longest one).
pub(super) fn fewest_collisions<G: Graph>(
    graph: &G,
    agent: Agent,
    distances: &[Option<u32>],
    table: &ConflictTable,
    last: Time,
) -> Option<Path> {
    distances[agent.start]?;

    let vertices = Time::try_from(graph.vertex_count()).unwrap_or(Time::MAX);
    let space = Soft {
        graph,
        goal: agent.goal,
        table,
        horizon: last.saturating_add(vertices),
    };
    let goal = |&(_, _, finished): &State| finished;
    let heuristic = heuristic_fn(|&(vertex, _, finished): &State| {
        let steps = if finished { 0 } else { distances[vertex]? };

        Some(Lexicographic {
            primary: 0,
            secondary: steps,
        })
    });

    let result = a_star(
        &space,
        (agent.start, 0, false),
        &goal,
        heuristic,
        Limits::NONE,
    );
    let solution = result.into_solution()?;
    let mut path: Path = solution
        .states
        .into_iter()
        .filter(|&(_, _, finished)| !finished)
        .map(|(vertex, _, _)| vertex)
        .collect();
    trim_trailing_waits(&mut path);

    Some(path)
}

/// The time-expanded graph of one agent among soft obstacles, cut at a horizon.
struct Soft<'a, G> {
    /// The graph the agent moves on.
    graph: &'a G,
    /// The agent's goal.
    goal: Vertex,
    /// The other agents' paths.
    table: &'a ConflictTable,
    /// No successor is generated at a later time than this.
    horizon: Time,
}

impl<G: Graph> Successors for Soft<'_, G> {
    type State = State;
    type Action = ();
    type Cost = Cost;

    /// Waits and moves, each costing a time step and its collisions; at the goal, also
    /// finishing, which costs the collisions of staying there for good.
    fn successors(&self, &(vertex, time, finished): &State, mut emit: impl FnMut((), State, Cost)) {
        if finished {
            return;
        }

        if vertex == self.goal {
            let cost = Lexicographic {
                primary: self.table.after(vertex, time),
                secondary: 0,
            };
            emit((), (vertex, time, true), cost);
        }

        if time >= self.horizon {
            return;
        }

        let mut step = |next: Vertex| {
            let cost = Lexicographic {
                primary: self.table.count(vertex, next, time),
                secondary: 1,
            };
            emit((), (next, time + 1, false), cost);
        };
        step(vertex);
        self.graph.neighbors(vertex, step);
    }
}

#[cfg(test)]
mod tests {
    //! The fewest-collision paths on small grids.

    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::graph::{Grid, distances_to};

    /// Plans `agent` on `grid` around `others`.
    fn plan(grid: &Grid, agent: Agent, others: &[Path]) -> Option<Path> {
        let table = ConflictTable::of(others);
        let last = others.iter().map(Vec::len).max().unwrap_or(0);
        let last = Time::try_from(last).expect("short paths");

        fewest_collisions(grid, agent, &distances_to(grid, agent.goal), &table, last)
    }

    /// The only way runs through a parked agent: the path takes it, once, and no longer
    /// than needed.
    #[test]
    fn passes_a_parked_agent_when_it_must() {
        // `0 1 2` over `3 @ 5`, the other agent parked at 1.
        let grid = Grid::parse(&["...", ".@."]);
        let path = plan(&grid, Agent { start: 3, goal: 5 }, &[vec![1]]).expect("reachable");

        assert_eq!(path, [3, 0, 1, 2, 5]);
    }

    /// A free detour beats a shorter path through a parked agent.
    #[test]
    fn detours_around_a_parked_agent() {
        // `0 1 2` over `3 4 5`, the other agent parked at 1.
        let grid = Grid::new(3, 2);
        let path = plan(&grid, Agent { start: 0, goal: 2 }, &[vec![1]]).expect("reachable");

        assert_eq!(path, [0, 3, 4, 5, 2]);
    }

    /// Waiting out a passing agent avoids the collision.
    #[test]
    fn waits_for_a_passing_agent() {
        let grid = Grid::new(3, 2);
        // `0 1 2` over `3 4 5`; the other agent crosses 1 at time 1 and parks at 3.
        let other = vec![2, 1, 0, 3];
        let path = plan(
            &grid,
            Agent { start: 4, goal: 1 },
            core::slice::from_ref(&other),
        )
        .expect("reachable");

        assert_eq!(
            crate::conflict::first_conflict(&[path.clone(), other]),
            None
        );
        assert_eq!(path.first(), Some(&4));
        assert_eq!(path.last(), Some(&1));
    }

    /// A goal that another agent passes later is a collision even if reached first; with
    /// no other way, the path still ends there and the collision is counted once.
    #[test]
    fn finishing_counts_later_visits() {
        let grid = Grid::new(3, 1);
        // The other agent runs 2 -> 1 -> 0 over the agent's goal 1.
        let path = plan(&grid, Agent { start: 0, goal: 1 }, &[vec![2, 2, 1, 0]]);

        assert!(path.is_some_and(|path| path.last() == Some(&1)));
    }

    /// An unreachable goal has no path.
    #[test]
    fn unreachable_goal_has_no_path() {
        let grid = Grid::parse(&[".@."]);

        assert_eq!(plan(&grid, Agent { start: 0, goal: 2 }, &[]), None);
    }
}
