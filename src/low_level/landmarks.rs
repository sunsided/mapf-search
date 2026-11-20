//! Landmarks ([`Landmarks`]): the vertices a positive constraint requires at given times,
//! and the admissible estimate they give the low level.
//!
//! An agent that must be at `v_k` at `t_k` cannot be farther than `t_k - t` from `v_k` at
//! time `t`. If its path must go on after `t_k` (`v_k` is not the goal, or something later
//! forces it away again), it also needs at least `t_k - t` more steps plus what the rest of
//! the chain needs. A landmark at the goal with nothing after it is different: goal-stay
//! lets the agent arrive early and wait, so it only prunes. The estimate is consistent: one
//! step lowers `t_k - t` by exactly 1.

use alloc::vec::Vec;

use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex, distances_to};
use crate::path::Time;

/// One required `(time, vertex)` with what it implies.
#[derive(Debug, Clone)]
struct Landmark {
    /// The time the vertex is required.
    time: Time,
    /// Distances to the vertex, as returned by [`distances_to`].
    distances: Vec<Option<u32>>,
    /// A lower bound on the cost from the landmark at its time to the final arrival. Positive
    /// if the path must go on after the landmark; `0` only for a landmark at the goal that
    /// the agent may meet by arriving early and staying.
    rest: u32,
}

/// The landmarks of one agent's constraints, sorted by time.
#[derive(Debug, Clone, Default)]
pub(super) struct Landmarks {
    /// The landmarks; empty without positive constraints.
    chain: Vec<Landmark>,
}

impl Landmarks {
    /// The landmarks of `constraints` for an agent with distance table `to_goal` that may
    /// finish at `earliest_finish`; `None` if they cannot all be met, so the agent has no
    /// path.
    pub(super) fn new<G: Graph>(
        graph: &G,
        constraints: &ConstraintSet,
        to_goal: &[Option<u32>],
        earliest_finish: Time,
    ) -> Option<Self> {
        let goal = to_goal.iter().position(|&d| d == Some(0));
        let required: Vec<(Time, Vertex)> = constraints.landmarks().collect();
        let mut chain: Vec<Landmark> = Vec::with_capacity(required.len());
        let mut later: Option<(Time, Vertex)> = None;
        for &(time, vertex) in required.iter().rev() {
            let distances = distances_to(graph, vertex);
            let mut rest = to_goal[vertex]?.max(earliest_finish.saturating_sub(time));
            if let Some((next_time, next_vertex)) = later {
                let gap = next_time - time;
                if distances_to(graph, next_vertex)[vertex]? > gap {
                    return None;
                }

                let next_rest = chain.last().expect("the later landmark").rest;
                if next_rest > 0 {
                    rest = rest.max(gap + next_rest);
                }
            }

            debug_assert!(
                rest > 0 || Some(vertex) == goal,
                "only the goal ends a path"
            );
            chain.push(Landmark {
                time,
                distances,
                rest,
            });
            later = Some((time, vertex));
        }

        chain.reverse();

        Some(Self { chain })
    }

    /// A lower bound on the remaining cost at `vertex` at `time` from the next landmark at
    /// or after `time`; `Some(0)` past the last one, `None` if the next one is out of reach.
    pub(super) fn estimate(&self, vertex: Vertex, time: Time) -> Option<u32> {
        let next = self.chain.partition_point(|landmark| landmark.time < time);
        let Some(landmark) = self.chain.get(next) else {
            return Some(0);
        };

        let slack = landmark.time - time;
        if landmark.distances[vertex]? > slack {
            return None;
        }

        // A landmark at the goal with nothing after it may be met by arriving early and
        // staying, so only one with a positive rest bounds the arrival from below.
        Some(if landmark.rest > 0 {
            slack + landmark.rest
        } else {
            0
        })
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the landmark estimate.

    use super::*;
    use crate::constraint::Constraint;
    use crate::graph::Grid;

    /// On a 5x1 corridor with the goal at 4, being required at 0 at time 4 forces a detour.
    #[test]
    fn estimates_through_a_landmark() {
        let grid = Grid::new(5, 1);
        let mut constraints = ConstraintSet::new();
        constraints.insert(&Constraint::PositiveVertex {
            agent: 0,
            vertex: 0,
            time: 4,
        });
        let to_goal = distances_to(&grid, 4);
        let landmarks = Landmarks::new(&grid, &constraints, &to_goal, 0).expect("feasible");

        assert_eq!(landmarks.estimate(2, 0), Some(4 + 4));
        assert_eq!(landmarks.estimate(4, 1), None, "too far from 0");
        assert_eq!(landmarks.estimate(0, 4), Some(4));
        assert_eq!(landmarks.estimate(3, 5), Some(0));
    }

    /// A landmark at the goal may be met by arriving early, so it bounds nothing; one that
    /// forces a later departure does.
    #[test]
    fn goal_landmarks_only_prune() {
        let grid = Grid::new(5, 1);
        let mut constraints = ConstraintSet::new();
        constraints.insert(&Constraint::PositiveVertex {
            agent: 0,
            vertex: 4,
            time: 6,
        });
        let to_goal = distances_to(&grid, 4);
        let landmarks = Landmarks::new(&grid, &constraints, &to_goal, 0).expect("feasible");

        assert_eq!(landmarks.estimate(2, 0), Some(0));
        assert_eq!(landmarks.estimate(0, 3), None, "too far from the goal");

        constraints.insert(&Constraint::PositiveVertex {
            agent: 0,
            vertex: 3,
            time: 7,
        });
        let landmarks = Landmarks::new(&grid, &constraints, &to_goal, 8).expect("feasible");
        assert_eq!(landmarks.estimate(2, 0), Some(6 + 1 + 1));
    }

    /// Two landmarks too far apart in time make the constraints unsatisfiable.
    #[test]
    fn rejects_unreachable_chains() {
        let grid = Grid::new(5, 1);
        let mut constraints = ConstraintSet::new();
        for (vertex, time) in [(0, 1), (4, 2)] {
            constraints.insert(&Constraint::PositiveVertex {
                agent: 0,
                vertex,
                time,
            });
        }

        assert!(Landmarks::new(&grid, &constraints, &distances_to(&grid, 4), 0).is_none());
    }
}
