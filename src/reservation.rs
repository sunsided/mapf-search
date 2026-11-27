//! Reservation tables: what another agent's path holds in space and time, as constraints
//! on one agent ([`reserve`]).
//!
//! PBS and cooperative A* plan an agent under the reservations of every agent above it,
//! for good. WHCA* reserves only a window: past its end, the other agents' positions are
//! not known yet, so nothing is reserved there.

use crate::constraint::{Constraint, ConstraintSet};
use crate::graph::Vertex;
use crate::instance::AgentId;
use crate::path::{Reservation, Time, reservations};

/// Adds to `constraints` (for `agent`) what keeps it out of the way of `path`: its vertex
/// at every time, the reverse of each of its moves, and its last vertex from its arrival
/// on.
///
/// With `until` set, nothing after time `until` is reserved: the last vertex is held only
/// through `until`, and visits and moves later than that are dropped. With `None`, the
/// last vertex is held for good, as goal-stay semantics demands of a finished path.
///
/// # Panics
///
/// Panics if `path` is empty.
pub(crate) fn reserve(
    constraints: &mut ConstraintSet,
    agent: AgentId,
    path: &[Vertex],
    until: Option<Time>,
) {
    let within = |time: Time| until.is_none_or(|until| time <= until);

    for reservation in reservations(path) {
        match reservation {
            Reservation::Visit { vertex, time } if within(time) => {
                constraints.insert(&Constraint::Vertex {
                    agent,
                    vertex,
                    time,
                });
            }
            Reservation::Move { from, to, time } if within(time.saturating_add(1)) => {
                constraints.insert(&Constraint::Edge {
                    agent,
                    from: to,
                    to: from,
                    time,
                });
            }
            Reservation::Park { vertex, time } => match until {
                None => constraints.block_from(vertex, time),
                Some(until) => {
                    for time in time..=until {
                        constraints.insert(&Constraint::Vertex {
                            agent,
                            vertex,
                            time,
                        });
                    }
                }
            },
            Reservation::Visit { .. } | Reservation::Move { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for reservations.

    use super::*;

    /// Unbounded, the last vertex is held for good.
    #[test]
    fn unbounded_parks_for_good() {
        let mut constraints = ConstraintSet::new();
        reserve(&mut constraints, 1, &[0, 1], None);

        assert!(!constraints.allows_vertex(0, 0));
        assert!(!constraints.allows_vertex(1, 1));
        assert!(!constraints.allows_vertex(1, 100));
        assert!(!constraints.allows(1, 0, 0), "the reverse move swaps");
    }

    /// Bounded, nothing after `until` is held: not the park, nor later visits or moves.
    #[test]
    fn bounded_holds_through_until() {
        let mut constraints = ConstraintSet::new();
        reserve(&mut constraints, 1, &[0, 1, 2, 3], Some(2));

        assert!(!constraints.allows_vertex(2, 2));
        assert!(constraints.allows_vertex(3, 3), "past the window");
        assert!(
            constraints.allows(3, 2, 2),
            "the move into time 3 is past the window"
        );
        assert!(!constraints.allows(2, 1, 1));

        let mut parked = ConstraintSet::new();
        reserve(&mut parked, 1, &[0, 1], Some(3));
        assert!(!parked.allows_vertex(1, 3));
        assert!(parked.allows_vertex(1, 4));
    }
}
