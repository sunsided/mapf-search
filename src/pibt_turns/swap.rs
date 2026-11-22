//! The swap pattern detector of PIBT*_T: in a corridor, an agent that pushes another one
//! the way the other must come from would push it to a dead end; the two must pass each
//! other instead, so the pusher reverses its preferences and pulls the other into its
//! vertex (the swap of Okumura's PIBT+ and `LaCAM`, on heading-free distances).
//!
//! A swap is *required* if pushing would only drive the other agent down a corridor
//! towards which the pusher itself is heading, and the two want to cross. It is *possible*
//! if, with the roles reversed, the pusher can back off: walking down the corridor from
//! the pusher away from the other agent, a vertex with room to step aside (two or more
//! ways on) comes before a dead end. Vertices with one
//! neighbour that hold an agent at its goal count as walls.

use super::Run;
use crate::graph::{OrientedGraph, Vertex};
use crate::instance::AgentId;

impl<G: OrientedGraph> Run<'_, G> {
    /// The agent `agent` should swap with given its preferences `candidates`, best first:
    /// the occupant of its first choice, or a neighbour that wants to pass it; `None` if no
    /// swap is required and possible. A partner must have an edge back to `agent`'s vertex,
    /// which it is pulled into; on a directed graph it may not.
    pub(super) fn swap_partner(&self, agent: AgentId, candidates: &[Vertex]) -> Option<AgentId> {
        let here = self.here[agent];
        let &first = candidates.first()?;
        if first == here {
            return None;
        }

        // `agent` would push the occupant of `first` down a corridor; the swap needs room
        // for `agent` itself to back off from `first` once it pulls the other through.
        if let Some(other) = self.occupant[first] {
            let undecided = self.claim[other].is_none() && self.decided[other].is_none();
            if undecided
                && self.has_edge(first, here)
                && self.swap_required(agent, other, here, first)
                && self.room_to_back_off(first, here)
            {
                return Some(other);
            }
        }

        let mut neighbors = alloc::vec::Vec::new();
        self.instance
            .graph()
            .neighbors(here, |next| neighbors.push(next));
        // The clear operation of push and swap: a neighbour behind `agent` wants through. It
        // is judged as if it stood at `here` and pushed `agent` on into `first`, not from its
        // own vertex, as in PIBT+ and LaCAM.
        neighbors.into_iter().find_map(|vertex| {
            let other = self.occupant[vertex]?;
            let wants_through = vertex != first
                && self.has_edge(vertex, here)
                && self.swap_required(other, agent, here, first)
                && self.room_to_back_off(first, here);

            wants_through.then_some(other)
        })
    }

    /// Returns `true` if the graph has an edge `from -> to`.
    fn has_edge(&self, from: Vertex, to: Vertex) -> bool {
        let mut found = false;
        self.instance
            .graph()
            .neighbors(from, |next| found |= next == to);

        found
    }

    /// Returns `true` if `pusher` at `pusher_at` pushing `puller` at `puller_at` would only
    /// drive it down a corridor the pusher is heading along, while both want to cross.
    fn swap_required(
        &self,
        pusher: AgentId,
        puller: AgentId,
        pusher_at: Vertex,
        puller_at: Vertex,
    ) -> bool {
        let least =
            |agent: AgentId, vertex: Vertex| self.costs.least(agent, vertex).unwrap_or(u32::MAX);
        let (mut behind, mut ahead) = (pusher_at, puller_at);

        for _ in 0..self.instance.graph().vertex_count() {
            if least(pusher, ahead) >= least(pusher, behind) {
                break;
            }

            match self.ways_on(ahead, behind) {
                Ways::Open => return false,
                Ways::DeadEnd => break,
                Ways::One(next) => (behind, ahead) = (ahead, next),
            }
        }

        let crosses = least(puller, behind) < least(puller, ahead);
        let pusher_done_or_closer =
            least(pusher, behind) == 0 || least(pusher, ahead) < least(pusher, behind);

        crosses && pusher_done_or_closer
    }

    /// Returns `true` if walking down the corridor from `start`, away from `away_from`,
    /// reaches a vertex with room to step aside before a dead end or a return to
    /// `away_from`. The swap rules call it with the pusher's vertex as `start`: the roles
    /// of the required-test are reversed (the paper's stage 2).
    fn room_to_back_off(&self, away_from: Vertex, start: Vertex) -> bool {
        let (mut behind, mut ahead) = (away_from, start);

        for _ in 0..self.instance.graph().vertex_count() {
            if ahead == away_from {
                return false;
            }

            match self.ways_on(ahead, behind) {
                Ways::Open => return true,
                Ways::DeadEnd => return false,
                Ways::One(next) => (behind, ahead) = (ahead, next),
            }
        }

        false
    }

    /// The ways on from `vertex` other than back to `from` and into dead ends that hold an
    /// agent at its goal.
    fn ways_on(&self, vertex: Vertex, from: Vertex) -> Ways {
        let graph = self.instance.graph();
        let mut count = 0;
        let mut last = None;
        graph.neighbors(vertex, |next| {
            if next != from && !self.is_parked_dead_end(next) {
                count += 1;
                last = Some(next);
            }
        });

        match (count, last) {
            (0, _) | (_, None) => Ways::DeadEnd,
            (1, Some(next)) => Ways::One(next),
            _ => Ways::Open,
        }
    }

    /// Returns `true` if `vertex` has a single neighbour and holds an agent at its goal.
    fn is_parked_dead_end(&self, vertex: Vertex) -> bool {
        let mut degree = 0;
        self.instance.graph().neighbors(vertex, |_| degree += 1);

        degree == 1
            && self.occupant[vertex]
                .is_some_and(|agent| self.instance.agents()[agent].goal == vertex)
    }
}

/// How a corridor goes on from a vertex.
enum Ways {
    /// Nowhere.
    DeadEnd,
    /// Exactly one way, to this vertex.
    One(Vertex),
    /// Two or more ways: room to step aside.
    Open,
}
