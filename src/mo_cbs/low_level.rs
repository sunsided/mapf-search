//! The multi-objective low level of MO-CBS: every cost-unique Pareto-optimal path of one
//! agent under a [`ConstraintSet`], found by NAMOA*dr or BOA* from `multiobjective-search`
//! on the agent's time-expanded graph ([`SpaceTime`]): NAMOA*dr-st and BOA*-st in the
//! paper's terms.
//!
//! States are `(vertex, time)`. A step from `(u, t)` to `(v, t + 1)` costs
//! `costs(u, v)`, a wait `costs(u, u)`. A state at the goal from the time no constraint
//! forbids the goal any more on is a goal: the agent stays there for free (goal-stay), so
//! such labels end their path and are not expanded.
//!
//! The time-expanded graph is infinite, so successors are cut at the horizon
//! `latest_step + vertex_count`, as in the single-objective low level. This loses no
//! Pareto-optimal cost vector: after the last constrained step `L` the agent moves on a
//! fixed graph, where removing a cycle or a wait from a path changes its cost by a
//! non-negative vector, so every cost vector of the Pareto set is reached by a path whose
//! part after `L` is simple, at most `vertex_count - 1` steps long.
//!
//! The heuristic is exact per objective on the graph without constraints and time: one
//! backward Dijkstra per objective from the goal ([`Heuristic::new`]). It ignores waits,
//! so it is consistent, as both NAMOA*dr and BOA* need.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use multiobjective_search::{Graph as MoGraph, Output, boa_star, heuristic_fn, namoa_star_dr};

use super::LowLevel;
use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex, reverse_adjacency};
use crate::instance::Agent;
use crate::path::{Path, Time};

/// A state of the time-expanded graph: a vertex and the time the agent is there.
type State = (Vertex, Time);

/// Per-objective lower bounds on the cost from every vertex to one goal: exact for each
/// objective alone, on the graph without constraints. Computed once per agent.
pub(super) struct Heuristic<const N: usize> {
    /// `bounds[v]` holds, per objective, the cheapest cost of a path from `v` to the goal;
    /// `None` if the goal is unreachable from `v`.
    bounds: Vec<Option<[u32; N]>>,
}

impl<const N: usize> Heuristic<N> {
    /// The bounds towards `goal`: one Dijkstra per objective along reversed edges, in
    /// `O(N (V + E) log V)` time.
    pub(super) fn new<G: Graph>(
        graph: &G,
        costs: &impl Fn(Vertex, Vertex) -> [u32; N],
        goal: Vertex,
    ) -> Self {
        let reverse = reverse_adjacency(graph);
        let per_objective: Vec<Vec<Option<u32>>> = (0..N)
            .map(|objective| dijkstra_to(&reverse, |u, v| costs(u, v)[objective], goal))
            .collect();
        let bounds = (0..graph.vertex_count())
            .map(|vertex| {
                let mut bound = [0; N];
                for (slot, distances) in bound.iter_mut().zip(&per_objective) {
                    *slot = distances[vertex]?;
                }

                Some(bound)
            })
            .collect();

        Self { bounds }
    }

    /// The bound at `vertex`; `None` if the goal is unreachable from it.
    fn at(&self, vertex: Vertex) -> Option<[u32; N]> {
        self.bounds[vertex]
    }
}

/// The cheapest `cost`-weighted distance from every vertex to `target`, by Dijkstra over
/// the in-neighbour lists `reverse`; `None` where `target` is unreachable.
fn dijkstra_to(
    reverse: &[Vec<Vertex>],
    cost: impl Fn(Vertex, Vertex) -> u32,
    target: Vertex,
) -> Vec<Option<u32>> {
    let mut distance = vec![None; reverse.len()];
    let mut queue = BinaryHeap::from([Reverse((0_u32, target))]);

    while let Some(Reverse((d, vertex))) = queue.pop() {
        if distance[vertex].is_some() {
            continue;
        }

        distance[vertex] = Some(d);
        for &previous in &reverse[vertex] {
            if distance[previous].is_none() {
                queue.push(Reverse((d + cost(previous, vertex), previous)));
            }
        }
    }

    distance
}

/// Every cost-unique Pareto-optimal path of `agent` under `constraints`, each with its
/// cost vector, sorted lexicographically by cost. Empty if the agent has no path.
///
/// Uses BOA* for two objectives unless `low_level` is [`LowLevel::NamoaDr`], else NAMOA*dr.
pub(super) fn pareto_paths<G: Graph, const N: usize>(
    graph: &G,
    costs: &impl Fn(Vertex, Vertex) -> [u32; N],
    agent: Agent,
    constraints: &ConstraintSet,
    heuristic: &Heuristic<N>,
    low_level: LowLevel,
) -> Vec<(Path, [u32; N])> {
    if !constraints.allows_vertex(agent.start, 0) {
        return Vec::new();
    }

    let space = SpaceTime::new(graph, costs, agent, constraints);
    let earliest_finish = space.earliest_finish;
    let is_goal = |&(vertex, time): &State| vertex == agent.goal && time >= earliest_finish;
    let start = (agent.start, 0);

    let paths: Vec<(Vec<State>, [u32; N])> = if N == 2 && low_level == LowLevel::Auto {
        let bound = |&(vertex, _): &State| heuristic.at(vertex).map(|b| narrow::<N, 2>(&b));
        let output = boa_star(&Bi(&space), heuristic_fn(bound), start, is_goal);

        unpack(output, |cost| narrow::<2, N>(&cost))
    } else {
        let bound = |&(vertex, _): &State| heuristic.at(vertex);
        let output = namoa_star_dr(&space, heuristic_fn(bound), start, is_goal);

        unpack(output, |cost| cost)
    };

    paths
        .into_iter()
        .map(|(states, cost)| (states.into_iter().map(|(v, _)| v).collect(), cost))
        .collect()
}

/// The paths and widened costs of a search output.
fn unpack<const M: usize, const N: usize>(
    output: Output<State, [u32; M]>,
    widen: impl Fn([u32; M]) -> [u32; N],
) -> Vec<(Vec<State>, [u32; N])> {
    output
        .solutions
        .into_iter()
        .map(|solution| (solution.path, widen(solution.cost)))
        .collect()
}

/// `vector` as an array of length `M`; only called with `M == N`.
fn narrow<const N: usize, const M: usize>(vector: &[u32; N]) -> [u32; M] {
    <[u32; M]>::try_from(&vector[..]).expect("only called when the lengths agree")
}

/// The time-expanded graph of one agent: `(vertex, time)` states, waits and moves priced
/// by the cost function, minus what the constraints forbid, cut at the horizon.
struct SpaceTime<'a, G, F> {
    /// The graph the agent moves on.
    graph: &'a G,
    /// The cost vector of a move `(from, to)`, or of a wait when `from == to`.
    costs: &'a F,
    /// The agent's constraints.
    constraints: &'a ConstraintSet,
    /// The agent may finish at its goal from this time on.
    earliest_finish: Time,
    /// No successor is generated at a later time than this.
    horizon: Time,
}

impl<'a, G: Graph, F> SpaceTime<'a, G, F> {
    /// The graph of `agent` on `graph` under `constraints`, with the horizon of the module
    /// docs.
    fn new(graph: &'a G, costs: &'a F, agent: Agent, constraints: &'a ConstraintSet) -> Self {
        let vertices = Time::try_from(graph.vertex_count()).unwrap_or(Time::MAX);

        Self {
            graph,
            costs,
            constraints,
            earliest_finish: constraints
                .blocked_until(agent.goal)
                .map_or(0, |t| t.saturating_add(1)),
            horizon: constraints.latest_step().saturating_add(vertices),
        }
    }
}

impl<G: Graph, F: Fn(Vertex, Vertex) -> [u32; N], const N: usize> MoGraph for SpaceTime<'_, G, F> {
    type State = State;
    type Weight = [u32; N];

    fn successors(&self, &(vertex, time): &State, out: &mut Vec<(State, [u32; N])>) {
        if time >= self.horizon {
            return;
        }

        let mut step = |next: Vertex| {
            if self.constraints.allows(vertex, next, time) {
                out.push(((next, time + 1), (self.costs)(vertex, next)));
            }
        };

        step(vertex);
        self.graph.neighbors(vertex, step);
    }
}

/// A [`SpaceTime`] graph with `N == 2` seen with two-element weights, for BOA*.
struct Bi<'a, T>(&'a T);

impl<T: MoGraph<State = State>> MoGraph for Bi<'_, T>
where
    T::Weight: AsRef<[u32]>,
{
    type State = State;
    type Weight = [u32; 2];

    fn successors(&self, state: &State, out: &mut Vec<(State, [u32; 2])>) {
        let mut wide = Vec::new();
        self.0.successors(state, &mut wide);
        out.extend(wide.into_iter().map(|(next, cost)| {
            let cost = <[u32; 2]>::try_from(cost.as_ref()).expect("BOA* runs only for N == 2");

            (next, cost)
        }));
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the multi-objective low level.

    use super::*;
    use crate::constraint::Constraint;
    use crate::graph::AdjacencyList;

    /// A diamond `0 -> {1, 2} -> 3` trading the two objectives; waits cost `[1, 1]`.
    fn diamond() -> (AdjacencyList, impl Fn(Vertex, Vertex) -> [u32; 2]) {
        let graph = AdjacencyList::undirected(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let costs = |from: Vertex, to: Vertex| match (from.min(to), from.max(to)) {
            (0, 1) | (1, 3) => [1, 3],
            (0, 2) | (2, 3) => [3, 1],
            _ => [1, 1],
        };

        (graph, costs)
    }

    /// Both routes are Pareto-optimal; BOA* and NAMOA*dr agree.
    #[test]
    fn finds_both_routes_with_either_planner() {
        let (graph, costs) = diamond();
        let agent = Agent { start: 0, goal: 3 };
        let heuristic = Heuristic::new(&graph, &costs, agent.goal);

        for low_level in [LowLevel::Auto, LowLevel::NamoaDr] {
            let paths = pareto_paths(
                &graph,
                &costs,
                agent,
                &ConstraintSet::new(),
                &heuristic,
                low_level,
            );

            assert_eq!(paths, [(vec![0, 1, 3], [2, 6]), (vec![0, 2, 3], [6, 2])]);
        }
    }

    /// Forbidding vertex 1 at time 1 leaves the other route and the wait-then-go route.
    #[test]
    fn respects_constraints_and_prices_waits() {
        let (graph, costs) = diamond();
        let agent = Agent { start: 0, goal: 3 };
        let heuristic = Heuristic::new(&graph, &costs, agent.goal);
        let mut constraints = ConstraintSet::new();
        constraints.insert(&Constraint::Vertex {
            agent: 0,
            vertex: 1,
            time: 1,
        });

        let paths = pareto_paths(
            &graph,
            &costs,
            agent,
            &constraints,
            &heuristic,
            LowLevel::Auto,
        );

        assert_eq!(paths, [(vec![0, 0, 1, 3], [3, 7]), (vec![0, 2, 3], [6, 2])]);
    }
}
