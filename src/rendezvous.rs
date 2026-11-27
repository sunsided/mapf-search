//! Coordination at physical rendezvous ([`rendezvous`]): agents that can exchange plans only
//! while they are connected in a communication graph plan in epochs that end with the team
//! connected again.
//!
//! This adapts Hollinger and Singh's periodic connectivity (ICRA 2010) to MAPF. Agents talk
//! over the links of a communication graph on the same vertices, relaying through each
//! other: a configuration is *connected* if the agents' vertices, linked by communication
//! edges in either direction, form one component. The team starts connected and must be
//! connected again every `interval` steps, the rendezvous times. Only then do agents
//! exchange plans, so each epoch's joint plan is agreed at its start and executed without
//! communication.
//!
//! An epoch plans the agents one after another, as the paper's Algorithm 1 does, with
//! conflicts added and the connectivity test made incremental:
//!
//! - a *pass* starts with some agent and plans the others outwards from it, breadth-first
//!   along the team's links at the start of the epoch. An agent takes the plan
//!   of `interval` steps that ends closest to its goal, where it is linked to the agents
//!   planned before it in this pass, and that avoids their plans ([`reserve`], through the
//!   epoch's end). It ignores the agents not planned yet, which plan around it later, so
//!   two agents can swap places. If an agent finds no such plan, the pass fails;
//! - the first pass starts with agent `e mod n` in epoch `e`. The epoch takes the first
//!   pass, from that agent on and then from each following one, that leaves the team
//!   strictly closer to its goals, summed over the agents. If none does, every agent stays
//!   put.
//!
//! Each agent of a pass avoids the plans before it, so the joint plan is conflict-free
//! through the epoch, and each agent ends linked to the agents before it, so the team ends
//! connected. Staying put has both properties too. By induction every rendezvous time finds
//! the team connected. Letting each agent check only the agents before it is what lets a
//! leader move ahead and its followers catch up in the same epoch; requiring every
//! accepted move to keep the whole team connected, as coordinate ascent from staying put
//! would, stalls a column of agents for good. The run ends at the first rendezvous with
//! every agent at its goal.
//!
//! The returned paths are trimmed of their trailing waits. Under goal-stay semantics that
//! changes no agent's position at any time ([`position_at`](crate::position_at)), so the
//! team is still connected at every multiple of `interval`. A single agent is connected by
//! itself.
//!
//! The safety argument is this crate's adaptation, not the paper's, and there is no
//! completeness result. Every epoch must bring the team closer to its goals, so an agent
//! never steps off its goal to let another pass, and the greedy passes can leave the team
//! waiting forever where a coordinated move would make progress.

use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use crate::conflict::first_conflict;
use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::low_level::distance_tables;
use crate::path::{Path, Time, trim_trailing_waits};
use crate::reservation::reserve;
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds a conflict-free plan for agents that exchange plans only at rendezvous: every
/// `interval` steps, the agents' vertices must be connected in `comm`, the communication
/// graph, which has the vertices of the instance's graph and links that work in either
/// direction.
///
/// The plan has no quality guarantee, and the solver is not complete: the team can wait
/// forever, so always pass a budget. The result is [`Outcome::Unsolvable`] only if an agent
/// cannot reach its goal even alone, and [`Outcome::Exhausted`] if the start or the goal
/// configuration is not connected, so the agents can never coordinate or never all finish at
/// a rendezvous.
///
/// `budget` is asked before every epoch; [`MapfResult::stats`] counts epochs in
/// [`Stats::expanded`] and single-agent searches in [`Stats::generated`].
///
/// # Panics
///
/// Panics if `interval` is `0`, or if `comm` does not have as many vertices as the graph of
/// `instance`.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, first_conflict, rendezvous};
///
/// // Two agents in a corridor talk only to an agent next to them.
/// let grid = Grid::new(5, 1);
/// let agents = vec![Agent { start: 0, goal: 3 }, Agent { start: 1, goal: 4 }];
/// let instance = Instance::new(grid.clone(), agents)?;
///
/// let result = rendezvous(&instance, &grid, 2, Limits::expansions(100));
/// let solution = result.solution().expect("they walk east side by side");
/// assert_eq!(first_conflict(&solution.paths), None);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn rendezvous<G: Graph, C: Graph>(
    instance: &Instance<G>,
    comm: &C,
    interval: Time,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    assert!(interval >= 1, "an epoch lasts at least one step");
    assert_eq!(
        comm.vertex_count(),
        instance.graph().vertex_count(),
        "the communication graph has the vertices of the instance's graph"
    );

    let mut run = Epochs {
        instance,
        comm,
        interval,
        distances: distance_tables(instance.graph(), instance.agents()),
        here: instance.agents().iter().map(|agent| agent.start).collect(),
        paths: instance
            .agents()
            .iter()
            .map(|agent| vec![agent.start])
            .collect(),
        stats: Stats::default(),
    };
    let outcome = run.solve(&mut budget);

    MapfResult {
        outcome,
        stats: run.stats,
    }
}

/// Returns `true` if `positions`, one distinct vertex per agent, are connected in `comm`: one
/// component when every link between two occupied vertices joins them, in either direction.
fn connected<C: Graph>(comm: &C, positions: &[Vertex]) -> bool {
    let mut occupant = vec![None; comm.vertex_count()];
    for (agent, &vertex) in positions.iter().enumerate() {
        occupant[vertex] = Some(agent);
    }

    let mut root: Vec<AgentId> = (0..positions.len()).collect();
    for (agent, &vertex) in positions.iter().enumerate() {
        comm.neighbors(vertex, |linked| {
            if let Some(other) = occupant[linked] {
                let (a, b) = (find(&mut root, agent), find(&mut root, other));
                root[a] = b;
            }
        });
    }

    (0..positions.len()).all(|agent| find(&mut root, agent) == find(&mut root, 0))
}

/// The agents at the connected `positions` in breadth-first order over their links in
/// `comm`, either direction, from agent `first`; ties by agent id. Each agent after the
/// first is linked to one before it, so planning in this order lets every agent stay close
/// to an agent already planned.
fn link_order<C: Graph>(comm: &C, positions: &[Vertex], first: AgentId) -> Vec<AgentId> {
    let mut occupant = vec![None; comm.vertex_count()];
    for (agent, &vertex) in positions.iter().enumerate() {
        occupant[vertex] = Some(agent);
    }

    let mut links: Vec<Vec<AgentId>> = vec![Vec::new(); positions.len()];
    for (agent, &vertex) in positions.iter().enumerate() {
        comm.neighbors(vertex, |linked| {
            if let Some(other) = occupant[linked] {
                links[agent].push(other);
                links[other].push(agent);
            }
        });
    }

    let mut order = Vec::from([first]);
    let mut seen = vec![false; positions.len()];
    seen[first] = true;
    let mut next = 0;
    while let Some(&agent) = order.get(next) {
        next += 1;
        links[agent].sort_unstable();
        for &other in &links[agent] {
            if !seen[other] {
                seen[other] = true;
                order.push(other);
            }
        }
    }

    debug_assert_eq!(order.len(), positions.len(), "the team is connected");
    order
}

/// The representative of `agent` in the union-find forest `root`, halving paths on the way.
fn find(root: &mut [AgentId], mut agent: AgentId) -> AgentId {
    while root[agent] != agent {
        root[agent] = root[root[agent]];
        agent = root[agent];
    }

    agent
}

/// The state of one rendezvous run.
struct Epochs<'a, G, C> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// The communication graph.
    comm: &'a C,
    /// The steps between two rendezvous.
    interval: Time,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// Where each agent stands at the current rendezvous.
    here: Vec<Vertex>,
    /// The executed path of each agent so far, ending at [`Epochs::here`].
    paths: Vec<Path>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph, C: Graph> Epochs<'_, G, C> {
    /// Runs epochs until every agent stands at its goal at a rendezvous or the budget runs
    /// out.
    fn solve(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let agents = self.instance.agents();
        if agents
            .iter()
            .zip(&self.distances)
            .any(|(agent, distances)| distances[agent.start].is_none())
        {
            return Outcome::Unsolvable;
        }

        let goals: Vec<Vertex> = agents.iter().map(|agent| agent.goal).collect();
        if !connected(self.comm, &self.here) || !connected(self.comm, &goals) {
            return Outcome::Exhausted;
        }

        let mut epoch = 0;
        while self.here != goals {
            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.stats.expanded += 1;
            let plans = self.plan_epoch(epoch);
            self.execute(&plans);
            epoch += 1;
        }

        Outcome::Solved(self.solution())
    }

    /// The joint plan of epoch `epoch`: `interval + 1` positions per agent, conflict-free
    /// and connected at the end. The first pass ([`Epochs::pass`]) from agent `epoch mod n`
    /// on, then from each following agent, that brings the team closer to its goals; every
    /// agent staying put if none does.
    fn plan_epoch(&mut self, epoch: usize) -> Vec<Path> {
        let steps = usize::try_from(self.interval).expect("the interval fits in usize");
        let stay: Vec<Path> = self.here.iter().map(|&v| vec![v; steps + 1]).collect();
        let count = stay.len();
        let before = self.remaining(&stay);

        for first in (0..count).map(|offset| (epoch + offset) % count) {
            let Some(plans) = self.pass(first) else {
                continue;
            };

            if self.remaining(&plans) < before {
                debug_assert_eq!(first_conflict(&plans), None, "plans avoid each other");
                return plans;
            }
        }

        stay
    }

    /// The sum of the agents' distances to their goals at the ends of `plans`.
    fn remaining(&self, plans: &[Path]) -> u32 {
        plans
            .iter()
            .zip(&self.distances)
            .map(|(plan, distance)| distance[plan[plan.len() - 1]].expect("reachable"))
            .sum()
    }

    /// Plans every agent in turn, from agent `first` outwards along the team's current links
    /// ([`link_order`]). Each takes the plan that ends closest to its goal, connected to the
    /// ends of the agents planned before it, avoiding their plans. `None` if an agent finds
    /// no such plan.
    fn pass(&mut self, first: AgentId) -> Option<Vec<Path>> {
        let count = self.here.len();
        let mut plans: Vec<Option<Path>> = vec![None; count];
        for agent in link_order(self.comm, &self.here, first) {
            let mut constraints = ConstraintSet::new();
            for (other, plan) in plans.iter().enumerate() {
                if let Some(plan) = plan {
                    reserve(&mut constraints, other, plan, Some(self.interval));
                }
            }

            let planned: Vec<Vertex> = plans.iter().flatten().map(|p| p[p.len() - 1]).collect();
            plans[agent] = Some(self.closest(agent, &constraints, planned)?);
        }

        Some(plans.into_iter().flatten().collect())
    }

    /// The plan of `agent` for the epoch under `constraints` that ends closest to its goal
    /// (ties: the smaller vertex) where it is connected to `planned`, the ends of the agents
    /// planned before it. `None` if there is none.
    fn closest(
        &mut self,
        agent: AgentId,
        constraints: &ConstraintSet,
        mut planned: Vec<Vertex>,
    ) -> Option<Path> {
        self.stats.generated += 1;
        let layers = self.reachable(self.here[agent], constraints);
        let last = layers.last().expect("an epoch has a final layer");
        let distance = &self.distances[agent];

        planned.push(0);
        let mut ends: Vec<Vertex> = (0..last.len())
            .filter(|&vertex| last[vertex].is_some() && distance[vertex].is_some())
            .collect();
        ends.sort_by_key(|&vertex| distance[vertex]);
        let end = ends.into_iter().find(|&vertex| {
            *planned.last_mut().expect("just pushed") = vertex;
            connected(self.comm, &planned)
        })?;

        Some(trace(&layers, end))
    }

    /// For each time `0..=interval`, the vertices `from` can occupy then under
    /// `constraints`, each with the vertex it came from (itself at time 0).
    fn reachable(&self, from: Vertex, constraints: &ConstraintSet) -> Vec<Vec<Option<Vertex>>> {
        let vertices = self.instance.graph().vertex_count();
        let mut layers = vec![vec![None; vertices]];
        layers[0][from] = Some(from);

        for time in 0..self.interval {
            let mut next = vec![None; vertices];
            let current = &layers[layers.len() - 1];
            for vertex in (0..vertices).filter(|&vertex| current[vertex].is_some()) {
                let mut step = |to: Vertex| {
                    if next[to].is_none() && constraints.allows(vertex, to, time) {
                        next[to] = Some(vertex);
                    }
                };
                step(vertex);
                self.instance.graph().neighbors(vertex, &mut step);
            }

            layers.push(next);
        }

        layers
    }

    /// Moves every agent along its epoch plan.
    fn execute(&mut self, plans: &[Path]) {
        for ((path, here), plan) in self.paths.iter_mut().zip(&mut self.here).zip(plans) {
            path.extend_from_slice(&plan[1..]);
            *here = plan[plan.len() - 1];
        }

        debug_assert!(
            connected(self.comm, &self.here),
            "every rendezvous finds the team connected"
        );
    }

    /// The executed paths, trimmed, as a solution.
    fn solution(&mut self) -> Solution {
        let mut paths = core::mem::take(&mut self.paths);
        paths.iter_mut().for_each(trim_trailing_waits);
        debug_assert_eq!(first_conflict(&paths), None, "epochs are conflict-free");

        Solution::new(paths)
    }
}

/// The path through `layers` (see [`Epochs::reachable`]) that ends at `end` at the last
/// time.
fn trace(layers: &[Vec<Option<Vertex>>], end: Vertex) -> Path {
    let mut path = vec![end];
    for layer in layers[1..].iter().rev() {
        let from = layer[path[path.len() - 1]].expect("traced vertices are reachable");
        path.push(from);
    }

    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    //! Unit tests for rendezvous coordination.

    use super::*;
    use crate::graph::{AdjacencyList, Grid};
    use crate::instance::Agent;
    use crate::path::position_at;
    use heuristic_search::Limits;

    /// Every agent pair is linked.
    fn complete(vertices: usize) -> AdjacencyList {
        let edges: Vec<(Vertex, Vertex)> = (0..vertices)
            .flat_map(|a| (0..vertices).filter(move |&b| b != a).map(move |b| (a, b)))
            .collect();

        AdjacencyList::directed(vertices, &edges)
    }

    /// Asserts that `paths` are connected in `comm` at every multiple of `interval`.
    fn assert_connected_at_rendezvous(comm: &impl Graph, paths: &[Path], interval: Time) {
        let end = paths.iter().map(Vec::len).max().unwrap_or(1);
        let end = Time::try_from(end).expect("short paths");
        for time in (0..=end).step_by(usize::try_from(interval).expect("small")) {
            let positions: Vec<Vertex> = paths.iter().map(|p| position_at(p, time)).collect();
            assert!(connected(comm, &positions), "disconnected at {time}");
        }
    }

    /// Two agents in a corridor that only talk to a neighbour walk east side by side.
    #[test]
    fn corridor_walk_stays_connected() {
        let grid = Grid::new(5, 1);
        let agents = vec![Agent { start: 0, goal: 3 }, Agent { start: 1, goal: 4 }];
        let instance = Instance::new(grid.clone(), agents).expect("valid");

        for interval in 1..=3 {
            let result = rendezvous(&instance, &grid, interval, Limits::expansions(50));
            let solution = result.solution().expect("solved");

            assert_eq!(first_conflict(&solution.paths), None);
            assert_connected_at_rendezvous(&grid, &solution.paths, interval);
        }
    }

    /// Three agents on a grid, linked within one cell, cross to the far side and arrive in a
    /// connected line.
    #[test]
    fn grid_team_keeps_its_relay() {
        let grid = Grid::new(4, 3);
        let agents = vec![
            Agent {
                start: grid.vertex(0, 0),
                goal: grid.vertex(3, 0),
            },
            Agent {
                start: grid.vertex(0, 1),
                goal: grid.vertex(3, 1),
            },
            Agent {
                start: grid.vertex(0, 2),
                goal: grid.vertex(3, 2),
            },
        ];
        let instance = Instance::new(grid.clone(), agents).expect("valid");
        let result = rendezvous(&instance, &grid, 2, Limits::expansions(50));
        let solution = result.solution().expect("solved");

        assert_eq!(first_conflict(&solution.paths), None);
        assert_connected_at_rendezvous(&grid, &solution.paths, 2);
    }

    /// A column of four agents linked to orthogonal neighbours crosses an 8x8 grid and shifts
    /// down into its goals, which takes passes from different first agents and agents that
    /// leave their vertices to those planned before them.
    #[test]
    fn column_crosses_for_every_interval() {
        let grid = Grid::new(8, 8);
        let agents = (0..4)
            .map(|y| Agent {
                start: grid.vertex(0, y),
                goal: grid.vertex(7, y + 4),
            })
            .collect();
        let instance = Instance::new(grid.clone(), agents).expect("valid");

        for interval in 1..=6 {
            let result = rendezvous(&instance, &grid, interval, Limits::expansions(100));
            let solution = result
                .solution()
                .unwrap_or_else(|| panic!("interval {interval}"));

            assert_eq!(first_conflict(&solution.paths), None);
            assert_connected_at_rendezvous(&grid, &solution.paths, interval);
        }
    }

    /// With every pair linked, connectivity never binds.
    #[test]
    fn complete_links_never_bind() {
        let grid = Grid::new(3, 3);
        let agents = vec![Agent { start: 0, goal: 8 }, Agent { start: 8, goal: 0 }];
        let instance = Instance::new(grid, agents).expect("valid");
        let result = rendezvous(&instance, &complete(9), 3, Limits::expansions(50));

        assert_eq!(
            result.solution().map(|s| first_conflict(&s.paths)),
            Some(None)
        );
    }

    /// Starts or goals out of range give up at once; an unreachable goal is unsolvable.
    #[test]
    fn disconnected_ends_and_unreachable_goals() {
        let grid = Grid::new(5, 1);
        let apart = vec![Agent { start: 0, goal: 1 }, Agent { start: 4, goal: 3 }];
        let instance = Instance::new(grid.clone(), apart).expect("valid");
        let result = rendezvous(&instance, &grid, 2, Limits::NONE);
        assert_eq!(result.outcome, Outcome::Exhausted);
        assert_eq!(result.stats, Stats::default());

        let spread = vec![Agent { start: 1, goal: 0 }, Agent { start: 2, goal: 4 }];
        let instance = Instance::new(grid.clone(), spread).expect("valid");
        let result = rendezvous(&instance, &grid, 2, Limits::NONE);
        assert_eq!(result.outcome, Outcome::Exhausted);

        let walled = Grid::parse(&[".@."]);
        let agents = vec![Agent { start: 0, goal: 2 }];
        let instance = Instance::new(walled.clone(), agents).expect("valid");
        let result = rendezvous(&instance, &walled, 2, Limits::NONE);
        assert_eq!(result.outcome, Outcome::Unsolvable);
    }

    /// A corridor swap never finishes; the budget stops it.
    #[test]
    fn corridor_swap_stops_at_the_budget() {
        let grid = Grid::new(3, 1);
        let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
        let instance = Instance::new(grid, agents).expect("valid");
        let result = rendezvous(&instance, &complete(3), 2, Limits::expansions(5));

        assert_eq!(result.outcome, Outcome::LimitReached);
        assert_eq!(result.stats.expanded, 5);
    }

    /// Connectivity follows links in either direction and through relays.
    #[test]
    fn connectivity_relays_both_ways() {
        let comm = AdjacencyList::directed(4, &[(0, 1), (2, 1)]);

        assert!(connected(&comm, &[0, 1, 2]));
        assert!(!connected(&comm, &[0, 2]));
        assert!(!connected(&comm, &[0, 1, 3]));
        assert!(connected(&comm, &[3]));
        assert!(connected(&comm, &[]));
    }

    /// An interval of 0 is rejected.
    #[test]
    #[should_panic(expected = "at least one step")]
    fn zero_interval_panics() {
        let grid = Grid::new(2, 1);
        let instance =
            Instance::new(grid.clone(), vec![Agent { start: 0, goal: 1 }]).expect("valid");
        let _ = rendezvous(&instance, &grid, 0, Limits::NONE);
    }
}
