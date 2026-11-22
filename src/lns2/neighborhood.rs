//! The neighborhood selection methods of MAPF-LNS2 ([`Method`]): which agents one repair
//! iteration replans.
//!
//! - **Collision-based**: a random colliding agent's connected component of the collision
//!   graph. A small component is filled up with agents that random walks from its agents'
//!   paths run into; a large one is sampled by a random walk on the collision graph.
//! - **Failure-based**: an agent picked by its number of collisions, with the agents that
//!   block it: those whose paths visit its start, and those whose goals lie on its path that
//!   crosses the fewest goals of other agents.
//! - **Random**: agents drawn with probability proportional to their number of collisions
//!   plus one.
//!
//! Each method returns distinct agents, at least one and at most the neighborhood size. A
//! method that cannot find enough agents returns fewer.

use alloc::collections::{BinaryHeap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use super::{Collisions, Repair};
use crate::graph::{Graph, Vertex};
use crate::instance::AgentId;
use crate::path::{Time, position_at};

/// How often in total a fill-up loop may fail to find a new agent before it returns what
/// it has; the walk on the collision graph may fail this many times per agent wanted.
const ATTEMPTS: usize = 64;

/// A neighborhood selection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Method {
    /// Colliding agents and the agents they might collide with.
    Collision,
    /// An agent that failed to avoid collisions and the agents in its way.
    Failure,
    /// Agents drawn by their number of collisions plus one.
    Random,
}

impl Method {
    /// Every method, in the order of [`Repair::weights`].
    pub(super) const ALL: [Self; 3] = [Self::Collision, Self::Failure, Self::Random];
}

impl<G: Graph> Repair<'_, G> {
    /// The agents to replan by `method`, for a plan with `collisions`, at least one of which
    /// collides.
    pub(super) fn neighborhood(&mut self, method: Method, collisions: &Collisions) -> Vec<AgentId> {
        match method {
            Method::Collision => self.collision_based(collisions),
            Method::Failure => self.failure_based(collisions),
            Method::Random => self.random_agents(collisions),
        }
    }

    /// The collision-based neighborhood: the component of the collision graph around a
    /// random colliding agent, filled up by random walks if smaller than the neighborhood,
    /// else sampled by a random walk on it.
    fn collision_based(&mut self, collisions: &Collisions) -> Vec<AgentId> {
        let colliding: Vec<AgentId> = (0..self.paths.len())
            .filter(|&agent| collisions.degree(agent) > 0)
            .collect();
        let start = colliding[self.random.below(colliding.len())];
        let component = component(collisions, start);

        if component.len() > self.size {
            return self.walk_collision_graph(collisions, start);
        }

        let mut agents = component;
        let mut failures = 0;
        while agents.len() < self.size && failures < ATTEMPTS {
            let walker = agents[self.random.below(agents.len())];
            match self.walk_into_another(walker, &agents) {
                Some(found) => agents.push(found),
                None => failures += 1,
            }
        }

        agents
    }

    /// Up to the neighborhood size of distinct agents met by a random walk on the
    /// collision graph from `start`.
    fn walk_collision_graph(&mut self, collisions: &Collisions, start: AgentId) -> Vec<AgentId> {
        let mut agents = vec![start];
        let (mut at, mut failures) = (start, 0);
        while agents.len() < self.size && failures < ATTEMPTS * self.size {
            let next = &collisions.neighbors[at];
            at = next[self.random.below(next.len())];
            if agents.contains(&at) {
                failures += 1;
            } else {
                agents.push(at);
            }
        }

        agents
    }

    /// The first agent outside `taken` that a random walk runs into, starting from a random
    /// position on `walker`'s path at its time; `None` if the walk meets nobody within as
    /// many steps as the graph has vertices.
    fn walk_into_another(&mut self, walker: AgentId, taken: &[AgentId]) -> Option<AgentId> {
        let path = &self.paths[walker];
        let index = self.random.below(path.len());
        let (mut vertex, mut time) = (path[index], Time::try_from(index).expect("short paths"));

        for _ in 0..self.instance.graph().vertex_count() {
            let mut moves = vec![vertex];
            self.instance
                .graph()
                .neighbors(vertex, |next| moves.push(next));
            vertex = moves[self.random.below(moves.len())];
            time += 1;

            let met = (0..self.paths.len())
                .filter(|agent| !taken.contains(agent))
                .find(|&agent| position_at(&self.paths[agent], time) == vertex);
            if met.is_some() {
                return met;
            }
        }

        None
    }

    /// The failure-based neighborhood: an agent drawn by its number of collisions, with
    /// the agents that visit its start and those whose goals lie on its least blocked path.
    fn failure_based(&mut self, collisions: &Collisions) -> Vec<AgentId> {
        let degrees: Vec<f64> = (0..self.paths.len())
            .map(|agent| f64::from(u32::try_from(collisions.degree(agent)).unwrap_or(u32::MAX)))
            .collect();
        let failed = self.random.weighted(&degrees).expect("some agent collides");
        let at_start = self.visiting_start(failed);
        let on_path = self.goals_on_least_blocked_path(failed);
        let mut agents = vec![failed];

        if at_start.is_empty() && on_path.is_empty() {
            return agents;
        }

        if at_start.len() + on_path.len() < self.size - 1 {
            agents.extend(on_path.iter().chain(&at_start).copied());
            agents.sort_unstable();
            agents.dedup();
            self.add_blockers(&mut agents);
        } else {
            agents.extend(self.blockers_by_rule(&at_start, &on_path));
        }

        agents
    }

    /// The other agents whose paths visit the start of `agent`, by the time they first do.
    fn visiting_start(&self, agent: AgentId) -> Vec<AgentId> {
        let start = self.instance.agents()[agent].start;
        let mut visits: Vec<(usize, AgentId)> = self
            .paths
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != agent)
            .filter_map(|(other, path)| Some((path.iter().position(|&v| v == start)?, other)))
            .collect();
        visits.sort_unstable();

        visits.into_iter().map(|(_, other)| other).collect()
    }

    /// The other agents whose goals lie on the path from the start of `agent` to its goal
    /// that crosses the fewest of them (a Dijkstra search weighted by goal vertices).
    fn goals_on_least_blocked_path(&self, agent: AgentId) -> Vec<AgentId> {
        let agents = self.instance.agents();
        let graph = self.instance.graph();
        let mut owner: Vec<Option<AgentId>> = vec![None; graph.vertex_count()];
        for (other, a) in agents
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != agent)
        {
            owner[a.goal] = Some(other);
        }

        let weight = |vertex: Vertex| u32::from(owner[vertex].is_some());
        let start = agents[agent].start;
        let mut cost = vec![u32::MAX; graph.vertex_count()];
        let mut previous: Vec<Option<Vertex>> = vec![None; graph.vertex_count()];
        let mut open = BinaryHeap::from([Reverse((weight(start), start))]);
        cost[start] = weight(start);
        while let Some(Reverse((here_cost, here))) = open.pop() {
            if here_cost > cost[here] {
                continue;
            }

            graph.neighbors(here, |next| {
                let next_cost = here_cost + weight(next);
                if next_cost < cost[next] {
                    cost[next] = next_cost;
                    previous[next] = Some(here);
                    open.push(Reverse((next_cost, next)));
                }
            });
        }

        let mut found = Vec::new();
        let mut at = Some(agents[agent].goal);
        while let Some(vertex) = at {
            found.extend(owner[vertex]);
            at = previous[vertex];
        }

        found
    }

    /// Adds agents to `agents` until the neighborhood is full: each time a random agent
    /// whose goal the path of a random member visits.
    fn add_blockers(&mut self, agents: &mut Vec<AgentId>) {
        let goals = self.instance.agents();
        let mut failures = 0;
        while agents.len() < self.size && failures < ATTEMPTS {
            let member = agents[self.random.below(agents.len())];
            let path = &self.paths[member];
            let blocked: Vec<AgentId> = (0..goals.len())
                .filter(|other| !agents.contains(other) && path.contains(&goals[*other].goal))
                .collect();
            if blocked.is_empty() {
                failures += 1;
            } else {
                agents.push(blocked[self.random.below(blocked.len())]);
            }
        }
    }

    /// The paper's rule for filling the rest of a failure-based neighborhood from the
    /// agents `at_start` (by the time they visit the start) and `on_path`, together at
    /// least one less than the neighborhood size.
    fn blockers_by_rule(&mut self, at_start: &[AgentId], on_path: &[AgentId]) -> Vec<AgentId> {
        let room = self.size - 1;
        let mut chosen = Vec::with_capacity(room);

        if room == 0 {
            return chosen;
        } else if at_start.is_empty() {
            chosen.extend(self.sample(on_path, room));
        } else if on_path.len() >= room {
            chosen.push(at_start[0]);
            let rest: Vec<AgentId> = on_path
                .iter()
                .copied()
                .filter(|&a| a != at_start[0])
                .collect();
            chosen.extend(self.sample(&rest, room.saturating_sub(1)));
        } else {
            chosen.extend_from_slice(on_path);
            for &agent in at_start {
                if chosen.len() == room {
                    break;
                }

                if !chosen.contains(&agent) {
                    chosen.push(agent);
                }
            }
        }

        chosen
    }

    /// Up to `count` distinct agents drawn uniformly from `agents`.
    fn sample(&mut self, agents: &[AgentId], count: usize) -> Vec<AgentId> {
        let mut pool = agents.to_vec();
        self.random.shuffle(&mut pool);
        pool.truncate(count);

        pool
    }

    /// The random neighborhood: agents drawn without replacement, each with probability
    /// proportional to its number of collisions plus one.
    fn random_agents(&mut self, collisions: &Collisions) -> Vec<AgentId> {
        let mut weights: Vec<f64> = (0..self.paths.len())
            .map(|agent| {
                f64::from(u32::try_from(collisions.degree(agent)).unwrap_or(u32::MAX)) + 1.0
            })
            .collect();
        let mut agents = Vec::with_capacity(self.size);
        while agents.len() < self.size {
            let Some(agent) = self.random.weighted(&weights) else {
                break;
            };

            weights[agent] = 0.0;
            agents.push(agent);
        }

        agents
    }
}

/// The connected component of `start` in the collision graph, in breadth-first order.
fn component(collisions: &Collisions, start: AgentId) -> Vec<AgentId> {
    let mut seen = vec![false; collisions.neighbors.len()];
    let mut found = Vec::new();
    let mut queue = VecDeque::from([start]);
    seen[start] = true;
    while let Some(agent) = queue.pop_front() {
        found.push(agent);
        for &next in &collisions.neighbors[agent] {
            if !seen[next] {
                seen[next] = true;
                queue.push_back(next);
            }
        }
    }

    found
}

#[cfg(test)]
mod tests {
    //! Each method's neighborhoods on a plan with collisions.

    use super::*;

    use heuristic_search::Stats;

    use crate::graph::Grid;
    use crate::instance::{Agent, Instance};
    use crate::low_level::distance_tables;
    use crate::random::Random;

    /// A repair of `instance` with neighborhoods of `size`, every agent on the straight path
    /// along its row, so that the two agents of each row collide.
    fn repair(instance: &Instance<Grid>, size: usize) -> Repair<'_, Grid> {
        let paths = instance
            .agents()
            .iter()
            .map(|agent| {
                if agent.start < agent.goal {
                    (agent.start..=agent.goal).collect()
                } else {
                    (agent.goal..=agent.start).rev().collect()
                }
            })
            .collect();

        Repair {
            instance,
            distances: distance_tables(instance.graph(), instance.agents()),
            size,
            random: Random::new(3),
            paths,
            weights: [1.0; 3],
            stats: Stats::default(),
        }
    }

    /// The instance of these tests: agents crossing a 4x3 grid in both directions.
    fn instance() -> Instance<Grid> {
        let grid = Grid::new(4, 3);
        let agents = [(0, 3), (3, 0), (4, 7), (7, 4), (8, 11), (11, 8)]
            .map(|(start, goal)| Agent { start, goal })
            .to_vec();

        Instance::new(grid, agents).expect("valid")
    }

    /// Every method returns distinct agents within the size, including a colliding one
    /// where the method starts from one.
    #[test]
    fn methods_return_distinct_bounded_neighborhoods() {
        let instance = instance();
        for size in 1..=6 {
            let mut repair = repair(&instance, size);
            let collisions = Collisions::of(&repair.paths);
            assert!(collisions.pairs > 0, "the straight paths collide");

            for method in Method::ALL {
                for _ in 0..20 {
                    let mut agents = repair.neighborhood(method, &collisions);
                    let len = agents.len();
                    agents.sort_unstable();
                    agents.dedup();

                    assert_eq!(agents.len(), len, "{method:?}: duplicates");
                    assert!(
                        (1..=size).contains(&len),
                        "{method:?}: {len} agents for {size}"
                    );
                    if method != Method::Random {
                        assert!(agents.iter().any(|&agent| collisions.degree(agent) > 0));
                    }
                }
            }
        }
    }
}
