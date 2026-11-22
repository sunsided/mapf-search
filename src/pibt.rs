//! PIBT ([`pibt`], Okumura et al. 2019, 2022): priority inheritance with backtracking. Plans
//! one time step for all agents at a time, very fast, neither optimal nor complete.
//!
//! Each step, every agent that has not reached its goal gains priority; one that has drops
//! back to its distinct tie-breaker. In decreasing priority, each agent not yet planned
//! takes the free neighbouring vertex (or its own) closest to its goal, preferring vertices
//! nobody occupies. If the vertex it wants is occupied by an agent not yet planned, that
//! agent inherits the priority and must move out of the way first, which may push further
//! agents; when it cannot, it reports back (*backtracking*) and the pusher tries its next
//! vertex. The run ends at the first step where every agent stands at its goal.
//!
//! ```mermaid
//! flowchart TD
//!     step[update priorities] --> order[agents by decreasing priority]
//!     order --> pick[an unplanned agent picks its best free vertex]
//!     pick -->|occupied by an unplanned agent| inherit[that agent inherits the priority, picks recursively]
//!     inherit -->|it is stuck| pick
//!     inherit -->|it moves| pick
//!     pick --> done{every agent at its goal?}
//!     done -->|yes| solved[Solved]
//!     done -->|no| budget{budget exhausted?}
//!     budget -->|no| step
//!     budget -->|yes| limit[LimitReached]
//! ```

use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::low_level::distance_tables;
use crate::path::Path;
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds a conflict-free plan with PIBT: one time step for every agent at a time, by
/// priority, with priority inheritance and backtracking.
///
/// The plan has no quality guarantee, and PIBT is not complete: agents can oscillate
/// forever, and on an unsolvable instance PIBT runs until its budget, so always pass one.
/// (Okumura et al. prove that every agent reaches its goal at some step on graphs where
/// every pair of adjacent vertices lies on a simple cycle, such as biconnected ones, but
/// not that all stand at their goals at the same step.) The result is
/// [`Outcome::Unsolvable`] only if an agent cannot reach its goal even alone, and never
/// [`Outcome::Exhausted`].
///
/// `budget` is asked before every time step; [`MapfResult::stats`] counts time steps in
/// [`Stats::expanded`] and per-agent decisions, those after an inheritance included, in
/// [`Stats::generated`].
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, first_conflict, pibt};
///
/// // Two agents swap the ends of the top row of a free 3x2 grid, which is a cycle.
/// let grid = Grid::new(3, 2);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = pibt(&instance, Limits::expansions(100));
/// let solution = result.solution().expect("one agent goes around");
/// assert_eq!(first_conflict(&solution.paths), None);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn pibt<G: Graph>(instance: &Instance<G>, mut budget: impl Budget<Stats>) -> MapfResult {
    let distances = distance_tables(instance.graph(), instance.agents());
    let reachable = instance
        .agents()
        .iter()
        .zip(&distances)
        .all(|(agent, distances)| distances[agent.start].is_some());
    if !reachable {
        return MapfResult {
            outcome: Outcome::Unsolvable,
            stats: Stats::default(),
        };
    }

    let agents = instance.agents().len();
    let mut run = Run {
        instance,
        distances,
        here: instance.agents().iter().map(|agent| agent.start).collect(),
        next: vec![None; agents],
        reserved: vec![None; instance.graph().vertex_count()],
        occupant: vec![None; instance.graph().vertex_count()],
        waiting: vec![0; agents],
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

/// The state of one [`pibt`] run.
struct Run<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// Distances to each agent's goal, indexed by agent; every start reaches its goal.
    distances: Vec<Vec<Option<u32>>>,
    /// Where each agent is at the current step, indexed by agent.
    here: Vec<Vertex>,
    /// Where each agent is at the next step, once decided, indexed by agent.
    next: Vec<Option<Vertex>>,
    /// The agent that has claimed each vertex for the next step, indexed by vertex.
    reserved: Vec<Option<AgentId>>,
    /// The agent at each vertex at the current step, indexed by vertex.
    occupant: Vec<Option<AgentId>>,
    /// How many steps each agent has spent away from its goal since it last stood there:
    /// the integer part of its priority, indexed by agent.
    waiting: Vec<u64>,
    /// Each agent's positions so far, indexed by agent.
    paths: Vec<Path>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Run<'_, G> {
    /// Takes steps until every agent stands at its goal or the budget ends.
    fn solve(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        loop {
            if self.all_at_goals() {
                return Outcome::Solved(Solution::new(self.trimmed_paths()));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.step();
        }
    }

    /// Returns `true` if every agent stands at its goal.
    fn all_at_goals(&self) -> bool {
        self.here
            .iter()
            .zip(self.instance.agents())
            .all(|(&here, agent)| here == agent.goal)
    }

    /// One time step: updates the priorities, plans every agent by decreasing priority,
    /// then moves them.
    fn step(&mut self) {
        self.stats.expanded += 1;

        for (agent, &here) in self.here.iter().enumerate() {
            self.occupant[here] = Some(agent);
            if here == self.instance.agents()[agent].goal {
                self.waiting[agent] = 0;
            } else {
                self.waiting[agent] += 1;
            }
        }

        for agent in self.by_priority() {
            if self.next[agent].is_none() {
                self.plan(agent, None);
            }
        }

        self.advance();
    }

    /// The agents by decreasing priority: more steps away from the goal first, then the
    /// larger agent id (the tie-breaker).
    fn by_priority(&self) -> Vec<AgentId> {
        let mut agents: Vec<AgentId> = (0..self.here.len()).collect();
        agents.sort_unstable_by_key(|&agent| core::cmp::Reverse((self.waiting[agent], agent)));

        agents
    }

    /// Moves every agent to its planned vertex and clears the step's bookkeeping.
    fn advance(&mut self) {
        for agent in 0..self.here.len() {
            let next = self.next[agent].take().expect("every agent was planned");
            self.occupant[self.here[agent]] = None;
            self.reserved[next] = None;
            self.here[agent] = next;
            self.paths[agent].push(next);
        }
    }

    /// The recursive procedure of PIBT: plans `agent`, which inherited its priority from
    /// `pusher` if any, and returns `true` if it found a vertex other than staying stuck.
    fn plan(&mut self, agent: AgentId, pusher: Option<AgentId>) -> bool {
        self.stats.generated += 1;

        for vertex in self.candidates(agent) {
            if self.reserved[vertex].is_some() {
                continue;
            }

            if pusher.is_some_and(|pusher| self.here[pusher] == vertex) {
                continue;
            }

            self.reserve(agent, vertex);
            let blocker =
                self.occupant[vertex].filter(|&other| other != agent && self.next[other].is_none());
            if let Some(blocker) = blocker {
                if !self.plan(blocker, Some(agent)) {
                    continue;
                }
            }

            return true;
        }

        self.reserve(agent, self.here[agent]);

        false
    }

    /// Claims `vertex` for `agent` at the next step, releasing what it claimed before.
    fn reserve(&mut self, agent: AgentId, vertex: Vertex) {
        if let Some(old) = self.next[agent].replace(vertex) {
            if self.reserved[old] == Some(agent) {
                self.reserved[old] = None;
            }
        }

        self.reserved[vertex] = Some(agent);
    }

    /// The vertices `agent` may take next, best first: its own and its neighbours from
    /// which the goal is reachable, by distance to the goal, unoccupied ones first, then by
    /// id.
    fn candidates(&self, agent: AgentId) -> Vec<Vertex> {
        let here = self.here[agent];
        let distances = &self.distances[agent];
        let mut candidates = vec![here];
        self.instance
            .graph()
            .neighbors(here, |next| candidates.push(next));
        candidates.retain(|&vertex| distances[vertex].is_some());
        candidates
            .sort_by_key(|&vertex| (distances[vertex], self.occupant[vertex].is_some(), vertex));

        candidates
    }

    /// Each agent's path, trimmed of trailing waits at its goal.
    fn trimmed_paths(&self) -> Vec<Path> {
        self.paths
            .iter()
            .map(|path| {
                let mut path = path.clone();
                while path.len() > 1 && path[path.len() - 2] == path[path.len() - 1] {
                    path.pop();
                }

                path
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! PIBT on small instances.

    use super::*;

    use heuristic_search::Limits;

    use crate::conflict::first_conflict;
    use crate::graph::Grid;
    use crate::instance::Agent;

    /// Checks that `result` solved `instance` with a valid, conflict-free plan.
    fn assert_solved(instance: &Instance<Grid>, result: &MapfResult) {
        let solution = result.solution().expect("solved");
        assert_eq!(first_conflict(&solution.paths), None);
        for (path, agent) in solution.paths.iter().zip(instance.agents()) {
            assert_eq!(path.first(), Some(&agent.start));
            assert_eq!(path.last(), Some(&agent.goal));
            for step in path.windows(2) {
                let ((x0, y0), (x1, y1)) = (
                    instance.graph().cell(step[0]),
                    instance.graph().cell(step[1]),
                );
                assert!(x0.abs_diff(x1) + y0.abs_diff(y1) <= 1, "a jump");
            }
        }
    }

    /// Twelve agents on a free 5x5 grid, from the first cells to the mirrored ones.
    #[test]
    fn solves_a_crowded_grid() {
        let agents = (0..12)
            .map(|i| Agent {
                start: i,
                goal: 24 - i,
            })
            .collect();
        let instance = Instance::new(Grid::new(5, 5), agents).expect("valid");

        let result = pibt(&instance, Limits::expansions(1_000));

        assert_solved(&instance, &result);
    }

    /// An agent sitting at its goal on a cycle is pushed off it so that another agent can
    /// take the short way, and comes back.
    #[test]
    fn pushes_an_agent_off_its_goal() {
        // `0 1 2` over `3 4 5`; agent 0 sits at 1, agent 1 crosses from 0 to 2.
        let agents = vec![Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }];
        let instance = Instance::new(Grid::new(3, 2), agents).expect("valid");

        let result = pibt(&instance, Limits::expansions(100));

        assert_solved(&instance, &result);
        assert!(result.solution().is_some_and(|s| s.paths[0].len() > 1));
    }

    /// Off a cycle PIBT can push in circles: in a corridor with a side pocket, an agent at
    /// its goal is pushed to the dead end, and the higher-priority agent keeps pushing it
    /// back there. This is the incompleteness the docs describe.
    #[test]
    fn can_livelock_off_a_cycle() {
        let grid = Grid::parse(&["...", "@.@"]);
        let agents = vec![Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }];
        let instance = Instance::new(grid, agents).expect("valid");

        let result = pibt(&instance, Limits::expansions(100));

        assert_eq!(result.outcome, Outcome::LimitReached);
    }

    /// The corridor swap is unsolvable; PIBT stops at its budget.
    #[test]
    fn stops_at_the_budget_when_unsolvable() {
        let grid = Grid::new(3, 1);
        let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
        let instance = Instance::new(grid, agents).expect("valid");

        let result = pibt(&instance, Limits::expansions(50));

        assert_eq!(result.outcome, Outcome::LimitReached);
        assert_eq!(result.stats.expanded, 50);
    }

    /// An unreachable goal is proven at once; agents already at their goals need no step.
    #[test]
    fn handles_trivial_instances() {
        let walled =
            Instance::new(Grid::parse(&[".@."]), vec![Agent { start: 0, goal: 2 }]).expect("valid");
        let home =
            Instance::new(Grid::new(2, 1), vec![Agent { start: 1, goal: 1 }]).expect("valid");

        assert_eq!(pibt(&walled, Limits::NONE).outcome, Outcome::Unsolvable);
        assert_eq!(
            pibt(&home, Limits::NONE).outcome,
            Outcome::Solved(Solution::new(vec![vec![1]]))
        );
    }
}
