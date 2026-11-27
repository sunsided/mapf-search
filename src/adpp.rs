//! Asynchronous decentralized prioritized planning ([`adpp`], Čáp et al. 2012, 2015): every
//! agent plans for itself and repairs its plan when it learns of a conflicting plan of an
//! agent above it, with no central solver.
//!
//! The run is a deterministic simulation of the protocol. Agent `i` has priority over every
//! agent `j > i`. At the start each agent plans alone and broadcasts its plan. A broadcast
//! reaches each lower agent after its own delay of 1 to [`AdppConfig::max_delay`] ticks,
//! drawn from the seed, so messages overtake each other. An agent keeps the newest plan it
//! has received from each agent above it (each broadcast carries a version, so an older
//! plan that arrives late is dropped). When a tick delivers it anything, it checks its plan
//! against those plans and, if one conflicts, replans under their reservations, held for
//! good ([`reserve`]), and broadcasts the new plan if it changed.
//!
//! ```mermaid
//! flowchart TD
//!     start[every agent plans alone and broadcasts] --> quiet{a message in flight?}
//!     quiet -->|no| solved[Solved]
//!     quiet -->|yes| budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| deliver[deliver the next tick's messages]
//!     deliver --> check[each recipient checks its plan against the newest plans above it]
//!     check -->|conflict, no path| exhausted[Exhausted]
//!     check -->|conflict| replan[replan under their reservations, broadcast if changed]
//!     replan --> quiet
//!     check -->|consistent| quiet
//! ```
//!
//! The run ends when no message is in flight. Then every agent holds the current plan of
//! every agent above it, since the last broadcast of each has arrived, and its own plan
//! avoids all of them: the plans are pairwise conflict-free. It does end: agent 0 never
//! replans, and once the agents above `i` stop broadcasting, `i` replans at most once
//! more. Like every prioritized planner it is incomplete: an agent can find no path around
//! the agents above it.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use crate::conflict::first_conflict;
use crate::constraint::ConstraintSet;
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::low_level::{distance_tables, plan};
use crate::path::{Path, trim_trailing_waits};
use crate::random::Random;
use crate::reservation::reserve;
use crate::solution::{MapfResult, Outcome, Solution};

/// The settings of [`adpp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdppConfig {
    /// The longest a broadcast takes to reach an agent, in ticks; at least 1. Each message
    /// takes between 1 and this many. With 1, every message arrives in the next tick: the
    /// synchronized variant (SD-PP). The default is 3.
    pub max_delay: u32,
    /// The seed of the message delays: the same instance, configuration and budget give
    /// the same result.
    pub seed: u64,
}

impl Default for AdppConfig {
    /// Delays of 1 to 3 ticks, seed `0`.
    fn default() -> Self {
        Self {
            max_delay: 3,
            seed: 0,
        }
    }
}

/// Finds a conflict-free plan with asynchronous decentralized prioritized planning: agents
/// with priorities by index plan for themselves and repair conflicts with the plans of the
/// agents above them as those plans arrive, after random delays.
///
/// The plan has no quality guarantee, and ADPP is not complete: an agent fails when the
/// agents above it block it for good, for example by finishing on a vertex it has to pass.
/// The result is [`Outcome::Unsolvable`] only if an agent cannot reach its goal even
/// alone, and [`Outcome::Exhausted`] if an agent finds no path around the agents above it.
///
/// `budget` is asked before every tick that delivers messages; [`MapfResult::stats`] counts
/// those ticks in [`Stats::expanded`] and single-agent searches, the initial ones included,
/// in [`Stats::generated`].
///
/// # Panics
///
/// Panics if `config.max_delay` is `0`.
///
/// # Examples
///
/// ```
/// use mapf_search::{AdppConfig, Agent, Grid, Instance, Limits, adpp, first_conflict};
///
/// let grid = Grid::parse(&["@.@", "...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 1), goal: grid.vertex(2, 1) },
///     Agent { start: grid.vertex(1, 0), goal: grid.vertex(1, 2) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = adpp(&instance, AdppConfig::default(), Limits::expansions(100));
/// let solution = result.solution().expect("the second agent waits");
/// assert_eq!(first_conflict(&solution.paths), None);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn adpp<G: Graph>(
    instance: &Instance<G>,
    config: AdppConfig,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    assert!(config.max_delay >= 1, "a message takes at least one tick");

    let mut run = Protocol {
        instance,
        distances: distance_tables(instance.graph(), instance.agents()),
        delays: Random::new(config.seed),
        max_delay: config.max_delay,
        history: Vec::new(),
        known: Vec::new(),
        in_flight: BTreeSet::new(),
        stats: Stats::default(),
    };
    let outcome = run.solve(&mut budget);

    MapfResult {
        outcome,
        stats: run.stats,
    }
}

/// A broadcast on its way: `(tick, to, from, version)`, the tick it arrives, its recipient,
/// its sender and the version of the sender's plan it carries. Ordered as a tuple, a set of
/// them yields each tick's messages together, grouped by recipient.
type Message = (u64, AgentId, AgentId, usize);

/// The state of one simulated ADPP run.
struct Protocol<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// Draws message delays.
    delays: Random,
    /// The longest delay of a message.
    max_delay: u32,
    /// Every plan each agent has broadcast, oldest first, indexed by agent; the last one is
    /// its current plan.
    history: Vec<Vec<Path>>,
    /// `known[i][j]` for `j < i`: the newest version of `j`'s plan that `i` has received.
    known: Vec<Vec<Option<usize>>>,
    /// The messages not yet delivered.
    in_flight: BTreeSet<Message>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Protocol<'_, G> {
    /// Runs the protocol until no message is in flight, an agent fails, or the budget runs
    /// out.
    fn solve(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let count = self.instance.agents().len();
        self.known = (0..count).map(|agent| alloc::vec![None; agent]).collect();
        for agent in 0..count {
            let Some(path) = self.plan(agent) else {
                return Outcome::Unsolvable;
            };

            self.history.push(Vec::from([path]));
            self.broadcast(agent, 0);
        }

        while let Some(&(tick, ..)) = self.in_flight.first() {
            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.stats.expanded += 1;
            let recipients = self.deliver(tick);
            for agent in recipients {
                if !self.repair(agent, tick) {
                    return Outcome::Exhausted;
                }
            }
        }

        Outcome::Solved(self.solution())
    }

    /// Delivers every message of `tick`; returns the recipients, in order. Messages come
    /// grouped by recipient ([`Message`]), so each recipient appears once.
    fn deliver(&mut self, tick: u64) -> Vec<AgentId> {
        let mut recipients = Vec::new();
        while let Some(&(when, to, from, version)) = self.in_flight.first() {
            if when != tick {
                break;
            }

            self.in_flight.pop_first();
            let known = &mut self.known[to][from];
            *known = (*known).max(Some(version));
            if recipients.last() != Some(&to) {
                recipients.push(to);
            }
        }

        recipients
    }

    /// Checks the plan of `agent` against the newest plans it knows of the agents above it
    /// and replans if one conflicts, broadcasting the new plan at `tick` if it changed.
    /// Returns `false` if it has no path around them.
    fn repair(&mut self, agent: AgentId, tick: u64) -> bool {
        let own = self.current(agent);
        let consistent = self
            .above(agent)
            .all(|other| first_conflict(&[own, other]).is_none());
        if consistent {
            return true;
        }

        let Some(path) = self.plan(agent) else {
            return false;
        };

        if path != *self.current(agent) {
            self.history[agent].push(path);
            self.broadcast(agent, tick);
        }

        true
    }

    /// The cheapest path of `agent` under the reservations of the newest plans it knows of
    /// the agents above it; `None` if there is none.
    fn plan(&mut self, agent: AgentId) -> Option<Path> {
        self.stats.generated += 1;
        let mut constraints = ConstraintSet::new();
        for path in self.above(agent) {
            reserve(&mut constraints, agent, path, None);
        }

        plan(
            self.instance.graph(),
            self.instance.agents()[agent],
            &constraints,
            &self.distances[agent],
        )
    }

    /// The newest plans `agent` knows of the agents above it.
    fn above(&self, agent: AgentId) -> impl Iterator<Item = &Path> {
        self.known[agent]
            .iter()
            .enumerate()
            .filter_map(|(other, version)| version.map(|version| &self.history[other][version]))
    }

    /// The current plan of `agent`.
    fn current(&self, agent: AgentId) -> &Path {
        self.history[agent].last().expect("every agent has planned")
    }

    /// Sends the current plan of `agent` to every agent below it, each arriving after its
    /// own delay from `tick`.
    fn broadcast(&mut self, agent: AgentId, tick: u64) {
        let version = self.history[agent].len() - 1;
        let delays = usize::try_from(self.max_delay).expect("u32 fits in usize");
        for to in agent + 1..self.instance.agents().len() {
            let delay = 1 + self.delays.below(delays) as u64;
            self.in_flight.insert((tick + delay, to, agent, version));
        }
    }

    /// The current plans, trimmed, as a solution.
    fn solution(&mut self) -> Solution {
        let mut paths: Vec<Path> = self
            .history
            .iter_mut()
            .map(|plans| plans.pop().expect("every agent has planned"))
            .collect();
        paths.iter_mut().for_each(trim_trailing_waits);
        debug_assert_eq!(
            first_conflict(&paths),
            None,
            "at quiescence every agent avoids the current plans above it"
        );

        Solution::new(paths)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for ADPP.

    use super::*;
    use crate::graph::{AdjacencyList, Grid};
    use crate::instance::Agent;
    use heuristic_search::Limits;

    /// Four agents crossing a free 4x4 grid in both directions.
    fn crossing() -> Instance<Grid> {
        let grid = Grid::new(4, 4);
        let agents = Vec::from([
            Agent { start: 0, goal: 15 },
            Agent { start: 15, goal: 0 },
            Agent { start: 3, goal: 12 },
            Agent { start: 12, goal: 3 },
        ]);

        Instance::new(grid, agents).expect("valid")
    }

    /// Every seed and delay converges to a conflict-free plan.
    #[test]
    fn converges_under_any_delays() {
        let instance = crossing();
        for max_delay in 1..=4 {
            for seed in 0..16 {
                let config = AdppConfig { max_delay, seed };
                let result = adpp(&instance, config, Limits::NONE);
                let solution = result.solution().expect("solved");

                assert_eq!(first_conflict(&solution.paths), None);
                assert!(result.stats.generated >= 4);
            }
        }
    }

    /// With delays of one tick, a conflict between the two agents repairs in one tick.
    #[test]
    fn synchronized_repairs_in_one_tick() {
        let grid = Grid::parse(&["@.@", "...", "@.@"]);
        let agents = Vec::from([
            Agent {
                start: grid.vertex(0, 1),
                goal: grid.vertex(2, 1),
            },
            Agent {
                start: grid.vertex(1, 0),
                goal: grid.vertex(1, 2),
            },
        ]);
        let instance = Instance::new(grid, agents).expect("valid");
        let config = AdppConfig {
            max_delay: 1,
            seed: 0,
        };
        let result = adpp(&instance, config, Limits::NONE);

        assert_eq!(result.solution().map(|s| s.sum_of_costs), Some(5));
        assert_eq!(result.stats.expanded, 1);
        assert_eq!(result.stats.generated, 3);
    }

    /// An agent above that finished on the only way through blocks the agent below.
    #[test]
    fn fails_behind_a_finished_agent() {
        let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 2), (1, 3)]);
        let agents = Vec::from([Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }]);
        let instance = Instance::new(graph, agents).expect("valid");

        let result = adpp(&instance, AdppConfig::default(), Limits::NONE);
        assert_eq!(result.outcome, Outcome::Exhausted);
    }

    /// An unreachable goal is unsolvable; a tiny budget stops the protocol.
    #[test]
    fn unsolvable_and_budget() {
        let grid = Grid::parse(&[".@."]);
        let agents = Vec::from([Agent { start: 0, goal: 2 }]);
        let instance = Instance::new(grid, agents).expect("valid");
        let result = adpp(&instance, AdppConfig::default(), Limits::NONE);
        assert_eq!(result.outcome, Outcome::Unsolvable);

        let result = adpp(&crossing(), AdppConfig::default(), Limits::expansions(0));
        assert_eq!(result.outcome, Outcome::LimitReached);
    }

    /// A delay of 0 is rejected.
    #[test]
    #[should_panic(expected = "at least one tick")]
    fn zero_delay_panics() {
        let config = AdppConfig {
            max_delay: 0,
            seed: 0,
        };
        let _ = adpp(&crossing(), config, Limits::NONE);
    }
}
