//! Cooperative A* ([`ca_star`]) and windowed hierarchical cooperative A* ([`whca_star`],
//! Silver 2005): agents plan one after another over a shared space-time reservation
//! table, the fast baseline for many agents.
//!
//! Each agent's plan is the cheapest path under the reservations of the agents planned
//! before it ([`reserve`]). The low level is space-time A* guided by the true distance to
//! the goal, which Silver's *hierarchical* variant computes with reverse resumable A*; a
//! distance table per agent gives it here.
//!
//! - **CA\*** plans every agent once, in index order, for good: an agent that finished
//!   holds its goal forever, so a later agent that needs to pass it has no path.
//! - **WHCA\*** reserves only a window of [`WhcaConfig::window`] steps. Every round, all
//!   agents plan from where they stand, the first agent of the round rotating, and then
//!   execute the first [`WhcaConfig::step`] steps of their plans. Beyond the window an
//!   agent ignores the others and follows its distance table. The run ends at the first
//!   round that starts with every agent at its goal.
//!
//! ```mermaid
//! flowchart TD
//!     start[every agent at its start] --> done{every agent at its goal?}
//!     done -->|yes| solved[Solved]
//!     done -->|no| budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| round[plan the window, agent after agent, from the rotating first one]
//!     round -->|an agent has no path| exhausted[Exhausted]
//!     round --> execute[execute the first steps of every plan]
//!     execute --> done
//! ```
//!
//! Within a round, every window plan avoids the reservations of the plans before it, so
//! the window plans are conflict-free up to the window's end, and the executed steps lie
//! inside it. Rounds follow each other in time, so the executed plan is conflict-free.

use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex};
use crate::instance::{Agent, AgentId, Instance};
use crate::low_level::{distance_tables, plan};
use crate::path::{Path, Time, position_at, trim_trailing_waits};
use crate::reservation::reserve;
use crate::solution::{MapfResult, Outcome, Solution};

/// The settings of [`whca_star`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WhcaConfig {
    /// How many time steps ahead each round reserves, the paper's `w`; at least
    /// [`WhcaConfig::step`]. The default is 16.
    pub window: Time,
    /// How many steps of each round's plans the agents execute before the next round
    /// replans them; at least 1 and at most [`WhcaConfig::window`]. The default is 8, half
    /// the window, as Silver suggests.
    pub step: Time,
}

impl Default for WhcaConfig {
    /// A window of 16 steps, replanned every 8.
    fn default() -> Self {
        Self {
            window: 16,
            step: 8,
        }
    }
}

/// Finds a conflict-free plan with cooperative A*: each agent, in index order, takes the
/// cheapest path that avoids the paths of the agents before it, including where they stay
/// at their goals for good.
///
/// The plan has no quality guarantee, and CA* is not complete: an agent fails when the
/// agents before it block it for good, for example by finishing on a vertex it has to
/// pass. The result is [`Outcome::Unsolvable`] only if an agent cannot reach its goal even
/// alone, and [`Outcome::Exhausted`] if an agent finds no path around the others.
///
/// `budget` is asked before every agent's search; [`MapfResult::stats`] counts searches in
/// both [`Stats::expanded`] and [`Stats::generated`].
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, ca_star, first_conflict};
///
/// let grid = Grid::parse(&["@.@", "...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 1), goal: grid.vertex(2, 1) },
///     Agent { start: grid.vertex(1, 0), goal: grid.vertex(1, 2) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = ca_star(&instance, Limits::expansions(100));
/// let solution = result.solution().expect("the second agent waits");
/// assert_eq!(first_conflict(&solution.paths), None);
/// assert_eq!(solution.sum_of_costs, 5);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn ca_star<G: Graph>(instance: &Instance<G>, mut budget: impl Budget<Stats>) -> MapfResult {
    let distances = distance_tables(instance.graph(), instance.agents());
    let mut stats = Stats::default();
    if !reachable(instance.agents(), &distances) {
        return MapfResult {
            outcome: Outcome::Unsolvable,
            stats,
        };
    }

    let mut paths: Vec<Path> = Vec::with_capacity(instance.agents().len());
    for (agent, &task) in instance.agents().iter().enumerate() {
        if budget.exhausted(&stats) {
            return MapfResult {
                outcome: Outcome::LimitReached,
                stats,
            };
        }

        stats.expanded += 1;
        stats.generated += 1;
        let mut constraints = ConstraintSet::new();
        for path in &paths {
            reserve(&mut constraints, agent, path, None);
        }

        let Some(path) = plan(instance.graph(), task, &constraints, &distances[agent]) else {
            return MapfResult {
                outcome: Outcome::Exhausted,
                stats,
            };
        };
        paths.push(path);
    }

    MapfResult {
        outcome: Outcome::Solved(solution(paths)),
        stats,
    }
}

/// Finds a conflict-free plan with windowed hierarchical cooperative A*: rounds of
/// cooperative planning over the next `config.window` steps, of which the agents execute
/// `config.step` before the next round.
///
/// The first agent of a round rotates (agent `r mod n` in round `r`), so an agent that
/// had to make way in one round goes first in a later one. The plan has no quality
/// guarantee, and WHCA* is not complete: agents can oscillate forever, so always pass a
/// budget, and a round fails when an agent finds no path around the window plans before
/// it. The result is [`Outcome::Unsolvable`] only if an agent cannot reach its goal even
/// alone, and [`Outcome::Exhausted`] if a round fails.
///
/// `budget` is asked before every round; [`MapfResult::stats`] counts rounds in
/// [`Stats::expanded`] and single-agent searches in [`Stats::generated`].
///
/// # Panics
///
/// Panics if `config.step` is `0` or greater than `config.window`.
///
/// # Examples
///
/// ```
/// use mapf_search::{AdjacencyList, Agent, Instance, Limits, WhcaConfig, first_conflict};
/// use mapf_search::whca_star;
///
/// // Agent 0 sits on its goal, vertex 1, where agent 1 must pass; vertex 3 is a pocket.
/// //
/// //   0 - 1 - 2
/// //       |
/// //       3
/// let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 2), (1, 3)]);
/// let agents = vec![Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }];
/// let instance = Instance::new(graph, agents)?;
///
/// let config = WhcaConfig { window: 4, step: 2 };
/// let result = whca_star(&instance, config, Limits::expansions(100));
/// let solution = result.solution().expect("agent 0 makes way in the second round");
/// assert_eq!(first_conflict(&solution.paths), None);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn whca_star<G: Graph>(
    instance: &Instance<G>,
    config: WhcaConfig,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    assert!(
        1 <= config.step && config.step <= config.window,
        "WHCA* executes between 1 and `window` steps per round"
    );

    let distances = distance_tables(instance.graph(), instance.agents());
    let mut run = Windowed {
        instance,
        config,
        distances,
        here: instance.agents().iter().map(|agent| agent.start).collect(),
        paths: instance
            .agents()
            .iter()
            .map(|agent| Vec::from([agent.start]))
            .collect(),
        stats: Stats::default(),
    };
    if !reachable(instance.agents(), &run.distances) {
        return MapfResult {
            outcome: Outcome::Unsolvable,
            stats: run.stats,
        };
    }

    let outcome = run.solve(&mut budget);

    MapfResult {
        outcome,
        stats: run.stats,
    }
}

/// Returns `true` if every agent can reach its goal alone.
fn reachable(agents: &[Agent], distances: &[Vec<Option<u32>>]) -> bool {
    agents
        .iter()
        .zip(distances)
        .all(|(agent, distances)| distances[agent.start].is_some())
}

/// The solution of conflict-free `paths`, each trimmed to its final arrival.
fn solution(mut paths: Vec<Path>) -> Solution {
    paths.iter_mut().for_each(trim_trailing_waits);
    debug_assert_eq!(
        crate::conflict::first_conflict(&paths),
        None,
        "reservations keep the plans apart"
    );

    Solution::new(paths)
}

/// The state of one WHCA* run.
struct Windowed<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// The window and step sizes.
    config: WhcaConfig,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// Where each agent stands at the start of the current round.
    here: Vec<Vertex>,
    /// The executed path of each agent so far, ending at [`Windowed::here`].
    paths: Vec<Path>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Windowed<'_, G> {
    /// Runs rounds until every agent stands at its goal, a round fails, or the budget runs
    /// out.
    fn solve(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let mut round = 0;
        loop {
            let finished = self
                .instance
                .agents()
                .iter()
                .zip(&self.here)
                .all(|(agent, &here)| here == agent.goal);
            if finished {
                return Outcome::Solved(solution(core::mem::take(&mut self.paths)));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.stats.expanded += 1;
            let Some(windows) = self.plan_round(round) else {
                return Outcome::Exhausted;
            };

            self.execute(&windows);
            round += 1;
        }
    }

    /// The window plan of every agent, indexed by agent: `window + 1` positions from where
    /// it stands. Agents plan in turn from agent `round mod n` on, each avoiding the window
    /// plans before it. `None` if one finds no path.
    fn plan_round(&mut self, round: usize) -> Option<Vec<Path>> {
        let count = self.instance.agents().len();
        let mut windows: Vec<Path> = alloc::vec![Vec::new(); count];
        let mut constraints = ConstraintSet::new();
        for agent in (0..count).map(|offset| (round + offset) % count) {
            let window = self.plan_window(agent, &constraints)?;
            reserve(&mut constraints, agent, &window, Some(self.config.window));
            windows[agent] = window;
        }

        Some(windows)
    }

    /// The first `window + 1` positions of the cheapest path of `agent` from where it
    /// stands under `constraints`, the reservations of the agents planned before it this
    /// round. `None` if there is no such path.
    fn plan_window(&mut self, agent: AgentId, constraints: &ConstraintSet) -> Option<Path> {
        self.stats.generated += 1;
        let task = Agent {
            start: self.here[agent],
            goal: self.instance.agents()[agent].goal,
        };
        let path = plan(
            self.instance.graph(),
            task,
            constraints,
            &self.distances[agent],
        )?;

        Some(
            (0..=self.config.window)
                .map(|time| position_at(&path, time))
                .collect(),
        )
    }

    /// Moves every agent along the first [`WhcaConfig::step`] steps of its window plan.
    fn execute(&mut self, windows: &[Path]) {
        let step = usize::try_from(self.config.step).expect("the step fits in usize");
        for ((path, here), window) in self.paths.iter_mut().zip(&mut self.here).zip(windows) {
            path.extend_from_slice(&window[1..=step]);
            *here = window[step];
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for CA* and WHCA*.

    use super::*;
    use crate::conflict::first_conflict;
    use crate::graph::{AdjacencyList, Grid};
    use heuristic_search::Limits;

    /// `0 - 1 - 2` with a pocket `3` off vertex 1; agent 0 sits on its goal, vertex 1, and
    /// agent 1 must pass it.
    fn pocket() -> Instance<AdjacencyList> {
        let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 2), (1, 3)]);
        let agents = Vec::from([Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }]);

        Instance::new(graph, agents).expect("valid")
    }

    /// CA* in index order lets agent 0 hold its goal for good, so agent 1 has no path.
    #[test]
    fn ca_star_fails_behind_a_finished_agent() {
        let result = ca_star(&pocket(), Limits::NONE);

        assert_eq!(result.outcome, Outcome::Exhausted);
        assert_eq!(result.stats.expanded, 2);
    }

    /// WHCA* reserves only the window, and agent 1 goes first in the second round, so
    /// agent 0 makes way.
    #[test]
    fn whca_star_rotates_past_a_finished_agent() {
        let config = WhcaConfig { window: 4, step: 2 };
        let result = whca_star(&pocket(), config, Limits::expansions(10));
        let solution = result.solution().expect("solved");

        assert_eq!(first_conflict(&solution.paths), None);
        assert_eq!(solution.paths[0].last(), Some(&1));
        assert_eq!(solution.paths[1].last(), Some(&2));
        assert!(result.stats.expanded >= 2);
    }

    /// Swapping the ends of a corridor never succeeds; the budget stops WHCA*.
    #[test]
    fn whca_star_stops_at_the_budget() {
        let grid = Grid::new(3, 1);
        let agents = Vec::from([Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }]);
        let instance = Instance::new(grid, agents).expect("valid");
        let result = whca_star(&instance, WhcaConfig::default(), Limits::expansions(5));

        assert!(
            matches!(result.outcome, Outcome::LimitReached | Outcome::Exhausted),
            "{:?}",
            result.outcome
        );
    }

    /// An unreachable goal is proven unsolvable before any search.
    #[test]
    fn unreachable_goal_is_unsolvable() {
        let grid = Grid::parse(&[".@."]);
        let agents = Vec::from([Agent { start: 0, goal: 2 }]);
        let instance = Instance::new(grid, agents).expect("valid");

        assert_eq!(
            ca_star(&instance, Limits::NONE).outcome,
            Outcome::Unsolvable
        );
        let result = whca_star(&instance, WhcaConfig::default(), Limits::NONE);
        assert_eq!(result.outcome, Outcome::Unsolvable);
        assert_eq!(result.stats, Stats::default());
    }

    /// A window as long as the step still keeps the executed plan conflict-free.
    #[test]
    fn whca_star_with_step_equal_to_window() {
        let grid = Grid::new(3, 3);
        let agents = Vec::from([
            Agent { start: 0, goal: 8 },
            Agent { start: 8, goal: 0 },
            Agent { start: 2, goal: 6 },
        ]);
        let instance = Instance::new(grid, agents).expect("valid");
        let config = WhcaConfig { window: 3, step: 3 };
        let result = whca_star(&instance, config, Limits::expansions(50));
        let solution = result.solution().expect("solved");

        assert_eq!(first_conflict(&solution.paths), None);
    }

    /// A step of 0 is rejected.
    #[test]
    #[should_panic(expected = "between 1 and `window`")]
    fn zero_step_panics() {
        let config = WhcaConfig { window: 4, step: 0 };
        let _ = whca_star(&pocket(), config, Limits::NONE);
    }
}
