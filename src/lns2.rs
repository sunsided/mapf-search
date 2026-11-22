//! MAPF-LNS2 ([`lns2`], Li et al. 2022): repairs a plan with collisions by large
//! neighborhood search until it has none. Fast, neither optimal nor complete.
//!
//! The initial plan comes from prioritized planning in a random order, where each agent
//! takes the path with the fewest collisions with the agents planned before it, so an
//! agent that cannot avoid them all still gets a path. Then, as long as two paths collide,
//! one iteration:
//!
//! 1. picks a *neighborhood* of agents with one of three methods (collision-based,
//!    failure-based or random; see [`neighborhood`]), chosen by adaptive LNS: each method
//!    has a weight, raised by the collisions it removes;
//! 2. replans those agents one after another in a random order, each with the fewest
//!    collisions with every other current path ([`low_level`]);
//! 3. keeps the new paths if the number of colliding agent pairs did not grow, else
//!    restores the old ones.
//!
//! ```mermaid
//! flowchart TD
//!     init[prioritized planning, fewest collisions] --> count{colliding pairs?}
//!     count -->|none| solved[Solved]
//!     count --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| pick[pick a method by weight, build a neighborhood]
//!     pick --> replan[replan its agents in random order]
//!     replan --> accept{more colliding pairs than before?}
//!     accept -->|no| keep[keep the new paths]
//!     accept -->|yes| restore[restore the old paths]
//!     keep --> weight[update the method's weight]
//!     restore --> weight
//!     weight --> count
//! ```

mod low_level;
mod neighborhood;
mod random;

use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use self::low_level::fewest_collisions;
use self::neighborhood::Method;
use self::random::Random;
use crate::conflict::conflicts;
use crate::conflict_table::ConflictTable;
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::low_level::distance_tables;
use crate::path::{Path, Time};
use crate::solution::{MapfResult, Outcome, Solution};

/// How fast the adaptive weights follow recent success: the paper's reaction factor `γ`.
const REACTION: f64 = 0.1;

/// The settings of [`lns2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lns2Config {
    /// How many agents each iteration replans, the paper's `N`; at least 1, at most the
    /// number of agents (larger values are capped). The default is the paper's 8.
    pub neighborhood_size: usize,
    /// The seed of every random choice: the same instance, configuration and budget give
    /// the same result.
    pub seed: u64,
}

impl Default for Lns2Config {
    /// Neighborhoods of 8 agents, seed `0`.
    fn default() -> Self {
        Self {
            neighborhood_size: 8,
            seed: 0,
        }
    }
}

/// Finds a conflict-free plan with MAPF-LNS2: an initial plan that may have collisions,
/// then large neighborhood search that replans a few agents at a time until no two paths
/// collide.
///
/// The plan has no quality guarantee; LNS2 aims at a feasible plan fast. It is not
/// complete either: on an unsolvable instance it runs until its budget, so always pass
/// one. The result is [`Outcome::Unsolvable`] only if an agent cannot reach its goal even
/// alone, and never [`Outcome::Exhausted`].
///
/// `budget` is asked before every repair iteration; [`MapfResult::stats`] counts
/// iterations in [`Stats::expanded`] and single-agent searches, the initial ones
/// included, in [`Stats::generated`].
///
/// # Panics
///
/// Panics if `config.neighborhood_size` is `0`.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Lns2Config, first_conflict, lns2};
///
/// // Four agents cross a free 4x4 grid.
/// let grid = Grid::new(4, 4);
/// let agents = vec![
///     Agent { start: 0, goal: 15 },
///     Agent { start: 3, goal: 12 },
///     Agent { start: 12, goal: 3 },
///     Agent { start: 15, goal: 0 },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let result = lns2(&instance, Lns2Config::default(), Limits::expansions(1_000));
/// let solution = result.solution().expect("LNS2 repairs every collision");
/// assert_eq!(first_conflict(&solution.paths), None);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn lns2<G: Graph>(
    instance: &Instance<G>,
    config: Lns2Config,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    assert!(config.neighborhood_size > 0, "a neighborhood has an agent");

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

    let mut repair = Repair {
        instance,
        distances,
        size: config.neighborhood_size.min(instance.agents().len()),
        random: Random::new(config.seed),
        paths: vec![Vec::new(); instance.agents().len()],
        weights: [1.0; Method::ALL.len()],
        stats: Stats::default(),
    };
    let outcome = repair.run(&mut budget);

    MapfResult {
        outcome,
        stats: repair.stats,
    }
}

/// The state of one [`lns2`] run.
struct Repair<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// Distances to each agent's goal, indexed by agent; every start reaches its goal.
    distances: Vec<Vec<Option<u32>>>,
    /// The neighborhood size, between 1 and the number of agents.
    size: usize,
    /// The source of every random choice.
    random: Random,
    /// The current path of each agent; empty only before the agent is first planned.
    paths: Vec<Path>,
    /// The adaptive weight of each neighborhood method, indexed like [`Method::ALL`].
    weights: [f64; Method::ALL.len()],
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Repair<'_, G> {
    /// Plans every agent, then repairs until no collision is left or the budget ends.
    fn run(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let mut order: Vec<AgentId> = (0..self.paths.len()).collect();
        self.random.shuffle(&mut order);
        self.replan(&order);

        loop {
            let collisions = Collisions::of(&self.paths);
            if collisions.pairs == 0 {
                return Outcome::Solved(Solution::new(self.paths.clone()));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.iterate(&collisions);
        }
    }

    /// One repair iteration on a plan with `collisions`: pick a method and a neighborhood,
    /// replan it, keep or restore, and update the method's weight.
    fn iterate(&mut self, collisions: &Collisions) {
        self.stats.expanded += 1;

        let index = self
            .random
            .weighted(&self.weights)
            .unwrap_or_else(|| self.random.below(Method::ALL.len()));
        let mut agents = self.neighborhood(Method::ALL[index], collisions);
        self.random.shuffle(&mut agents);

        let old: Vec<Path> = agents
            .iter()
            .map(|&agent| self.paths[agent].clone())
            .collect();
        self.replan(&agents);
        let after = Collisions::of(&self.paths).pairs;
        if after > collisions.pairs {
            for (&agent, path) in agents.iter().zip(old) {
                self.paths[agent] = path;
            }
        }

        let gain =
            f64::from(u32::try_from(collisions.pairs.saturating_sub(after)).unwrap_or(u32::MAX));
        self.weights[index] = REACTION * gain + (1.0 - REACTION) * self.weights[index];
    }

    /// Replans `agents` in their order, each with the fewest collisions with every other
    /// agent's path, the ones not yet replanned among `agents` left out.
    fn replan(&mut self, agents: &[AgentId]) {
        for (position, &agent) in agents.iter().enumerate() {
            let pending = &agents[position..];
            let others = self
                .paths
                .iter()
                .enumerate()
                .filter(|(other, path)| !path.is_empty() && !pending.contains(other))
                .map(|(_, path)| path);
            let table = ConflictTable::of(others.clone());
            let last = others.map(Vec::len).max().unwrap_or(0);
            let last = Time::try_from(last).expect("path lengths fit in u32 time steps");

            self.stats.generated += 1;
            self.paths[agent] = fewest_collisions(
                self.instance.graph(),
                self.instance.agents()[agent],
                &self.distances[agent],
                &table,
                last,
            )
            .expect("every goal is reachable");
        }
    }
}

/// The collision graph of a plan: which agents' paths collide.
struct Collisions {
    /// The agents each agent collides with, indexed by agent; sorted, without duplicates.
    neighbors: Vec<Vec<AgentId>>,
    /// The number of colliding agent pairs, the paper's `CP`.
    pairs: usize,
}

impl Collisions {
    /// The collision graph of `paths`.
    ///
    /// shortcut: scans every pair of paths at every time step, `O(m^2 T)` for `m` agents;
    /// fine at this crate's instance sizes, upgrade to an incremental table for hundreds.
    fn of(paths: &[Path]) -> Self {
        let mut neighbors = vec![Vec::new(); paths.len()];
        for conflict in conflicts(paths) {
            let (a, b) = conflict.agents();
            neighbors[a].push(b);
            neighbors[b].push(a);
        }

        for list in &mut neighbors {
            list.sort_unstable();
            list.dedup();
        }

        let pairs = neighbors.iter().map(Vec::len).sum::<usize>() / 2;

        Self { neighbors, pairs }
    }

    /// The number of agents `agent` collides with.
    fn degree(&self, agent: AgentId) -> usize {
        self.neighbors[agent].len()
    }
}

#[cfg(test)]
mod tests {
    //! Runs of LNS2 on small instances.

    use super::*;

    use heuristic_search::Limits;

    use crate::conflict::first_conflict;
    use crate::graph::Grid;
    use crate::instance::Agent;

    /// Twelve agents on a free 5x5 grid, from the first cells to the mirrored ones: the
    /// initial plan collides for most seeds.
    fn crowded() -> Instance<Grid> {
        let agents = (0..12)
            .map(|i| Agent {
                start: i,
                goal: 24 - i,
            })
            .collect();

        Instance::new(Grid::new(5, 5), agents).expect("valid")
    }

    /// Every seed and neighborhood size repairs the crowded grid, some only after repair
    /// iterations, and the same seed gives the same plan.
    #[test]
    fn repairs_deterministically() {
        let instance = crowded();
        let mut iterations = 0;
        for seed in 0..4 {
            for neighborhood_size in [2, 4, 8, 100] {
                let config = Lns2Config {
                    neighborhood_size,
                    seed,
                };
                let first = lns2(&instance, config, Limits::expansions(10_000));
                let second = lns2(&instance, config, Limits::expansions(10_000));
                let solution = first.solution().expect("solvable");

                assert_eq!(first, second);
                assert_eq!(first_conflict(&solution.paths), None);
                for (path, agent) in solution.paths.iter().zip(instance.agents()) {
                    assert_eq!(path.first(), Some(&agent.start));
                    assert_eq!(path.last(), Some(&agent.goal));
                }

                iterations += first.stats.expanded;
            }
        }

        assert!(iterations > 0, "some initial plan had collisions");
    }

    /// The corridor swap is unsolvable; LNS2 stops at its budget.
    #[test]
    fn stops_at_the_budget_when_unsolvable() {
        let grid = Grid::new(3, 1);
        let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
        let instance = Instance::new(grid, agents).expect("valid");

        let result = lns2(&instance, Lns2Config::default(), Limits::expansions(50));

        assert_eq!(result.outcome, Outcome::LimitReached);
        assert_eq!(result.stats.expanded, 50);
    }

    /// An unreachable goal is proven at once.
    #[test]
    fn proves_an_unreachable_goal() {
        let grid = Grid::parse(&[".@."]);
        let instance = Instance::new(grid, vec![Agent { start: 0, goal: 2 }]).expect("valid");

        let result = lns2(&instance, Lns2Config::default(), Limits::NONE);

        assert_eq!(result.outcome, Outcome::Unsolvable);
    }
}
