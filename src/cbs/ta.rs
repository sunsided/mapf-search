//! CBS-TA (Hönig et al. 2018): conflict-based search with optimal task assignment, over a
//! [`TaskInstance`] whose agents each have a set of candidate goals.
//!
//! The search runs on a forest: one constraint tree per agent-to-goal assignment, each
//! searched like plain CBS. Assignments come cheapest first from Murty's ranking
//! ([`Ranked`]) and a tree is opened only once its assignment is cheaper than the best
//! open node, so most of the factorially many trees are never built.

use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use super::{CbsConfig, Search};
use crate::assignment::Ranked;
use crate::graph::{Graph, Vertex, distances_to};
use crate::instance::TaskInstance;
use crate::path::Objective;
use crate::solution::{MapfResult, Outcome};

/// Assigns each agent one of its candidate goals and finds a conflict-free plan to them,
/// optimal for `objective` over all assignments and plans, with conflict-based search
/// with task assignment and every improvement of [`CbsConfig::improved`].
///
/// See [`cbs_ta_with`] for the budget and the tie-breaking.
///
/// # Examples
///
/// ```
/// use mapf_search::{Grid, Limits, Objective, TaskAgent, TaskInstance, cbs_ta};
///
/// // Two agents in a corridor, both allowed at either end: keeping their sides is free
/// // of conflicts, swapping is not even possible.
/// let grid = Grid::new(4, 1);
/// let ends = vec![0, 3];
/// let agents = vec![
///     TaskAgent { start: 1, goals: ends.clone() },
///     TaskAgent { start: 2, goals: ends },
/// ];
/// let instance = TaskInstance::new(grid, agents)?;
///
/// let result = cbs_ta(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
/// let solution = result.solution().expect("solvable");
/// assert_eq!(solution.sum_of_costs, 2);
/// assert_eq!(solution.paths[0].last(), Some(&0));
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn cbs_ta<G: Graph>(
    instance: &TaskInstance<G>,
    objective: Objective,
    budget: impl Budget<Stats>,
) -> MapfResult {
    cbs_ta_with(instance, CbsConfig::improved(objective), budget)
}

/// Assigns each agent one of its candidate goals and finds a conflict-free plan to them,
/// optimal for `config.objective` over all assignments and plans, with conflict-based
/// search with task assignment and the improvements `config` selects.
///
/// Each assignment opens its own constraint tree, searched as by [`cbs_with`]. An
/// assignment costs the objective of its agents' shortest paths alone; trees are opened
/// cheapest first, each once it is cheaper than every open node, so ties between trees
/// go to the earlier one. The goal of agent `i` in the result is the last vertex of
/// `paths[i]`.
///
/// `budget` is asked before every constraint-tree expansion; [`MapfResult::stats`] counts
/// expanded nodes in [`Stats::expanded`] and nodes with a path, each tree's root included,
/// in [`Stats::generated`]. Ranking the assignments runs outside the budget: one
/// Hungarian-method solve per agent and opened tree, a few more per tree for
/// [`Objective::Makespan`].
///
/// The result is [`Outcome::Unsolvable`] if no assignment gives every agent a distinct
/// reachable goal, or if every tree is proven unsolvable.
///
/// # Examples
///
/// ```
/// use mapf_search::{CbsConfig, Grid, Limits, Objective, TaskAgent, TaskInstance, cbs_ta_with};
///
/// // Three agents and three goals on a free grid; any agent may take any goal.
/// let grid = Grid::new(3, 3);
/// let goals = vec![6, 7, 8];
/// let agents = [0, 1, 2]
///     .map(|start| TaskAgent { start, goals: goals.clone() })
///     .to_vec();
/// let instance = TaskInstance::new(grid, agents)?;
///
/// let config = CbsConfig::plain(Objective::Makespan);
/// let result = cbs_ta_with(&instance, config, Limits::NONE);
/// assert_eq!(result.solution().expect("solvable").makespan, 2);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
///
/// [`cbs_with`]: crate::cbs_with
pub fn cbs_ta_with<G: Graph>(
    instance: &TaskInstance<G>,
    config: CbsConfig,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    let graph = instance.graph();
    let starts: Vec<Vertex> = instance.agents().iter().map(|agent| agent.start).collect();
    let goals = candidate_goals(instance);
    let distances: Vec<Vec<Option<u32>>> = goals
        .iter()
        .map(|&goal| distances_to(graph, goal))
        .collect();
    let costs = cost_matrix(instance, &goals, &distances);

    let ranked = Ranked::new(&costs, config.objective);
    if ranked.peek().is_none() {
        return MapfResult {
            outcome: Outcome::Unsolvable,
            stats: Stats::default(),
        };
    }

    let mut search = Search::forest(graph, starts, goals, distances, Some(ranked), config);
    let outcome = search.run(&mut budget);

    MapfResult {
        outcome,
        stats: search.stats,
    }
}

/// Every vertex some agent of `instance` may end at, sorted and without duplicates.
fn candidate_goals<G: Graph>(instance: &TaskInstance<G>) -> Vec<Vertex> {
    let mut goals: Vec<Vertex> = instance
        .agents()
        .iter()
        .flat_map(|agent| agent.goals.iter().copied())
        .collect();
    goals.sort_unstable();
    goals.dedup();

    goals
}

/// The assignment costs: for each agent and each of `goals`, the distance from the agent's
/// start to the goal (`distances` is indexed like `goals`), or `None` if the goal is not a
/// candidate of the agent or out of its reach.
fn cost_matrix<G: Graph>(
    instance: &TaskInstance<G>,
    goals: &[Vertex],
    distances: &[Vec<Option<u32>>],
) -> Vec<Vec<Option<u32>>> {
    instance
        .agents()
        .iter()
        .map(|agent| {
            goals
                .iter()
                .zip(distances)
                .map(|(goal, table)| {
                    agent
                        .goals
                        .contains(goal)
                        .then(|| table[agent.start])
                        .flatten()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! CBS-TA against CBS over every assignment, by brute force.

    use super::*;
    use alloc::vec;

    use heuristic_search::Limits;
    use proptest::prelude::*;

    use crate::cbs::{CbsConfig, cbs_with};
    use crate::graph::Grid;
    use crate::instance::{Agent, Instance, TaskAgent};
    use crate::solution::Solution;

    /// The expansion budget of every run in these tests.
    const BUDGET: u64 = 2_000;

    /// What CBS over every assignment says about an instance.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Expected {
        /// The best cost over all assignments.
        Cost(u32),
        /// Every assignment was proven unsolvable.
        Unsolvable,
        /// A run hit its budget, so the optimum is unknown.
        Unknown,
    }

    /// The optimum of `config.objective` over every assignment of `instance`, each solved
    /// by CBS with `config`.
    fn brute_force(instance: &TaskInstance<Grid>, config: CbsConfig) -> Expected {
        let mut best: Option<u32> = None;
        for goals in injective(instance) {
            let agents = instance
                .agents()
                .iter()
                .zip(&goals)
                .map(|(agent, &goal)| Agent {
                    start: agent.start,
                    goal,
                })
                .collect();
            let fixed = Instance::new(instance.graph().clone(), agents).expect("distinct goals");
            match cbs_with(&fixed, config, Limits::expansions(BUDGET)).outcome {
                Outcome::Solved(solution) => {
                    let cost = solution.cost(config.objective);
                    best = Some(best.map_or(cost, |b| b.min(cost)));
                }
                Outcome::Unsolvable => {}
                Outcome::LimitReached | Outcome::Exhausted => return Expected::Unknown,
            }
        }

        best.map_or(Expected::Unsolvable, Expected::Cost)
    }

    /// Every choice of one candidate goal per agent with no goal chosen twice.
    fn injective(instance: &TaskInstance<Grid>) -> Vec<Vec<Vertex>> {
        let mut choices = vec![Vec::new()];
        for agent in instance.agents() {
            let mut extended = Vec::new();
            for chosen in &choices {
                for &goal in &agent.goals {
                    if !chosen.contains(&goal) {
                        let mut next: Vec<Vertex> = chosen.clone();
                        next.push(goal);
                        extended.push(next);
                    }
                }
            }

            choices = extended;
        }

        choices
    }

    /// Up to 3 agents on a 3x3 grid with up to 2 walls, each with 1 to 3 candidate goals.
    fn instance() -> impl Strategy<Value = TaskInstance<Grid>> {
        let walls = prop::collection::vec(0_usize..9, 0..=2);
        let agents = prop::collection::vec(
            (0_usize..9, prop::collection::vec(0_usize..9, 1..=3)),
            1..=3,
        );

        (walls, agents).prop_map(|(walls, agents)| {
            let mut rows = [*b"...", *b"...", *b"..."];
            for wall in walls {
                rows[wall / 3][wall % 3] = b'@';
            }

            let rows: Vec<&str> = rows
                .iter()
                .map(|row| core::str::from_utf8(row).expect("ascii"))
                .collect();
            let grid = Grid::parse(&rows);
            let mut seen = Vec::new();
            let agents = agents
                .into_iter()
                .filter(|(start, _)| {
                    let fresh = !seen.contains(start);
                    seen.push(*start);
                    fresh
                })
                .map(|(start, goals)| TaskAgent { start, goals })
                .collect();

            TaskInstance::new(grid, agents).expect("distinct starts on the grid")
        })
    }

    /// Checks that `solution` starts every agent at its start and ends it at a distinct
    /// candidate goal of its own.
    fn assigns_candidates(instance: &TaskInstance<Grid>, solution: &Solution) -> bool {
        let ends: Vec<Vertex> = solution
            .paths
            .iter()
            .map(|path| *path.last().expect("non-empty"))
            .collect();
        let distinct = ends
            .iter()
            .enumerate()
            .all(|(i, end)| !ends[..i].contains(end));

        distinct
            && instance
                .agents()
                .iter()
                .zip(&ends)
                .zip(&solution.paths)
                .all(|((agent, end), path)| {
                    agent.goals.contains(end) && path.first() == Some(&agent.start)
                })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// CBS-TA finds the best cost over every assignment, for both objectives and with
        /// and without the improvements, and its plan uses distinct candidate goals.
        #[test]
        fn matches_cbs_over_every_assignment(
            instance in instance(),
            makespan in any::<bool>(),
            improved in any::<bool>(),
        ) {
            let objective = if makespan { Objective::Makespan } else { Objective::SumOfCosts };
            let config = if improved { CbsConfig::improved(objective) } else { CbsConfig::plain(objective) };
            let expected = brute_force(&instance, config);
            if expected == Expected::Unknown {
                return Ok(());
            }

            let result = cbs_ta_with(&instance, config, Limits::expansions(BUDGET * 64));
            match (&result.outcome, expected) {
                (Outcome::Solved(solution), Expected::Cost(cost)) => {
                    prop_assert_eq!(solution.cost(objective), cost);
                    prop_assert!(assigns_candidates(&instance, solution));
                    prop_assert_eq!(crate::conflict::first_conflict(&solution.paths), None);
                }
                (Outcome::Unsolvable | Outcome::LimitReached, Expected::Unsolvable) => {}
                (outcome, expected) => prop_assert!(false, "{outcome:?}, expected {expected:?}"),
            }
        }
    }

    /// With a single candidate goal per agent, CBS-TA is CBS.
    #[test]
    fn single_candidates_are_cbs() {
        let grid = Grid::parse(&["...", "@.@"]);
        let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
        let fixed = Instance::new(grid.clone(), agents.clone()).expect("valid");
        let tasks = TaskInstance::new(
            grid,
            agents
                .iter()
                .map(|agent| TaskAgent {
                    start: agent.start,
                    goals: vec![agent.goal],
                })
                .collect(),
        )
        .expect("valid");

        for objective in [Objective::SumOfCosts, Objective::Makespan] {
            let config = CbsConfig::improved(objective);
            let ta = cbs_ta_with(&tasks, config, Limits::expansions(BUDGET));
            let plain = cbs_with(&fixed, config, Limits::expansions(BUDGET));

            assert_eq!(ta.outcome, plain.outcome);
            assert_eq!(ta.stats, plain.stats);
        }
    }

    /// Without agents, both objectives solve with the empty plan, as CBS does.
    #[test]
    fn no_agents_is_the_empty_plan() {
        let tasks = TaskInstance::new(Grid::new(2, 1), Vec::new()).expect("valid");
        let fixed = Instance::new(Grid::new(2, 1), Vec::new()).expect("valid");

        for objective in [Objective::SumOfCosts, Objective::Makespan] {
            let ta = cbs_ta(&tasks, objective, Limits::NONE);
            let plain = crate::cbs::cbs(&fixed, objective, Limits::NONE);

            assert_eq!(ta.outcome, Outcome::Solved(Solution::new(Vec::new())));
            assert_eq!(ta.outcome, plain.outcome);
        }
    }

    /// More agents than distinct candidate goals is unsolvable without any search.
    #[test]
    fn too_few_goals_is_unsolvable() {
        let grid = Grid::new(3, 1);
        let agents = vec![
            TaskAgent {
                start: 0,
                goals: vec![1],
            },
            TaskAgent {
                start: 2,
                goals: vec![1],
            },
        ];
        let instance = TaskInstance::new(grid, agents).expect("valid");

        let result = cbs_ta(&instance, Objective::SumOfCosts, Limits::NONE);

        assert_eq!(result.outcome, Outcome::Unsolvable);
        assert_eq!(result.stats, Stats::default());
    }
}
