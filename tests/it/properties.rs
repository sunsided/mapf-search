//! Property tests: on random tiny grids with 2-3 agents, CBS in every configuration agrees
//! with the joint-space oracle on solvability and on the optimal sum of costs and makespan.

use heuristic_search::Limits;
use mapf_search::{
    Agent, CbsConfig, Grid, Instance, MapfResult, Objective, Outcome, Vertex, cbs_with,
    first_conflict,
};
use proptest::prelude::*;

use crate::oracle::{assert_valid_plan, optimal_cost};

/// Constraint-tree expansions CBS may use on solvable instances.
const SOLVABLE_BUDGET: u64 = 20_000;

/// Constraint-tree expansions CBS may use on instances the oracle proves unsolvable.
const UNSOLVABLE_BUDGET: u64 = 300;

/// A random grid of 2-3 by 2-3 cells (3 by 4 for two agents), about one cell in six
/// blocked, with 2-3 agents on distinct free starts and distinct free goals.
fn instance() -> impl Strategy<Value = (Grid, Vec<Agent>)> {
    (2..=3_usize)
        .prop_flat_map(|agents| {
            let height = if agents == 2 { 2..=4_usize } else { 2..=3 };
            (Just(agents), 2..=3_usize, height)
        })
        .prop_flat_map(|(agents, width, height)| {
            let blocked = prop::collection::vec(prop::bool::weighted(0.15), width * height);
            (Just(agents), Just(width), Just(height), blocked)
        })
        .prop_filter("enough free cells", |(agents, _, _, blocked)| {
            blocked.iter().filter(|&&b| !b).count() >= *agents
        })
        .prop_flat_map(|(agents, width, height, blocked)| {
            let grid = grid(width, height, &blocked);
            let free: Vec<Vertex> = (0..blocked.len()).filter(|&v| !blocked[v]).collect();
            let starts = Just(free.clone()).prop_shuffle();
            let goals = Just(free).prop_shuffle();

            (Just(grid), Just(agents), starts, goals)
        })
        .prop_map(|(grid, count, starts, goals)| {
            let agents = starts
                .iter()
                .zip(&goals)
                .take(count)
                .map(|(&start, &goal)| Agent { start, goal })
                .collect();

            (grid, agents)
        })
}

/// A `width` by `height` grid with the cells marked in `blocked` (row-major) blocked.
fn grid(width: usize, height: usize, blocked: &[bool]) -> Grid {
    let mut grid = Grid::new(width, height);
    for (vertex, _) in blocked.iter().enumerate().filter(|(_, b)| **b) {
        let (x, y) = grid.cell(vertex);
        grid.block(x, y);
    }

    grid
}

/// Runs CBS with `config` and compares it with the oracle: the same solvability, and on
/// solvable instances a valid, conflict-free plan of the optimal cost.
fn check(grid: &Grid, agents: &[Agent], config: CbsConfig) -> Result<(), TestCaseError> {
    check_optimal(grid, agents, config.objective, |instance, budget| {
        cbs_with(instance, config, Limits::expansions(budget))
    })
}

/// Runs an optimal solver for `objective` (given an instance and a budget) and compares it
/// with the oracle, as [`check`] describes.
pub fn check_optimal(
    grid: &Grid,
    agents: &[Agent],
    objective: Objective,
    solve: impl Fn(&Instance<Grid>, u64) -> MapfResult,
) -> Result<(), TestCaseError> {
    let optimum = optimal_cost(grid, agents, objective);
    let budget = if optimum.is_some() {
        SOLVABLE_BUDGET
    } else {
        UNSOLVABLE_BUDGET
    };
    let instance = Instance::new(grid.clone(), agents.to_vec()).expect("valid by construction");
    let result = solve(&instance, budget);

    match (optimum, result.outcome) {
        (Some(optimum), Outcome::Solved(solution)) => {
            let (soc, makespan) = assert_valid_plan(grid, agents, &solution.paths);
            prop_assert_eq!(first_conflict(&solution.paths), None);
            prop_assert_eq!((solution.sum_of_costs, solution.makespan), (soc, makespan));
            prop_assert_eq!(solution.cost(objective), optimum);
        }
        (None, Outcome::Unsolvable | Outcome::LimitReached) => {}
        (Some(_), Outcome::LimitReached) => {
            // Plain CBS blows up on a few tightly coupled instances (about 1 in 600 here,
            // see `cases::hard_instance_needs_many_expansions`); discard them.
            prop_assume!(false, "needs more than {} expansions", SOLVABLE_BUDGET);
        }
        (optimum, outcome) => {
            prop_assert!(false, "oracle {optimum:?}, solver {outcome:?}");
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        // Bounds the hard instances discarded in `check`, so a CBS regression cannot hide.
        max_global_rejects: 8,
        ..ProptestConfig::default()
    })]

    /// Plain CBS for sum of costs agrees with the oracle.
    #[test]
    fn plain_cbs_matches_joint_search_for_sum_of_costs((grid, agents) in instance()) {
        check(&grid, &agents, CbsConfig::plain(Objective::SumOfCosts))?;
    }

    /// Plain CBS for makespan agrees with the oracle.
    #[test]
    fn plain_cbs_matches_joint_search_for_makespan((grid, agents) in instance()) {
        check(&grid, &agents, CbsConfig::plain(Objective::Makespan))?;
    }

    /// Improved CBS for sum of costs agrees with the oracle.
    #[test]
    fn improved_cbs_matches_joint_search_for_sum_of_costs((grid, agents) in instance()) {
        check(&grid, &agents, CbsConfig::improved(Objective::SumOfCosts))?;
    }

    /// Improved CBS for makespan (the CG heuristic is off there) agrees with the oracle.
    #[test]
    fn improved_cbs_matches_joint_search_for_makespan((grid, agents) in instance()) {
        check(&grid, &agents, CbsConfig::improved(Objective::Makespan))?;
    }

    /// Each improvement alone keeps CBS optimal, for both objectives.
    #[test]
    fn each_cbs_improvement_alone_matches_joint_search(
        (grid, agents) in instance(),
        improvement in 0..3_usize,
        makespan in any::<bool>(),
    ) {
        let objective = if makespan { Objective::Makespan } else { Objective::SumOfCosts };
        let mut config = CbsConfig::plain(objective);
        match improvement {
            0 => config.prioritize_conflicts = true,
            1 => config.bypass = true,
            _ => config.cardinal_heuristic = true,
        }

        check(&grid, &agents, config)?;
    }
}
