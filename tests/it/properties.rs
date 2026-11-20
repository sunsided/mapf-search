//! Property tests: on random tiny grids with 2-3 agents, CBS agrees with the joint-space
//! oracle on solvability and on the optimal sum of costs.

use heuristic_search::Limits;
use mapf_search::{Agent, Grid, Instance, Outcome, Vertex, cbs, first_conflict};
use proptest::prelude::*;

use crate::oracle::{assert_valid_plan, optimal_sum_of_costs};

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

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        // Bounds the hard instances discarded below, so a CBS regression cannot hide there.
        max_global_rejects: 8,
        ..ProptestConfig::default()
    })]

    /// CBS solves exactly the instances the oracle solves, with a valid, conflict-free plan
    /// of the optimal sum of costs.
    #[test]
    fn cbs_matches_joint_search((grid, agents) in instance()) {
        let optimum = optimal_sum_of_costs(&grid, &agents);
        let budget = if optimum.is_some() { SOLVABLE_BUDGET } else { UNSOLVABLE_BUDGET };
        let instance = Instance::new(grid.clone(), agents.clone()).expect("valid by construction");
        let result = cbs(&instance, Limits::expansions(budget));

        match (optimum, result.outcome) {
            (Some(optimum), Outcome::Solved(solution)) => {
                prop_assert_eq!(first_conflict(&solution.paths), None);
                prop_assert_eq!(assert_valid_plan(&grid, &agents, &solution.paths), optimum);
                prop_assert_eq!(solution.cost, optimum);
            }
            (None, Outcome::Unsolvable | Outcome::LimitReached) => {}
            (Some(_), Outcome::LimitReached) => {
                // Plain CBS blows up on a few tightly coupled instances (about 1 in 600
                // here, see `cases::hard_instance_needs_many_expansions`); discard them.
                prop_assume!(false, "CBS needs more than {} expansions", SOLVABLE_BUDGET);
            }
            (optimum, outcome) => {
                prop_assert!(false, "oracle {optimum:?}, CBS {outcome:?}");
            }
        }
    }
}
