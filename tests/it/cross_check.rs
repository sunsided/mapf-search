//! Cross-checks on random grids too large for the oracle: the optimal solvers (CBS, ICTS
//! and M*) agree on the optimal sum of costs, ECBS stays within its factor of it, PBS is no
//! cheaper, and every plan passes the independent plan checker.

use mapf_search::{Agent, CbsConfig, Grid, Instance, Objective, Outcome, Vertex, Weight};
use proptest::prelude::*;

use crate::oracle::assert_valid_plan;
use crate::solvers::{Solver, assert_within};

/// High-level expansions each solver may use.
const BUDGET: u64 = 5_000;

/// A random grid of 4-5 by 4-5 cells, about one cell in five blocked, with 3-4 agents on
/// distinct free starts and distinct free goals.
fn instance() -> impl Strategy<Value = (Grid, Vec<Agent>)> {
    (3..=4_usize, 4..=5_usize, 4..=5_usize)
        .prop_flat_map(|(agents, width, height)| {
            let blocked = prop::collection::vec(prop::bool::weighted(0.2), width * height);
            (Just(agents), Just(width), Just(height), blocked)
        })
        .prop_filter("enough free cells", |(agents, _, _, blocked)| {
            blocked.iter().filter(|&&b| !b).count() >= *agents
        })
        .prop_flat_map(|(agents, width, height, blocked)| {
            let mut grid = Grid::new(width, height);
            for (vertex, _) in blocked.iter().enumerate().filter(|(_, b)| **b) {
                let (x, y) = grid.cell(vertex);
                grid.block(x, y);
            }

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

/// Runs `solver` and checks a plan, if any, with the independent plan checker.
fn solve(grid: &Grid, agents: &[Agent], solver: Solver) -> Outcome {
    let instance = Instance::new(grid.clone(), agents.to_vec()).expect("valid by construction");
    let outcome = solver.run(&instance, BUDGET).outcome;
    if let Outcome::Solved(solution) = &outcome {
        let (soc, _) = assert_valid_plan(grid, agents, &solution.paths);
        assert_eq!(soc, solution.sum_of_costs, "{solver:?}");
    }

    outcome
}

/// The sum of costs of a solved outcome.
fn cost(outcome: &Outcome) -> Option<u32> {
    match outcome {
        Outcome::Solved(solution) => Some(solution.sum_of_costs),
        _ => None,
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    /// CBS, ICTS and M* agree; ECBS and PBS respect their guarantees relative to them.
    #[test]
    fn solvers_agree_on_larger_grids((grid, agents) in instance()) {
        let optimal = [
            Solver::Cbs(CbsConfig::improved(Objective::SumOfCosts)),
            Solver::Icts(Objective::SumOfCosts),
            Solver::MStar(Objective::SumOfCosts),
        ];
        let others = [
            Solver::Ecbs(Objective::SumOfCosts, Weight::new(3, 2)),
            Solver::Ecbs(Objective::SumOfCosts, Weight::ONE),
            Solver::Pbs(Objective::SumOfCosts),
        ];
        let outcomes: Vec<(Solver, Outcome)> = optimal
            .into_iter()
            .chain(others)
            .map(|solver| (solver, solve(&grid, &agents, solver)))
            .collect();

        let solved = outcomes.iter().any(|(_, outcome)| matches!(outcome, Outcome::Solved(_)));
        let proven = outcomes.iter().any(|(_, outcome)| *outcome == Outcome::Unsolvable);
        prop_assert!(!(solved && proven), "contradicting outcomes: {outcomes:?}");

        let costs: Vec<u32> = outcomes[..optimal.len()].iter().filter_map(|(_, o)| cost(o)).collect();
        // Unsolvable instances, or all optimal solvers at their budget: nothing to compare.
        let Some(&optimum) = costs.first() else {
            return Ok(());
        };
        prop_assert!(costs.iter().all(|&c| c == optimum), "optimal solvers disagree: {costs:?}");

        for (solver, outcome) in &outcomes {
            if let Some(cost) = cost(outcome) {
                assert_within(*solver, cost, optimum);
            }
        }
    }
}
