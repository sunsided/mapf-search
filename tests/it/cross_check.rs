//! Cross-checks on random grids too large for the oracle: the optimal solvers (CBS, ICTS,
//! basic, recursive and OD-recursive M*) agree on the optimum, ECBS stays within its factor of it, PBS is
//! no cheaper, and every plan passes the independent plan checker.

use core::ops::RangeInclusive;

use mapf_search::{
    Agent, CbsConfig, Grid, Instance, MStarConfig, Objective, Outcome, Vertex, Weight,
};
use proptest::prelude::*;

use crate::oracle::assert_valid_plan;
use crate::solvers::{Solver, assert_within};

/// High-level expansions each solver may use.
const BUDGET: u64 = 5_000;

/// Cost vectors makespan ICTS may evaluate. Its `m`-th vector searches MDDs `m` levels
/// deep, so on an unsolvable instance the run grows with the cube of the budget: 5 000
/// vectors take minutes. 100 reaches far past any optimal makespan on these grids.
const MAKESPAN_ICTS_BUDGET: u64 = 100;

/// A random grid of `sides` by `sides` cells, about one cell in five blocked, with
/// `agents` agents on distinct free starts and distinct free goals.
fn instance(
    agents: RangeInclusive<usize>,
    sides: RangeInclusive<usize>,
) -> impl Strategy<Value = (Grid, Vec<Agent>)> {
    (agents, sides.clone(), sides)
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
    let budget = match solver {
        Solver::Icts(Objective::Makespan) => MAKESPAN_ICTS_BUDGET,
        _ => BUDGET,
    };
    let outcome = solver.run(&instance, budget).outcome;
    if let Outcome::Solved(solution) = &outcome {
        let checked = assert_valid_plan(grid, agents, &solution.paths);
        assert_eq!(
            checked,
            (solution.sum_of_costs, solution.makespan),
            "{solver:?}"
        );
    }

    outcome
}

/// The value of `objective` of a solved outcome.
fn cost(outcome: &Outcome, objective: Objective) -> Option<u32> {
    match outcome {
        Outcome::Solved(solution) => Some(solution.cost(objective)),
        _ => None,
    }
}

/// Runs the `optimal` solvers and the `others`, all for `objective`: no two outcomes
/// contradict each other on solvability, the optimal solvers agree on the optimum where
/// they solve, and the others respect their guarantee relative to it.
fn agree(
    grid: &Grid,
    agents: &[Agent],
    objective: Objective,
    optimal: &[Solver],
    others: &[Solver],
) -> Result<(), TestCaseError> {
    let outcomes: Vec<(Solver, Outcome)> = optimal
        .iter()
        .chain(others)
        .map(|&solver| (solver, solve(grid, agents, solver)))
        .collect();

    let solved = outcomes
        .iter()
        .any(|(_, outcome)| matches!(outcome, Outcome::Solved(_)));
    let proven = outcomes
        .iter()
        .any(|(_, outcome)| *outcome == Outcome::Unsolvable);
    prop_assert!(!(solved && proven), "contradicting outcomes: {outcomes:?}");

    let costs: Vec<u32> = outcomes[..optimal.len()]
        .iter()
        .filter_map(|(_, outcome)| cost(outcome, objective))
        .collect();
    // Unsolvable instances, or all optimal solvers at their budget: nothing to compare.
    let Some(&optimum) = costs.first() else {
        return Ok(());
    };
    prop_assert!(
        costs.iter().all(|&c| c == optimum),
        "optimal solvers disagree: {outcomes:?}"
    );

    for (solver, outcome) in &outcomes {
        if let Some(cost) = cost(outcome, objective) {
            assert_within(*solver, cost, optimum);
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    /// CBS, ICTS and both forms of M* agree; ECBS and PBS respect their guarantees
    /// relative to them. For both objectives.
    #[test]
    fn solvers_agree_on_larger_grids(
        (grid, agents) in instance(3..=4, 4..=5),
        makespan in any::<bool>(),
    ) {
        let objective = if makespan { Objective::Makespan } else { Objective::SumOfCosts };
        let optimal = [
            Solver::Cbs(CbsConfig::improved(objective)),
            Solver::Icts(objective),
            Solver::MStar(MStarConfig::basic(objective)),
            Solver::MStar(MStarConfig::recursive(objective)),
            Solver::MStar(MStarConfig::od_recursive(objective)),
            Solver::Epea(objective),
        ];
        let others = [
            Solver::Ecbs(objective, Weight::new(3, 2)),
            Solver::Ecbs(objective, Weight::ONE),
            Solver::Pbs(objective),
            Solver::Lns2(objective),
        ];

        agree(&grid, &agents, objective, &optimal, &others)?;
    }

    /// With more agents, recursive M* with and without operator decomposition agrees with
    /// CBS, for both objectives.
    #[test]
    fn recursive_mstar_agrees_with_cbs_on_more_agents(
        (grid, agents) in instance(5..=6, 5..=6),
        makespan in any::<bool>(),
    ) {
        let objective = if makespan { Objective::Makespan } else { Objective::SumOfCosts };
        let optimal = [
            Solver::Cbs(CbsConfig::improved(objective)),
            Solver::MStar(MStarConfig::recursive(objective)),
            Solver::MStar(MStarConfig::od_recursive(objective)),
        ];

        agree(&grid, &agents, objective, &optimal, &[])?;
    }
}
