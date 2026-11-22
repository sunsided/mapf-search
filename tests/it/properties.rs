//! Property tests: on random tiny grids with 2-3 agents, every solver agrees with the
//! joint-space oracle on solvability (as far as it can prove it), and its plans are valid,
//! conflict-free and within its guarantee over the optimal sum of costs or makespan.

use mapf_search::{
    Agent, CbsConfig, CbsHeuristic, Grid, Instance, MStarConfig, Objective, Outcome, Splitting,
    Vertex, Weight, first_conflict,
};
use proptest::prelude::*;

use crate::oracle::{assert_valid_plan, optimal_cost};
use crate::solvers::{Solver, assert_within};

/// High-level expansions a solver may use on solvable instances.
const SOLVABLE_BUDGET: u64 = 20_000;

/// High-level expansions a solver may use on instances the oracle proves unsolvable.
const UNSOLVABLE_BUDGET: u64 = 60;

/// A random grid of 2-3 by 2-3 cells (3 by 4 for two agents), about one cell in six
/// blocked, with 2-3 agents on distinct free starts and distinct free goals.
pub(crate) fn instance() -> impl Strategy<Value = (Grid, Vec<Agent>)> {
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

/// Either objective.
fn objective() -> impl Strategy<Value = Objective> {
    prop::sample::select(vec![Objective::SumOfCosts, Objective::Makespan])
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

/// Runs `solver` and compares it with the oracle: never `Solved` on an unsolvable instance,
/// never `Unsolvable` on a solvable one, and when solved a valid, conflict-free plan within
/// the solver's guarantee over the optimum.
fn check(grid: &Grid, agents: &[Agent], solver: Solver) -> Result<(), TestCaseError> {
    let optimum = optimal_cost(grid, agents, solver.objective());
    let budget = if optimum.is_some() {
        SOLVABLE_BUDGET
    } else {
        UNSOLVABLE_BUDGET
    };
    let instance = Instance::new(grid.clone(), agents.to_vec()).expect("valid by construction");
    let result = solver.run(&instance, budget);

    match (optimum, result.outcome) {
        (Some(optimum), Outcome::Solved(solution)) => {
            let (soc, makespan) = assert_valid_plan(grid, agents, &solution.paths);
            prop_assert_eq!(first_conflict(&solution.paths), None);
            prop_assert_eq!((solution.sum_of_costs, solution.makespan), (soc, makespan));
            assert_within(solver, solution.cost(solver.objective()), optimum);
        }
        (None, Outcome::Unsolvable | Outcome::LimitReached | Outcome::Exhausted) => {}
        (Some(_), Outcome::LimitReached | Outcome::Exhausted) if !solver.is_complete() => {}
        (Some(_), Outcome::LimitReached) if solver.is_complete() => {
            // Plain CBS blows up on a few tightly coupled instances (about 1 in 600 here,
            // see `cases::hard_instance_needs_many_expansions`); discard them.
            prop_assume!(false, "needs more than {} expansions", SOLVABLE_BUDGET);
        }
        (optimum, outcome) => {
            prop_assert!(false, "oracle {optimum:?}, {solver:?}: {outcome:?}");
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        // Bounds the hard instances discarded in `check`, so a regression cannot hide.
        max_global_rejects: 8,
        ..ProptestConfig::default()
    })]

    /// Plain CBS for sum of costs agrees with the oracle.
    #[test]
    fn plain_cbs_matches_joint_search_for_sum_of_costs((grid, agents) in instance()) {
        check(&grid, &agents, Solver::Cbs(CbsConfig::plain(Objective::SumOfCosts)))?;
    }

    /// Plain CBS for makespan agrees with the oracle.
    #[test]
    fn plain_cbs_matches_joint_search_for_makespan((grid, agents) in instance()) {
        check(&grid, &agents, Solver::Cbs(CbsConfig::plain(Objective::Makespan)))?;
    }

    /// Improved CBS for sum of costs agrees with the oracle.
    #[test]
    fn improved_cbs_matches_joint_search_for_sum_of_costs((grid, agents) in instance()) {
        check(&grid, &agents, Solver::Cbs(CbsConfig::improved(Objective::SumOfCosts)))?;
    }

    /// Improved CBS with WDG instead of DG agrees with the oracle: WDG with disjoint
    /// splitting, bypass and inherited MDDs.
    #[test]
    fn improved_cbs_with_wdg_matches_joint_search((grid, agents) in instance()) {
        let config = CbsConfig {
            heuristic: CbsHeuristic::Wdg,
            ..CbsConfig::improved(Objective::SumOfCosts)
        };
        check(&grid, &agents, Solver::Cbs(config))?;
    }

    /// Improved CBS for makespan (the heuristic is off there) agrees with the oracle.
    #[test]
    fn improved_cbs_matches_joint_search_for_makespan((grid, agents) in instance()) {
        check(&grid, &agents, Solver::Cbs(CbsConfig::improved(Objective::Makespan)))?;
    }

    /// Each improvement alone keeps CBS optimal, for both objectives, and so does bypass
    /// with disjoint splitting.
    #[test]
    fn each_cbs_improvement_alone_matches_joint_search(
        (grid, agents) in instance(),
        improvement in 0..9_usize,
        makespan in any::<bool>(),
    ) {
        let objective = if makespan { Objective::Makespan } else { Objective::SumOfCosts };
        let mut config = CbsConfig::plain(objective);
        match improvement {
            0 => config.prioritize_conflicts = true,
            1 => config.bypass = true,
            2 => config.heuristic = CbsHeuristic::Cg,
            3 => config.conflict_avoidance = true,
            4 => config.splitting = Splitting::Disjoint,
            5 => config.heuristic = CbsHeuristic::Dg,
            6 => config.heuristic = CbsHeuristic::Wdg,
            7 => {
                // WDG over positive constraints and the MDDs a positive child keeps.
                config.heuristic = CbsHeuristic::Wdg;
                config.splitting = Splitting::Disjoint;
            }
            _ => {
                // Bypass adopts the several paths a positive child replans.
                config.bypass = true;
                config.splitting = Splitting::Disjoint;
            }
        }

        check(&grid, &agents, Solver::Cbs(config))?;
    }

    /// ICTS agrees with the oracle, for both objectives.
    #[test]
    fn icts_matches_joint_search((grid, agents) in instance(), objective in objective()) {
        check(&grid, &agents, Solver::Icts(objective))?;
    }

    /// Basic and recursive M*, each with and without operator decomposition, agree with the
    /// oracle, for both objectives.
    #[test]
    fn mstar_matches_joint_search(
        (grid, agents) in instance(),
        objective in objective(),
        recursive in any::<bool>(),
        operator_decomposition in any::<bool>(),
    ) {
        let config = MStarConfig { objective, recursive, operator_decomposition };
        check(&grid, &agents, Solver::MStar(config))?;
    }

    /// EPEA* agrees with the oracle, for both objectives.
    #[test]
    fn epea_matches_joint_search((grid, agents) in instance(), objective in objective()) {
        check(&grid, &agents, Solver::Epea(objective))?;
    }

    /// LNS2 plans are valid and no cheaper than the optimum; LNS2 may stop at its budget.
    #[test]
    fn lns2_plans_are_valid((grid, agents) in instance(), objective in objective()) {
        check(&grid, &agents, Solver::Lns2(objective))?;
    }

    /// PBS plans are valid and no cheaper than the optimum, for both objectives; PBS may
    /// give up.
    #[test]
    fn pbs_plans_are_valid((grid, agents) in instance(), objective in objective()) {
        check(&grid, &agents, Solver::Pbs(objective))?;
    }

    /// ECBS with `w = 1` is optimal, for both objectives.
    #[test]
    fn ecbs_with_weight_one_is_optimal((grid, agents) in instance(), objective in objective()) {
        check(&grid, &agents, Solver::Ecbs(objective, Weight::ONE))?;
    }

    /// ECBS stays within its suboptimality factor, for both objectives.
    #[test]
    fn ecbs_stays_within_its_bound(
        (grid, agents) in instance(),
        objective in objective(),
        numerator in prop::sample::select(vec![11_u32, 15, 20, 30]),
    ) {
        check(&grid, &agents, Solver::Ecbs(objective, Weight::new(numerator, 10)))?;
    }
}
