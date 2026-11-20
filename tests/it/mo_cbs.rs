//! MO-CBS against the joint-space Pareto oracle: on random tiny grids with random cost
//! vectors, every variant returns exactly the oracle's front, every plan is valid,
//! conflict-free and priced as claimed, and with one objective and unit costs the front is
//! CBS's optimal sum of costs. Plus hand-made cases.

use heuristic_search::Limits;
use mapf_search::{
    AdjacencyList, Agent, Graph, Grid, Instance, LowLevel, MoCbsConfig, Objective, ParetoOutcome,
    Vertex, cbs, first_conflict, mo_cbs, mo_cbs_with,
};
use proptest::prelude::*;

use crate::oracle::{assert_valid_plan, optimal_cost};
use crate::pareto_oracle::{pareto_front, plan_cost};
use crate::properties::instance;

/// High-level expansions MO-CBS may use on solvable instances.
const SOLVABLE_BUDGET: u64 = 20_000;

/// High-level expansions MO-CBS may use on instances the oracle proves unsolvable.
const UNSOLVABLE_BUDGET: u64 = 60;

/// A cost table for a graph of `vertices` vertices: `table[from * vertices + to]` is the
/// cost vector of the move `from -> to` (a wait if equal). Components are 1 to 3.
fn cost_table<const N: usize>(vertices: usize) -> impl Strategy<Value = Vec<[u32; N]>> {
    prop::collection::vec(prop::array::uniform::<_, N>(1..=3_u32), vertices * vertices)
}

/// A random instance of [`instance`] with a random [`cost_table`].
fn weighted<const N: usize>() -> impl Strategy<Value = (Grid, Vec<Agent>, Vec<[u32; N]>)> {
    instance().prop_flat_map(|(grid, agents)| {
        let table = cost_table::<N>(grid.vertex_count());

        (Just(grid), Just(agents), table)
    })
}

/// Any of the four configurations: plain or tree-by-tree, BOA* (two objectives only) or
/// NAMOA*dr.
fn config() -> impl Strategy<Value = MoCbsConfig> {
    (any::<bool>(), any::<bool>()).prop_map(|(tree_by_tree, namoa)| MoCbsConfig {
        tree_by_tree,
        low_level: if namoa {
            LowLevel::NamoaDr
        } else {
            LowLevel::Auto
        },
    })
}

/// Runs MO-CBS with `config` and compares it with the oracle: on an unsolvable instance
/// never a front, on a solvable one the complete front, each plan valid, conflict-free and
/// costing what it claims.
fn check<const N: usize>(
    grid: &Grid,
    agents: &[Agent],
    table: &[[u32; N]],
    config: MoCbsConfig,
) -> Result<(), TestCaseError> {
    let vertices = grid.vertex_count();
    let costs = |from: Vertex, to: Vertex| table[from * vertices + to];
    let expected = pareto_front(grid, agents, &costs);
    let budget = if expected.is_empty() {
        UNSOLVABLE_BUDGET
    } else {
        SOLVABLE_BUDGET
    };
    let instance = Instance::new(grid.clone(), agents.to_vec()).expect("valid by construction");
    let result = mo_cbs_with(&instance, &costs, config, Limits::expansions(budget));

    if expected.is_empty() {
        prop_assert_ne!(result.outcome, ParetoOutcome::Complete);
        prop_assert!(result.front.is_empty());
        return Ok(());
    }

    if result.outcome == ParetoOutcome::LimitReached {
        // MO-CBS splits like plain CBS, which blows up on a few tightly coupled instances
        // (see `cases::hard_instance_needs_many_expansions`); discard them.
        prop_assume!(false, "needs more than {} expansions", SOLVABLE_BUDGET);
    }

    prop_assert_eq!(result.outcome, ParetoOutcome::Complete);
    let found: Vec<[u32; N]> = result.front.iter().map(|plan| plan.cost).collect();
    prop_assert_eq!(found, expected);
    for plan in &result.front {
        assert_valid_plan(grid, agents, &plan.paths);
        prop_assert_eq!(first_conflict(&plan.paths), None);
        prop_assert_eq!(plan_cost(agents, &plan.paths, &costs), plan.cost);
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        // Bounds the hard instances discarded in `check`, so a regression cannot hide.
        max_global_rejects: 8,
        ..ProptestConfig::default()
    })]

    /// Two objectives: all four variants (MO-CBS-b, -n, -tb, -tn) return the oracle's front.
    #[test]
    fn two_objectives_match_the_oracle(
        (grid, agents, table) in weighted::<2>(),
        config in config(),
    ) {
        check(&grid, &agents, &table, config)?;
    }

    /// Three objectives (NAMOA*dr), plain and tree-by-tree, return the oracle's front.
    #[test]
    fn three_objectives_match_the_oracle(
        (grid, agents, table) in weighted::<3>(),
        tree_by_tree in any::<bool>(),
    ) {
        let config = MoCbsConfig { tree_by_tree, low_level: LowLevel::Auto };
        check(&grid, &agents, &table, config)?;
    }

    /// One objective with random costs returns the oracle's single optimum.
    #[test]
    fn one_weighted_objective_matches_the_oracle(
        (grid, agents, table) in weighted::<1>(),
        config in config(),
    ) {
        check(&grid, &agents, &table, config)?;
    }

    /// One objective with unit costs is CBS: the front is the optimal sum of costs.
    #[test]
    fn unit_costs_reduce_to_cbs((grid, agents) in instance(), config in config()) {
        let unit = |_: Vertex, _: Vertex| [1];
        let optimum = optimal_cost(&grid, &agents, Objective::SumOfCosts);
        let instance = Instance::new(grid, agents).expect("valid by construction");
        let budget = if optimum.is_some() { SOLVABLE_BUDGET } else { UNSOLVABLE_BUDGET };
        let result = mo_cbs_with(&instance, &unit, config, Limits::expansions(budget));

        if let Some(optimum) = optimum {
            prop_assume!(result.outcome != ParetoOutcome::LimitReached, "a hard instance");
            let found: Vec<_> = result.front.iter().map(|plan| plan.cost).collect();
            prop_assert_eq!(found, [[optimum]]);
            let solved = cbs(&instance, Objective::SumOfCosts, Limits::expansions(budget));
            prop_assert_eq!(solved.solution().map(|s| s.sum_of_costs), Some(optimum));
        } else {
            prop_assert!(result.front.is_empty());
        }
    }
}

/// Two agents swap the ends of a corridor `0 - 1 - 2 - 3` with a risky side pocket `4` at
/// `1` and a long, safe loop `0 - 5 - ... - 9 - 3`. Alone, each agent runs straight down
/// the corridor, so every root conflicts and the whole front comes from conflict
/// resolution: either one agent dodges into the pocket (fast, risky: `[8, 16]`) or one
/// takes the loop (slow, safe: `[9, 9]`).
#[test]
fn front_comes_from_conflict_resolution() {
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 3),
    ];
    let graph = AdjacencyList::undirected(10, &edges);
    let costs = |_: Vertex, to: Vertex| [1, if to == 4 { 9 } else { 1 }];
    let agents = vec![Agent { start: 0, goal: 3 }, Agent { start: 3, goal: 0 }];
    let instance = Instance::new(graph.clone(), agents.clone()).expect("valid");

    assert_eq!(pareto_front(&graph, &agents, &costs), [[8, 16], [9, 9]]);
    for config in [MoCbsConfig::plain(), MoCbsConfig::tree_by_tree()] {
        let result = mo_cbs_with(&instance, &costs, config, Limits::expansions(10_000));
        let found: Vec<_> = result.front.iter().map(|plan| plan.cost).collect();

        assert_eq!(result.outcome, ParetoOutcome::Complete);
        assert_eq!(found, [[8, 16], [9, 9]]);
        assert!(result.stats.expanded > 0, "the single root conflicts");
        for plan in &result.front {
            assert_valid_plan(&graph, &agents, &plan.paths);
        }
    }
}

/// An agent whose goal is unreachable makes the instance unsolvable before any search.
#[test]
fn unreachable_goal_is_unsolvable() {
    let graph = AdjacencyList::undirected(3, &[(0, 1)]);
    let instance = Instance::new(graph, vec![Agent { start: 0, goal: 2 }]).expect("valid");
    let result = mo_cbs(&instance, &|_, _| [1, 1], Limits::NONE);

    assert_eq!(result.outcome, ParetoOutcome::Unsolvable);
    assert_eq!(result.front, []);
}

/// Swapping the ends of a bare corridor is unsolvable, but the constraint forest is
/// infinite: the budget ends the run, with no front.
#[test]
fn infinite_forest_hits_the_budget() {
    let grid = Grid::new(3, 1);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
    let instance = Instance::new(grid, agents).expect("valid");
    let result = mo_cbs(&instance, &|_, _| [1, 2], Limits::expansions(200));

    assert_eq!(result.outcome, ParetoOutcome::LimitReached);
    assert_eq!(result.front, []);
}
