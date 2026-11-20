//! MO-CBS against the joint-space Pareto oracle: on random tiny grids with random cost
//! vectors, shared or per agent, every configuration (the paper's four variants and every
//! combination of the improvements) returns exactly the oracle's front, every plan is
//! valid, conflict-free and priced as claimed, the feasibility check proves unsolvable
//! instances, and with one objective and unit costs the front is CBS's optimal sum of
//! costs. Plus hand-made cases.

use heuristic_search::{Limits, Stats};
use mapf_search::{
    AdjacencyList, Agent, Graph, Grid, Instance, LowLevel, MoCbsConfig, Objective, ParetoOutcome,
    Splitting, Vertex, cbs, first_conflict, mo_cbs, mo_cbs_per_agent, mo_cbs_with,
};
use proptest::prelude::*;

use crate::oracle::{assert_valid_plan, optimal_cost};
use crate::pareto_oracle::{pareto_front, plan_cost};
use crate::properties::instance;

/// High-level expansions MO-CBS may use on solvable instances, and on unsolvable ones
/// with the feasibility check (which spends it on M*).
const SOLVABLE_BUDGET: u64 = 20_000;

/// High-level expansions MO-CBS may use on instances the oracle proves unsolvable, without
/// the feasibility check.
const UNSOLVABLE_BUDGET: u64 = 60;

/// Cost tables for `agents` agents on a graph of `vertices` vertices:
/// `table[(agent * vertices + from) * vertices + to]` is what the move `from -> to` (a wait
/// if equal) costs `agent`. Components are 1 to 3.
fn cost_table<const N: usize>(
    agents: usize,
    vertices: usize,
) -> impl Strategy<Value = Vec<[u32; N]>> {
    let entries = agents * vertices * vertices;

    prop::collection::vec(prop::array::uniform::<_, N>(1..=3_u32), entries)
}

/// A random instance of [`instance`] with random [`cost_table`]s.
fn weighted<const N: usize>() -> impl Strategy<Value = (Grid, Vec<Agent>, Vec<[u32; N]>)> {
    instance().prop_flat_map(|(grid, agents)| {
        let table = cost_table::<N>(agents.len(), grid.vertex_count());

        (Just(grid), Just(agents), table)
    })
}

/// Any configuration: plain or tree-by-tree, BOA* (two objectives only) or NAMOA*dr, and
/// each improvement on or off.
fn config() -> impl Strategy<Value = MoCbsConfig> {
    prop::array::uniform6(any::<bool>()).prop_map(|bits| MoCbsConfig {
        tree_by_tree: bits[0],
        low_level: if bits[1] {
            LowLevel::NamoaDr
        } else {
            LowLevel::Auto
        },
        feasibility_check: bits[2],
        prioritize_conflicts: bits[3],
        bypass: bits[4],
        splitting: if bits[5] {
            Splitting::Disjoint
        } else {
            Splitting::Standard
        },
    })
}

/// Runs MO-CBS with `config` and compares it with the oracle: on an unsolvable instance
/// never a front, and `Unsolvable` with the feasibility check; on a solvable one the
/// complete front, each plan valid, conflict-free and costing what it claims.
///
/// With `per_agent`, each agent prices moves by its own table, else all use agent 0's.
fn check<const N: usize>(
    grid: &Grid,
    agents: &[Agent],
    table: &[[u32; N]],
    config: MoCbsConfig,
    per_agent: bool,
) -> Result<(), TestCaseError> {
    let vertices = grid.vertex_count();
    let costs = |agent: usize, from: Vertex, to: Vertex| {
        let agent = if per_agent { agent } else { 0 };

        table[(agent * vertices + from) * vertices + to]
    };
    let expected = pareto_front(grid, agents, &costs);
    let budget = if expected.is_empty() && !config.feasibility_check {
        UNSOLVABLE_BUDGET
    } else {
        SOLVABLE_BUDGET
    };
    let instance = Instance::new(grid.clone(), agents.to_vec()).expect("valid by construction");
    let result = if per_agent {
        mo_cbs_per_agent(&instance, &costs, config, Limits::expansions(budget))
    } else {
        let shared = |from: Vertex, to: Vertex| costs(0, from, to);
        mo_cbs_with(&instance, &shared, config, Limits::expansions(budget))
    };

    if expected.is_empty() {
        prop_assert!(result.front.is_empty());
        if config.feasibility_check {
            prop_assert_eq!(result.outcome, ParetoOutcome::Unsolvable);
        } else {
            prop_assert_ne!(result.outcome, ParetoOutcome::Complete);
        }

        return Ok(());
    }

    if result.outcome == ParetoOutcome::LimitReached {
        // Without the improvements MO-CBS splits like plain CBS, which blows up on a few
        // tightly coupled instances (see `cases::hard_instance_needs_many_expansions`);
        // discard them.
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

    /// Two objectives with shared costs: every configuration returns the oracle's front.
    #[test]
    fn two_objectives_match_the_oracle(
        (grid, agents, table) in weighted::<2>(),
        config in config(),
    ) {
        check(&grid, &agents, &table, config, false)?;
    }

    /// Two objectives with per-agent costs: every configuration returns the oracle's front.
    #[test]
    fn per_agent_costs_match_the_oracle(
        (grid, agents, table) in weighted::<2>(),
        config in config(),
    ) {
        check(&grid, &agents, &table, config, true)?;
    }

    /// The paper's plain MO-CBS and the improved configuration, both with NAMOA*dr, return
    /// the oracle's front for three objectives.
    #[test]
    fn three_objectives_match_the_oracle(
        (grid, agents, table) in weighted::<3>(),
        improved in any::<bool>(),
    ) {
        let config = if improved { MoCbsConfig::improved() } else { MoCbsConfig::plain() };
        check(&grid, &agents, &table, config, false)?;
    }

    /// One objective with random costs returns the oracle's single optimum.
    #[test]
    fn one_weighted_objective_matches_the_oracle(
        (grid, agents, table) in weighted::<1>(),
        config in config(),
    ) {
        check(&grid, &agents, &table, config, false)?;
    }

    /// One objective with unit costs is CBS: the front is the optimal sum of costs.
    #[test]
    fn unit_costs_reduce_to_cbs((grid, agents) in instance(), config in config()) {
        let unit = |_: Vertex, _: Vertex| [1];
        let optimum = optimal_cost(&grid, &agents, Objective::SumOfCosts);
        let instance = Instance::new(grid, agents).expect("valid by construction");
        let budget = if optimum.is_some() || config.feasibility_check {
            SOLVABLE_BUDGET
        } else {
            UNSOLVABLE_BUDGET
        };
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

    assert_eq!(
        pareto_front(&graph, &agents, &|_, u, v| costs(u, v)),
        [[8, 16], [9, 9]]
    );
    let configs = [
        MoCbsConfig::plain(),
        MoCbsConfig::tree_by_tree(),
        MoCbsConfig::improved(),
    ];
    for config in configs {
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
/// infinite: without the feasibility check the budget ends the run, with no front.
#[test]
fn infinite_forest_hits_the_budget() {
    let grid = Grid::new(3, 1);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
    let instance = Instance::new(grid, agents).expect("valid");
    let config = MoCbsConfig::tree_by_tree();
    let result = mo_cbs_with(&instance, &|_, _| [1, 2], config, Limits::expansions(200));

    assert_eq!(result.outcome, ParetoOutcome::LimitReached);
    assert_eq!(result.front, []);
}

/// The budget is asked inside the low level too: a single agent has no conflict, so the
/// high level never asks, and still a budget that always says stop ends the run.
#[test]
fn budget_interrupts_the_low_level() {
    let grid = Grid::new(4, 4);
    let instance = Instance::new(grid, vec![Agent { start: 0, goal: 15 }]).expect("valid");
    let costs = |from: Vertex, to: Vertex| [1, u32::try_from((from + to) % 3).expect("small") + 1];
    let mut asked = 0;
    let stop = |_: &Stats| {
        asked += 1;
        true
    };

    let config = MoCbsConfig::tree_by_tree();
    let result = mo_cbs_with(&instance, &costs, config, stop);
    let unlimited = mo_cbs_with(&instance, &costs, config, Limits::NONE);

    assert_eq!(result.outcome, ParetoOutcome::LimitReached);
    assert_eq!(asked, 1);
    assert_eq!(unlimited.outcome, ParetoOutcome::Complete);
}

/// With the feasibility check, a corridor swap is proven unsolvable; the paper's MO-CBS
/// only stops at its budget.
#[test]
fn feasibility_check_proves_the_corridor_swap_unsolvable() {
    let grid = Grid::new(3, 1);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
    let instance = Instance::new(grid, agents).expect("valid");
    let costs = |_: Vertex, _: Vertex| [1, 2];

    let improved = mo_cbs(&instance, &costs, Limits::NONE);
    let plain = mo_cbs_with(
        &instance,
        &costs,
        MoCbsConfig::plain(),
        Limits::expansions(200),
    );

    assert_eq!(improved.outcome, ParetoOutcome::Unsolvable);
    assert_eq!(plain.outcome, ParetoOutcome::LimitReached);
}

/// A budget of exactly the expansions a run needs lets it complete: the checks inside the
/// low level see only completed expansions.
#[test]
fn exact_expansion_budget_completes() {
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
    let instance = Instance::new(graph, agents).expect("valid");

    for config in [MoCbsConfig::plain(), MoCbsConfig::improved()] {
        let free = mo_cbs_with(&instance, &costs, config, Limits::NONE);
        let needed = free.stats.expanded;
        let exact = mo_cbs_with(&instance, &costs, config, Limits::expansions(needed));
        let short = mo_cbs_with(&instance, &costs, config, Limits::expansions(needed - 1));

        assert!(needed > 0);
        assert_eq!(exact.outcome, ParetoOutcome::Complete, "{config:?}");
        assert_eq!(exact.front, free.front);
        assert_eq!(short.outcome, ParetoOutcome::LimitReached, "{config:?}");
    }
}
