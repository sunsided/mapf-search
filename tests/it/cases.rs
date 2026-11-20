//! Hand-made instances: conflict detection, goal-stay semantics, constraints in the low
//! level, unsolvable instances and instance validation.

use heuristic_search::Limits;
use mapf_search::{
    AdjacencyList, Agent, CbsConfig, Conflict, Constraint, ConstraintSet, Grid, Instance,
    InstanceError, MStarConfig, Objective, Outcome, Splitting, cbs, cbs_with, conflicts,
    first_conflict, mstar_with, space_time_astar,
};

use crate::oracle::{assert_valid_plan, optimal_cost};

/// Two agents swapping along an edge is a conflict, reported for the first agent's move.
#[test]
fn swap_is_detected() {
    let paths = [vec![0, 1], vec![1, 0]];

    assert_eq!(
        first_conflict(&paths),
        Some(Conflict::Edge {
            agents: (0, 1),
            from: 0,
            to: 1,
            time: 0,
        })
    );
}

/// Entering a vertex another agent leaves in the same step is allowed.
#[test]
fn following_is_not_a_conflict() {
    let paths = [vec![0, 1, 2], vec![1, 2, 3]];

    assert_eq!(first_conflict(&paths), None);
}

/// Two agents at the same vertex at the same time conflict.
#[test]
fn vertex_conflict_is_detected() {
    let paths = [vec![0, 1, 2], vec![2, 1, 0]];

    assert_eq!(
        first_conflict(&paths),
        Some(Conflict::Vertex {
            agents: (0, 1),
            vertex: 1,
            time: 1,
        })
    );
}

/// An agent whose path has ended keeps blocking its goal.
#[test]
fn finished_agent_blocks_its_goal() {
    let paths = [vec![0, 1], vec![3, 3, 3, 1]];

    assert_eq!(
        conflicts(&paths),
        [Conflict::Vertex {
            agents: (0, 1),
            vertex: 1,
            time: 3,
        }]
    );
}

/// The low level does not finish at a goal that a later constraint forbids.
#[test]
fn low_level_respects_late_goal_constraint() {
    let grid = Grid::new(3, 1);
    let agent = Agent { start: 0, goal: 2 };
    let mut constraints = ConstraintSet::new();
    constraints.insert(&Constraint::Vertex {
        agent: 0,
        vertex: 2,
        time: 5,
    });

    let path = space_time_astar(&grid, agent, &constraints).expect("solvable");

    assert_eq!(path.len(), 7, "arrives at time 6: {path:?}");
    assert_ne!(path[5], 2);
    assert_eq!(path.last(), Some(&2));
}

/// The low level obeys edge constraints, also on the way back.
#[test]
fn low_level_respects_edge_constraint() {
    let grid = Grid::new(2, 1);
    let agent = Agent { start: 0, goal: 1 };
    let mut constraints = ConstraintSet::new();
    constraints.insert(&Constraint::Edge {
        agent: 0,
        from: 0,
        to: 1,
        time: 0,
    });

    assert_eq!(
        space_time_astar(&grid, agent, &constraints),
        Some(vec![0, 0, 1])
    );
}

/// An agent that starts at its goal must step aside into a pocket to let another agent
/// pass, and pays for the time it is away.
///
/// ```text
/// 0 - 1 - 2
///     |
///     3
/// ```
#[test]
fn agent_steps_off_its_goal() {
    let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 2), (1, 3)]);
    let agents = vec![Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }];
    let instance = Instance::new(graph.clone(), agents.clone()).expect("valid");

    let result = cbs(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
    let solution = result.solution().expect("solvable");

    assert_eq!(solution.sum_of_costs, 4);
    assert_eq!(assert_valid_plan(&graph, &agents, &solution.paths).0, 4);
    assert_eq!(
        optimal_cost(&graph, &agents, Objective::SumOfCosts),
        Some(4)
    );
}

/// A goal behind a wall is proven unreachable at the root.
#[test]
fn unreachable_goal_is_unsolvable() {
    let grid = Grid::parse(&[".@."]);
    let agents = vec![Agent { start: 0, goal: 2 }];
    let instance = Instance::new(grid, agents).expect("valid");

    assert_eq!(
        cbs(&instance, Objective::SumOfCosts, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

/// Two agents swapping the ends of a corridor without room to pass: the constraint tree is
/// infinite, so CBS stops at its budget, and the oracle proves the instance unsolvable.
#[test]
fn corridor_swap_hits_the_budget() {
    let grid = Grid::new(3, 1);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
    let instance = Instance::new(grid.clone(), agents.clone()).expect("valid");

    let result = cbs(&instance, Objective::SumOfCosts, Limits::expansions(200));

    assert_eq!(result.outcome, Outcome::LimitReached);
    assert_eq!(result.stats.expanded, 200);
    assert_eq!(optimal_cost(&grid, &agents, Objective::SumOfCosts), None);
}

/// Instances with shared starts, shared goals or vertices outside the graph are rejected.
#[test]
fn invalid_instances_are_rejected() {
    let grid = Grid::new(2, 2);
    let at = |start, goal| Agent { start, goal };

    assert_eq!(
        Instance::new(grid.clone(), vec![at(0, 1), at(0, 2)]),
        Err(InstanceError::SharedStart(0, 1))
    );
    assert_eq!(
        Instance::new(grid.clone(), vec![at(0, 1), at(2, 1)]),
        Err(InstanceError::SharedGoal(0, 1))
    );
    assert_eq!(
        Instance::new(grid, vec![at(0, 4)]),
        Err(InstanceError::OutOfRange { agent: 0 })
    );
}

/// The hard instance: the three agents must thread past each other through the centre of
/// a nearly blocked 3x3 grid. Found by the property test.
///
/// ```text
/// @ . @
/// . . .
/// . @ .
/// ```
fn hard_instance() -> (Grid, Vec<Agent>) {
    let grid = Grid::parse(&["@.@", "...", ".@."]);
    let agents = vec![
        Agent { start: 1, goal: 3 },
        Agent { start: 6, goal: 5 },
        Agent { start: 4, goal: 1 },
    ];

    (grid, agents)
}

/// Plain CBS needs about 63 000 expansions on [`hard_instance`]. Slow in debug builds; run
/// with `--release -- --ignored`.
#[test]
#[ignore = "slow: plain CBS needs about 63 000 expansions"]
fn hard_instance_needs_many_expansions() {
    let (grid, agents) = hard_instance();
    let instance = Instance::new(grid.clone(), agents.clone()).expect("valid");

    let result = cbs_with(
        &instance,
        CbsConfig::plain(Objective::SumOfCosts),
        Limits::expansions(200_000),
    );
    let solution = result.solution().expect("solvable");

    assert_eq!(solution.sum_of_costs, 19);
    assert_eq!(assert_valid_plan(&grid, &agents, &solution.paths).0, 19);
    assert_eq!(
        optimal_cost(&grid, &agents, Objective::SumOfCosts),
        Some(19)
    );
}

/// Improved CBS (prioritized conflicts, bypass, CG, conflict avoidance, disjoint
/// splitting) solves [`hard_instance`] with about 1% of plain CBS's expansions; disjoint
/// splitting alone saves about nine in ten of the rest.
#[test]
fn improved_cbs_solves_the_hard_instance() {
    let (grid, agents) = hard_instance();
    let instance = Instance::new(grid.clone(), agents.clone()).expect("valid");
    let improved = CbsConfig::improved(Objective::SumOfCosts);
    let standard = CbsConfig {
        splitting: Splitting::Standard,
        ..improved
    };

    let result = cbs_with(&instance, improved, Limits::expansions(200_000));
    let solution = result.solution().expect("solvable");
    assert_eq!(assert_valid_plan(&grid, &agents, &solution.paths).0, 19);
    assert!(result.stats.expanded < 1_500, "{:?}", result.stats);

    let without = cbs_with(&instance, standard, Limits::expansions(200_000));
    let solution = without.solution().expect("solvable");
    assert_eq!(assert_valid_plan(&grid, &agents, &solution.paths).0, 19);
    assert!(
        result.stats.expanded * 5 < without.stats.expanded,
        "{:?} vs {:?}",
        result.stats,
        without.stats
    );
}

/// A positive constraint makes the low level detour through the required vertex, and a
/// positive constraint on another agent keeps the agent away from it.
#[test]
fn low_level_respects_positive_constraints() {
    let grid = Grid::new(5, 1);
    let agent = Agent { start: 2, goal: 4 };
    let required = Constraint::PositiveVertex {
        agent: 0,
        vertex: 0,
        time: 3,
    };

    let own = ConstraintSet::for_agent(0, &[required]);
    let path = space_time_astar(&grid, agent, &own).expect("detour");
    assert_eq!(
        path.len(),
        8,
        "to 0 by time 3, then 4 steps to the goal: {path:?}"
    );
    assert_eq!(path[3], 0);

    let others = ConstraintSet::for_agent(1, &[required]);
    let agent = Agent { start: 1, goal: 0 };
    let path = space_time_astar(&grid, agent, &others).expect("arrives after 3");
    assert_eq!(path.len(), 5, "may not sit at 0 at time 3: {path:?}");
}

/// Two pairs of agents swap places in two rooms, the right pair later than the left one.
/// Recursive M* plans the pairs apart and needs fewer expansions than basic M*, which
/// branches over all four agents once both pairs collided; both are optimal, for both
/// objectives, and the oracle agrees.
#[test]
fn recursive_mstar_plans_independent_pairs_apart() {
    let grid = Grid::parse(&["...@.....", "@.@@@@@.@"]);
    let at = |x, y| grid.vertex(x, y);
    let agents = vec![
        Agent {
            start: at(0, 0),
            goal: at(2, 0),
        },
        Agent {
            start: at(2, 0),
            goal: at(0, 0),
        },
        Agent {
            start: at(4, 0),
            goal: at(8, 0),
        },
        Agent {
            start: at(8, 0),
            goal: at(4, 0),
        },
    ];
    let instance = Instance::new(grid.clone(), agents.clone()).expect("valid");

    for objective in [Objective::SumOfCosts, Objective::Makespan] {
        let optimum = optimal_cost(&grid, &agents, objective);
        let basic = mstar_with(&instance, MStarConfig::basic(objective), Limits::NONE);
        let recursive = mstar_with(&instance, MStarConfig::recursive(objective), Limits::NONE);

        for result in [&basic, &recursive] {
            let solution = result.solution().expect("solvable");
            assert_valid_plan(&grid, &agents, &solution.paths);
            assert_eq!(Some(solution.cost(objective)), optimum, "{objective:?}");
        }

        assert!(
            recursive.stats.expanded < basic.stats.expanded,
            "{objective:?}: {:?} vs {:?}",
            recursive.stats,
            basic.stats
        );
    }
}

/// The benchmark's instance (`benches/solvers.rs`): eight agents crossing an 8x8 map with
/// a few walls. Optimal sum of costs 92.
pub fn benchmark_instance() -> Instance<Grid> {
    let grid = Grid::parse(&[
        "........", "..@@....", "......@.", ".@......", "....@...", "..@.....", "......@.",
        "........",
    ]);
    let at = |(sx, sy), (gx, gy)| Agent {
        start: grid.vertex(sx, sy),
        goal: grid.vertex(gx, gy),
    };
    let agents = vec![
        at((0, 0), (7, 7)),
        at((7, 0), (0, 7)),
        at((0, 7), (7, 0)),
        at((7, 7), (0, 0)),
        at((3, 0), (3, 7)),
        at((0, 4), (7, 4)),
        at((7, 3), (0, 3)),
        at((4, 7), (4, 0)),
    ];

    Instance::new(grid, agents).expect("valid")
}

/// The conflict avoidance table cuts plain CBS's expansions on the benchmark instance, for
/// both objectives, without changing the optimum.
#[test]
fn conflict_avoidance_reduces_expansions() {
    let instance = benchmark_instance();
    for objective in [Objective::SumOfCosts, Objective::Makespan] {
        let plain = CbsConfig::plain(objective);
        let avoiding = CbsConfig {
            conflict_avoidance: true,
            ..plain
        };

        let without = cbs_with(&instance, plain, Limits::NONE);
        let with = cbs_with(&instance, avoiding, Limits::NONE);
        let cost = |result: &mapf_search::MapfResult| {
            let solution = result.solution().expect("solvable");
            assert_valid_plan(instance.graph(), instance.agents(), &solution.paths);
            solution.cost(objective)
        };

        assert_eq!(cost(&with), cost(&without), "{objective:?}");
        assert!(
            with.stats.expanded < without.stats.expanded,
            "{objective:?}: {:?} vs {:?}",
            with.stats,
            without.stats
        );
    }
}
