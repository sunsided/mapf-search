//! Hand-made instances: conflict detection, goal-stay semantics, constraints in the low
//! level, unsolvable instances and instance validation.

use heuristic_search::Limits;
use mapf_search::{
    AdjacencyList, Agent, Conflict, Constraint, ConstraintSet, Grid, Instance, InstanceError,
    Outcome, cbs, conflicts, first_conflict, space_time_astar,
};

use crate::oracle::{assert_valid_plan, optimal_sum_of_costs};

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

    let result = cbs(&instance, Limits::expansions(1_000));
    let solution = result.solution().expect("solvable");

    assert_eq!(solution.cost, 4);
    assert_eq!(assert_valid_plan(&graph, &agents, &solution.paths), 4);
    assert_eq!(optimal_sum_of_costs(&graph, &agents), Some(4));
}

/// A goal behind a wall is proven unreachable at the root.
#[test]
fn unreachable_goal_is_unsolvable() {
    let grid = Grid::parse(&[".@."]);
    let agents = vec![Agent { start: 0, goal: 2 }];
    let instance = Instance::new(grid, agents).expect("valid");

    assert_eq!(cbs(&instance, Limits::NONE).outcome, Outcome::Unsolvable);
}

/// Two agents swapping the ends of a corridor without room to pass: the constraint tree is
/// infinite, so CBS stops at its budget, and the oracle proves the instance unsolvable.
#[test]
fn corridor_swap_hits_the_budget() {
    let grid = Grid::new(3, 1);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
    let instance = Instance::new(grid.clone(), agents.clone()).expect("valid");

    let result = cbs(&instance, Limits::expansions(200));

    assert_eq!(result.outcome, Outcome::LimitReached);
    assert_eq!(result.stats.expanded, 200);
    assert_eq!(optimal_sum_of_costs(&grid, &agents), None);
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

/// A solvable instance on which plain CBS needs about 73 000 expansions: the three agents
/// must thread past each other through the centre of a nearly blocked 3x3 grid. Found by
/// the property test. Slow in debug builds; run with `--release -- --ignored`.
///
/// ```text
/// @ . @
/// . . .
/// . @ .
/// ```
#[test]
#[ignore = "slow: plain CBS needs about 73 000 expansions"]
fn hard_instance_needs_many_expansions() {
    let grid = Grid::parse(&["@.@", "...", ".@."]);
    let agents = vec![
        Agent { start: 1, goal: 3 },
        Agent { start: 6, goal: 5 },
        Agent { start: 4, goal: 1 },
    ];
    let instance = Instance::new(grid.clone(), agents.clone()).expect("valid");

    let result = cbs(&instance, Limits::expansions(200_000));
    let solution = result.solution().expect("solvable");

    assert_eq!(solution.cost, 19);
    assert_eq!(assert_valid_plan(&grid, &agents, &solution.paths), 19);
    assert_eq!(optimal_sum_of_costs(&grid, &agents), Some(19));
}
