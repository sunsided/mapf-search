//! Fuzzes the solvers on tiny grids built from raw bytes: every plan a solver returns must be
//! valid (each path runs from the agent's start to its goal, each step waits or moves to a
//! neighbouring cell, moves run between free cells, and no two paths conflict), and the optimal solvers (CBS, ICTS, M* and EPEA*) must
//! report the same cost whenever more than one of them solves the instance within its
//! budget. CBS-TA runs on the same agents with every agent's goal as a candidate of every
//! agent: its plan must end each agent at a distinct one, and cost no more than CBS's. ECBS,
//! PBS, LNS2 and PIBT must return valid plans. Blocked cells are vertices without edges, so an agent whose start is its goal
//! may wait on a blocked cell. Instances stay at the scale of the property tests, so budgets stay small.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;
use mapf_search::{
    Agent, Grid, Instance, Limits, Lns2Config, MapfResult, Objective, TaskAgent, TaskInstance,
    Weight, cbs, cbs_ta, ecbs, epea, first_conflict, icts, lns2, mstar, pbs, pibt,
};

/// The most columns and rows a fuzzed grid has.
const MAX_SIDE: usize = 4;

/// The most agents a fuzzed instance has.
const MAX_AGENTS: usize = 3;

/// High-level expansions each solver may use. Kept small: on unsolvable instances whose
/// search space is infinite (two agents swapping the ends of a corridor) only the budget
/// stops CBS and ICTS, and ICTS's MDDs deepen with every cost it tries (about 1 GB at 2000
/// expansions on a 1 by 2 corridor).
const BUDGET: u64 = 200;

/// Builds an objective and an instance from bytes: the objective, the width, the height and
/// the agent count, one bit per cell where a set bit blocks it (two bytes, enough for the
/// largest grid), then a start and a goal byte per agent. Returns `None` for too short input
/// or agents that [`Instance::new`] rejects (shared starts or goals).
fn parse(data: &[u8]) -> Option<(Objective, Instance<Grid>)> {
    let (&[objective, width, height, agents, low, high], rest) = data.split_first_chunk::<6>()?;
    let objective = if objective % 2 == 0 {
        Objective::SumOfCosts
    } else {
        Objective::Makespan
    };
    let width = 1 + usize::from(width) % MAX_SIDE;
    let height = 1 + usize::from(height) % MAX_SIDE;
    let agents = 1 + usize::from(agents) % MAX_AGENTS;
    let cells = width * height;
    let grid = grid(width, height, u16::from_le_bytes([low, high]));

    let agents = rest
        .as_chunks::<2>()
        .0
        .iter()
        .take(agents)
        .map(|&[start, goal]| Agent {
            start: usize::from(start) % cells,
            goal: usize::from(goal) % cells,
        })
        .collect();

    Some((objective, Instance::new(grid, agents).ok()?))
}

/// A `width` by `height` grid (at most 16 cells) whose cell `v` is blocked if bit `v` of
/// `blocked` is set.
fn grid(width: usize, height: usize, blocked: u16) -> Grid {
    let mut grid = Grid::new(width, height);
    for vertex in (0..width * height).filter(|&vertex| blocked >> vertex & 1 == 1) {
        let (x, y) = grid.cell(vertex);
        grid.block(x, y);
    }

    grid
}

/// Asserts that `result`'s plan, if any, is valid for `instance`.
fn assert_valid(instance: &Instance<Grid>, result: &MapfResult, solver: &str) {
    let Some(solution) = result.solution() else {
        return;
    };
    let grid = instance.graph();

    assert_eq!(
        solution.paths.len(),
        instance.agents().len(),
        "{solver}: path count"
    );
    for (path, agent) in solution.paths.iter().zip(instance.agents()) {
        assert_eq!(path.first(), Some(&agent.start), "{solver}: start");
        assert_eq!(path.last(), Some(&agent.goal), "{solver}: goal");
        for step in path.windows(2).filter(|step| step[0] != step[1]) {
            let ((x0, y0), (x1, y1)) = (grid.cell(step[0]), grid.cell(step[1]));
            assert_eq!(x0.abs_diff(x1) + y0.abs_diff(y1), 1, "{solver}: jump");
            assert!(
                !grid.is_blocked(step[0]),
                "{solver}: move off a blocked cell"
            );
            assert!(
                !grid.is_blocked(step[1]),
                "{solver}: move onto a blocked cell"
            );
        }
    }

    assert_eq!(first_conflict(&solution.paths), None, "{solver}: conflict");
}

/// Runs CBS-TA with every goal of `instance` as a candidate of every agent and checks its
/// plan against `cbs`, CBS's result on the fixed assignment.
fn check_task_assignment(instance: &Instance<Grid>, objective: Objective, cbs: &MapfResult) {
    let goals: Vec<_> = instance.agents().iter().map(|agent| agent.goal).collect();
    let agents = instance
        .agents()
        .iter()
        .map(|agent| TaskAgent {
            start: agent.start,
            goals: goals.clone(),
        })
        .collect();
    let tasks = TaskInstance::new(instance.graph().clone(), agents).expect("starts are distinct");
    let result = cbs_ta(&tasks, objective, Limits::expansions(BUDGET));
    let Some(solution) = result.solution() else {
        return;
    };

    let mut ends: Vec<_> = solution
        .paths
        .iter()
        .map(|path| path.last().copied())
        .collect();
    ends.sort_unstable();
    ends.dedup();
    assert_eq!(ends.len(), goals.len(), "cbs_ta: shared goal");
    for (path, agent) in solution.paths.iter().zip(instance.agents()) {
        assert_eq!(path.first(), Some(&agent.start), "cbs_ta: start");
    }

    assert_eq!(first_conflict(&solution.paths), None, "cbs_ta: conflict");
    if let Some(fixed) = cbs.solution() {
        assert!(
            solution.cost(objective) <= fixed.cost(objective),
            "cbs_ta costs more than cbs"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((objective, instance)) = parse(data) else {
        return;
    };
    let budget = || Limits::expansions(BUDGET);

    let optimal = [
        ("cbs", cbs(&instance, objective, budget())),
        ("icts", icts(&instance, objective, budget())),
        ("mstar", mstar(&instance, objective, budget())),
        ("epea", epea(&instance, objective, budget())),
    ];
    for (solver, result) in &optimal {
        assert_valid(&instance, result, solver);
    }

    let mut costs = optimal
        .iter()
        .filter_map(|(solver, result)| Some((solver, result.solution()?.cost(objective))));
    if let Some((first, cost)) = costs.next() {
        for (solver, other) in costs {
            assert_eq!(cost, other, "{first} and {solver} disagree on the optimum");
        }
    }

    check_task_assignment(&instance, objective, &optimal[0].1);

    let weight = Weight::new(3, 2);
    assert_valid(
        &instance,
        &ecbs(&instance, objective, weight, budget()),
        "ecbs",
    );
    assert_valid(&instance, &pbs(&instance, objective, budget()), "pbs");
    let config = Lns2Config {
        seed: u64::from(data[0]),
        ..Lns2Config::default()
    };
    assert_valid(&instance, &lns2(&instance, config, budget()), "lns2");
    assert_valid(&instance, &pibt(&instance, budget()), "pibt");
});
