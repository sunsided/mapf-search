//! Every solver behind one interface ([`Solver`]), and hand-made instances every solver
//! must get right: an agent stepping off its goal, an agent passing through a finished
//! agent's goal, an unreachable goal and an unsolvable corridor swap.

use heuristic_search::Limits;
use mapf_search::{
    AdjacencyList, Agent, CbsConfig, Graph, Grid, Instance, MapfResult, Objective, Outcome, Weight,
    cbs_with, ecbs, icts, pbs,
};

use crate::oracle::{assert_valid_plan, optimal_cost};

/// A solver and its settings.
#[derive(Debug, Clone, Copy)]
pub enum Solver {
    /// CBS with a configuration; optimal for its objective.
    Cbs(CbsConfig),
    /// ECBS with a suboptimality factor; bounded-suboptimal for sum of costs.
    Ecbs(Weight),
    /// PBS; no guarantee, incomplete.
    Pbs,
    /// ICTS; optimal for sum of costs.
    Icts,
}

impl Solver {
    /// Every solver in a representative configuration, for sum of costs.
    pub fn all() -> [Self; 5] {
        [
            Self::Cbs(CbsConfig::plain(Objective::SumOfCosts)),
            Self::Cbs(CbsConfig::improved(Objective::SumOfCosts)),
            Self::Ecbs(Weight::new(3, 2)),
            Self::Pbs,
            Self::Icts,
        ]
    }

    /// Runs the solver with a budget of `budget` high-level expansions.
    pub fn run<G: Graph>(self, instance: &Instance<G>, budget: u64) -> MapfResult {
        let limits = Limits::expansions(budget);
        match self {
            Self::Cbs(config) => cbs_with(instance, config, limits),
            Self::Ecbs(weight) => ecbs(instance, weight, limits),
            Self::Pbs => pbs(instance, limits),
            Self::Icts => icts(instance, limits),
        }
    }

    /// The objective the solver minimizes.
    pub fn objective(self) -> Objective {
        match self {
            Self::Cbs(config) => config.objective,
            Self::Ecbs(_) | Self::Pbs | Self::Icts => Objective::SumOfCosts,
        }
    }

    /// The guaranteed factor over the optimum as `(numerator, denominator)`; `None` if the
    /// solver guarantees no quality.
    pub fn bound(self) -> Option<(u32, u32)> {
        match self {
            Self::Cbs(_) | Self::Icts => Some((1, 1)),
            Self::Ecbs(weight) => Some((weight.numerator(), weight.denominator())),
            Self::Pbs => None,
        }
    }

    /// Returns `true` if the solver never gives up before its budget, so it returns
    /// `Solved` on every solvable instance given enough budget.
    pub fn is_complete(self) -> bool {
        !matches!(self, Self::Pbs)
    }
}

/// Checks that `cost` is within `solver`'s guarantee over `optimum`.
pub fn assert_within(solver: Solver, cost: u32, optimum: u32) {
    if let Some((numerator, denominator)) = solver.bound() {
        assert!(
            u64::from(cost) * u64::from(denominator) <= u64::from(optimum) * u64::from(numerator),
            "{solver:?}: cost {cost} exceeds the bound over the optimum {optimum}"
        );
    }

    assert!(
        cost >= optimum,
        "{solver:?}: cost {cost} below the optimum {optimum}"
    );
}

/// Runs every solver on `agents` over `graph` and checks the result against the oracle.
fn check_all<G: Graph + Clone>(graph: &G, agents: &[Agent]) {
    let instance = Instance::new(graph.clone(), agents.to_vec()).expect("valid");
    for solver in Solver::all() {
        let optimum = optimal_cost(graph, agents, solver.objective()).expect("solvable");
        let result = solver.run(&instance, 1_000);
        let solution = result
            .solution()
            .unwrap_or_else(|| panic!("{solver:?}: {result:?}"));
        let (soc, makespan) = assert_valid_plan(graph, agents, &solution.paths);

        assert_eq!((soc, makespan), (solution.sum_of_costs, solution.makespan));
        assert_within(solver, solution.cost(solver.objective()), optimum);
    }
}

/// An agent that starts at its goal steps aside into a pocket to let another agent pass.
///
/// ```text
/// 0 - 1 - 2
///     |
///     3
/// ```
#[test]
fn every_solver_steps_off_a_goal() {
    let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 2), (1, 3)]);
    let agents = [Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }];

    check_all(&graph, &agents);
}

/// Agent 0 would reach its goal, vertex 1, before agent 1 passes through it; one of them
/// waits (optimal sum of costs 5).
///
/// ```text
/// 0 - 1 - 2 - 3
///     |
///     4
/// ```
#[test]
fn every_solver_passes_a_finished_goal() {
    let graph = AdjacencyList::undirected(5, &[(0, 1), (1, 2), (2, 3), (1, 4)]);
    let agents = [Agent { start: 4, goal: 1 }, Agent { start: 0, goal: 3 }];

    assert_eq!(
        optimal_cost(&graph, &agents, Objective::SumOfCosts),
        Some(5)
    );
    check_all(&graph, &agents);
}

/// A goal behind a wall is proven unreachable.
#[test]
fn every_solver_proves_an_unreachable_goal() {
    let grid = Grid::parse(&[".@.", "@@@", "..."]);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 6, goal: 8 }];
    let instance = Instance::new(grid, agents).expect("valid");

    for solver in Solver::all() {
        assert_eq!(
            solver.run(&instance, 100).outcome,
            Outcome::Unsolvable,
            "{solver:?}"
        );
    }
}

/// Swapping the ends of a corridor is unsolvable; no solver claims a solution.
#[test]
fn no_solver_solves_a_corridor_swap() {
    let grid = Grid::new(3, 1);
    let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
    let instance = Instance::new(grid, agents).expect("valid");

    for solver in Solver::all() {
        let outcome = solver.run(&instance, 100).outcome;
        assert!(
            !matches!(outcome, Outcome::Solved(_)),
            "{solver:?}: {outcome:?}"
        );
    }
}
