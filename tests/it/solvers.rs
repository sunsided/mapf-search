//! Every solver behind one interface ([`Solver`]), and hand-made instances every solver
//! must get right: an agent stepping off its goal, an agent passing through a finished
//! agent's goal, an unreachable goal and an unsolvable corridor swap.

use heuristic_search::Limits;
use mapf_search::{
    AdjacencyList, AdppConfig, Agent, CbsConfig, Graph, Grid, Instance, Lns2Config, MStarConfig,
    MapfResult, Objective, Outcome, Weight, WhcaConfig, adpp, ca_star, cbs_id, cbs_with, ecbs,
    epea, icts, lns2, mstar_with, pbs, pibt, rendezvous, whca_star,
};

use crate::oracle::{assert_valid_plan, optimal_cost};

/// A solver and its settings.
#[derive(Debug, Clone, Copy)]
pub enum Solver {
    /// CBS with a configuration; optimal for its objective.
    Cbs(CbsConfig),
    /// CBS with independence detection and a configuration; optimal for its objective.
    CbsId(CbsConfig),
    /// ECBS with a suboptimality factor; bounded-suboptimal for its objective.
    Ecbs(Objective, Weight),
    /// PBS; no guarantee, incomplete.
    Pbs(Objective),
    /// ICTS; optimal for its objective.
    Icts(Objective),
    /// M*, basic or recursive; optimal for its objective.
    MStar(MStarConfig),
    /// EPEA*; optimal for its objective.
    Epea(Objective),
    /// MAPF-LNS2 with its default configuration; no guarantee, incomplete. It optimizes
    /// nothing; the objective is the one its plans are checked against.
    Lns2(Objective),
    /// PIBT; no guarantee, incomplete. It optimizes nothing; the objective is the one its
    /// plans are checked against.
    Pibt(Objective),
    /// Cooperative A*; no guarantee, incomplete. It optimizes nothing; the objective is the
    /// one its plans are checked against.
    CaStar(Objective),
    /// WHCA* with a window of 4 and a step of 2; no guarantee, incomplete. It optimizes
    /// nothing; the objective is the one its plans are checked against.
    Whca(Objective),
    /// ADPP with its default configuration; no guarantee, incomplete. It optimizes nothing;
    /// the objective is the one its plans are checked against.
    Adpp(Objective),
    /// Rendezvous every 3 steps with every pair of vertices linked; no guarantee,
    /// incomplete. It optimizes nothing; the objective is the one its plans are checked
    /// against.
    Rendezvous(Objective),
}

impl Solver {
    /// Every solver in a representative configuration, for both objectives.
    pub fn all() -> Vec<Self> {
        [Objective::SumOfCosts, Objective::Makespan]
            .into_iter()
            .flat_map(Self::all_for)
            .collect()
    }

    /// Every solver in a representative configuration, for `objective`.
    pub fn all_for(objective: Objective) -> [Self; 16] {
        [
            Self::Cbs(CbsConfig::plain(objective)),
            Self::Cbs(CbsConfig::improved(objective)),
            Self::CbsId(CbsConfig::improved(objective)),
            Self::Ecbs(objective, Weight::new(3, 2)),
            Self::Pbs(objective),
            Self::Icts(objective),
            Self::MStar(MStarConfig::basic(objective)),
            Self::MStar(MStarConfig::recursive(objective)),
            Self::MStar(MStarConfig::od_recursive(objective)),
            Self::Epea(objective),
            Self::Lns2(objective),
            Self::Pibt(objective),
            Self::CaStar(objective),
            Self::Whca(objective),
            Self::Adpp(objective),
            Self::Rendezvous(objective),
        ]
    }

    /// Runs the solver with a budget of `budget` high-level expansions.
    pub fn run<G: Graph>(self, instance: &Instance<G>, budget: u64) -> MapfResult {
        let limits = Limits::expansions(budget);
        match self {
            Self::Cbs(config) => cbs_with(instance, config, limits),
            Self::CbsId(config) => cbs_id(instance, config, limits),
            Self::Ecbs(objective, weight) => ecbs(instance, objective, weight, limits),
            Self::Pbs(objective) => pbs(instance, objective, limits),
            Self::Icts(objective) => icts(instance, objective, limits),
            Self::MStar(config) => mstar_with(instance, config, limits),
            Self::Epea(objective) => epea(instance, objective, limits),
            // LNS2 that has not repaired a tiny instance in 500 iterations is stuck in a
            // local minimum (such as two agents and a side pocket); more only costs time.
            Self::Lns2(_) => lns2(
                instance,
                Lns2Config::default(),
                Limits::expansions(budget.min(500)),
            ),
            // PIBT that has not finished a tiny instance in 500 steps oscillates.
            Self::Pibt(_) => pibt(instance, Limits::expansions(budget.min(500))),
            Self::CaStar(_) => ca_star(instance, limits),
            // Like PIBT, WHCA* that has not finished a tiny instance in 500 rounds oscillates.
            Self::Whca(_) => whca_star(
                instance,
                WhcaConfig { window: 4, step: 2 },
                Limits::expansions(budget.min(500)),
            ),
            Self::Adpp(_) => adpp(instance, AdppConfig::default(), limits),
            // Like PIBT, rendezvous planning that has not finished a tiny instance in 500
            // epochs waits forever.
            Self::Rendezvous(_) => rendezvous(
                instance,
                &complete(instance.graph().vertex_count()),
                3,
                Limits::expansions(budget.min(500)),
            ),
        }
    }

    /// The objective the solver minimizes.
    pub fn objective(self) -> Objective {
        match self {
            Self::Cbs(config) | Self::CbsId(config) => config.objective,
            Self::MStar(config) => config.objective,
            Self::Ecbs(objective, _)
            | Self::Pbs(objective)
            | Self::Icts(objective)
            | Self::Epea(objective)
            | Self::Lns2(objective)
            | Self::Pibt(objective)
            | Self::CaStar(objective)
            | Self::Whca(objective)
            | Self::Adpp(objective)
            | Self::Rendezvous(objective) => objective,
        }
    }

    /// The guaranteed factor over the optimum as `(numerator, denominator)`; `None` if the
    /// solver guarantees no quality.
    pub fn bound(self) -> Option<(u32, u32)> {
        match self {
            Self::Cbs(_) | Self::CbsId(_) | Self::Icts(_) | Self::MStar(_) | Self::Epea(_) => {
                Some((1, 1))
            }
            Self::Ecbs(_, weight) => Some((weight.numerator(), weight.denominator())),
            Self::Pbs(_)
            | Self::Lns2(_)
            | Self::Pibt(_)
            | Self::CaStar(_)
            | Self::Whca(_)
            | Self::Adpp(_)
            | Self::Rendezvous(_) => None,
        }
    }

    /// Returns `true` if the solver never gives up before its budget, so it returns
    /// `Solved` on every solvable instance given enough budget.
    pub fn is_complete(self) -> bool {
        !matches!(
            self,
            Self::Pbs(_)
                | Self::Lns2(_)
                | Self::Pibt(_)
                | Self::CaStar(_)
                | Self::Whca(_)
                | Self::Adpp(_)
                | Self::Rendezvous(_)
        )
    }
}

/// The communication graph that links every pair of the `vertices`.
fn complete(vertices: usize) -> AdjacencyList {
    let edges: Vec<(usize, usize)> = (0..vertices)
        .flat_map(|a| (0..vertices).filter(move |&b| b != a).map(move |b| (a, b)))
        .collect();

    AdjacencyList::directed(vertices, &edges)
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
    check_solvers(graph, agents, Solver::all());
}

/// Runs `solvers` on `agents` over `graph` and checks each result against the oracle.
fn check_solvers<G: Graph + Clone>(graph: &G, agents: &[Agent], solvers: Vec<Solver>) {
    let instance = Instance::new(graph.clone(), agents.to_vec()).expect("valid");
    for solver in solvers {
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
///
/// Every solver but PIBT, CA*, ADPP and rendezvous planning: the graph has no cycle, so
/// PIBT pushes the agent at the goal to a dead end and back forever
/// (`pibt::tests::can_livelock_off_a_cycle`), the fixed-priority planners CA* and ADPP let
/// agent 0 hold its goal for good (`cooperative::tests`, `adpp::tests`), and rendezvous
/// planning takes only epochs that bring the team closer to its goals, which stepping off
/// one does not.
#[test]
fn every_solver_steps_off_a_goal() {
    let graph = AdjacencyList::undirected(4, &[(0, 1), (1, 2), (1, 3)]);
    let agents = [Agent { start: 1, goal: 1 }, Agent { start: 0, goal: 2 }];
    let solvers = Solver::all()
        .into_iter()
        .filter(|solver| {
            !matches!(
                solver,
                Solver::Pibt(_) | Solver::CaStar(_) | Solver::Adpp(_) | Solver::Rendezvous(_)
            )
        })
        .collect();

    check_solvers(&graph, &agents, solvers);
}

/// Agent 0 would reach its goal, vertex 1, before agent 1 passes through it; one of them
/// waits (optimal sum of costs 5).
///
/// ```text
/// 0 - 1 - 2 - 3
///     |
///     4
/// ```
///
/// Every solver but CA* and ADPP, which give agent 0 priority and let it hold its goal for
/// good, and rendezvous planning, which takes only epochs that bring the team closer to its
/// goals: agent 0 would have to step off its goal and back.
#[test]
fn every_solver_passes_a_finished_goal() {
    let graph = AdjacencyList::undirected(5, &[(0, 1), (1, 2), (2, 3), (1, 4)]);
    let agents = [Agent { start: 4, goal: 1 }, Agent { start: 0, goal: 3 }];
    let solvers = Solver::all()
        .into_iter()
        .filter(|solver| {
            !matches!(
                solver,
                Solver::CaStar(_) | Solver::Adpp(_) | Solver::Rendezvous(_)
            )
        })
        .collect();

    assert_eq!(
        optimal_cost(&graph, &agents, Objective::SumOfCosts),
        Some(5)
    );
    check_solvers(&graph, &agents, solvers);
}

/// On a directed graph a step can raise the sum-of-distances `f` by more than 2: the edge
/// `0 -> 1` leads into a one-way detour three steps longer than the shortcut `0 -> 4`.
///
/// ```text
/// 0 -> 4, 0 -> 1 -> 2 -> 3 -> 4
/// ```
#[test]
fn every_solver_handles_a_one_way_detour() {
    let graph = AdjacencyList::directed(5, &[(0, 4), (0, 1), (1, 2), (2, 3), (3, 4)]);

    check_all(&graph, &[Agent { start: 0, goal: 4 }]);
    check_all(
        &graph,
        &[Agent { start: 0, goal: 4 }, Agent { start: 1, goal: 3 }],
    );
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
