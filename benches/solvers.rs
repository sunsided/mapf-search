//! Benchmarks: every solver on one fixed 8x8 instance with eight agents. Run with
//! `cargo bench`.

use std::hint::black_box;

use criterion::Criterion;
use mapf_search::{
    Agent, CbsConfig, Grid, Instance, Limits, MStarConfig, MapfResult, Objective, TaskAgent,
    TaskInstance, Weight, cbs_ta, cbs_with, ecbs, icts, mstar_with, pbs,
};

/// The benchmark instance: eight agents crossing an 8x8 map with a few walls.
fn instance() -> Instance<Grid> {
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

/// The benchmark instance with every agent allowed at every agent's goal.
fn anonymous(instance: &Instance<Grid>) -> TaskInstance<Grid> {
    let goals: Vec<_> = instance.agents().iter().map(|agent| agent.goal).collect();
    let agents = instance
        .agents()
        .iter()
        .map(|agent| TaskAgent {
            start: agent.start,
            goals: goals.clone(),
        })
        .collect();

    TaskInstance::new(instance.graph().clone(), agents).expect("valid")
}

/// Benchmarks every solver.
fn main() {
    let instance = instance();
    let tasks = anonymous(&instance);
    let budget = || Limits::expansions(100_000);
    let soc = Objective::SumOfCosts;
    let solvers: [(&str, &dyn Fn() -> MapfResult); 10] = [
        ("cbs plain", &|| {
            cbs_with(&instance, CbsConfig::plain(soc), budget())
        }),
        ("cbs improved", &|| {
            cbs_with(&instance, CbsConfig::improved(soc), budget())
        }),
        ("cbs-ta anonymous", &|| cbs_ta(&tasks, soc, budget())),
        ("ecbs w=1.5", &|| {
            ecbs(&instance, soc, Weight::new(3, 2), budget())
        }),
        ("pbs", &|| pbs(&instance, soc, budget())),
        ("icts", &|| icts(&instance, soc, budget())),
        ("mstar basic", &|| {
            mstar_with(&instance, MStarConfig::basic(soc), budget())
        }),
        // ODrM*, what `mstar` uses, needs about 12 s and 5.4 million expansions here (see
        // docs/design.md), too slow for this benchmark; its rM* form is measured instead.
        ("mstar recursive", &|| {
            mstar_with(&instance, MStarConfig::recursive(soc), budget())
        }),
        ("mstar basic od", &|| {
            let config = MStarConfig {
                operator_decomposition: true,
                ..MStarConfig::basic(soc)
            };
            mstar_with(&instance, config, budget())
        }),
        ("mstar recursive makespan", &|| {
            mstar_with(
                &instance,
                MStarConfig::recursive(Objective::Makespan),
                budget(),
            )
        }),
    ];

    // Basic M* takes about 0.3 s per run; ten samples keep the whole run short.
    let mut c = Criterion::default().sample_size(10).configure_from_args();
    for (name, solve) in solvers {
        assert!(solve().solution().is_some(), "{name} solves the benchmark");
        c.bench_function(name, |b| b.iter(|| black_box(solve())));
    }

    c.final_summary();
}
