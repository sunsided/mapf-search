//! Benchmarks: every solver on one fixed 8x8 instance with eight agents. Run with
//! `cargo bench`.

use std::hint::black_box;

use criterion::Criterion;
use mapf_search::{
    Agent, CbsConfig, Grid, Instance, Limits, MapfResult, Objective, Weight, cbs_with, ecbs, icts,
    mstar, pbs,
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

/// Benchmarks every solver.
fn main() {
    let instance = instance();
    let budget = || Limits::expansions(100_000);
    let soc = Objective::SumOfCosts;
    let solvers: [(&str, &dyn Fn() -> MapfResult); 6] = [
        ("cbs plain", &|| {
            cbs_with(&instance, CbsConfig::plain(soc), budget())
        }),
        ("cbs improved", &|| {
            cbs_with(&instance, CbsConfig::improved(soc), budget())
        }),
        ("ecbs w=1.5", &|| {
            ecbs(&instance, soc, Weight::new(3, 2), budget())
        }),
        ("pbs", &|| pbs(&instance, soc, budget())),
        ("icts", &|| icts(&instance, soc, budget())),
        ("mstar", &|| mstar(&instance, soc, budget())),
    ];

    // M* takes about 0.3 s per run; ten samples keep the whole run short.
    let mut c = Criterion::default().sample_size(10).configure_from_args();
    for (name, solve) in solvers {
        assert!(solve().solution().is_some(), "{name} solves the benchmark");
        c.bench_function(name, |b| b.iter(|| black_box(solve())));
    }

    c.final_summary();
}
