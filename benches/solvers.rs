//! A dependency-free benchmark: every solver on one fixed 8x8 instance with eight agents,
//! timed with `std::time::Instant` over a few runs. Run with `cargo bench`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use mapf_search::{
    Agent, CbsConfig, Grid, Instance, Limits, MapfResult, Objective, Weight, cbs_with, ecbs, icts,
    mstar, pbs,
};

/// How often each solver runs; the fastest run is reported.
const RUNS: usize = 5;

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

/// The fastest of [`RUNS`] runs of `solve`, and its last result.
fn time(solve: impl Fn() -> MapfResult) -> (Duration, MapfResult) {
    let mut best = Duration::MAX;
    let mut result = None;
    for _ in 0..RUNS {
        let start = Instant::now();
        let outcome = black_box(solve());
        best = best.min(start.elapsed());
        result = Some(outcome);
    }

    (best, result.expect("at least one run"))
}

/// Times every solver and prints one line each.
fn main() {
    let instance = instance();
    let budget = || Limits::expansions(100_000);
    let soc = Objective::SumOfCosts;
    let solvers: [(&str, &dyn Fn() -> MapfResult); 6] = [
        ("cbs (plain)", &|| {
            cbs_with(&instance, CbsConfig::plain(soc), budget())
        }),
        ("cbs (improved)", &|| {
            cbs_with(&instance, CbsConfig::improved(soc), budget())
        }),
        ("ecbs (w = 1.5)", &|| {
            ecbs(&instance, Weight::new(3, 2), budget())
        }),
        ("pbs", &|| pbs(&instance, budget())),
        ("icts", &|| icts(&instance, budget())),
        ("m*", &|| mstar(&instance, budget())),
    ];

    for (name, solve) in solvers {
        let (elapsed, result) = time(solve);
        let cost = result.solution().map(|s| s.sum_of_costs);
        println!(
            "{name:<16} {elapsed:>12.3?}  sum of costs {cost:?}, {} expanded",
            result.stats.expanded
        );
    }
}
