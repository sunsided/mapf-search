//! Benchmarks: every solver on one fixed 8x8 instance with eight agents, and MAPF-LNS2 also
//! on a crowded one whose initial plan collides. Run with `cargo bench`.

use std::hint::black_box;

use criterion::Criterion;
use mapf_search::{
    AdppConfig, Agent, CbsConfig, Grid, Instance, Limits, Lns2Config, MStarConfig, MapfResult,
    Objective, TaskAgent, TaskInstance, UaInstance, Weight, WhcaConfig, adpp, ca_star, cbs_id,
    cbs_ta, cbs_with, ecbs, epea, focbs, icts, lns2, mstar_with, pbs, pibt, pibt_turns, rendezvous,
    whca_star,
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

/// Twenty-four agents on a free 8x8 grid, from the first three rows to the mirrored cells
/// of the last three: prioritized planning leaves collisions that LNS2 must repair.
fn crowded() -> Instance<Grid> {
    let agents = (0..24)
        .map(|i| Agent {
            start: i,
            goal: 63 - i,
        })
        .collect();

    Instance::new(Grid::new(8, 8), agents).expect("valid")
}

/// Benchmarks CBS with independence detection on `instance`, and with the `parallel`
/// feature its parallel form on four threads.
fn bench_independence(c: &mut Criterion, instance: &Instance<Grid>) {
    let config = CbsConfig::improved(Objective::SumOfCosts);
    let limits = Limits::expansions(100_000);
    let solve = || cbs_id(instance, config, limits);
    assert!(solve().solution().is_some(), "cbs_id solves the benchmark");
    c.bench_function("cbs_id", |b| b.iter(|| black_box(solve())));

    #[cfg(feature = "parallel")]
    c.bench_function("cbs_id_par 4 threads", |b| {
        b.iter(|| black_box(mapf_search::cbs_id_par(instance, config, limits, 4)));
    });
}

/// Benchmarks PIBT*_T on `instance`, every agent facing east at the start.
fn bench_turns(c: &mut Criterion, instance: &Instance<Grid>) {
    let east = vec![Grid::EAST; instance.agents().len()];
    let solve = || pibt_turns(instance, &east, 0, Limits::expansions(100_000));
    assert!(
        solve().result.solution().is_some(),
        "pibt_turns solves the benchmark"
    );
    c.bench_function("pibt_turns", |b| b.iter(|| black_box(solve())));
}

/// Benchmarks CA* and ADPP on `instance`, WHCA* (default window) on `instance` and
/// `crowded`, and rendezvous planning on a column of four agents crossing an 8x8 grid,
/// linked only to orthogonal neighbours.
fn bench_cooperative(c: &mut Criterion, instance: &Instance<Grid>, crowded: &Instance<Grid>) {
    let budget = || Limits::expansions(100_000);
    let grid = Grid::new(8, 8);
    let column = (0..4)
        .map(|y| Agent {
            start: grid.vertex(0, y),
            goal: grid.vertex(7, y + 4),
        })
        .collect();
    let column = Instance::new(grid.clone(), column).expect("valid");
    let solvers: [(&str, &dyn Fn() -> MapfResult); 5] = [
        ("ca_star", &|| ca_star(instance, budget())),
        ("adpp", &|| adpp(instance, AdppConfig::default(), budget())),
        ("whca_star", &|| {
            whca_star(instance, WhcaConfig::default(), budget())
        }),
        ("whca_star crowded", &|| {
            whca_star(crowded, WhcaConfig::default(), budget())
        }),
        ("rendezvous column", &|| {
            rendezvous(&column, &grid, 4, budget())
        }),
    ];
    for (name, solve) in solvers {
        assert!(solve().solution().is_some(), "{name} solves the benchmark");
        c.bench_function(name, |b| b.iter(|| black_box(solve())));
    }
}

/// Benchmarks FOCBS on `idle`, an instance with agents without goals.
fn bench_focbs(c: &mut Criterion, idle: &UaInstance<Grid>) {
    let solve = || focbs(idle, Limits::expansions(100_000));
    assert!(solve().solution().is_some(), "focbs solves the benchmark");
    c.bench_function("focbs two idle", |b| b.iter(|| black_box(solve())));
}

/// The benchmark instance with its last two agents left without goals.
fn with_idle_agents(instance: &Instance<Grid>) -> UaInstance<Grid> {
    let (assigned, idle) = instance.agents().split_at(6);
    let starts = idle.iter().map(|agent| agent.start).collect();

    UaInstance::new(instance.graph().clone(), assigned.to_vec(), starts).expect("valid")
}

/// Benchmarks every solver.
fn main() {
    let instance = instance();
    let tasks = anonymous(&instance);
    let crowded = crowded();
    let idle = with_idle_agents(&instance);
    let budget = || Limits::expansions(100_000);
    let soc = Objective::SumOfCosts;
    let solvers: [(&str, &dyn Fn() -> MapfResult); 14] = [
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
        ("lns2", &|| lns2(&instance, Lns2Config::default(), budget())),
        ("lns2 crowded", &|| {
            lns2(&crowded, Lns2Config::default(), budget())
        }),
        ("pibt", &|| pibt(&instance, budget())),
        ("icts", &|| icts(&instance, soc, budget())),
        ("epea", &|| epea(&instance, soc, budget())),
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

    bench_focbs(&mut c, &idle);

    bench_turns(&mut c, &instance);
    bench_independence(&mut c, &instance);
    bench_cooperative(&mut c, &instance, &crowded);

    c.final_summary();
}
