//! Benchmarks: the four MO-CBS variants of the paper (plain or tree-by-tree, BOA* or
//! NAMOA*dr) and the improved configuration on one fixed 6x6 instance with three agents and two objectives and hashed edge costs. Run with
//! `cargo bench --features multi-objective --bench mo_cbs`.

use std::hint::black_box;

use criterion::Criterion;
use mapf_search::{
    Agent, Grid, Instance, Limits, LowLevel, MoCbsConfig, ParetoOutcome, ParetoResult, Vertex,
    mo_cbs_with,
};

/// The benchmark instance: three agents crossing a 6x6 map with a few walls.
fn instance() -> Instance<Grid> {
    let grid = Grid::parse(&["......", ".@@...", "...@..", "..@...", "...@@.", "......"]);
    let at = |(sx, sy), (gx, gy)| Agent {
        start: grid.vertex(sx, sy),
        goal: grid.vertex(gx, gy),
    };
    let agents = vec![at((0, 0), (5, 5)), at((5, 0), (0, 5)), at((0, 5), (5, 0))];

    Instance::new(grid, agents).expect("valid")
}

/// Edge costs as in the paper's main experiments (Sec. VIII-A): each component an integer
/// in `1..=4`, here from a fixed hash of the undirected edge instead of a random draw.
/// Waits are self-loops and priced the same way.
fn costs(from: Vertex, to: Vertex) -> [u32; 2] {
    let (a, b) = (from.min(to), from.max(to));
    let component = |x: usize, y: usize| u32::try_from((a * x + b * y) % 4).expect("small") + 1;

    [component(31, 17), component(13, 29)]
}

/// Benchmarks every variant.
fn main() {
    let instance = instance();
    let budget = || Limits::expansions(100_000);
    let variant = |tree_by_tree, low_level| MoCbsConfig {
        tree_by_tree,
        low_level,
        ..MoCbsConfig::plain()
    };
    let solvers: [(&str, &dyn Fn() -> ParetoResult<2>); 5] = [
        ("mo-cbs-b", &|| {
            mo_cbs_with(&instance, &costs, variant(false, LowLevel::Auto), budget())
        }),
        ("mo-cbs-n", &|| {
            mo_cbs_with(
                &instance,
                &costs,
                variant(false, LowLevel::NamoaDr),
                budget(),
            )
        }),
        ("mo-cbs-tb", &|| {
            mo_cbs_with(&instance, &costs, variant(true, LowLevel::Auto), budget())
        }),
        ("mo-cbs improved", &|| {
            mo_cbs_with(&instance, &costs, MoCbsConfig::improved(), budget())
        }),
        ("mo-cbs-tn", &|| {
            mo_cbs_with(
                &instance,
                &costs,
                variant(true, LowLevel::NamoaDr),
                budget(),
            )
        }),
    ];

    let mut c = Criterion::default().sample_size(10).configure_from_args();
    for (name, solve) in solvers {
        let result = solve();
        assert_eq!(result.outcome, ParetoOutcome::Complete, "{name} completes");
        eprintln!(
            "{name}: front of {}, {} expansions",
            result.front.len(),
            result.stats.expanded
        );
        c.bench_function(name, |b| b.iter(|| black_box(solve())));
    }

    c.final_summary();
}
