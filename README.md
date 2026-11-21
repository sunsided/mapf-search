# mapf-search

[![crates.io](https://img.shields.io/crates/v/mapf-search.svg)](https://crates.io/crates/mapf-search)
[![docs.rs](https://img.shields.io/docsrs/mapf-search)](https://docs.rs/mapf-search)
[![CI](https://github.com/sunsided/mapf-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/mapf-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/mapf-search/graph/badge.svg)](https://codecov.io/gh/sunsided/mapf-search)
[![MSRV](https://img.shields.io/crates/msrv/mapf-search)](https://crates.io/crates/mapf-search)
[![License](https://img.shields.io/crates/l/mapf-search)](https://github.com/sunsided/mapf-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Multi-agent pathfinding: CBS, ECBS, CBS-TA, ICTS, PBS, M*, MO-CBS and MAPF-LNS2.

## Implemented

- CBS, optimal for sum of costs or makespan, with prioritized (cardinal) conflicts,
  bypass, the CG, DG and WDG heuristics, a conflict avoidance table and disjoint
  splitting (with positive constraints), each switchable (`cbs`, `cbs_with`,
  `CbsConfig`)
- ECBS, bounded-suboptimal, focal search at both levels (`ecbs`)
- PBS, priority-based search: fast, neither optimal nor complete (`pbs`)
- ICTS with pairwise pruning, optimal (`icts`)
- M*, basic and recursive (rM*), each with or without operator decomposition (OD-M*,
  ODrM*), optimal, proves unsolvability (`mstar`, `mstar_with`, `MStarConfig`)
- MO-CBS, multi-objective CBS: every Pareto-optimal plan when moves and waits cost
  vectors, shared or per agent; plain or tree-by-tree (MO-CBS-t), over BOA* or NAMOA*dr
  from `multiobjective-search`, with an opt-in feasibility check (proves unsolvability),
  conflict prioritization, bypass and disjoint splitting (`mo_cbs`, `mo_cbs_with`,
  `mo_cbs_per_agent`, `MoCbsConfig`; optional feature `multi-objective`)
- Every single-objective solver takes an `Objective`: sum of costs or makespan
- Shared parts: conflict detection, constraints, space-time A* low level, MDDs, Moving AI
  `.map`/`.scen` parsers

## Planned scope

- Conflict-based: CBS-TA
- Local repair: MAPF-LNS2
- Partial expansion: EPEA*

## Status

Pre-release. Milestones 1 to 6 are drafted; the API will change. See
[docs/design.md](https://github.com/sunsided/mapf-search/blob/main/docs/design.md).

## Usage

```rust
use mapf_search::{Agent, Grid, Instance, Limits, Objective, cbs, first_conflict};

// Two agents swap the ends of a corridor that has one side pocket.
let grid = Grid::parse(&["...", "@.@"]);
let agents = vec![
    Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
    Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
];
let instance = Instance::new(grid, agents)?;

let result = cbs(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
let solution = result.solution().expect("the pocket lets them pass");

assert_eq!(first_conflict(&solution.paths), None);
assert_eq!(solution.sum_of_costs, 7);
# Ok::<(), mapf_search::InstanceError>(())
```

The crate is `no_std` and needs only `alloc`. The feature `multi-objective` adds MO-CBS
over `multiobjective-search`.

## MSRV

Rust 1.85 (`rust-version` in `Cargo.toml`). The feature `multi-objective` needs Rust 1.86,
the MSRV of `multiobjective-search`. The MSRV is a signal, not a contract: it may rise in a
minor release.

## Breaking changes in this draft

Nothing is published yet, but code written against earlier states of this branch breaks:

- **Budgets are `impl Budget<Stats>`** (since `4b4f43b`). heuristic-search adopted
  search-core's `Budget<T>`, which is generic over the statistics type, so every solver
  here takes `budget: impl Budget<Stats>` instead of `impl Budget`. `Limits` and closures
  over `&Stats` still work unchanged; a hand-written `impl Budget for MyBudget` must become
  `impl Budget<Stats> for MyBudget`.
- **Every solver takes an `Objective`**: `ecbs(instance, objective, weight, budget)`,
  `pbs(instance, objective, budget)`, `icts(instance, objective, budget)` and
  `mstar(instance, objective, budget)`.
- **`mstar` runs recursive M\* with operator decomposition (ODrM\*)**, as does
  `MStarConfig::default()` (for sum of costs); `mstar_with` and `MStarConfig` (now with
  `operator_decomposition`) select basic M*, rM* and OD-M*.
- **`CbsConfig`**: `cardinal_heuristic: bool` became `heuristic: CbsHeuristic` (`None`,
  `Cg`, `Dg`, `Wdg`), and the struct gained `splitting: Splitting` and
  `conflict_avoidance: bool`. `CbsConfig::improved` (what `cbs` uses) now means DG,
  disjoint splitting and the conflict avoidance table on top of the earlier improvements.
- **`MoCbsConfig`** gained `feasibility_check`, `prioritize_conflicts`, `bypass` and
  `splitting`, so struct literals need them (or `..MoCbsConfig::plain()`), and `mo_cbs`
  and `MoCbsConfig::default()` now mean `MoCbsConfig::improved()` instead of
  `MoCbsConfig::tree_by_tree()`. The front is the same.
- **`Constraint`** gained `PositiveVertex` and `PositiveEdge`, so exhaustive matches on it
  need new arms, and `ConstraintSet::for_agent` adds the negative constraints that other
  agents' positive constraints imply.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/mapf-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/mapf-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions. See
[CONTRIBUTING.md](https://github.com/sunsided/mapf-search/blob/main/CONTRIBUTING.md).
