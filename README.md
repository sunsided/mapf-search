# mapf-search

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
[docs/design.md](docs/design.md).

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

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
