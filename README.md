# mapf-search

Multi-agent pathfinding: CBS, ECBS, CBS-TA, ICTS, PBS, M* and MAPF-LNS2.

## Implemented

- CBS, optimal for sum of costs or makespan, with prioritized (cardinal) conflicts,
  bypass and the CG heuristic, each switchable (`cbs`, `cbs_with`, `CbsConfig`)
- ECBS, bounded-suboptimal for sum of costs, focal search at both levels (`ecbs`)
- PBS, priority-based search: fast, neither optimal nor complete (`pbs`)
- ICTS with pairwise pruning, optimal for sum of costs (`icts`)
- M* (basic subdimensional expansion), optimal for sum of costs, proves unsolvability
  (`mstar`)
- Shared parts: conflict detection, constraints, space-time A* low level, MDDs, Moving AI
  `.map`/`.scen` parsers

## Planned scope

- Conflict-based: CBS-TA; DG/WDG heuristics and disjoint splitting for CBS
- Recursive M*
- Local repair: MAPF-LNS2
- Partial expansion: EPEA*

## Status

Pre-release. Milestones 1 to 5 are drafted; the API will change. See
[docs/design.md](docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
