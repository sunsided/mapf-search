# mapf-search

[![crates.io](https://img.shields.io/crates/v/mapf-search.svg)](https://crates.io/crates/mapf-search)
[![docs.rs](https://img.shields.io/docsrs/mapf-search)](https://docs.rs/mapf-search)
[![CI](https://github.com/sunsided/mapf-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/mapf-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/mapf-search/graph/badge.svg)](https://codecov.io/gh/sunsided/mapf-search)
[![MSRV](https://img.shields.io/crates/msrv/mapf-search)](https://crates.io/crates/mapf-search)
[![License](https://img.shields.io/crates/l/mapf-search)](https://github.com/sunsided/mapf-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Multi-agent pathfinding: CBS, CBS-TA, ECBS, ICTS, PBS, PIBT (also for agents that need time
to turn), M\*, EPEA\*, MAPF-LNS2 and multi-objective MO-CBS on grids and general graphs.

Given a graph and a set of agents with start and goal vertices, the solvers find one path per
agent so that no two agents ever share a vertex or swap along an edge. CBS, ICTS, M\* and
EPEA\* minimize the sum of costs or the makespan, ECBS stays within a given factor of the optimum,
and PBS, PIBT and MAPF-LNS2 trade optimality for speed. CBS-TA also picks the goals: each agent gets one of its
candidate goals, optimally over all assignments. The optimal solvers are checked against a joint-space oracle by
property tests, and M\* and EPEA\* also prove when an instance has no solution. The crate is
`no_std` with `alloc`, uses the A\* of
[heuristic-search](https://github.com/sunsided/heuristic-search) as its single-agent low
level, and reads the Moving AI `.map` and `.scen` benchmark formats.

Part of the [planning and search crates](https://sunsided.github.io/planning-and-search/); see also this crate's [overview page](https://sunsided.github.io/planning-and-search/crates/mapf-search/).

## Who this is for

- Robotics and warehouse automation: fleets of mobile robots or AGVs on a grid or a roadmap
  graph that must reach their goals without collisions.
- Game developers moving groups of units through tight corridors, where independent A\*
  per unit deadlocks or collides.
- Researchers who want CBS and its improvements (prioritized conflicts, bypass, CG/DG/WDG
  heuristics, disjoint splitting) switchable one by one, next to ECBS, ICTS, PBS, M\* and EPEA\*
  on the same instance type.
- Planners that trade off several costs per agent (time, energy, risk): MO-CBS returns every
  Pareto-optimal joint plan, using
  [multiobjective-search](https://github.com/sunsided/multiobjective-search) as its low
  level.
- For a single agent, use [heuristic-search](https://github.com/sunsided/heuristic-search)
  directly; for many connectors on a diagram that may cross at a cost rather than never
  meet, [connector-routing](https://github.com/sunsided/connector-routing) is the better
  fit.

## Installation

```sh
cargo add mapf-search
```

Or in `Cargo.toml`:

```toml
[dependencies]
mapf-search = "0.1"
```

## Algorithms

- CBS, optimal for sum of costs or makespan, with prioritized (cardinal) conflicts,
  bypass, the CG, DG and WDG heuristics, a conflict avoidance table and disjoint
  splitting (with positive constraints), each switchable (`cbs`, `cbs_with`,
  `CbsConfig`)
- CBS-TA, conflict-based search with task assignment: each agent has candidate goals, and
  the plan is optimal over all assignments, for sum of costs or makespan (`cbs_ta`,
  `cbs_ta_with`, `TaskInstance`)
- FOCBS, flow-oracle CBS for agents without goals (idle robots that only have to get out of
  the way): CBS over the agents with goals, a min-cost flow routes the idle ones, and the
  flow's minimum cut tells CBS which reservations block them; optimal sum of costs for the
  agents with goals, fewest moves for the idle ones given those paths (`focbs`,
  `focbs_with`, `UaInstance`)
- ECBS, bounded-suboptimal, focal search at both levels (`ecbs`)
- PBS, priority-based search: fast, neither optimal nor complete (`pbs`)
- PIBT, priority inheritance with backtracking: plans one time step for all agents at a
  time, very fast, neither optimal nor complete (`pibt`)
- PIBT\*_T, PIBT for robots that face a heading, move only forward and need time to turn in
  place, with cycle handling, swaps and a push escape trigger, on grids or any
  `OrientedGraph`; seeded, neither optimal nor complete (`pibt_turns`, `TurnResult`)
- MAPF-LNS2, large neighborhood search repair: starts from a plan with collisions and
  replans a few agents at a time with SIPPS (safe interval path planning with soft
  obstacles) until none are left; seeded, neither optimal nor complete (`lns2`,
  `Lns2Config`)
- ICTS with pairwise pruning, optimal (`icts`)
- CBS with independence detection: agents that do not interact are solved as separate,
  smaller problems, and conflicting groups merge and are solved again; optimal (`cbs_id`;
  `cbs_id_par` solves the independent groups on several threads, optional feature
  `parallel`)
- M*, basic and recursive (rM*), each with or without operator decomposition (OD-M*,
  ODrM*), optimal, proves unsolvability (`mstar`, `mstar_with`, `MStarConfig`)
- EPEA\*, enhanced partial expansion A\* in the joint space: generates only the children
  at the node's stored `f`, optimal, proves unsolvability (`epea`)
- MO-CBS, multi-objective CBS: every Pareto-optimal plan when moves and waits cost
  vectors, shared or per agent; plain or tree-by-tree (MO-CBS-t), over BOA* or NAMOA*dr
  from `multiobjective-search`, with an opt-in feasibility check (proves unsolvability),
  conflict prioritization, bypass and disjoint splitting (`mo_cbs`, `mo_cbs_with`,
  `mo_cbs_per_agent`, `MoCbsConfig`; optional feature `multi-objective`)
- Every single-objective solver except MAPF-LNS2, PIBT and PIBT\*_T, which only seek a
  feasible plan, and FOCBS, which minimizes the sum of costs, takes an `Objective`: sum of
  costs or makespan
- Shared parts: conflict detection, constraints, space-time A* low level, MDDs, Moving AI
  `.map`/`.scen` parsers

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
over `multiobjective-search`; the feature `parallel` adds `cbs_id_par` and needs `std`.

## MSRV

Rust 1.86 (`rust-version` in `Cargo.toml`), for every feature. The MSRV is a signal, not a
contract: it may rise in a minor release.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/mapf-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/mapf-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions. See
[CONTRIBUTING.md](https://github.com/sunsided/mapf-search/blob/main/CONTRIBUTING.md).
