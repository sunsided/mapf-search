# mapf-search design

Multi-agent pathfinding (MAPF): find one path per agent so that no two agents collide,
minimizing a joint objective. This crate hosts the conflict-based family (CBS, ECBS,
CBS-TA), ICTS, PBS, M*, MAPF-LNS2 and EPEA*.

These algorithms share the problem model, path conflicts, joint objectives, constraints
and reservations, and the single-agent low-level search. They do **not** share one search
engine: CBS searches a tree of constraint sets, ICTS a tree of cost vectors, M* the joint
state space with collision sets, PBS a tree of priority orderings and LNS2 repairs a full
plan. So the crate is a toolbox of shared parts plus one module per algorithm, not one
engine with pluggable policies.

## Problem model

- **Graph.** A finite directed graph with dense vertex ids `0..n` ([`Vertex`] = `usize`).
  The [`Graph`] trait reports the vertex count and the out-neighbours of a vertex through
  a callback (like `heuristic_search::Successors`), so grids need no adjacency lists.
  Two implementations ship: `AdjacencyList` (explicit lists) and `Grid` (4-connected,
  row-major `y * width + x`, with blocked cells). Generic vertex types (coordinates,
  roadmap nodes) can map onto dense ids; a generic `V` was dropped from the placeholder's
  `Conflict<V>` because distance tables and constraint sets want dense indices.
- **Agents.** Each agent has a start and a goal vertex. Starts are pairwise distinct and
  goals are pairwise distinct; `Instance::new` rejects anything else, and vertices out of
  range.
- **Time.** Discrete, `t = 0, 1, 2, ...`. Every time step each agent either moves along
  one edge or waits at its vertex. Move and wait both cost 1.
- **Paths.** A path is the list of vertices an agent occupies at `t = 0..=T`, so
  `path[t]` is the position at time `t`. A path ends at the agent's goal.
- **Goal-stay semantics.** After its path ends, an agent stays at its goal forever and
  keeps blocking it. Position at `t > T` is `path[T]`. This is the standard model of the
  Moving AI benchmarks and of Sharon et al. (2015). The "disappear at goal" variant is
  not supported.
- **Conflicts** between agents `i` and `j`:
  - *vertex conflict* `(i, j, v, t)`: both occupy `v` at time `t`;
  - *edge (swapping) conflict* `(i, j, u, v, t)`: `i` moves `u -> v` while `j` moves
    `v -> u` between `t` and `t + 1`. Waits never swap, since `u != v`.

  Following conflicts (entering a vertex another agent leaves in the same step) and
  cycle conflicts are allowed. Detection pads the shorter path with its goal, so an agent
  sitting at its goal conflicts with anyone passing through it.

## Objectives

- **Sum of costs (SoC)**: sum over agents of the arrival time `T_i` at the goal, counting
  waits before the final arrival, including waits *at the goal* before leaving it again.
  Waits after the final arrival are free. With paths trimmed to their final arrival this
  is `sum(len_i - 1)`.
- **Makespan**: `max_i T_i`.

`cbs` takes an `Objective` (`SumOfCosts` or `Makespan`); the other solvers minimize sum
of costs only. Both objectives use the same low level:
each agent's path is the cheapest one under its constraints, so a CT node's makespan is
a lower bound on the makespan of every solution below it, just as its SoC is for SoC.
Ordering the CT by makespan is therefore makespan-optimal. Ties go to the lower SoC, but
the result is not guaranteed to have the lowest SoC among all makespan-optimal plans:
that would need a low level that may take longer paths up to the makespan bound.

Makespan CBS also terminates on solvable instances: a node with makespan at most `M*`
only has conflicts, and so constraints, at times up to `M*`, and there are finitely many
such constraint sets. A `Solution` reports both values, `sum_of_costs` and `makespan`.

## Constraints

A constraint forbids one agent something at one time:

- `Vertex { agent, vertex, time }`: agent must not be at `vertex` at `time`;
- `Edge { agent, from, to, time }`: agent must not traverse `from -> to` between `time` and
  `time + 1`.

Per agent, the low level gets a `ConstraintSet`: two `BTreeSet`s, of `(time, vertex)` and
`(time, from, to)`. Lookups are `O(log c)`; c is small (it is the depth of the CT branch).
The set also answers two summary questions the low level needs:

- `latest_step()`: the last time step any constraint touches (an edge constraint at `t`
  touches `t + 1`); sets the search horizon;
- `blocked_until(goal)`: the latest time the goal vertex is forbidden; the agent may
  not finish at or before it.

PBS also needs "blocked from `t` on, forever" for a vertex where a higher-priority agent
stays at its goal. `ConstraintSet::block_from(vertex, t)` stores the earliest such `t` per
vertex. It counts towards `latest_step()`, so the horizon argument below still holds on the
graph minus the blocked vertices, and `blocked_until` reports `Time::MAX` for such a vertex:
an agent can never finish there.

Positive constraints (disjoint splitting, Li et al. 2019) are later work; the
representation keeps room for them as new variants.

## CBS (Sharon et al. 2015)

High level: best-first search over a **constraint tree (CT)**. A CT node holds a set of
constraints, one path per agent that is optimal under its constraints, and the SoC of
those paths.

1. Root: no constraints; plan every agent independently. If an agent has no path at all,
   the instance is unsolvable.
2. Pop the node with the lowest objective value (ties: the other objective, then fewer
   conflicts, then FIFO).
3. Find a conflict between its paths. None: the paths are an optimal solution.
4. Split: for a vertex conflict `(i, j, v, t)` create two children, one with
   `Vertex(i, v, t)` and one with `Vertex(j, v, t)`. For an edge conflict
   `(i, j, u, v, t)` the children get `Edge(i, u, v, t)` and `Edge(j, v, u, t)`.
5. In each child, replan only the constrained agent. If it has no path, drop the child.

**Improvements** (milestone 3). `cbs` uses all of them; `cbs_with` takes a `CbsConfig`
that switches each one (`CbsConfig::plain` is milestone 1's CBS). None changes the
objective value of the result.

- **MDDs.** `Mdd` holds every path of one agent that arrives at its goal at exactly a
  given time under its constraints, layered by time: a forward pass under the constraints
  and the distance bound, then a backward prune. Every remaining node lies on a full path.
  Levels past the depth are the goal alone (goal-stay), so a conflict at a finished
  agent's goal is classified correctly. A node builds MDDs only for its conflicting agents
  and shares the others with its parent through `Rc`.
- **Prioritized conflicts** (Boyarski et al. 2015). A conflict is cardinal for an agent if
  its MDD at its current cost has the conflicting vertex (for a swap: both ends) as the
  only node at that time. Cardinal for both is cardinal, for one semi-cardinal. CBS splits
  the earliest conflict of the most cardinal class. Without prioritizing, it picks the
  earliest conflict (smallest `t`, then the smallest agent pair). Any choice keeps
  optimality.
- **Bypass.** If a child is as cheap *for its replanned agent* as the parent and has
  strictly fewer conflicts, the parent takes over the child's paths (keeping its own
  constraints and MDDs) and is reopened instead of being split. Comparing the agent's own
  cost matters for makespan: a node's makespan can stay the same while one path gets
  dearer, and then the path is no longer the cheapest under the node's constraints. Each
  bypass counts as an expansion.
- **CG heuristic** (Felner et al. 2018). The cardinal conflict graph has agents as
  vertices and an edge per pair with a cardinal conflict. Every vertex cover of it is a set
  of agents whose costs must rise, so its minimum size is an admissible bound on the
  sum-of-costs increase. `vertex_cover_bound` starts from a greedy maximal matching (a
  lower bound) and searches exactly up to size 12, which is exponential in the size. Above
  that it returns the largest size proven necessary. A greedy *cover* would overestimate
  and break optimality. Admissible for sum of costs only, so it is off for makespan.
- On the hard 3x3 instance below, plain CBS needs about 73 000 expansions; with all
  improvements about 10 000. DG/WDG heuristics (Li et al. 2019) and disjoint splitting
  would cut this further.

**Duplicate handling.** A child never re-adds a constraint its parent already holds: the
parent's paths satisfy all of its constraints, and the new constraint is violated by
them. Two CT nodes in different branches can still end up with the same constraint set,
added in different orders (sibling A adds `c_i`, sibling B adds `c_j`, and later
descendants of each add the other). Such duplicates cost work, not optimality. Milestone 1
does no CT duplicate detection; each node stores its full constraint list (a CT node is
small next to its paths). A later milestone can hash the sorted constraint set and skip
known ones. Low-level duplicates are handled by the A* closed list on `(vertex, time)`.

**Termination.** CBS terminates with an optimal solution if one exists. On an unsolvable
instance where every agent alone can reach its goal (two agents swapping ends of a
corridor) the CT is infinite and CBS never terminates. So `cbs` takes a
`heuristic_search::Budget` on CT expansions and reports one of `Solved`, `Unsolvable`
(proven: root replan failed, or the CT ran empty) and `LimitReached`. A proof of
unsolvability in general needs a different method (joint search or the
pebble-motion feasibility test of Kornhauser et al. 1984) and is not attempted.

## Low level: space-time A* with constraints

The low level is `heuristic_search::astar` over the time-expanded graph, through an
adapter that implements `Successors`:

- **State** `(vertex, time)`, both small integers.
- **Successors** of `(v, t)`: wait `(v, t + 1)` and every edge `(u, t + 1)` for
  out-neighbours `u`, each at cost 1, minus those forbidden by a vertex constraint on
  `(target, t + 1)` or an edge constraint on `(v, target, t)`.
- **Goal test** `v == goal && t > blocked_until(goal)`: goal-stay means the agent
  must not need to leave the goal after arriving, so it cannot arrive before the last
  vertex constraint on its goal.
- **Heuristic** `h(v, t) = max(dist(v, goal), blocked_until(goal) + 1 - t)`, where `dist`
  is an exact backward BFS table per agent computed once per CBS run. `None` for
  vertices that cannot reach the goal, so they are dropped.
- **Horizon.** The time-expanded graph is infinite, so A* alone would never report
  `Unsolvable`. After `latest_step()` the constraints are gone, and from any vertex that
  can reach the goal, it is reached within `n - 1` more steps. So successors beyond
  `latest_step() + n` are cut. Any path that exists also exists within the horizon, so
  `Unsolvable` from A* is a proof.

Tie-breaking: heuristic-search breaks `f` ties towards larger `g`, which is what space-time
A* wants (deeper, closer to the goal). A conflict-avoidance-table tie-break (fewer
conflicts with other agents' paths, Sharon et al. 2015) needs a secondary key; see
"Cross-crate needs".

### Decision: reuse heuristic-search

Its API fits without changes: `Successors` with a callback is the time-expanded adapter,
`Goal` and `Heuristic` are closures, `Cost = u32`, and `Outcome::Unsolvable` with the
horizon is a proof. The crate depends on it through a path dependency on the sibling
draft (`../heuristic-search`, branch `draft/initial`, at `58ba6ec` when this was last updated).
**The path dependency must become a crates.io version before `mapf-search` is published.**

The CBS high level reuses heuristic-search's `Budget`, `Limits` and `Stats` for its own
CT expansion counts, so callers limit both layers with the same types, and ECBS takes its
`Weight`. These types are re-exported, which makes heuristic-search a **public dependency**: its breaking releases
are breaking releases of this crate.

## Shared result types

Every solver returns a `MapfResult`: an `Outcome` and high-level `Stats`. `Outcome` is
`Solved(Solution)`, `Unsolvable` (a proof), `LimitReached` (the budget stopped the run) or
`Exhausted`. Only PBS returns `Exhausted`: it ran out of options, which proves nothing.
Solvers that do not support makespan (ECBS, PBS, ICTS, M*) minimize sum of costs and take
no `Objective`.

## ECBS (Barer et al. 2014)

`ecbs(instance, weight, budget)` returns a plan whose sum of costs is at most `w` times the
optimum. `w` is heuristic-search's rational `Weight` (re-exported), compared in integers.

- **Focal list.** A local `FocalList` (`focal.rs`). Each item has an admissible `f` that
  orders the open list, and a *value* (at least `f`) that decides focal membership: value
  at most `floor(w * f_min)`. The focal sublist is ordered by a free secondary key. Items
  can be removed by handle, and the sublist is resynchronized lazily when `f_min` moves
  either way.
- **Low level.** Focal search over `(vertex, time)`. Here `f` is both order and value, and
  the focal key is the accumulated number of conflicts with the other agents' current
  paths (vertex conflicts, swaps, agents parked at their goals), then `f`, then larger `g`.
  `g` is the time, so it never improves: no state is reopened, and `f_min` stays a valid
  lower bound. The search reports `f_min` at the pop of the goal as the agent's lower
  bound. The horizon is the A* low level's. Not counted: future conflicts at the agent's
  own goal after it finishes (the high level still resolves them).
- **High level.** A node's `f` is the sum of its agents' lower bounds; its value is its sum
  of costs; its focal key is the number of conflicting agent pairs, then the sum of costs.
  A child keeps the larger of its parent's and its own bound for the replanned agent; both
  are valid under the child's constraints. Every node satisfies `cost <= w * f`, so the
  node with the smallest `f` is always in focal, and the result costs at most
  `w * f_min <= w * optimum`. The root plans agents one after another, each avoiding the
  earlier ones' paths.
- `w = 1` is optimal; the property tests check this against the oracle.

TODO: switch to heuristic-search's `FocalList` (added in its `b5cc485`) once it settles.
Both crates are changing in parallel, so this crate keeps a small local copy for now.

## PBS (Ma et al. 2019)

`pbs(instance, budget)`: depth-first search over partial priority orders. A node stores the
transitive closure of its order (`higher[a]`, every agent above `a`) and one path per
agent, the cheapest one that avoids the paths of all agents above it. Those paths become
constraints: their vertex at each time, the reverse of each move, and `block_from` at their
goal from their arrival on.

The earliest conflict `(a, b)` gives two children, `a` above `b` and `b` above `a`, unless
the order already says otherwise. A child replans the new lower agent. Then it replans, in
topological order, every agent below that one whose path conflicts with an agent above
it. In a transitive closure an agent above another has strictly fewer agents above it, so
sorting by `|higher|` is a topological order. A failed replan drops the child. The cheaper
child is explored first. PBS is neither optimal nor complete; `Unsolvable` only means an
agent cannot reach its goal even alone.

## ICTS (Sharon et al. 2013)

`icts(instance, budget)`: the high level visits cost vectors in order of their sum
(`BTreeSet` of `(sum, vector)`, deduplicated), starting at every agent's distance to its
goal. Each child raises one agent's cost by 1. For each vector, the agents' MDDs at their
costs (without constraints, built once per agent and cost) are searched for a
conflict-free combination: first every pair alone (simple pairwise pruning), then all
agents. The joint MDD search is a depth-first search over `(time, MDD node per agent)`
with a visited set. An agent past its MDD's depth stays at its goal and still blocks the
others.

An MDD of cost `c` includes paths that reach the goal early and wait. That is fine: such a
plan is also found at the cheaper vector of its actual costs, which is visited first. So
the first solved vector gives an optimal plan. Its sum of costs is recomputed from the
trimmed paths. Unsolvable instances where every agent can reach its goal alone give an
infinite tree, so ICTS ends at its budget.

## M* (Wagner and Choset 2015)

`mstar(instance, budget)`: A* over joint states (every agent's vertex plus a finished
flag), with the oracle's goal-stay pricing written independently. An unfinished agent at
its goal may finish and then stays there for free; every other unfinished agent pays 1 per
step. The heuristic is the sum of the unfinished agents' distances; it is consistent.

Agents outside a state's collision set follow their individual policy: the
smallest-numbered neighbour one step closer to their goal, or finish at the goal. Agents
in it branch over every move to a vertex that can still reach their goal, plus finishing at
the goal. Generating a successor:

- If the joint step collides (same vertex, or a swap), the colliders are merged into the
  expanding state's collision set and passed back. The successor is not stored.
- Otherwise the expanding state is recorded in the successor's back set, the successor's
  current collision set is passed back to it right away, and the successor's `g` is
  relaxed.

Passing back merges a set into a state's set. Wherever that set grows, it reopens the state
and continues through the state's back set (a worklist, not recursion). A popped state is
skipped if it was already expanded with the same `g` and collision-set size.

The joint space is finite, so M* also proves unsolvability, for example on the corridor
swap where CBS and ICTS only hit their budget. This is basic M*: one merged collision set
per state, not recursive M*'s independent subsets, so strongly coupled groups of more than
a few agents are slow (on the benchmark's 8-agent instance it is about 250 times slower than ICTS).

## Later algorithms

- **CBS-TA** (Hönig et al. 2018): task assignment plus pathfinding. The root enumerates
  assignments in order of cost (Murty's k-best assignments); each CT node also carries
  an assignment. Agents get a set of candidate goals.
- **Recursive M*** and ODrM*: split collision sets into independent subproblems.
- **ICTS for makespan**: every agent's cost is the makespan; a simple variant.
- **MAPF-LNS2** (Li et al. 2022): start from a plan with collisions (prioritized planning
  with soft constraints), repeatedly pick a neighbourhood of agents, replan them with
  SIPPS, keep the result if collisions drop. Needs SIPP (safe-interval path planning) as
  a second low level; SIPP intervals are a different state space, so it is a local
  module, not a heuristic-search adapter.
- **EPEA*** (Goldenberg et al. 2014): partial expansion A* for the joint space, using
  operator selection functions to generate only children with `f = F`. Useful for joint
  search and as an ICTS/M* alternative. Belongs in heuristic-search if it is generic.

## Benchmarks: Moving AI format

Support the grid benchmarks of Stern et al. (2019), <https://movingai.com/benchmarks/mapf/>:

- `.map`: header `type octile`, `height H`, `width W`, `map`, then `H` rows of `W`
  characters. Passable: `.`, `G`, `S`; blocked: `@`, `O`, `T`, `W`.
- `.scen`: first line `version 1`, then one agent per line, tab-separated: bucket, map
  name, map width, map height, start x, start y, goal x, goal y, optimal single-agent
  length (octile; for 4-connected grids it is only a lower bound check).

Implemented in `movingai.rs`: `parse_map` -> `Grid`, `parse_scenario` ->
`Vec<ScenarioEntry>`, and `scenario_instance(grid, &entries, k)`, which builds an
`Instance` from the first `k` entries. The parsers read strings, not files, so they stay
`no_std` with `alloc`; reading files is the caller's job. They are lenient about what does
not change meaning and strict about the rest:

- header keys in any order, empty lines and CRLF line ends are accepted;
- scenario fields are split on any whitespace;
- every error carries its 1-based line and a `ParseErrorKind`;
- `scenario_instance` checks the map size each entry declares, blocked start or goal cells,
  and `Instance::new`'s rules (`ScenarioError`).

The `optimal_length` column is kept as `f64`. It is the octile length, which is only a
lower bound on 4-connected grids. Moves stay 4-connected, as in the MAPF benchmarks.

## Testing strategy

- **Independent oracle.** A* (Dijkstra) over the joint state of all agents on tiny grids
  (2-3 agents, up to 3x4 cells), using heuristic-search's `dijkstra` but its own collision
  rules, not this crate's conflict detector. The joint state is the agents' positions plus
  a "finished" bit per agent; an unfinished agent at its goal may finish (stays there
  forever, pays nothing from then on), every other unfinished agent pays 1 per step. This
  prices SoC with goal-stay exactly, including waits at the goal before leaving. For
  makespan, a joint step costs 1 if any agent pays.
- **Properties** (proptest, case count fixed in code), for every solver behind one
  `Solver` enum in the tests: plain and improved CBS for both objectives, each CBS
  improvement alone, ECBS at `w = 1` and at `w` in {1.1, 1.5, 2, 3}, ICTS, M* and PBS. When
  the oracle finds an optimum, a solved plan must pass the plan checker and be within the
  solver's guarantee: equal for the optimal solvers, at most `w` times for ECBS, at least
  the optimum for PBS. When the oracle proves the instance unsolvable, no solver returns
  `Solved`. A complete solver must not return `Unsolvable` on a solvable instance. About
  1 in 600 random solvable instances needs more than the 20 000 expansions the property
  test allows (plain CBS blows up on tightly coupled agents); those cases are discarded, and
  one of them (73 000 expansions for plain CBS, about 10 000 improved) is kept as a
  regression test. Each property passed 1 500 cases once in release mode.
- **Cross-checks** on random 4x4 to 5x5 grids with 3-4 agents, too large for the oracle:
  improved CBS, ICTS and M* agree on the sum of costs whenever they solve, ECBS and PBS
  respect their guarantees relative to it, no solver claims `Unsolvable` while another
  solves, and every plan passes the plan checker (1 500 cases once in release mode).
- **Hand-made cases for every solver:** an agent that steps off its goal, an agent passing
  through a finished agent's goal, an unreachable goal (`Unsolvable`), and a corridor swap
  (never `Solved`; M* proves it `Unsolvable`).
- **Hand-made cases:** swapping conflict detected, goal-stay respected (another agent's
  path crosses a finished agent's goal), an agent that must step off its goal, unreachable
  goal (`Unsolvable`), corridor swap (`LimitReached` under a budget).
- **Parsers:** a hand-written 8x8 map with CRLF line ends and six agents, every
  `ParseErrorKind` and `ScenarioError` variant, and CBS (both objectives) on the parsed
  instance, checked by the oracle's plan checker.
- **Benchmark:** `benches/solvers.rs` (criterion, as in heuristic-search) runs every
  solver on one fixed 8x8 instance with eight agents. All optimal solvers find sum of
  costs 92. Typical times: CBS 0.3 ms plain and 0.4 ms improved, ECBS 0.4 ms, PBS
  0.3 ms, ICTS 1.3 ms, M* 320 ms.
- **Later:** real Moving AI instances with known optimal SoC from published results, as
  slow ignored tests (the benchmark files are not vendored).

## Milestones

1. **Done in this draft.** Instance types (graph trait, adjacency list, grid), conflict
   detection, constraints, space-time A* via heuristic-search, optimal CBS for SoC, oracle
   and property tests.
2. **Done in this draft.** Makespan-optimal CBS; Moving AI `.map`/`.scen` parser.
3. **Done in this draft:** MDDs, prioritized conflicts, bypass, CG heuristic. Open:
   conflict avoidance table (needs the lexicographic cost from heuristic-search), DG/WDG,
   disjoint splitting.
4. **Done in this draft:** ECBS (local focal list), PBS (with `block_from`).
5. **Done in this draft:** ICTS (shares MDDs), basic M*.
6. SIPP and MAPF-LNS2; CBS-TA; EPEA*.

## Cross-crate needs

None blocks the milestones done so far. The two API additions below belong in
heuristic-search. Focal search is implemented locally for now (see ECBS) because both
crates change in parallel; the lexicographic cost is not implemented here.

Milestones 3 to 5 were developed against an export of heuristic-search's `b5cc485`,
because its working tree was mid-change and did not compile at the time. The final checks
ran against the path dependency at `58ba6ec`.

### Secondary tie-break: a lexicographic cost

CBS breaks low-level ties with a conflict avoidance table (CAT): among the cheapest paths,
prefer the one with the fewest conflicts with other agents' current paths (Sharon et al.
2015). The conflict count accumulates along the path, like a cost, so a state-only
tie-break key is not enough: two paths to the same `(vertex, time)` with equal `g` but
different conflict counts must be treated as an improvement, or the search returns a
path with more conflicts than necessary. A lexicographic cost type does exactly that,
with no engine change:

```rust
/// A cost compared lexicographically: `primary` first, then `secondary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Lexicographic<A, B> {
    pub primary: A,
    pub secondary: B,
}

impl<A: Cost, B: Cost> Cost for Lexicographic<A, B> {
    const ZERO: Self = Self { primary: A::ZERO, secondary: B::ZERO };
    /// `primary.cmp_cost`, then `secondary.cmp_cost`.
    fn cmp_cost(&self, other: &Self) -> Ordering;
    /// Component-wise.
    fn times(self, factor: u32) -> Self;
    /// Component-wise.
    fn saturating_add(self, other: Self) -> Self;
}
// plus `Add` component-wise.
```

Here, the low level would use `Successors<Cost = Lexicographic<u32, u32>>` with edge cost
`(1, conflicts_with_cat(from, to, t))` and a heuristic `(dist, 0)`. A* then returns a
cost-optimal path with the fewest CAT conflicts among the cost-optimal ones. The engine's
existing "larger `g` first" tie-break compares the whole lexicographic `g`. That is fine.
A primary-only comparison would also be acceptable; the docs should say which one it is.

### Focal search

ECBS (Barer et al. 2014) runs focal search at both levels. The low level needs:

```rust
/// A secondary cost that orders the focal list; summed along the path like a cost.
pub trait FocalCost<S> {
    /// The secondary cost of the edge `from -> to`, e.g. the number of conflicts it causes.
    fn edge(&mut self, from: &S, to: &S) -> u32;
}

/// Focal search: expands, among open states with `f <= w * f_min`, the one with the
/// smallest accumulated focal cost (ties: smaller `f`, then larger `g`, then FIFO).
/// Returns a path that costs at most `w` times the optimum.
pub fn focal_search<G, T, H, F>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    focal: F,
    weight: Weight,
    budget: impl Budget,
) -> FocalResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    F: FocalCost<G::State>;

/// A `GraphResult` plus the bound ECBS's high level needs.
pub struct FocalResult<G: Successors> {
    pub result: GraphResult<G>,
    /// `f_min` of the open list when the search ended: a lower bound on the optimal cost.
    /// It is `None` when the open list is empty.
    pub lower_bound: Option<G::Cost>,
}
```

`lower_bound` is essential: ECBS's high-level bound is the sum of the agents' `f_min`
values, not their path costs. `Weight` is the existing rational weight. The ECBS high level
is local to this crate. heuristic-search's `b5cc485` adds a `FocalList`; switching the
local one to it is a TODO in `focal.rs`.

### Other needs

- **Reuse across calls.** CBS calls the low level thousands of times, and each call
  allocates a fresh search space and open list. A reusable search context would save
  those allocations. Measure first.
- **Publish.** The path dependency must become a crates.io version before this crate is
  published. `Budget`, `Limits` and `Stats` are re-exported, so heuristic-search is a
  public dependency.

## References

- G. Sharon, R. Stern, A. Felner, N. Sturtevant. Conflict-based search for optimal
  multi-agent pathfinding. Artificial Intelligence 219, 2015.
- M. Barer, G. Sharon, R. Stern, A. Felner. Suboptimal variants of the conflict-based
  search algorithm for the multi-agent pathfinding problem. SoCS 2014. (ECBS)
- E. Boyarski et al. ICBS: improved conflict-based search algorithm for multi-agent
  pathfinding. IJCAI 2015.
- A. Felner et al. Adding heuristics to conflict-based search for multi-agent path
  finding. ICAPS 2018.
- J. Li, A. Felner, E. Boyarski, H. Ma, S. Koenig. Improved heuristics for multi-agent path
  finding with conflict-based search. IJCAI 2019.
- J. Li, D. Harabor, P. Stuckey, H. Ma, S. Koenig. Disjoint splitting for multi-agent path
  finding with conflict-based search. ICAPS 2019.
- W. Hönig et al. Conflict-based search with optimal task assignment. AAMAS 2018. (CBS-TA)
- G. Sharon, R. Stern, M. Goldenberg, A. Felner. The increasing cost tree search for
  optimal multi-agent pathfinding. Artificial Intelligence 195, 2013. (ICTS)
- H. Ma, D. Harabor, P. Stuckey, J. Li, S. Koenig. Searching with consistent
  prioritization for multi-agent path finding. AAAI 2019. (PBS)
- G. Wagner, H. Choset. Subdimensional expansion for multirobot path planning.
  Artificial Intelligence 219, 2015. (M*)
- J. Li, Z. Chen, D. Harabor, P. Stuckey, S. Koenig. MAPF-LNS2: fast repairing for
  multi-agent path finding via large neighborhood search. AAAI 2022.
- M. Goldenberg et al. Enhanced partial expansion A*. JAIR 50, 2014. (EPEA*)
- D. Silver. Cooperative pathfinding. AIIDE 2005. (space-time A*, reservation tables)
- M. Phillips, M. Likhachev. SIPP: safe interval path planning for dynamic
  environments. ICRA 2011.
- R. Stern et al. Multi-agent pathfinding: definitions, variants, and benchmarks. SoCS
  2019. (Moving AI MAPF benchmarks)
- D. Kornhauser, G. Miller, P. Spirakis. Coordinating pebble motion on graphs, the
  diameter of permutation groups, and applications. FOCS 1984.
