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

Milestone 1 optimizes SoC only. Makespan-optimal CBS needs a different high-level key
(max instead of sum) and the same low level; it comes with milestone 2.

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

Positive constraints (disjoint splitting, Li et al. 2019) and range constraints are
later work; the representation keeps room for them as new variants.

## CBS (Sharon et al. 2015)

High level: best-first search over a **constraint tree (CT)**. A CT node holds a set of
constraints, one path per agent that is optimal under its constraints, and the SoC of
those paths.

1. Root: no constraints; plan every agent independently. If an agent has no path at all,
   the instance is unsolvable.
2. Pop the node with the lowest SoC (ties: fewer conflicts, then FIFO).
3. Find a conflict between its paths. None: the paths are an optimal solution.
4. Split: for a vertex conflict `(i, j, v, t)` create two children, one with
   `Vertex(i, v, t)` and one with `Vertex(j, v, t)`. For an edge conflict
   `(i, j, u, v, t)` the children get `Edge(i, u, v, t)` and `Edge(j, v, u, t)`.
5. In each child, replan only the constrained agent. If it has no path, drop the child.

**Conflict selection.** Milestone 1 picks the earliest conflict (smallest `t`, then the
smallest agent pair). Any choice keeps optimality. Later: prioritized conflicts via MDDs
(cardinal > semi-cardinal > non-cardinal, Boyarski et al. 2015), bypassing, and the
admissible CG/DG/WDG heuristics (Felner et al. 2018; Li et al. 2019).

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
draft (`../heuristic-search`, branch `draft/initial`, at `60d01f4` when this was last updated).
**The path dependency must become a crates.io version before `mapf-search` is published.**

The CBS high level reuses heuristic-search's `Budget`, `Limits` and `Stats` for its own
CT expansion counts, so callers limit both layers with the same types. These types are
re-exported, which makes heuristic-search a **public dependency**: its breaking releases
are breaking releases of this crate.

## Later algorithms

- **ECBS** (Barer et al. 2014): bounded-suboptimal CBS with factor `w`. Both levels use
  focal search: the low level orders a focal list (`f <= w * f_min`) by number of
  conflicts with other agents' paths; the high level by number of conflicting agent
  pairs. Needs a focal open list from heuristic-search, or a local one.
- **CBS-TA** (Hönig et al. 2018): task assignment plus pathfinding. The root enumerates
  assignments in order of cost (Murty's k-best assignments); each CT node also carries
  an assignment. Agents get a set of candidate goals.
- **ICTS** (Sharon et al. 2013): high level is a breadth-first search over cost vectors
  `(c_1, ..., c_k)` starting at individual optima; low level builds an MDD per agent for
  its cost and searches the cross product of MDDs for a conflict-free joint path,
  with pairwise pruning. Shares the MDD type with prioritized CBS conflicts.
- **PBS** (Ma et al. 2019): depth-first search over partial priority orders; on a
  conflict, branch on which agent has priority and replan the lower agents with the
  higher agents' paths as moving obstacles. Needs a reservation table (vertex and edge
  occupancy by time, plus goal-stay "occupied from `t` on").
- **M*** (Wagner and Choset 2011/2015): A* in the joint space where each agent follows its
  individual optimal policy until it collides; colliding agents join a collision set and
  only they branch. Shares the joint-state type with the test oracle.
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

Instances take the first `k` agents of a scenario. The parser reads strings, not files,
so it stays `no_std` with `alloc`; reading files is the caller's job. Milestone 2.

## Testing strategy

- **Independent oracle.** A* (Dijkstra) over the joint state of all agents on tiny grids
  (2-3 agents, up to 3x4 cells), using heuristic-search's `dijkstra` but its own collision
  rules, not this crate's conflict detector. The joint state is the agents' positions plus
  a "finished" bit per agent; an unfinished agent at its goal may finish (stays there
  forever, pays nothing from then on), every other unfinished agent pays 1 per step. This
  prices SoC with goal-stay exactly, including waits at the goal before leaving.
- **Properties** (proptest, case count fixed in code): when the oracle finds an optimum,
  CBS returns `Solved` with the same SoC and conflict-free paths that start at the
  starts, end at the goals and use only edges and waits. When the oracle proves the
  instance unsolvable, CBS does not return `Solved`. About 1 in 600 random solvable
  instances needs more than the 20 000 CT expansions the property test allows (plain CBS
  blows up on tightly coupled agents); those cases are discarded, and one of them
  (73 000 expansions) is kept as an ignored regression test.
- **Hand-made cases:** swapping conflict detected, goal-stay respected (another agent's
  path crosses a finished agent's goal), an agent that must step off its goal, unreachable
  goal (`Unsolvable`), corridor swap (`LimitReached` under a budget).
- **Later:** Moving AI instances with known optimal SoC from published results, as slow
  ignored tests; cross-checks between CBS, ICTS and M* once they exist.

## Milestones

1. **Done in this draft.** Instance types (graph trait, adjacency list, grid), conflict
   detection, constraints, space-time A* via heuristic-search, optimal CBS for SoC, oracle
   and property tests.
2. Moving AI parser, makespan objective, MDDs, prioritized conflicts, CG heuristic,
   conflict avoidance table, disjoint splitting.
3. ECBS (focal search), PBS with a reservation table.
4. ICTS (shares MDDs), M* (shares the joint state).
5. SIPP and MAPF-LNS2; CBS-TA; EPEA*.

## Cross-crate needs

None blocks the milestones done so far; heuristic-search's API (as of `60d01f4`) covers
them. The two API additions below belong in heuristic-search, not here. This crate will
not implement them locally.

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
is local to this crate.

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
