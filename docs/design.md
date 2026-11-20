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

Every solver takes an `Objective` (`SumOfCosts` or `Makespan`). For CBS, both objectives
use the same low level: each agent's path is the cheapest one under its constraints, so a CT node's makespan is
a lower bound on the makespan of every solution below it, just as its SoC is for SoC.
Ordering the CT by makespan is therefore makespan-optimal. Ties go to the lower SoC, but
the result is not guaranteed to have the lowest SoC among all makespan-optimal plans:
that would need a low level that may take longer paths up to the makespan bound.

Makespan CBS also terminates on solvable instances: a node with makespan at most `M*`
only has conflicts, and so constraints, at times up to `M*`, and there are finitely many
such constraint sets. A `Solution` reports both values, `sum_of_costs` and `makespan`.

The other solvers:

- **ECBS** orders nodes by the maximum of the agents' lower bounds instead of their sum,
  and admits a node to focal by its makespan instead of its sum of costs. Each path costs
  at most `w` times its agent's bound, so a node's makespan is at most `w` times the
  largest bound, and the bound guarantee carries over.
- **PBS** explores the child with the lower objective value first (ties: the other
  objective). Its low level is unchanged.
- **ICTS** searches the chain of vectors `(m, ..., m)` from the largest individual optimum
  on. An MDD of depth `m` contains the paths that arrive earlier and wait, so the first
  solvable `m` is the optimal makespan.
- **M\*** solves makespan through sum of costs under deadlines (see M*). Its plan also
  has the least sum of costs among the makespan-optimal plans.

The other solvers do not minimize the sum of costs among the makespan-optimal plans.

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

**Positive constraints** (disjoint splitting, Li et al. 2019):

- `PositiveVertex { agent, vertex, time }`: agent must be at `vertex` at `time`;
- `PositiveEdge { agent, from, to, time }`: agent must move `from -> to` then.

A positive constraint implies negative ones for every *other* agent: nobody else at
`vertex` at `time`; for an edge nobody else at `from` at `time`, at `to` at `time + 1`, or
moving `to -> from`. `ConstraintSet::for_agent` adds those, so the A* low level, the MDDs
and every other user of a `ConstraintSet` see them without knowing about positive
constraints. In the set, a positive edge is two required `(time, vertex)` pairs (being at
`from` at `t` and at `to != from` at `t + 1` is that move). A required pair forbids every
other vertex at its time, so two different ones at one time make the set unsatisfiable;
it counts towards `latest_step()`, and towards `blocked_until(goal)` when it requires
another vertex.

The low level also uses them as **landmarks** (`low_level/landmarks.rs`): at time `t`, a
state farther than `t_k - t` from the next required vertex `v_k` is pruned, and if the
path must go on after `t_k`, the heuristic is at least `t_k - t` plus the rest of the chain.
A landmark at the goal with nothing after it only prunes: goal-stay lets the agent arrive
early and wait there. (Bounding through it anyway was a bug the benchmark caught: CBS
returned sum of costs 94 instead of 92.) The landmark estimate is consistent, and the
maximum of it and the plain heuristic is too.

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
- **DG and WDG heuristics** (Li et al. 2019, `CbsHeuristic::Dg` and `::Wdg`; `cbs` uses
  DG). Over the node's conflicting pairs (`cbs/heuristic.rs`):
  - DG: an edge for every *dependent* pair, whose MDDs at their current costs have no
    conflict-free combination (`mdd::joint_paths`, the joint search ICTS also uses). A
    cardinal conflict implies dependence, so the MDD product is only searched otherwise.
    The bound is the same vertex cover as CG's, over a superset of its edges.
  - WDG: each DG edge weighted by `Δ = optimum of the pair alone under its constraints -
    current costs`, at least 1 (`cbs/pairs.rs`: a best-first CBS over two agents with
    standard splitting). The pair search can be infinite (a corridor swap), so it stops
    after 64 expansions and returns the cost of its cheapest open node, a lower bound. If
    its open list runs empty, the pair has no joint plan and the node is dropped. The bound
    is a minimum edge-weighted vertex cover, `min sum x_i` with `x_a + x_b >= Δ_ab`
    (`cbs/cover.rs`): exact per connected component by branch and bound while the product
    of `1 + max weight` over its agents stays below 65 536, else the weight of a greedy
    set of disjoint edges.
  - Pair weights depend only on the two agents' constraint sets (implied ones included),
    so they are cached by them for the whole run.
  - Pairs without a conflict at the node count as independent. MDDs a positive child did
    not rebuild may be too wide, which can only hide a dependence, so the bound stays
    admissible.

  Tests: a crate-internal property expands plain CBS (standard and disjoint splitting) on
  random 3x3 grids and checks, at each node, CG <= DG <= WDG and `cost + WDG <=` the optimum
  below the node (plain CBS from it); a WDG weight one too high fails it at once. On the
  hard instance, improved CBS needs about 980 expansions without a heuristic, 740 with CG,
  590 with DG and 370 with WDG. WDG's pair searches make it about three times slower than
  DG there, so `cbs` uses DG.
- **Disjoint splitting** (Li et al. 2019, `Splitting::Disjoint`). For a conflict of `a`
  and `b`, pick one agent `i`; one child gets the negative constraint on `i` (as in
  standard splitting), the other the positive one, which forbids the conflict to everyone
  else. The two subtrees share no solution. The positive child keeps `i`'s path (it takes
  part in the conflict, so it meets the constraint, and stays optimal under more
  constraints) but drops `i`'s MDD, and replans every other agent whose path breaks an
  implied negative constraint. `i` is the agent whose MDD forces it through the conflict,
  if exactly one is and both MDDs are known, else the smaller id: then the positive child
  keeps `i`'s cost and the negative one raises it. Bypass takes over all the paths a
  child replanned when each of them keeps its cost. Agents that are not replanned keep
  their MDDs, which may be wider than under the new implied constraints; a wider MDD only
  makes conflicts look less cardinal, so the CG bound stays admissible.
- On the hard 3x3 instance below, plain CBS needs about 73 000 expansions (63 000 since
  the low level's tie-breaking changed), plain CBS with disjoint splitting about 1 200,
  and with all improvements about 590 (with CG: 740; 7 000 with standard splitting).

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
A* wants (deeper, closer to the goal).

**Conflict avoidance table** (`CbsConfig::conflict_avoidance`, on in `cbs`). Among the
cheapest paths, prefer the one with the fewest conflicts with the other agents' current
paths (Sharon et al. 2015). The conflict count accumulates along the path like a cost, so
the low level runs A* over a local `Lexicographic` cost `(time steps, conflicts)`
(`low_level/lexicographic.rs`) with edge cost `(1, conflicts of the move)` and heuristic
`(h, 0)`; that heuristic stays consistent, so A* returns a cheapest path with the fewest
conflicts among the cheapest. The counts come from `ConflictTable` (`conflict_table.rs`,
shared with ECBS): vertex conflicts, swaps and agents parked at their goals. Conflicts the
agent causes by staying at its own goal are not counted: every cheapest path arrives at
the same goal at the same time, so they add the same number to each. With the table, the
root plans the agents one after another, each avoiding the earlier ones. Without it, the
low level keeps a plain `u32` cost (a `Pricing` type parameter picks one), the hot path of
plain CBS and PBS. Plain CBS needs 6 instead of 14 expansions on the benchmark instance
and about 41 000 instead of 63 000 on the hard 3x3 instance; improved CBS about 7 000
instead of 10 000 there.

### Decision: reuse heuristic-search

Its API fits without changes: `Successors` with a callback is the time-expanded adapter,
`Goal` and `Heuristic` are closures, `Cost = u32`, and `Outcome::Unsolvable` with the
horizon is a proof. The crate depends on it through a path dependency on the sibling
draft (`../heuristic-search`, branch `draft/initial`, at `dbc85f8` when this was last
updated).
**The path dependency must become a crates.io version before `mapf-search` is published.**

The CBS high level reuses heuristic-search's `Budget`, `Limits` and `Stats` for its own
CT expansion counts, so callers limit both layers with the same types, and ECBS takes its
`Weight`. These types are re-exported, which makes heuristic-search a **public
dependency**: its breaking releases are breaking releases of this crate. That already
happened once on this branch: heuristic-search moved `Budget` to search-core, where it is
`Budget<T>`, generic over the statistics type, so every solver here now takes
`impl Budget<Stats>` (`4b4f43b`). `Limits` and closures over `&Stats` work as before; a
caller's own `impl Budget` must name `Stats`. The README lists this and the other breaking
changes of the draft.

## Shared result types

Every solver returns a `MapfResult`: an `Outcome` and high-level `Stats`. `Outcome` is
`Solved(Solution)`, `Unsolvable` (a proof), `LimitReached` (the budget stopped the run) or
`Exhausted`. Only PBS returns `Exhausted`: it ran out of options, which proves nothing.

## ECBS (Barer et al. 2014)

`ecbs(instance, objective, weight, budget)` returns a plan whose objective value is at
most `w` times the optimum (see Objectives for makespan). `w` is heuristic-search's rational `Weight` (re-exported), compared in integers.

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
  bound. The horizon is the A* low level's. Finishing is a step of its own: from the goal,
  once the agent may stay, a free step leads to a *finished* copy of the state, the only
  goal state. Its focal key adds the conflicts the agent causes by staying: every later
  visit of another agent to the goal, and another agent parking there later. The copy has
  the same `f`, so the lower bound is unchanged. The counts come from a shared
  `ConflictTable` (`conflict_table.rs`), also used by CBS's tie-break.
- **High level.** A node's `f` is the sum of its agents' lower bounds; its value is its sum
  of costs; its focal key is the number of conflicting agent pairs, then the sum of costs.
  A child keeps the larger of its parent's and its own bound for the replanned agent; both
  are valid under the child's constraints. Every node satisfies `cost <= w * f`, so the
  node with the smallest `f` is always in focal, and the result costs at most
  `w * f_min <= w * optimum`. The root plans agents one after another, each avoiding the
  earlier ones' paths.
- `w = 1` is optimal; the property tests check this against the oracle.

**Focal lists.** The low level uses heuristic-search's `FocalList` (`push(item, f, g,
key)` over dense arena ids, `pop`, `min_f`, `contains`): `f` orders it and decides focal
membership, the time is `g`, the conflict count is the key, and its tie-breaks (smaller `f`,
then larger `g`, then first in) are what the local list did. Re-keying an open state with
fewer conflicts is a second `push`; `min_f` before the pop is the lower bound.

The high level keeps the local `FocalList` (`focal.rs`). The upstream list admits an item
to focal by its `f`, but ECBS's high level needs two values: the lower bound orders the
list and is the `f_min` the guarantee is measured against, while the node's *cost*, which
may be larger, must be within `w * f_min`. With the bound as `f`, upstream would admit
nodes whose cost exceeds `w * f_min`, which breaks the guarantee; with the cost as `f`,
`f_min` is no longer a lower bound on the optimum. Upstream would fit with a separate
focal value (`push(item, f, value, g, key)` with `f <= value`), as the local list has.

## PBS (Ma et al. 2019)

`pbs(instance, objective, budget)`: depth-first search over partial priority orders. A node stores the
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

`icts(instance, objective, budget)`: the high level visits cost vectors in order of their sum
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

`mstar(instance, objective, budget)`: A* over joint states (every agent's vertex plus a finished
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
swap where CBS and ICTS only hit their budget.

### Recursive M* (rM*)

`mstar` runs recursive M*; `mstar_with(instance, MStarConfig, budget)` selects basic M*
(`MStarConfig::basic`) or rM* (`MStarConfig::recursive`). Both share one planner
(`mstar/planner.rs`); they differ only in how collision sets merge and what a group does:

- **Collision sets** (`mstar/collision.rs`) are disjoint groups of agents. A collision
  between `i` and `j` merges `{i, j}` with every group that shares an agent. Several
  pairs colliding in one step are merged pair by pair, so unrelated pairs stay apart.
  Basic M* merges everything into one group.
- **Successors.** Agents outside every group follow their individual policy. A group that
  holds every agent of the current search branches over all moves. In rM*, a smaller group
  moves one step along an optimal joint path of *that group alone*, from its current
  joint state. A separate planner over just the group computes that path, recursively by
  the same rules. If the group cannot reach its goals from there, the state has no
  successors.
- **Subgroup planners** live for the whole run, keyed by the sorted group. A query from a
  new start state bumps the planner's query counter; `g`, the parent and the expansion
  record of a node only count for the query that set them. Collision sets and predecessor
  sets are properties of the joint state, not of the query, so they carry over. Every
  query records the optimal path it found as forward pointers with the exact cost to go.
  A later query takes such a state's exact cost as its heuristic, which is admissible, and
  stops when it pops one: its `f` is exact and lowest on the open list, so no cheaper path
  exists. The rest of the path follows the pointers. A query that exhausts its space
  caches `None`.
- **Stale pops.** A node remembers the `g` and the *version* of its collision set when it
  was expanded; the version is bumped on every change. The set size is not enough: two
  groups merging keep the agent count.
- **Budget.** All planners share one `Stats` and one budget, asked before every
  expansion in any planner. A query stopped by the budget reports `LimitReached` all the
  way up; only an exhausted query prunes a state.
Optimality is Wagner and Choset's result, stated for a joint cost that sums the robots'
costs, which is the only cost the search uses (see makespan below). The oracle properties
check every form for both objectives; cross-checks compare the forms with CBS on 3-4 and
5-6 agents, and a hand-made case with two separate pairs checks that rM* expands fewer
states than basic M*.

### Operator decomposition (OD-M*, ODrM*)

`MStarConfig::operator_decomposition` (Ferner, Wagner and Choset 2013; `od_recursive`
with rM*). A state is a `Key`: the time, every agent's position (`base`) and the new
positions of the first `k` agents of the planner's order (`moved`; empty for a standard
state). Where a standard state would branch (basic M*: its collision set is not empty;
rM*: a group of all the planner's agents), it is expanded one agent at a time instead:

- the next agent `k` branches if the state's *own* collision set holds it, else takes its
  individual policy move; the last agent's move yields the standard successor;
- a move that collides with an agent moved earlier in the step (same vertex, or a swap)
  passes the pair back from this intermediate state, through the same back sets as
  standard states, so the agent that moved earlier branches at its state too;
- intermediate states start with empty collision sets, so an agent only branches once a
  collision downstream needs it to;
- the heuristic is the sum of the distances at the current positions (moved agents at
  their new ones), consistent like the standard one; only standard states are goals and
  enter the forward cache of exact costs.

Measurements (release build):

| | basic M* | OD-M* | rM* | ODrM* |
|---|---|---|---|---|
| benchmark, 8 agents | 350 ms | 25 ms | 30 ms | 12 s (5.4 M expansions) |
| 300 random 5x5-6x6 grids, 5-7 agents, total | | | 36 s | 8 s |

OD wins where groups are large and tightly coupled: on the random instances the slowest
rM* cases drop from 8 s to 1 s and from 17 s to 3 s. But ODrM* is not uniformly better:
on easy instances it is two to three times slower (every agent of a branching step costs
an expansion), on one of the random instances it needs over 2 million expansions where
rM* needs 112 000, and on the benchmark its subgroup searches form different, larger
groups (five agents queried 24 times) than rM*'s. Applying OD only in the top-level
planner removes the slowdown and the gain alike. So `mstar` keeps rM*, and ODrM* is
opt-in (see the open questions).

### Makespan

Pricing a joint step at 1 whenever any agent pays (as the oracle does) is exact but slow:
an agent with slack can wander without raising `f`, collisions found on those plateaus
merge groups, and rM* took 83 s on the benchmark instance (over 3.6 million top-level
states). Tie-breaking by a sum-of-costs estimate, or comparing paths by `(makespan, sum
of costs)`, did not change that.

Instead, `mstar_with` solves makespan through sum of costs (`Run::makespan`):

1. Solve for sum of costs. Unsolvable or out of budget ends the run. Otherwise its
   makespan `U` is an upper bound on the optimum.
2. The largest individual distance `L` is a lower bound. For each deadline `M` from `L`
   to `U - 1`, solve for sum of costs under the deadline: states carry their time
   (without a deadline the time is always `0`), and a state from which an unfinished
   agent cannot reach its goal by `M` is pruned, in the choices and in the heuristic.
   Policy moves and subgroup paths keep a feasible state feasible. The first `M` with a
   plan is the optimal makespan; a search that exhausts its space proves `M` too small.
3. If no deadline below `U` works, the first plan is optimal.

Each deadline search is a fresh run of the planners (their caches depend on the
deadline); the counters and the budget carry over. The plan has the optimal makespan and,
among such plans, the least sum of costs. On the benchmark `L = U = 14`, so makespan costs
one sum-of-costs search: rM* 30 ms instead of 83 s, basic M* 330 ms, OD-M* 25 ms. A
deadline search that fails must exhaust its time-expanded space, so instances whose
optimal makespan is far above `L` cost more.

## Later algorithms

- **CBS-TA** (Hönig et al. 2018): task assignment plus pathfinding. The root enumerates
  assignments in order of cost (Murty's k-best assignments); each CT node also carries
  an assignment. Agents get a set of candidate goals.
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
  `Solver` enum in the tests: plain and improved CBS (also with WDG), each CBS
  improvement alone (plus bypass and WDG each with disjoint splitting), ECBS at `w = 1`
  and at `w` in {1.1, 1.5, 2, 3}, ICTS, basic and recursive M* and PBS, each for both
  objectives. When
  the oracle finds an optimum, a solved plan must pass the plan checker and be within the
  solver's guarantee: equal for the optimal solvers, at most `w` times for ECBS, at least
  the optimum for PBS. When the oracle proves the instance unsolvable, no solver returns
  `Solved`. A complete solver must not return `Unsolvable` on a solvable instance. About
  1 in 600 random solvable instances needs more than the 20 000 expansions the property
  test allows (plain CBS blows up on tightly coupled agents); those cases are discarded, and
  one of them (63 000 expansions for plain CBS, about 740 improved) is kept as a
  regression test. Each property passed 1 500 cases once in release mode (at `8290dab`,
  and the WDG ones and the each-improvement one again at `08a75a1`), as did the
  crate-internal heuristic admissibility property.
- **Cross-checks** on random 4x4 to 5x5 grids with 3-4 agents, too large for the oracle:
  improved CBS, ICTS and every form of M* agree on the optimum (either objective)
  whenever they solve, ECBS and PBS respect their guarantees relative to it, no solver
  claims `Unsolvable` while another solves, and every plan passes the plan checker. Makespan
  ICTS gets 100 cost vectors instead of 5 000 expansions: on an unsolvable instance its
  run grows with the cube of the budget (1 000 vectors take 4 s), which, not M*, is what
  made 1 500 cases take over half an hour before; now they take 30 s in release. A second
  cross-check runs improved CBS against rM* and ODrM* with 5-6 agents on 5x5 to 6x6 grids,
  for both objectives. In a sample of 200 cases both CBS and rM* solved, and so were
  compared, in 59 of 100 sum-of-costs cases and 65 of 100 makespan cases. Both
  cross-checks and the M* oracle property (every form, both objectives) passed 1 500
  cases once in release mode at the commit that added operator decomposition.
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
  costs 92. Typical times on an idle machine, before DG, disjoint splitting and the
  conflict avoidance table: CBS 0.3 ms plain and 0.4 ms improved, ECBS 0.4 ms, PBS
  0.3 ms, ICTS 1.3 ms, basic M* 320 ms; recursive M* is about ten times faster than
  basic M*. Later wall times were too noisy (the machine was shared), so the newer
  changes were measured in instructions (cachegrind): improved CBS now needs 6 instead
  of 8 expansions on this instance but about a third more instructions per run (the
  conflict table and the DG checks cost more than the two expansions save), while on the
  hard 3x3 instance it needs 590 instead of 7 000 expansions.
- **Later:** real Moving AI instances with known optimal SoC from published results, as
  slow ignored tests (the benchmark files are not vendored).

## Milestones

1. **Done in this draft.** Instance types (graph trait, adjacency list, grid), conflict
   detection, constraints, space-time A* via heuristic-search, optimal CBS for SoC, oracle
   and property tests.
2. **Done in this draft.** Makespan-optimal CBS; Moving AI `.map`/`.scen` parser.
3. **Done in this draft:** MDDs, prioritized conflicts, bypass, CG, DG and WDG
   heuristics, conflict avoidance table, disjoint splitting.
4. **Done in this draft:** ECBS (local focal list), PBS (with `block_from`).
5. **Done in this draft:** ICTS (shares MDDs), basic and recursive M*, every solver for
   both objectives, operator decomposition (OD-M*, ODrM*).
6. SIPP and MAPF-LNS2; CBS-TA; EPEA*.

## Cross-crate needs

None blocks the milestones done so far. The two API additions below belong in
heuristic-search. The high level's focal list is local (see ECBS for why the upstream one
does not fit it). The lexicographic cost is implemented locally too
(`low_level/lexicographic.rs`, a `(u32, u32)` pair implementing `heuristic_search::Cost`);
a generic `Lexicographic<A, B>` in search-core or heuristic-search would replace it.

Milestones 3 to 5 were developed against an export of heuristic-search's `b5cc485`,
because its working tree was mid-change and did not compile at the time. The final checks
ran against the path dependency at `58ba6ec`. Disjoint splitting and DG/WDG were developed
against an export of its `2d1f1ba` for the same reason; everything since was checked
against the path dependency at `dbc85f8`.

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
    budget: impl Budget<Stats>,
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
is local to this crate. The low level now uses heuristic-search's `FocalList`; the high
level cannot yet (see ECBS).

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
