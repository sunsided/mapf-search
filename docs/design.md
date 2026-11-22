# mapf-search design

Multi-agent pathfinding (MAPF): find one path per agent so that no two agents collide,
minimizing a joint objective. This crate hosts the conflict-based family (CBS, CBS-TA,
FOCBS, ECBS, MO-CBS), ICTS, PBS, PIBT and the turn-aware PIBT*_T, M*, EPEA* and MAPF-LNS2
(over SIPPS).

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
  roadmap nodes) can map onto dense ids; `Conflict` has no generic `V`, because distance tables and constraint sets want dense indices.
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

Every solver but LNS2, PIBT and PIBT*_T, which only seek a feasible plan, and FOCBS, which
minimizes the sum of costs, takes an `Objective` (`SumOfCosts` or `Makespan`). For CBS, both objectives
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
- **M\*** and **EPEA\*** solve makespan through sum of costs under deadlines (see M*).
  Their plans also have the least sum of costs among the makespan-optimal plans.

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

**Improvements.** `cbs` uses all of them; `cbs_with` takes a `CbsConfig`
that switches each one (`CbsConfig::plain` is the original CBS). None changes the
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
descendants of each add the other). Such duplicates cost work, not optimality. CBS does
no CT duplicate detection; each node stores its full constraint list (a CT node is
small next to its paths). A later version can hash the sorted constraint set and skip
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
the low level runs A* over search-core's `Lexicographic<u32, u32>` cost
`(time steps, conflicts)` with edge cost `(1, conflicts of the move)` and heuristic
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
horizon is a proof.

The CBS high level reuses heuristic-search's `Budget`, `Limits` and `Stats` for its own
CT expansion counts, so callers limit both layers with the same types, and ECBS takes its
`Weight`. These types are re-exported, which makes heuristic-search a **public
dependency**: its breaking releases are breaking releases of this crate. `Budget` itself is
search-core's `Budget<T>`, generic over the statistics type, so every solver here takes
`impl Budget<Stats>`; `Limits` and closures over `&Stats` implement it, and a caller's own
budget implements `Budget<Stats>`.

## Shared result types

Every solver but FOCBS returns a `MapfResult`: an `Outcome` and high-level `Stats`. FOCBS
returns a `UaResult`, whose `UaOutcome::Solved` carries the assigned agents' `Solution`
and the unassigned agents' paths separately, so the sum of costs stays that of the agents
with goals. `Outcome` is
`Solved(Solution)`, `Unsolvable` (a proof), `LimitReached` (the budget stopped the run) or
`Exhausted`. Only PBS returns `Exhausted`: it ran out of options, which proves nothing.

## CBS-TA (Hönig et al. 2018)

`cbs_ta(instance, objective, budget)` and `cbs_ta_with(instance, config, budget)` solve a
`TaskInstance`: each agent has a start and a list of candidate goals, and a solution ends
every agent at a distinct one of its candidates. Agents may share candidates. The result is
optimal over all assignments and plans; agent `i`'s goal is the last vertex of its path.

- **Forest.** One constraint tree per assignment, each searched exactly as by `cbs_with`
  (all of `CbsConfig` applies). The CBS `Search` holds the starts, the candidate goals with
  one distance table each, and the assignment of every opened tree; a node carries its tree
  index, and the low level, the MDDs and WDG's pair weights look up the agent's goal
  through it. WDG's cache key includes the pair's goals, so trees that assign a pair the
  same goals share weights. Plain CBS is the forest with one tree, agent `i` at goal `i`.
- **Assignments in order of cost.** `assignment.rs` ranks the assignments of the cost
  matrix (agent to candidate: the shortest distance, or none if the goal is not a
  candidate or unreachable) with Murty's partitioning over the Hungarian method. For sum
  of costs the key is the sum. For makespan it is the bottleneck, then the sum: the
  smallest threshold that still admits an assignment, found by a binary search over the
  distinct entries, each step a Hungarian solve with the larger entries removed.
- **Opening trees.** Before each pop, the next ranked assignment becomes a tree while its
  cost is below the best open key, or while nothing is open. An assignment's cost is the
  objective of its root without heuristic, which bounds every node of its tree from below,
  so no closed tree can beat the node popped next.

### Paper comparison

| Paper (Hönig et al. 2018) | Here | Status |
| --- | --- | --- |
| Binary `N x M` assignment matrix `A`; goals may outnumber agents | candidate goal list per agent; goals are their union | same |
| An agent without an assigned goal only has to stop somewhere | every agent needs a goal; fewer goals than agents is `Unsolvable` | narrower |
| Sum of individual costs | `Objective::SumOfCosts`, plus makespan via bottleneck assignments | generalized |
| Forest of constraint trees, one per assignment | the same, in one open list | same |
| Next root created when a root node is expanded | next root opened when its cost is below the best open key | lazier, same optimality; also correct with CBS heuristics and bypass |
| `constrainedAssignment(I, O, C)`: forced entries cost 0, forbidden ones infinity, then the Hungarian method | forced pairs clear the rest of their row and column, forbidden pairs are removed | same partition |
| Plain CBS high level | every `CbsConfig` improvement | beyond the paper |
| ECBS-TA (bounded suboptimal) | not implemented | missing |

### Tests

A property test compares `cbs_ta_with` on random 3x3 instances (up to three agents, one to
three candidates each, both objectives, plain and improved) with CBS run on every
assignment. Another checks that the ranked assignments match brute-force enumeration in
order and count. With one candidate per agent, CBS-TA returns what CBS returns, with the
same statistics. The fuzz target runs CBS-TA with every agent allowed at every goal and
checks the plan and that it costs no more than CBS's. The benchmark runs it on the
eight-agent instance with all goals shared: about 53 µs (bench profile, shared machine),
against about 1.1 ms for `cbs` on the fixed assignment.

## FOCBS (Jin et al. 2026)

`focbs(instance, budget)` and `focbs_with(instance, config, budget)` solve a `UaInstance`
(MAPF with unassigned agents, MAPF-UA): assigned agents have a start and a goal,
unassigned ones only a start; they may end anywhere but must not collide with anyone.
Starts are distinct over all agents. The objective is the assigned agents' sum of costs
(the paper's sum of service time, last arrivals with goal-stay); the unassigned agents'
moves (SMU) are minimized second, for the chosen assigned paths, and counted only up to the
latest assigned arrival `T`, after which nobody moves. Only sum of costs: `focbs_with`
panics on `Objective::Makespan`.

- **High level.** The CBS `Search` over the assigned agents, every `CbsConfig` improvement
  included. A third hook next to CBS-TA's forest: at a conflict-free node the budget is
  asked, then the flow oracle routes the unassigned agents. If it succeeds, the node is the
  solution; if not, the node is expanded by *flow-conflict branching*: one child per
  blocker, adding a negative vertex or edge constraint on its assigned agent, through the
  same `child` as a CBS split. Blockers at step 0 (fixed by the starts) and constraints the
  node already has are skipped; without any other blocker the node is dropped. The paper's
  Lemma 2: every feasible plan below the node releases one of the blockers, so the split
  loses nothing; children may overlap.
- **Flow oracle** (`flow.rs`). A time-expanded network over steps `0..=T`: each `(v, t)` is
  a unit arc `v_t^in -> v_t^out`, waits are free arcs to `v_{t+1}^in`, and each vertex pair
  joined by an edge has a gadget per step whose central arc has capacity 1 and cost 1 (a
  move counts once, and two agents cannot swap along the edge). The source feeds the
  unassigned starts at step 0, and every vertex at step `T` drains into the sink. The
  assigned agents' reservations (their vertex at every step up to `T`, goal waits included,
  and the central arc of each of their moves) have capacity 0 and carry a label naming the
  agent. Successive shortest paths with Dijkstra on reduced costs sends one unit per
  unassigned agent at the least total cost; the units are followed back into paths.
- **One network for both jobs.** The paper routes on a network without the reservations
  and computes the cut on a second, diagnostic one where they are labelled zero-capacity
  arcs. Here the routing network already is the diagnostic one: zero-capacity arcs carry no
  flow, and successive shortest paths that stops short of `m` units has reached a maximum
  flow, so the nodes the source reaches in the residual network give a minimum cut. Its
  labelled arcs are the blockers.
- **Directed graphs.** The paper assumes an undirected graph. Here a gadget's entry arcs
  exist only from an end with an edge to the other end, and its exit arcs only into an end
  the other has an edge to.

### Paper comparison

| Paper (Jin et al. 2026, arXiv:2611.24108v1) | Here | Status |
| --- | --- | --- |
| MAPF-UA: assigned and unassigned agents, SST first, SMU second, sequential optimality (Def. 2) | the same; SST is the goal-stay sum of costs | same |
| CBS over the assigned agents, ordered by SST | the CBS `Search`, with every `CbsConfig` improvement | beyond the paper |
| Flow oracle: time-expanded network to `T_N`, vertex split, 5-arc edge gadget, min-cost flow | the same network | same |
| BZ-SSP: zero-cost wait paths first, then SSP | plain successive shortest paths | simpler, slower |
| Ordinary network for routing, diagnostic network for the cut | one network serving both | same result |
| Flow-conflict branching on min-cut blockers, skipping `t = 0` and already forbidden ones | the same | same |
| Start-avoiding tie-breaks (CT key and low-level key) | not implemented | missing; affects SMU only |
| Undirected graphs | directed graphs too | generalized |

### Tests and measurements

A property test compares FOCBS (plain and improved) on random grids up to 3x3 with up to
three agents, one or two of them assigned, against a joint Dijkstra over all agents with
goal-stay pricing: the same assigned sum of costs, a conflict-free plan, and an unassigned
routing whose number of moves equals a separate brute force over the returned assigned
paths. Branching is rare in random cases (a few in 256), so a fixed case requires it: an
idle robot on the goal at the end of a corridor with a pocket, where the conflict-free
path alone costs 2 and the answer is 3. Further tests: no unassigned agents (the same
outcome and statistics as `cbs_with`), no agents at all, a one-way ring, and makespan
rejected. The fuzz target runs FOCBS with the last agent left without its goal. On the
benchmark's eight-agent instance with the last two agents idle it takes about 0.8 ms
(bench profile, noisy machine).

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

## PIBT (Okumura et al. 2019, 2022)

`pibt(instance, budget)`: priority inheritance with backtracking, Algorithm 1 of the
paper. It plans one time step for all agents at a time and stops at the first step at
which every agent stands at its goal. No objective, no quality guarantee, not complete.

- **Priorities.** Each step, an agent away from its goal gains 1; one at its goal drops to
  its tie-breaker. Here the priority is `(steps away from the goal, distance from the
  start to the goal, agent id)` lexicographically: the integer part, then a distinct
  tie-breaker in `[0, 1)` of the paper, set by distance as in LaCAM, so that at the first
  step the agents with the longest way go first. PIBT*_T shares these priorities, with its
  turn-aware distance.
- **One step.** In decreasing priority, each agent not yet planned runs the recursive
  procedure: candidates are its vertex and its neighbours (those that reach the goal), by
  distance to the goal, unoccupied ones first, then by id. It skips vertices claimed for
  the next step and the current vertex of the agent it inherited from (no swap), claims
  the next one, and, if an unplanned agent stands there, lets it inherit and plan first.
  If that agent is stuck, it tries its next candidate; with none left it stays and reports
  stuck to its own pusher.
- **Completeness.** The paper proves that on graphs where every pair of adjacent vertices
  lies on a simple cycle (biconnected ones, for example) every agent reaches its goal at
  some step; it does not prove that all stand at their goals at once. Off such graphs PIBT
  can push in circles: in a corridor with a side pocket an agent at its goal is pushed to
  the dead end and back forever. That case is a unit test, and the hand-made
  `every_solver_steps_off_a_goal` case skips PIBT for it. `Unsolvable` only means an agent
  cannot reach its goal even alone.
- **Budget.** Asked before every time step; `Stats::expanded` counts steps and
  `Stats::generated` per-agent decisions.

### Paper comparison

| Paper (Okumura et al.) | Here | Status |
| --- | --- | --- |
| Priority `p_i`: `+1` away from the goal, reset to `ε_i` at it | `(steps away, agent id)` | same order |
| Candidates by distance to the goal, presence of an agent as tie-breaker | the same, then by vertex id | same |
| Skip claimed vertices and the inheritor's current vertex | the same | same |
| Priority inheritance to an unplanned occupant, backtracking on failure | the same | same |
| Iterative (lifelong) MAPF and MAPD | one-shot only: stop when all agents stand at their goals | narrower |
| Random tie-breaks in later versions (as in LaCAM) | deterministic | our choice |

### Tests and measurements

Unit tests: a crowded 5x5 grid with twelve agents, an agent pushed off its goal on a
cycle, the livelock in a corridor with a pocket, the corridor swap stopping at the budget,
and trivial instances. PIBT is one of the shared test solvers (oracle property test,
cross-check, the other hand-made cases) with no guarantee and at most 500 steps in the
tests, and it runs in the solver fuzz target. On the benchmark's eight-agent instance it
takes about 58 µs (bench profile, shared machine) for 26 steps, with sum of costs 123
against the optimum 92.

## PIBT*_T (Tao, Yu and Hu 2026)

`pibt_turns(instance, headings, seed, budget)` plans agents that face a heading, move only
forward and turn in place, on any `OrientedGraph`: a `Graph` with a fixed number of
headings, `forward(vertex, heading)` and `turns(heading)`, whose neighbours are exactly
the forward vertices. `Grid` is one, with four headings and quarter turns. The result is
a `TurnResult`: a `MapfResult` whose paths list every step of the run (turns are waits, so
conflicts and costs are checked as for any plan) and each agent's heading per step.

- **Model.** Each step an agent moves forward keeping its heading, turns once, or waits; a
  turning agent keeps its vertex. Conflicts are the usual vertex and swap conflicts; the
  goal needs no heading. A half turn on a grid takes two steps.
- **Costs** (`pibt_turns/costs.rs`). The fewest turns between headings (breadth-first over
  `turns`), and per agent an exact backward breadth-first search over `(vertex, heading)`
  states to its goal. `TC` for a request is the turns and the move to get there (or one
  step for staying) plus the remaining steps from the state it arrives in.
  `Compute_action` moves if the agent faces the request, else turns once along a shortest
  turn sequence (on a grid: counter-clockwise first for a half turn, as in the paper).
- **Step.** PIBT's procedure with the paper's additions: candidates by `TC` (unoccupied
  first, random among equals after a seeded shuffle), the k-push escape trigger (`k = 2`,
  shuffle and reset), the swap detector, and the remembered request of an agent that turned
  or waited towards a vertex (first again at the next step). An agent claims a vertex,
  lets an undecided occupant decide first, then decides; if the occupant stays (it only
  turns), the agent waits instead of moving in. A request that reaches the vertex of the
  agent that started the chain closes a cycle: all its agents move at once if each faces
  its request, else the misaligned turn and the others wait. A cycle cannot close on a
  later agent of the chain, since each stands on a vertex its pusher has claimed.
- **Swaps** (`pibt_turns/swap.rs`): the PIBT+/LaCAM rule on heading-free distances (the
  least over headings), as in the reference code: required if pushing only drives the
  other agent down a corridor the pusher is heading along and the two want to cross,
  possible if, with the roles reversed, the pusher can back off down its corridor to room
  to step aside before a dead end. A neighbour behind the agent that wants through (the
  clear operation of push and swap) is judged as if it stood at the agent's vertex. The pusher then
  reverses its preferences and pulls the other agent into its vertex (turning it first if
  needed), unless the pusher stays. Corridor walks are capped at the vertex count, since
  on directed graphs they need not return.

### Paper comparison

| Paper (Tao, Yu and Hu 2026, arXiv:2611.23911v1) / reference code (MIT) | Here | Status |
| --- | --- | --- |
| 4-connected grids, 4 orientations, 90° turns, forward moves | any `OrientedGraph`; `Grid` is the paper's model | generalized |
| Algorithm 1, `Compute_action`, LGS node selection (`TC`) | the same | same |
| Cost tables by BFS over the state graph | the same, one table per agent to its goal facing any way | same |
| Cycle handling (Definition 1, Propositions 1-2), detected at the chain's initiator | the same | same |
| Swap pattern detector (prose only; reference code) | ported from the reference, with capped corridor walks | same rule |
| k-push escape trigger, `k = 2`; reference resets the count after a shuffle | the same | same |
| Remembered target of a turning agent (reference `reserved_nodes`) | `intent`, requested first next step | same idea, simpler state |
| Priorities: elapsed, then initial distance, then a random tie-breaker (reference) | elapsed, then agent id | simpler |
| ADG execution model | not part of the planner | not applicable |

### Tests and measurements

An independent validator (only `forward` and `turns`, not `Compute_action`) checks every
plan: each step waits, turns once, or moves forward without turning, and no two agents
conflict. A property test on random grids (3x3 to 4x4, walls, 2-5 agents, random headings
and seeds) checks every step of a 100-step run whether or not it finishes, the returned
plans, and that the same seed gives the same run. In a 20 000-case run it took 30 000
cycle resolutions, 7 600 pulls and 118 000 escape shuffles, and solved 85% of the cases
within 300 steps. Unit tests cover the cost tables and actions, an aligned and a
misaligned 2x2 carousel (a four-agent cycle), turning before moving, a one-way ring with a
single heading (a non-grid `OrientedGraph`), an unreachable goal and a missing heading.
The fuzz target runs it with headings from the input. On the benchmark's eight-agent
instance, all facing east, it takes about 0.2 ms (bench profile, noisy machine), against
about 56 µs for PIBT without turns.

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

`mstar` runs ODrM* (recursive M* with operator decomposition, below; also
`MStarConfig::default()`); `mstar_with(instance, MStarConfig, budget)` selects basic M*
(`MStarConfig::basic`), rM* (`MStarConfig::recursive`) or either with operator
decomposition. All share one planner
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
planner removes the slowdown and the gain alike. `mstar` and `MStarConfig::default()`
use ODrM* anyway (the user's decision, for the large wins on coupled groups); rM* stays
one `MStarConfig::recursive` away, and is the better choice on the benchmark.

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

## EPEA* (Goldenberg et al. 2014)

`epea(instance, objective, budget)`: A* in the joint space of all agents, with enhanced
partial expansion. States and pricing are those of M*: every agent's vertex and finished
flag, finishing at the goal is free, every other unfinished agent pays 1 per step, and the
heuristic is the sum of the unfinished agents' distances. Makespan uses M*'s deadline
searches. The joint space is finite, so EPEA* proves unsolvability, like M*.

- **Partial expansion.** A node stores `F`, initially its `f`. Expanding it generates only
  the children with `f = F`; then `F` becomes the next larger `f` among its children and
  the node goes back into the open list, or is closed if there is none. Each partial
  expansion counts once in `Stats::expanded`. Ties in `F` go to the smaller `h`.
- **Operator selection function** (`epea/osf.rs`). With the sum of distances as
  heuristic, a joint step raises `f` by the sum of the agents' own rises: `1 + d(next) -
  d(here)` for a move or wait, never negative and at most `2` on an undirected graph (a
  directed edge into a one-way detour can raise it more), and `0` for finishing or staying
  finished. Each agent's moves are sorted by rise; a depth-first search over the agents
  picks one move each, keeps the remaining target within the remaining agents' least and
  most total rise, and drops colliding combinations (same vertex, or a swap) as it goes.
  The next `F` is the smallest achievable total rise above the current one, from a
  subset-sum over the agents' rise sets, collisions ignored: a node whose children at
  some `F` all collide is just expanded again at the next one.
- **Duplicates.** States are kept in a `BTreeMap` with their best `g`. With a consistent
  heuristic and pops in order of `F`, a state is first generated with its optimal `g`; a
  cheaper path would still reopen it (`Stats::reopened`), but none occurs in the tests.

### Paper comparison

| Paper (Goldenberg et al. 2014) | Here | Status |
| --- | --- | --- |
| Stored value `F(n)`, initially `f(n)`; expand generates children with `f = F(n)`, then `F(n)` becomes the next value or `n` is closed | the same | same |
| MAPF operator selection: per-agent rises of `f` under the sum of individual costs, combined to the target | the same, with goal-stay finishing at rise 0 | same |
| Collisions checked on generated children | checked while combining agent moves | same result, earlier pruning |
| Next `F` from the OSF, regardless of collisions | subset-sum over the agents' rise sets | same |
| Sum of individual costs | sum of costs, plus makespan via M*'s deadline searches | generalized |
| Plain joint state per time step | M*'s states with finished flags (goal-stay); time only under a deadline | same model as the oracle |

### Tests and measurements

EPEA* is one of the solvers of the shared tests: the hand-made cases (stepping off a goal,
passing a finished goal, an unreachable goal, the corridor swap, which it proves
unsolvable), the property test against the joint-space oracle for both objectives, the
cross-check against CBS, ICTS and M* on larger grids, and the solver fuzz target. On the
benchmark's eight-agent instance it takes about 6 ms (bench profile, shared machine):
1 592 expansions and 11 686 generated children for the optimal sum of costs 92, against
about 12 s and 5.4 million expansions for ODrM*. That makespan is also the lower bound, so
no deadline search runs.

## MO-CBS (Ren, Rathinam and Choset 2021)

Multi-objective MAPF: every move and wait costs a vector of `N` objectives, and the
answer is the Pareto-optimal front, one plan per Pareto-optimal cost vector.
`mo_cbs(instance, costs, budget)` and `mo_cbs_with(.., MoCbsConfig, ..)` implement
"A Conflict-Based Search Framework for Multi-Objective Multi-Agent Path Finding"
(arXiv:2101.03805v5), behind the optional `multi-objective` feature (`src/mo_cbs.rs`,
`src/mo_cbs/low_level.rs`).

- **Costs.** A closure `costs(from, to) -> [u32; N]`; `costs(v, v)` is the wait at `v`
  (the paper's self-loops). `N` is a const generic, so cost vectors are plain arrays.
  `mo_cbs_per_agent` takes `costs(agent, from, to)` instead, one cost model per agent;
  the shared form is a wrapper around it.
- **Configurations.** `MoCbsConfig::plain()` is the paper's Alg. 1 and
  `MoCbsConfig::tree_by_tree()` its MO-CBS-t, both with no improvements.
  `MoCbsConfig::improved()` (what `mo_cbs` and `MoCbsConfig::default()` use) is MO-CBS-t
  with the feasibility check, conflict prioritization, bypass and disjoint splitting; each
  is a separate switch.
- **High level.** Alg. 1 as written: the roots are the product of the agents' Pareto
  sets; nodes are popped in lexicographic cost order (a `BTreeMap` keyed by cost and
  creation order); Filter drops a node that a found solution weakly dominates, at pop and
  at insert; a conflict-free node runs Update (drop the solutions it dominates) and joins
  `C`; the search ends when the open list is empty. Conflicts, constraints
  (`Constraint::resolving`, standard splitting) and conflict detection
  (`first_conflict`) are the crate's.
- **Result.** `ParetoResult { outcome, front, stats }`: `Complete` (the front is exact),
  `Unsolvable` (an agent has no path alone, or the search ended without a solution) or
  `LimitReached` (the plans found so far: conflict-free and mutually non-dominated, but
  not proven Pareto-optimal, since a later tree may dominate them).
- **Goal-stay.** An agent pays until its final arrival and nothing afterwards, as
  everywhere in the crate. With one objective and unit costs the front is CBS's optimal
  sum of costs.

### Decision: the low level from multiobjective-search

The low level must return all cost-unique Pareto-optimal paths of one agent under its
constraints. multiobjective-search
already has NAMOA*dr and BOA*, and its interfaces fit a space-time graph without
changes: `Graph` with `State = (Vertex, Time)` and `Weight = [u32; N]` is the
time-expanded adapter, the heuristic is a `search_core::Heuristic` closure, and goal
labels are recorded, not expanded, which is what goal-stay needs. Writing a local
NAMOA*-st would have duplicated a tested frontier implementation. The cost: the
dependency is optional so the default build keeps one dependency, and the crate's MSRV
rose to multiobjective-search's 1.86.

- **Graph.** `SpaceTime` in `mo_cbs/low_level.rs`: successors are the wait and every
  edge, priced by `costs`, minus what `ConstraintSet::allows` forbids; the goal test is
  `vertex == goal && time >= blocked_until(goal) + 1`.
- **Horizon.** Successors are cut at `latest_step + vertex_count`, as in the
  single-objective low level. After the last constrained step the graph is fixed, and
  removing a cycle or a wait changes a path's cost by a non-negative vector, so every
  Pareto-optimal cost vector is reached by a path whose unconstrained tail is simple.
- **Heuristic.** Per objective, the exact cost to the goal on the static graph: one
  backward Dijkstra per objective and agent, computed once per run. It ignores waits and
  time, so it is consistent, as NAMOA*dr and BOA* require.
- **BOA\* adapter.** BOA* is typed for `[C; 2]`; for `N == 2` a small wrapper re-types
  the weights (`LowLevel::Auto`). `LowLevel::NamoaDr` forces NAMOA*dr for any `N`.
- **Budget.** Neither search takes a budget; the adapter asks it instead (see Budget
  inside the low level).

### Improvements

None of them changes the front; each is checked against the oracle in every combination.

- **Feasibility check** (`feasibility_check`, the paper's Sec. VII-B). Feasibility does
  not depend on the costs, so recursive M* (`MStarConfig::recursive`) with unit costs
  decides it; M* proves unsolvability because its joint space is finite. The paper cites a
  polynomial-time test (Kornhauser et al. 1984); M* is exponential in the worst case but
  already in the crate. M* gets the caller's budget with its own counters; if the budget
  stops it, the search goes on without the proof. rM*, not ODrM*, because ODrM* was much
  slower on the 8-agent benchmark (see M*).
- **Conflict prioritization** (`prioritize_conflicts`). A conflict is cardinal for an
  agent if every path of the agent's current Pareto set takes part in it (the
  multi-objective reading of "every cheapest path", from the Pareto sets the low level
  returns anyway, so no MDD is built); then whichever of its Pareto paths the agent keeps,
  the child that forbids the conflict must replace it. Cardinal for both, then for one,
  then the rest; the earliest within a class. The Pareto sets stored per node predate the
  negative constraints that other agents' later positive constraints imply, so this is a
  heuristic order, which is all it needs to be: Lemma 2 holds for any split conflict.
- **Bypass** (`bypass`). If every agent a child replans has a path of exactly its cost in
  the parent and those paths leave fewer conflicts, the parent takes them over and is
  reopened instead of being split. The node keeps its constraints and its per-agent cost
  vectors, which is all a P-node (Def. 3) depends on, so Lemmas 2 and 3 still hold.
- **Disjoint splitting** (`splitting: Splitting::Disjoint`, Li et al. 2019): one child
  forbids the conflict to one agent, the other requires it of that agent, which forbids it
  to the others; every other agent whose path breaks that is replanned, and the child
  becomes one node per combination of their Pareto paths. The two children partition the
  solutions, and the positive child keeps, for each replanned agent, a path weakly
  dominating its path in each permitted solution. The split agent is the one for which the
  conflict is cardinal if only one is, else the conflict's first agent.

### Budget inside the low level

NAMOA*dr and BOA* take no budget, and multiobjective-search is read-only here, so the hook
is in the space-time adapter: `SpaceTime` asks the caller's budget (with the high-level
counters) before its first expansion and every 256 expansions after. Once the budget says
stop it generates no more successors, the search drains its open list, and the run ends
with `LimitReached`. A `Limits` budget sees no change during one low-level call, so it
still bounds high-level expansions only; a closure over a deadline or a cancellation flag
stops a long low-level call. High-level expansions are counted once their children are
built, so the checks inside the low level see completed expansions only and
`Limits::expansions(n)` lets a run that needs exactly `n` expansions complete.

### Paper comparison

| Paper (arXiv:2101.03805v5) | Here | Status |
| --- | --- | --- |
| Problem: vector edge costs in `(0, inf)^M`, waits are self-loops with costs | `costs(from, to) -> [u32; N]`, `costs(v, v)` for waits; positive components documented as a requirement for termination, not checked | same |
| Goal behaviour: not specified | goal-stay: pay until final arrival, free afterwards; the goal state needs `t > blocked_until(goal)` | our choice |
| Vertex and edge (swap) conflicts, first conflict split into two constraints | `first_conflict`, `Constraint::resolving` | same |
| Initialization: product of all agents' Pareto sets as roots (MO-CBS) | `Roots`, generated up front for `MoCbsConfig::plain()`; budget asked before each root | same |
| Pop a non-dominated node, in practice lexicographic order | lexicographic, ties by creation order | same |
| Filter: drop a node weakly dominated by `C` (at pop and before insert) | `is_filtered` at pop and in `push` | same |
| Update: drop solutions dominated by a new one, then add it | `record` | same |
| One child per Pareto path of the constrained agent | `expand` | same |
| Terminate when OPEN is empty; `C` is the front | `ParetoOutcome::Complete` | same |
| MO-CBS-t: tree-by-tree, roots generated on demand | `MoCbsConfig::tree_by_tree()`; `improved()`, the default of `mo_cbs` and `MoCbsConfig`, is tree-by-tree too; `C` shared across trees | same |
| MO-CBS-n: NAMOA*dr-st low level | `LowLevel::NamoaDr` (and `Auto` for `N != 2`) | same, via multiobjective-search |
| MO-CBS-b: BOA*-st low level, two objectives | `LowLevel::Auto` with `N == 2` | same, via multiobjective-search |
| Time horizon `T`, "a large positive integer" | `latest_step + vertex_count`, which loses no Pareto-optimal cost vector | sharper |
| Heuristic: `M` backward Dijkstra searches per agent | the same | same |
| Feasibility pre-check for infeasible instances (unit-cost MAPF, polynomial test of Kornhauser et al.) | `feasibility_check`: recursive M* with unit costs proves `Unsolvable`; exponential in the worst case, under the budget | same purpose, different test |
| Expansion: any detected conflict, standard splitting | `MoCbsConfig::plain()` and `tree_by_tree()`; `improved()` adds conflict prioritization, bypass and disjoint splitting | beyond the paper, opt-in |
| Edge costs shared by all agents | shared (`mo_cbs_with`) or per agent (`mo_cbs_per_agent`) | generalized |
| Runtime limit | `impl Budget<Stats>`, asked before each high-level expansion and inside the low level | equivalent |
| A cost-unification variant ("MO-CBS-c") | not in 2101.03805v5, which defines only MO-CBS, -t, -n and -b | not applicable |

### Tests and measurements

- **Pareto oracle** (`tests/it/pareto_oracle.rs`): a label-setting search (Martins 1984)
  over the joint state of the testing oracle (`tests/it/oracle.rs`), with vector step costs, written
  without this crate's code and without multiobjective-search. It returns the exact
  front without a horizon: the joint space is finite and every antichain of integer
  vectors is finite. This replaces the brief's "enumerate joint plans up to a horizon",
  which would need a horizon argument of its own.
- **Properties** (`tests/it/mo_cbs.rs`, 48 cases each): random tiny grids with 2-3 agents
  and random cost tables (components 1 to 3, per directed move and per wait): two
  objectives with shared and with per-agent costs, each over random configurations (all
  six switches drawn independently), three objectives (plain and improved), one weighted
  objective, and unit costs against the joint oracle's and CBS's optimal sum of costs.
  On an unsolvable instance a run with the feasibility check must return `Unsolvable`.
  On a solvable instance the run must be `Complete`, its cost vectors equal to the
  oracle's, and every plan must pass the oracle's plan checker and `first_conflict`, with
  its cost recomputed from the paths. Without the improvements MO-CBS splits like plain
  CBS (standard splitting, no bypass, no heuristic), so it shares plain CBS's blow-ups on a few tightly coupled
  instances: those that hit the 20 000-expansion budget are discarded, at most 8 per
  property. One such instance (3x3, middle row walled except the centre, three agents)
  needs over 200 000 expansions with unit costs, as does plain CBS, while improved CBS
  needs 1 203. Keeping only one Pareto path per replanned agent makes the two- and
  three-objective properties fail with missing front points. Skipping Update does not:
  in lexicographic pop order a dominating solution is almost always found before the
  ones it dominates, so the properties never reached that case. A unit test of
  `record` covers Update and Filter directly and fails without Update. Two mutations of
  the improvements fail the properties as well: bypass adopting a path of a different
  cost, and a positive child that replans no other agent.
- **Hand cases:** a corridor swap with a risky pocket and a long safe loop, where every
  root conflicts and the front `[[8, 16], [9, 9]]` comes from conflict resolution (checked
  by hand), for plain, `-t` and improved; an unreachable goal (`Unsolvable`); a bare
  corridor swap (`LimitReached` without the feasibility check, `Unsolvable` with it); a
  single agent whose run a budget stops inside the low level. Unit tests cover the
  combinations, conflict choice, bypass and the low-level interrupt.
- **Benchmark** (`benches/mo_cbs.rs`, `--features multi-objective`): three agents on a
  6x6 grid with walls, two objectives with hashed edge costs in `1..=4` as in the paper's
  experiments. Front of 2, 2 110 expansions for each of the paper's variants; about
  175 ms for MO-CBS-b and -tb, 285 ms for -n and -tn at first (bench profile, shared
  machine), so BOA*-st is about 1.6 times faster than NAMOA*dr-st here, in line with the
  paper's Table I. In a later, quieter run: about 105 ms for -b and -tb, 130 ms for -n and
  -tn, and 6 ms and 126 expansions for the improved configuration, feasibility check
  included.

## MAPF-LNS2 (Li et al. 2022)

`lns2(instance, config, budget)` repairs a plan with collisions until it has none.
`Lns2Config` holds the neighborhood size `N` (default 8, the paper's) and a seed; every
random choice comes from a local SplitMix64 generator (`lns2/random.rs`), so a run is
reproducible and the crate stays `no_std` without a dependency. LNS2 has no quality
guarantee and is incomplete: on an unsolvable instance it runs until its budget, and some
solvable ones are local minima it never leaves (two agents swapping through a side pocket:
whichever agent is planned first takes the straight path, and neither can then avoid the
other alone). It returns `Unsolvable` only for a goal unreachable even alone.

- **Initial plan.** Prioritized planning in a random order, each agent with the fewest
  collisions with the agents before it. Where an agent can avoid them all, this is the
  shortest collision-free path that PP with SIPP would find; where it cannot, it is the
  paper's fallback of planning the rest with the fewest collisions.
- **Low level: SIPPS** (`lns2/sipps.rs`), the paper's Algorithms 1-3. The other paths
  become soft obstacles: steps at vertices, parked agents (target obstacles) and swaps
  (edge obstacles); there are no hard ones. Each vertex's timeline is cut into maximal
  safe intervals that are dirty (an obstacle at every step) or clean, built lazily and
  shared for vertices nobody visits. Nodes are `(vertex, interval, goal flag)` with the
  earliest arrival as `g` and a collision count `c` (one per dirty interval entered, one
  per swap); the open list orders by `c`, then `f`. A move whose early arrivals swap
  splits into two nodes, before and after the last swapping step. Twins (same vertex,
  interval and goal flag) prune each other by dominance and trim overlapping intervals.
  `h` is the distance to the goal, raised for `c = 0` to the time the goal is last
  visited. A node at the goal finishes at once if nobody comes later, else becomes a goal
  node charged with the later visits.
- **Exactness.** SIPPS guarantees a path when one exists and the shortest collision-free
  one when one exists; otherwise its count is an approximation (waiting through several
  dirty steps of one interval counts once), as the paper notes. The former space-time A*
  with a (collisions, time) `Lexicographic` cost, which counts exactly, stays as a test
  oracle (`lns2/low_level.rs`, test builds only).
- **Iteration.** A method is drawn by its adaptive weight, builds a neighborhood, and the
  neighborhood is replanned by prioritized planning in a random order against every other
  current path. The new plan is kept if its number of colliding pairs did not grow. The
  method's weight becomes `γ · max(0, c⁻ - c⁺) + (1 - γ) · w` with `γ = 0.1`.
- **Neighborhoods** (`lns2/neighborhood.rs`): collision-based, failure-based and random,
  as in the paper, each returning distinct agents, at most `N`. The searches for more
  agents are bounded and then return fewer: the fill-up loops after 64 failed draws in
  total, the walk on the collision graph after `64 N` revisits, and each random walk from
  a path after as many steps as the graph has vertices.
- **Counting.** The collision graph comes from `conflicts` on the whole plan after every
  iteration, `O(m^2 T)`; fine at the sizes here, to be made incremental for hundreds of
  agents.

### Paper comparison

| Paper (Li et al. 2022) | Here | Status |
| --- | --- | --- |
| Initial plan from a MAPF algorithm (PP), the rest planned with fewest collisions | PP where every agent takes its fewest-collision path | same result |
| SIPPS low level (soft obstacles, safe intervals), Algorithms 1-3 | the same, soft obstacles only | same |
| Space-time A* as the baseline of Experiment 1 | kept as the test oracle of SIPPS | same role |
| Replan the neighborhood with PP in a random order | the same | same |
| Accept if the colliding pairs do not grow | the same | same |
| Collision-based, failure-based and random neighborhoods | the same | same |
| ALNS with reaction factor `γ = 0.1`, weights start at 1 | the same; all-zero weights fall back to a uniform pick | same |
| `N = 8` | `Lns2Config::default()` | same |
| Stop when feasible, or at a time limit | stop when feasible, or at the `Budget` (iterations) | same |

### Tests and measurements

Unit tests cover the low level (detours, waiting out an agent, passing a parked agent when
it must, finishing collisions), the generator, every neighborhood method (distinct, at
most `N`, a colliding agent included where the method starts from one) and full runs on a
crowded 5x5 grid with twelve agents, for several seeds and sizes: valid, and the same
result for the same seed. LNS2 is one of the shared test solvers (hand-made cases, oracle
property test, cross-check) with no guarantee and, in the tests, at most 500 iterations,
and it runs in the solver fuzz target. A property test checks SIPPS against the exact
space-time A* on 4x4 grids with random walks: a path exactly when A* finds one, valid,
never fewer collisions than the minimum, and the same zero-collision length where A*
finds a path without collisions (the paper's Theorem 2).

The benchmark has two LNS2 cases: the eight-agent instance, whose initial plan is already
collision-free (prioritized planning only), and `lns2 crowded`, 24 agents across a free
8x8 grid, whose plan needs repairs. Crowded grids are where SIPPS pays off: with 64
agents on a free 16x16 grid (from the first rows to the mirrored cells), one run took
about 2-4 s with SIPPS and 30 s with the space-time A* (shared machine, noisy), roughly
3 ms against 37 ms per single-agent search; with 96 agents, SIPPS took about 15 s and the
A* run was stopped after 9 minutes. On sparse instances SIPPS's interval bookkeeping
makes it up to about twice as slow per call as A*, at a few microseconds.

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
  and at `w` in {1.1, 1.5, 2, 3}, ICTS, basic and recursive M*, EPEA* and PBS, each for both
  objectives. When
  the oracle finds an optimum, a solved plan must pass the plan checker and be within the
  solver's guarantee: equal for the optimal solvers, at most `w` times for ECBS, at least
  the optimum for PBS. When the oracle proves the instance unsolvable, no solver returns
  `Solved`. A complete solver must not return `Unsolvable` on a solvable instance. About
  1 in 600 random solvable instances needs more than the 20 000 expansions the property
  test allows (plain CBS blows up on tightly coupled agents); those cases are discarded, and
  one of them (63 000 expansions for plain CBS, about 740 improved) is kept as a
  regression test. Each property passed 1 500 cases once in release mode, as did the
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

## Roadmap

Nothing is planned at the moment.

## Cross-crate needs

The high level's focal list is local (see ECBS
for why the upstream one does not fit it). The lexicographic cost below is now
search-core's `Lexicographic<P, S>`; this crate depends on
search-core directly for it and dropped its local `low_level/lexicographic.rs`.

### Secondary tie-break: a lexicographic cost (done in search-core)

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
- W. Hönig, S. Kiesel, A. Tinka, J. W. Durham, N. Ayanian. Conflict-based search with
  optimal task assignment. AAMAS 2018. (CBS-TA)
- K. G. Murty. An algorithm for ranking all the assignments in order of increasing cost.
  Operations Research 16, 1968.
- G. Sharon, R. Stern, M. Goldenberg, A. Felner. The increasing cost tree search for
  optimal multi-agent pathfinding. Artificial Intelligence 195, 2013. (ICTS)
- H. Ma, D. Harabor, P. Stuckey, J. Li, S. Koenig. Searching with consistent
  prioritization for multi-agent path finding. AAAI 2019. (PBS)
- G. Wagner, H. Choset. Subdimensional expansion for multirobot path planning.
  Artificial Intelligence 219, 2015. (M*)
- K. Okumura, M. Machida, X. Défago, Y. Tamura. Priority inheritance with backtracking
  for iterative multi-agent path finding. IJCAI 2019; extended in arXiv:1901.11282v5,
  2022. (PIBT)
- X. Jin, C. Qian, R. Zhang, G. Wu, X. Li, J. Jiang, W. Shang. Asymmetric decomposition
  for multi-agent path finding with unassigned agents. arXiv:2611.24108v1, 2026. (FOCBS)
- Z. Tao, C. Yu, Y. Hu. Scalable iterative multi-agent path finding under turn
  constraints. arXiv:2611.23911v1, 2026; reference code https://github.com/tzy82065/PIBTT2
  (MIT). (PIBT*_T)
- J. Li, Z. Chen, D. Harabor, P. Stuckey, S. Koenig. MAPF-LNS2: fast repairing for
  multi-agent path finding via large neighborhood search. AAAI 2022.
- M. Goldenberg, A. Felner, R. Stern, G. Sharon, N. Sturtevant, R. Holte, J. Schaeffer.
  Enhanced partial expansion A*. JAIR 50, 2014. (EPEA*)
- D. Silver. Cooperative pathfinding. AIIDE 2005. (space-time A*, reservation tables)
- M. Phillips, M. Likhachev. SIPP: safe interval path planning for dynamic
  environments. ICRA 2011.
- R. Stern et al. Multi-agent pathfinding: definitions, variants, and benchmarks. SoCS
  2019. (Moving AI MAPF benchmarks)
- Z. Ren, S. Rathinam, H. Choset. A conflict-based search framework for multi-objective
  multi-agent path finding. arXiv:2101.03805v5, 2022. (MO-CBS)
- E. Martins. On a multicriteria shortest path problem. EJOR 16, 1984.
- D. Kornhauser, G. Miller, P. Spirakis. Coordinating pebble motion on graphs, the
  diameter of permutation groups, and applications. FOCS 1984.
