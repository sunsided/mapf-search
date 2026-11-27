# Rendezvous-based coordination for mapf-search (issue #21): literature research

Background for [`rendezvous`](../../src/rendezvous.rs) and the stale-information notes in
[design.md](../design.md). Written for sunsided/mapf-search#21.

Date: 2026-11-27. Scope: the "rendezvous-based coordination" bullet of sunsided/mapf-search#21, plus the
two bullets it depends on (asynchronous decentralized prioritized planning, stale information).

Verification legend:
- **[V]** I read the claim in the primary source (full text or the relevant section).
- **[A]** Verified from the abstract or the publisher/author page only.
- **[U]** Not verified from a primary source. Treat it as a lead.

## TL;DR

- **Which reading the literature means.** Wherever the robotics literature says "rendezvous", it
  means **(B), physical**: robots exchange information only while they are co-located or in range
  at agreed places and times. The fully read sources here are Kantaros & Zavlanos [V] and Ribeiro
  da Silva et al. [V, html]. Reading (A), "communicate every T steps", is studied under other names:
  *periodic / delayed communication* (Atzmon et al. JAIR 2020 §7 [V]) and *rolling-horizon*
  planning (RHCR [V]).
- **The bridge between the two.** Hollinger & Singh's *periodic connectivity* [V] works on a
  time-unfolded graph with unit steps. Every `T_I` steps the team must reach a configuration that is
  connected in a communication graph `G_C`. It replans only at those moments, by sequential
  (prioritized) path updates.
- **Reading (A), smallest algorithm.** RHCR with replanning period `h = T` and window `w ≥ T`.
  - The inner Windowed-MAPF solver is one of the existing solvers, or a decentralized SD-RPP /
    AD-RPP run (Čáp et al. 2015) that runs to quiescence inside the sync.
  - Fallback when the inner solve fails: every agent waits for the period. This is always
    conflict-free.
  - If execution can slip, the committed segment must also be T-robust (Atzmon Theorem 7.1).
- **Reading (B), smallest algorithm.** Hollinger & Singh's Algorithm 1 with MAPF conflicts added.
  - At each connected configuration, agents plan their next `T_I` steps one at a time, in priority
    order, under the reservations of the agents already planned.
  - Candidate paths whose joint configuration at `+T_I` is disconnected in `G_C` are discarded.
  - "Everyone stays put" is the initial plan and the fallback. It is conflict-free and stays
    connected.
  - Once (B) holds, it reduces to (A): the physics guarantees a full sync every `T_I`.
- **Not supported by any source.** Subgroup rendezvous (Kantaros-style teams, never all connected
  at once) with collision avoidance has no primary-source algorithm. It stays an unproven
  extension (§5.3).

---

## 1. Asynchronous / decentralized prioritized planning

### 1.1 Čáp, Novák, Kleiner, Selecký (2015): RPP, SD-RPP, AD-RPP

- Citation: M. Čáp, P. Novák, A. Kleiner, M. Selecký, "Prioritized Planning Algorithms for Trajectory
  Coordination of Multiple Mobile Robots", *IEEE Transactions on Automation Science and Engineering*,
  July 2015. [U] I did not check the volume and issue. The TU Delft copy also lists M. Pěchouček.
- URLs: https://arxiv.org/abs/1409.2399 · https://pure.tudelft.nl/ws/portalfiles/portal/67074672/07138650.pdf
- Model [V, ar5iv full text, read through a summarizer]:
  - **RPP.** Robot i plans in the workspace minus the *start* regions of all lower-priority robots,
    and it avoids the space-time occupancy of the higher-priority robots.
    - Theorem 2 and Corollary 3 give a sufficient condition for success: every robot has a path that
      avoids lower-priority starts and higher-priority goals. Under that condition, RPP with a
      complete single-robot planner terminates with a conflict-free solution.
    - "Valid infrastructures" (endpoints pairwise connectable while keeping clearance from the other
      endpoints) satisfy this condition.
  - **SD-RPP.** Globally synchronized rounds, after Velagapudi et al.
    - In each round, a robot whose trajectory is inconsistent with the higher-priority trajectories
      it holds replans and broadcasts `INFORM(i, trajectory)`. Only robots that changed broadcast.
    - Prop. 4: by induction on priority, every robot eventually stops sending. Hence the protocol
      terminates (Cor. 5).
    - Soundness (Prop. 6): on success, all trajectories are pairwise conflict-free. Completeness
      (Prop. 7) holds under the Theorem 2 condition.
  - **AD-RPP.** No rounds. A robot replans only when it receives a higher-priority INFORM that breaks
    its consistency, and broadcasts only if its trajectory changed. Termination (Prop. 8), soundness
    (Prop. 9) and completeness (Prop. 10) carry over.
  - **Communication assumption.** Broadcast is reliable and order-preserving: every message is
    eventually received by all robots. The fielded UAV variant (CLAD-PP) re-broadcasts periodically
    because radio delivery is not guaranteed, and it has *no* proof under message loss.
- Mapping onto this crate:
  - The "Best-traj" step is `space_time_astar` under `reserve(...)`-generated constraints from the
    higher-priority paths, plus vertex constraints on lower-priority start vertices (the RPP twist).
  - An INFORM message is one `Path`.
  - The protocol is a deterministic fixed-point loop. Simulated in-process, SD-RPP is a loop over
    rounds that ends when no path changes. That loop is deterministic and `no_std`-friendly.
  - Round bound: with strict priorities, the protocol runs at most N rounds (each round settles at
    least the highest-priority unsettled agent). This bound is stated in arXiv:1210.6855, §1.2.

### 1.2 Čáp, Novák, Vokřínek, Pěchouček (2012): ADPP / IADPP

- Citation: "Asynchronous Decentralized Algorithm for Space-Time Cooperative Pathfinding", arXiv:1210.6855, 2012.
- URL: https://arxiv.org/abs/1210.6855
- Model [V, ar5iv, read through a summarizer]:
  - The original ADPP: agents send their paths to lower-priority agents and replan on receipt.
    IADPP also restarts a planning run that a new message interrupts.
  - Termination is proven by induction on priority. SDPP makes at most N iterations. Global
    termination detection assumes a snapshot algorithm (Chandy–Lamport).
  - The algorithm is sound but **incomplete**: no backtracking.
  - Experiments run on 4- and 8-connected grids, with move = 1 s and wait = 0.5 s, so time is not
    in unit steps.
- Mapping: same as §1.1, but without the start-avoidance rule. This is plain prioritized planning,
  computed decentrally.

### 1.3 Velagapudi, Sycara, Scerri (2010): decentralized prioritized planning

- Citation: P. Velagapudi, K. Sycara, P. Scerri, "Decentralized Prioritized Planning in Large Multirobot Teams", IROS 2010, pp. 4603–4609.
- URLs: https://ri.cmu.edu/?p=10563 · https://core.ac.uk/download/357646196.pdf
- Model [A]:
  - Two distributed algorithms. The first is an "efficient, complete" method, shown to converge to
    the centralized prioritized-planning solution. This is the synchronized-rounds scheme that Čáp
    et al. call SD-PP.
  - The second is a sparse variant in which robots detect collisions probabilistically.
  - Experiments used teams of up to 240 robots in 2D.
- [U] I did not read the full text, so I have not checked the exact round and message details.
  "Complete" here appears to mean "reaches the same outcome as centralized PP", not MAPF completeness.

### 1.4 Čáp, Vokřínek, Kleiner (2015): COBRA, on-line decentralized planning

- Citation: "Complete Decentralized Method for On-Line Multi-Robot Trajectory Planning in Valid Infrastructures", ICAPS 2015. arXiv:1501.07704.
- URL: https://arxiv.org/abs/1501.07704
- Model [A]:
  - When a task is issued, the assigned robot plans a trajectory that avoids the robots already
    executing earlier tasks, so the order of issue is the priority.
  - The method is complete and deadlock-free in valid infrastructures, with time quadratic in the
    number of robots.
  - [U] The abstract does not describe the token or shared-reservation mechanism.
- Mapping: a lifelong, first-come prioritized planner over a shared reservation table.

---

## 2. Rendezvous / intermittent communication

### 2.1 Kantaros & Zavlanos: intermittent communication at rendezvous points (reading B)

- Citations:
  - Y. Kantaros, M. M. Zavlanos, "Distributed Intermittent Connectivity Control of Mobile Robot
    Networks", *IEEE Transactions on Automatic Control* 62(7):3109–3121, 2017.
  - The preprint I read: "Simultaneous Intermittent Communication Control and Path Optimization in
    Networks of Mobile Robots", arXiv:1609.07038, 2016. [U] I did not check which conference
    version corresponds to it.
- URLs: https://arxiv.org/abs/1609.07038 · https://dukespace.lib.duke.edu/items/e4fe0047-913e-400b-a81f-eb8960c4ebcb
- Model [V, preprint]:
  - Robots move on a *mobility graph* G and are split into M teams `T_m`. A robot may belong to
    several teams.
  - Team m communicates **only when all its members are simultaneously at one common location** in
    its set of communication points `C_m`.
  - The team graph `G_T` (edge when two teams share a robot) is assumed connected, so information
    spreads through shared members.
  - The requirement is an LTL formula: every team meets infinitely often (□◇) at some point of `C_m`.
  - Each team plans its own meeting tour. A distributed conflict-resolution scheme orders each
    robot's per-team sub-paths, padded with waits, so that no robot must be in two places at once
    (Prop. 3.6 / 3.8; the number of slots is `max deg(G_T) + 1`).
  - Inserting delays (waits) lets execution be asynchronous.
- Important caveats:
  - "Conflict" here means a **scheduling** conflict for a robot, not a collision between robots.
  - Robots are points that *co-locate*, which MAPF vertex conflicts forbid.
  - The paper makes no collision-freedom claim. [V: "collision" does not occur anywhere in the
    preprint text.]
  - Footnote 1 assumes paths between a robot's own communication points never pass through other
    teams' communication points. This is a routing assumption, not spatial exclusivity.
- Mapping:
  - The meeting points become *timed waypoints* `(agent, vertex set, time)`.
  - Co-location must be replaced by "in range": all members on vertices of a cluster `C_m` at the
    same t, which with vertex conflicts must be distinct vertices.
  - The schedule is a periodic tour, so it is not a one-shot MAPF instance.

### 2.2 Hollinger & Singh: periodic connectivity. Best discrete-time fit for (B)

- Citations:
  - G. Hollinger, S. Singh, "Multi-Robot Coordination with Periodic Connectivity", ICRA 2010,
    pp. 4457–4462.
  - Journal version: "Multirobot Coordination With Periodic Connectivity: Theory and Experiments",
    *IEEE Transactions on Robotics* 28(4):967–973, 2012 [A].
- URLs: https://publications.ri.cmu.edu/storage/publications/pub_files/2010/5/HollingerICRA10.pdf · https://ri.cmu.edu/?p=128025
- Model [V, ICRA 2010 full text]:
  - The environment is a graph G. Planning runs on its *time-unfolded* graph, with paths
    `r_k(0..T)` in discrete unit steps.
  - A separate connectivity graph `G_C` over the same nodes is given, or derived from range or
    line-of-sight constraints.
  - A configuration is connected when every robot pair is joined by a `G_C` path whose nodes are all
    occupied, so robots can act as relays.
  - The robots start connected and must be connected again at times `T_I, 2T_I, …`. With `T_I = 1`
    this is continual connectivity.
  - Robots *may* share cells. The paper assumes low-level collision avoidance, but notes that it
    "could be incorporated by disallowing robots from occupying the same cell at the same time".
  - Algorithm 1, "implicit coordination":
    - Initialize every path to *stay stationary*.
    - Repeat until convergence: robot k removes its own path from the visited set, enumerates
      feasible paths of length `T_I`, discards those that are disconnected at `T_I`, takes the
      argmax of the objective given the others' paths, and shares it.
    - The algorithm runs at the start and again each time the robots reach the planned connected
      configuration. Synchronization happens then, because "the network is guaranteed to be
      connected every T_I".
    - It is a coordinate ascent in a random planning order. Convergence: finite paths and a
      monotone, bounded reward.
  - Theorem 2 bounds sequential allocation when the connected configurations are given in advance.
- Mapping onto MAPF:
  - Replace "maximize information" with "progress to goal", for example the sum of distances to the
    goals at `+T_I`.
  - Replace "cells may be shared" with `reserve()` of the already-planned agents.
  - Add a final-configuration filter: connected in `G_C`.
  - The stay-put initialization is a safe fallback in MAPF too. Distinct vertices, no moves: no
    vertex or swap conflict, and connectivity is unchanged.

### 2.3 Ribeiro da Silva, Chaimowicz, Silva, Hsieh (2023/24): exploration with intermittent rendezvous

- Citation: "Communication-Constrained Multi-Robot Exploration with Intermittent Rendezvous", arXiv:2309.13494 (v6, 2024).
- URL: https://arxiv.org/abs/2309.13494
- Model [V, arXiv html v5, read through a summarizer]:
  - A DEC-POMDP for exploration. A rendezvous plan is a set of agreements. Each agreement names its
    participating robots, a location, and a timing: the number of exploration steps a robot takes
    before it goes to the location.
  - Robots "fulfill an agreement" when they "meet at their assigned location". A link exists
    within communication range ϕ.
  - The agreements are built by reducing to Job Shop Scheduling.
  - Evidence is empirical only. I saw no formal guarantees.
- This confirms the physical sense (B).

### 2.4 Connected MAPF (Queffelec, Sankur, Schwarzentruber; Calviac et al.)

- Citations:
  - A. Queffelec, O. Sankur, F. Schwarzentruber, "Conflict-Based Search for Connected Multi-Agent
    Path Finding", arXiv:2006.03280, 2020.
  - The same authors, "Connected Multi-Agent Path Finding: Generation and Visualization", IJCAI 2021
    [U] I did not check the track (demo or main).
  - Calviac, Sankur, Schwarzentruber, "Improved Complexity Results and an Efficient Solution for
    Connected Multi-Agent Path Finding", AAMAS 2023.
- URLs: https://arxiv.org/abs/2006.03280 · https://www.ijcai.org/proceedings/2021/714 · https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p896.pdf
- Model [A]:
  - Agents on a movement graph must stay connected to each other and to a base, under a separate
    communication graph.
  - The CBS variant branches on *disconnection* conflicts rather than collisions.
  - [U] From the abstracts I could not verify whether connectivity is required at every timestep.
    I believe it is, but have not checked.
- Relevance: this is the "always connected" extreme. Rendezvous (B) is the "connected only at
  scheduled (t, C_m)" relaxation. CMAPF's CBS adaptation (connectivity checked at a timestep, as a
  conflict) is the natural way to *enforce* rendezvous constraints inside CBS.

### 2.5 Multi-Agent Meeting (Atzmon et al.)

- Citations:
  - D. Atzmon et al., the MM* paper, IJCAI 2020, doi:10.24963/ijcai.2020/562. [U] I did not check
    the exact title.
  - The conflict-free extension, CF-MAM.
- URLs: https://www.ijcai.org/proceedings/2020/562 · https://webdocs.cs.ualberta.ca/~nathanst/papers/atzmon2020mam.pdf · https://webdocs.cs.ualberta.ca/~nathanst/papers/atzmon23conflict.pdf
- Model [A]:
  - The input is a graph and k start vertices. The task is to find the meeting vertex that is
    optimal, by shortest path from each start.
  - CF-MAM additionally asks for conflict-free MAPF paths to that meeting.
- Relevance: this is the "choose the meeting point" sub-problem of (B). It uses the word
  "meeting", not "rendezvous", and involves no information exchange, so it does not count toward
  §4. [U] I did not read how CF-MAM defines "meeting" under vertex conflicts.

---

## 3. Planning under stale information

### 3.1 Atzmon, Stern, Felner, Wagner, Barták, Zhou (JAIR 2020): k-robust MAPF and delayed communication. Best fit for (A)

- Citation: "Robust Multi-Agent Path Finding and Executing", *Journal of Artificial Intelligence Research* 67:549–579, 2020. Conference precursors: SoCS 2017 (k-robust MAPF) and SoCS 2018.
- URLs: https://jair.org/index.php/jair/article/view/11734 · https://ojs.aaai.org/index.php/SOCS/article/view/18415
- Model [V]:
  - A plan is k-robust if it stays valid under any ≤k delays per agent.
  - Def. 3.2, the *k-delay conflict*: `π_i(t) = π_j(t+Δ)` for some `Δ ∈ {0..k}`.
  - Obs. 1: a plan is k-robust iff it has no k-delay conflict. A k-robust plan with k>0 therefore
    also excludes swaps and train-like following.
  - The authors adapt A*, CBS and a CP/SAT encoding.
  - **§7, Delayed Communication.** Agents exchange status only at every T-th "communication time
    step".
    - The execution policies combine when to modify the plan (Eager / Reasonable / Lazy) with how to
      modify it (T-replan / T-All / T-Selective).
    - Theorem 7.1: Eager, Reasonable and Lazy T-replan are robust given a T-robust initial plan. At every
      communication step the executed plan has no T-delay conflict in the next T steps, so no
      collision happens before the next communication step.
    - Theorem 7.2: the same holds for Eager T-All, Reasonable T-All and Reasonable T-Selective.
    - MCP stays robust under delayed communication because it preserves the order of vertex visits.
- Mapping:
  - k-delay conflicts are a widened conflict check. Def. 3.2 applies to both orders of the pair, so
    another agent's visit to v at time t must reserve v over `t-k..=t+k`.
  - With `reserve()` this means inserting vertex constraints over that range, which needs no new
    machinery.
  - A deterministic executor needs k = 0. The T-robust layer matters only if real execution can
    slip.

### 3.1b Atzmon, Stern, Felner, Sturtevant, Koenig (ICAPS 2020): p-robust MAPF

- Citation: "Probabilistic Robust Multi-Agent Path Finding", ICAPS 2020, pp. 29–37, doi:10.1609/icaps.v30i1.6642.
- URL: https://ojs.aaai.org/index.php/ICAPS/article/view/6642
- Model [A]:
  - Plans are found for which the probability of no collision is at least p, under per-agent delay
    probabilities.
  - pR-CBS is optimal. pR-GCBS is a faster suboptimal variant. Two methods verify p-robustness.
- Mapping: the result is probabilistic, so it gives no hard collision-freedom guarantee. It does not
  fit a library whose invariant is "the executed joint plan is conflict-free".

### 3.1c Soft vs. time-bounded reservations of stale plans

I found **no primary source** that defines "soft reservations of another agent's last known plan"
with a guarantee. What the sources do give:

- **Hard, time-bounded reservations.** WHCA*-style windows, RHCR's `w`, and Atzmon's T-robust
  segment between communication steps guarantee collision-freedom only up to the bound. The
  guarantee needs every reserving agent to replan, or be re-informed, before the bound expires,
  which is the `w ≥ h` rule.
- **Soft reservations.** These are conflict counts used as tie-breakers or penalties, as in focal
  search in ECBS, or p-robust probabilities. They improve plan quality but carry **no**
  collision-freedom guarantee.

Consequence: under stale information, a sound library either treats the last known plans as hard
reservations up to a bound that every party knows, or it provides no guarantee.

### 3.2 Ma, Kumar, Koenig (AAAI 2017): MAPF with delay probabilities, MCP

- Citation: H. Ma, T. K. S. Kumar, S. Koenig, "Multi-Agent Path Finding with Delay Probabilities", AAAI 2017. arXiv:1612.05309.
- URLs: https://ojs.aaai.org/index.php/AAAI/article/view/11035 · https://arxiv.org/abs/1612.05309
- Model:
  - [A] Each agent has a delay probability. The AME solver minimizes expected cost, and
    plan-execution policies are attached.
  - [V, via the JAIR 2020 summary of it] MCP (Minimal Communication Policy): an agent enters vertex
    v only when it is the next agent the plan has entering v. Otherwise it waits until the
    predecessor has left.
- Mapping: MCP is a post-plan execution layer, a precedence table per vertex. It is not a solver
  change.

### 3.3 Hönig, Kiesel, Tinka, Durham, Ayanian (RA-L 2019): Action Dependency Graph

- Citation: "Persistent and Robust Execution of MAPF Schedules in Warehouses", *IEEE RA-L* 4(2):1125–1131, 2019.
- URL: https://act.usc.edu/publications/Hoenig_RAL2019.pdf
- Model [A]:
  - The ADG puts precedence on *actions*. Robots communicate only when an action completes.
  - The ADG lets replanning overlap with execution.
  - It is collision-free under unknown slow-downs.
- Mapping: a generalization of MCP. It is useful later for "execute a committed plan robustly".

### 3.4 Li, Tinka, Kiesel, Durham, Kumar, Koenig (AAAI 2021): RHCR / Windowed MAPF. Structural template for (A)

- Citation: "Lifelong Multi-Agent Path Finding in Large-Scale Warehouses", AAAI 2021, pp. 11272–11281. arXiv:2005.07371.
- URLs: https://ojs.aaai.org/index.php/AAAI/article/view/17344 · https://arxiv.org/abs/2005.07371
- Model [V]:
  - The solver replans every h steps and must resolve collisions within the first w steps. Paths
    are "collision-free for the first w timesteps", and the paper states that w must be at least h.
  - Agents move h steps along the plan, then the next episode starts from their actual positions.
  - Small w can produce deadlocks (Example 2). §4.4 proposes a potential-function fix.
  - The method is incomplete, inheriting the incompleteness of PBS/ECBS/CA*.
- Mapping:
  - This is exactly reading (A) with h = T. The crate's `whca_star` / `reserve(.., until)` already
    implements the "reserve only through `until`" half.

---

## 4. Which reading does the literature mean?

Mostly **(B), physical**. Both sources that I read in full and that use the word "rendezvous"
mean meeting physically, by co-location or range, at an agreed place and time to exchange
information: Kantaros & Zavlanos [V] and Ribeiro da Silva et al. [V]. The classical
"rendezvous search" line (e.g. Meghjani & Dudek) presumably means the same, but I did not verify it
[U]. Reading (A) is real and well
studied, but under different names:

- "periodic" or "delayed communication" (Atzmon et al. §7);
- rolling or receding horizon (RHCR);
- synchronized rounds (Velagapudi; SD-PP).

Hollinger & Singh sit between the two: their times are periodic as in (A), but connectivity is
physical, by range, as in (B).

Recommendation for the issue text: rename (A) to **"periodic-sync (windowed) coordination"**.
Reserve "rendezvous" for (B).

---

## 5. Recommendations

### 5.1 Reading (A): periodic-sync windowed planning

**Algorithm.** RHCR with `h = T`, `w ≥ T`. The Windowed-MAPF solver is either the existing
centralized solvers or SD-RPP run to quiescence (§1.1). This is Atzmon §7's "Eager T-replan" in the
no-delay case.

- Inputs:
  - the graph;
  - the agents' current vertices at the sync time `t`;
  - the goals: one per agent, with goal-stay;
  - the period `T ≥ 1`;
  - the window `w ≥ T` (default `w = T`, or `None` for a full horizon);
  - an optional strict priority order;
  - optionally, `k` for k-robustness.
- Output: per agent, a `Path` that starts at its current vertex. The joint prefix on `[t, t+w]` is
  conflict-free. Everyone commits to the prefix `[t, t+T]`.
- Invariant: at every sync time `t`, the committed joint prefix on `[t, t+T]` has no vertex or swap
  conflict, or no k-delay conflict if robust. The prefix is either the solver's output or
  all-agents-wait. Because `w ≥ T` and execution is deterministic, every
  executed step lies inside some committed prefix. The executed joint plan is therefore
  conflict-free by induction over sync epochs (the argument of RHCR's `w ≥ h` and Atzmon Theorem 7.1).
- Decentralized variant:
  - Inside a sync, agents run SD-RPP. Each round, every agent whose path conflicts with
    higher-priority paths replans under `reserve(.., Some(t+w))` of those paths, then broadcasts it.
  - The sync ends when a round changes nothing. This terminates in ≤ n rounds with strict
    priorities (Čáp Prop. 4–5).
  - Soundness on success: Prop. 6.
  - Failure means some agent found no path. Then **all agents wait for the period**. This is
    conflict-free, because positions at a sync are distinct and nobody moves.
  - "Keep the previous committed plan" is **not** a safe fallback when `w = T`: that plan was
    checked only through the current sync time. It is safe only if `w ≥ 2T`.
- Termination of the whole run:
  - The run stops when all agents are at their goals.
  - Windowed planning is **incomplete and can livelock or deadlock** (RHCR §4.4).
  - The cheapest sound guard is `w = ∞`. Each sync then plans full paths, so the first sync already
    solves the instance and later syncs change nothing unless goals change. Otherwise, add an
    epoch cap and report failure.
  - Completeness is then that of the inner solver. For RPP it holds under the Čáp Theorem 2 /
    valid-infrastructure condition.
- No changes needed beyond the solver:
  - `reserve(.., until)`, `space_time_astar`, `whca_star`, `pbs` and the conflict detection already
    exist.
  - The new parts are a driver loop (deterministic, allocation via `alloc`) and an optional RPP
    start-avoidance constraint.
- Caveat: one-shot offline MAPF with a deterministic executor gains nothing from (A). It pays off
  only when information changes between syncs (new goals, lifelong operation, deviations).

### 5.2 Reading (B): physical rendezvous with periodic connectivity

**Algorithm.** Hollinger & Singh's Algorithm 1 (§2.2) with MAPF conflict constraints added. I
adapted it to MAPF myself. The parts that carry its soundness (stay-put initialization, sequential
updates under shared paths, the connectivity filter) are taken from the paper.

- Inputs:
  - the graph;
  - a communication relation `G_C` on vertices, as an adjacency list or a predicate;
  - the start vertices, whose configuration must be connected in `G_C`;
  - the goals;
  - the interval `T_I ≥ 1`;
  - a strict priority order;
  - an epoch cap.
- Per epoch, at a connected configuration at time t:
  - For each agent in priority order, run `space_time_astar` over `[t, t+T_I]` under `reserve()`
    of the agents already planned. Use the RPP start-avoidance of lower-priority agents if wanted.
  - Accept a path only if the configuration at `t+T_I` is connected in `G_C`, given the vertices
    already fixed for higher-priority agents. Make this a goal test at `t+T_I`, not a branch.
  - Optional coordinate-ascent sweeps can follow (Hollinger), each under the same constraints.
  - If any agent finds no acceptable path, the epoch falls back to all-stay-put.
- Output: the joint `T_I`-step plan for the epoch, and over all epochs the executed joint plan.
- Invariant: each epoch's joint plan has no vertex or swap conflict, and ends in a configuration
  connected in `G_C`.
  - The stay-put fallback has both properties trivially.
  - By induction, every epoch starts connected, so all agents can exchange full information then.
  - Between epochs, no information is needed, because the plans are fixed and execution is
    deterministic.
  - The executed plan is therefore conflict-free and connected at every multiple of `T_I`. These
    are exactly the rendezvous times.
- Termination:
  - Each epoch is finite: one bounded A* per agent, plus finitely many sweeps under a cap.
  - Reaching the goals is **not** guaranteed. Connectivity can make the goals mutually
    unreachable, and stay-put can repeat forever. Stop on "all at goal" or on the epoch cap, and
    report failure on the cap.
  - [U] No completeness result exists for this MAPF adaptation.
  - The sequential greedy step is incomplete. It can fail even when connected solutions exist. CBS
    with connectivity-at-`T_I` as a conflict, after CMAPF (§2.4), would be the complete,
    heavier option.
- Smallest useful slice: rendezvous vertex sets given in advance. Agent i must be in `C_m` at
  `t_m`, as in Hollinger Theorem 2's "prespecified connected configurations".
  - This is a per-agent *timed-waypoint* constraint. Enforce it in the low level, as a goal test or
    an intermediate goal at `t_m`, without CBS branching.
  - Any existing solver then gives a conflict-free joint plan that satisfies the rendezvous.

### 5.3 Unproven extension: subgroup rendezvous (Kantaros-style)

Some teams meet at different times, and the team is never all connected at once. Then an agent's
plan can change at a meeting while non-members still hold its stale version.

- A sound rule has to keep each changed plan away from every space-time cell that a non-member
  might occupy before it is next informed. One way is to give each group an exclusive space-time
  region until its next meeting.
- This is my synthesis. **No primary source proves it.** Kantaros & Zavlanos do not model
  collisions.
- Do not build it before a written proof exists, especially for swaps at region borders.

---

## 6. Unverified or open items

- Velagapudi et al. 2010: I read only the abstract and summary, not the round or message details.
- Hollinger & Singh: the T-RO 2012 journal version was not read. The ICRA 2010 version was read.
- CMAPF: whether connectivity must hold at every timestep, and its graph model, are unverified.
- COBRA (ICAPS 2015): the internal mechanism (token or reservation) is unverified.
- CF-MAM: the definition of "meeting" under vertex conflicts is unverified.
- Meghjani & Dudek (rendezvous search for multi-robot exploration) was named in the task. I did not
  locate or verify it.
- §5.2 adapts Hollinger's algorithm to MAPF, and the invariant argument for it is mine. §5.3 is
  synthesis only.
- The MM* title, the CMAPF IJCAI 2021 track, and the conference version of Kantaros's preprint are
  unverified.
