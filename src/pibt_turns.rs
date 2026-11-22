//! PIBT*_T ([`pibt_turns`], Tao, Yu and Hu 2026): PIBT for agents that face a heading,
//! move only forward, and need time to turn in place, on any [`OrientedGraph`].
//!
//! Plain PIBT ([`pibt`](crate::pibt())) assumes that an agent that accepts a push leaves its
//! vertex at the next step. With turns it may first have to rotate, and still stand there.
//! PIBT*_T keeps PIBT's priority inheritance and backtracking and adds:
//!
//! - **Turn-aware choices.** An agent requests the neighbour (or its own vertex) with the
//!   least `TC`: the turns and the move to get there plus the exact remaining time from the
//!   state it arrives in (`costs`); `Compute_action` then moves forward if it faces the
//!   request, else turns once towards it.
//! - **Staying behind a turning agent.** An agent whose requested vertex stays occupied,
//!   because its occupant only turns, waits instead of moving in.
//! - **Cycles.** When a chain of requests closes back on the agent that started it, the
//!   agents of the cycle all move at once if each faces its request, else the misaligned
//!   ones turn and the others wait (the paper's Definition 1 and Proposition 2). It cannot
//!   close on a later agent of the chain: each of those stands on the vertex its pusher
//!   has claimed.
//! - **Swaps** (`swap`): in a corridor where two agents must pass each other, the pusher
//!   reverses its preferences and pulls the other agent into its vertex.
//! - **The k-push escape trigger.** An agent pushed by the same agent `k = 2` times shuffles
//!   its preferences the next time, which breaks livelocks of repeated pushing.
//! - **Remembered requests.** An agent that turned (or waited) towards a vertex requests it
//!   first again at the next step, so it finishes the turn instead of changing its mind.
//!
//! The run ends at the first step at which every agent stands at its goal, facing any way.

mod costs;
mod swap;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use self::costs::Costs;
use crate::graph::{Heading, OrientedGraph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::path::Path;
use crate::random::Random;
use crate::solution::{MapfResult, Outcome, Solution};

/// How often the same pusher may push an agent before the agent shuffles its preferences:
/// the paper's `k`.
const PUSH_LIMIT: u32 = 2;

/// The result of a [`pibt_turns`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnResult {
    /// How the run ended, and its counters. A solution's paths hold each agent's vertex at
    /// every step of the run, turns as waits, so [`first_conflict`](crate::first_conflict)
    /// and the costs apply as for any plan.
    pub result: MapfResult,
    /// Each agent's heading at every step of the run, indexed like the paths: the same
    /// length as its path in a solution, empty otherwise.
    pub headings: Vec<Vec<Heading>>,
}

/// Finds a conflict-free plan for agents that need time to turn, with PIBT*_T: one step
/// for all agents at a time, by priority, with priority inheritance, backtracking, cycle
/// handling, swaps and the push escape trigger.
///
/// `headings[i]` is the heading agent `i` starts with. Every random choice comes from
/// `seed`. The plan has no quality guarantee, and the method is not complete: on an
/// unsolvable instance it runs until its budget, so always pass one. The result is
/// [`Outcome::Unsolvable`] only if an agent cannot reach its goal even alone, and never
/// [`Outcome::Exhausted`].
///
/// `budget` is asked before every step; [`MapfResult::stats`] counts steps in
/// [`Stats::expanded`] and per-agent decisions in [`Stats::generated`].
///
/// # Panics
///
/// Panics if `headings` does not hold one heading below
/// [`OrientedGraph::heading_count`] per agent, or if turning cannot reach every heading.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, first_conflict, pibt_turns};
///
/// // Two robots swap the ends of the top row of a free 3x2 grid, both facing east.
/// let grid = Grid::new(3, 2);
/// let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
/// let instance = Instance::new(grid, agents)?;
///
/// let run = pibt_turns(&instance, &[Grid::EAST, Grid::EAST], 0, Limits::expansions(200));
/// let solution = run.result.solution().expect("one robot goes around");
/// assert_eq!(first_conflict(&solution.paths), None);
/// assert_eq!(run.headings[1][0], Grid::EAST);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn pibt_turns<G: OrientedGraph>(
    instance: &Instance<G>,
    headings: &[Heading],
    seed: u64,
    mut budget: impl Budget<Stats>,
) -> TurnResult {
    let graph = instance.graph();
    let agents = instance.agents();
    assert_eq!(headings.len(), agents.len(), "one heading per agent");
    assert!(
        headings
            .iter()
            .all(|&heading| heading < graph.heading_count()),
        "headings are below the heading count"
    );

    let costs = Costs::new(graph, agents);
    let reachable = agents
        .iter()
        .enumerate()
        .all(|(index, agent)| costs.least(index, agent.start).is_some());
    if !reachable {
        return TurnResult {
            result: MapfResult {
                outcome: Outcome::Unsolvable,
                stats: Stats::default(),
            },
            headings: Vec::new(),
        };
    }

    let mut run = Run::new(instance, costs, headings, seed);
    let outcome = run.solve(&mut budget);
    let headings = if matches!(outcome, Outcome::Solved(_)) {
        run.heading_paths
    } else {
        Vec::new()
    };

    TurnResult {
        result: MapfResult {
            outcome,
            stats: run.stats,
        },
        headings,
    }
}

/// The state of one [`pibt_turns`] run.
struct Run<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// The turn-aware cost tables.
    costs: Costs<'a, G>,
    /// Each agent's vertex at the current step.
    here: Vec<Vertex>,
    /// Each agent's heading at the current step.
    heading: Vec<Heading>,
    /// The vertex each agent has claimed while its decision is in progress.
    claim: Vec<Option<Vertex>>,
    /// Each agent's decided vertex and heading for the next step.
    decided: Vec<Option<(Vertex, Heading)>>,
    /// The agent holding each vertex for the next step, by vertex.
    reserved: Vec<Option<AgentId>>,
    /// The agent at each vertex at the current step, by vertex.
    occupant: Vec<Option<AgentId>>,
    /// The vertex each agent turned or waited towards last, requested first next time.
    intent: Vec<Option<Vertex>>,
    /// How many steps each agent has spent away from its goal: its priority.
    waiting: Vec<u64>,
    /// How often an agent (first) was pushed into a new vertex by another (second).
    pushes: BTreeMap<(AgentId, AgentId), u32>,
    /// The requests of the current inheritance chain: each agent on it and the vertex it
    /// requests, from the agent planned first.
    chain: Vec<(AgentId, Vertex)>,
    /// The source of the random tie-breaks and shuffles.
    random: Random,
    /// Each agent's vertex at every step so far.
    paths: Vec<Path>,
    /// Each agent's heading at every step so far.
    heading_paths: Vec<Vec<Heading>>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<'a, G: OrientedGraph> Run<'a, G> {
    /// A run from the starts of `instance`, facing `headings`.
    fn new(
        instance: &'a Instance<G>,
        costs: Costs<'a, G>,
        headings: &[Heading],
        seed: u64,
    ) -> Self {
        let agents = instance.agents().len();
        let vertices = instance.graph().vertex_count();
        let here: Vec<Vertex> = instance.agents().iter().map(|agent| agent.start).collect();

        Self {
            instance,
            costs,
            paths: here.iter().map(|&vertex| vec![vertex]).collect(),
            heading_paths: headings.iter().map(|&heading| vec![heading]).collect(),
            heading: headings.to_vec(),
            here,
            claim: vec![None; agents],
            decided: vec![None; agents],
            reserved: vec![None; vertices],
            occupant: vec![None; vertices],
            intent: vec![None; agents],
            waiting: vec![0; agents],
            pushes: BTreeMap::new(),
            chain: Vec::new(),
            random: Random::new(seed),
            stats: Stats::default(),
        }
    }

    /// Takes steps until every agent stands at its goal or the budget ends.
    fn solve(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        loop {
            if self.all_at_goals() {
                return Outcome::Solved(Solution::new(self.paths.clone()));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.step();
        }
    }

    /// Returns `true` if every agent stands at its goal.
    fn all_at_goals(&self) -> bool {
        self.here
            .iter()
            .zip(self.instance.agents())
            .all(|(&here, agent)| here == agent.goal)
    }

    /// One step: updates the priorities, decides every agent by decreasing priority, then
    /// carries the decisions out.
    fn step(&mut self) {
        self.stats.expanded += 1;

        for (agent, &here) in self.here.iter().enumerate() {
            self.occupant[here] = Some(agent);
            if here == self.instance.agents()[agent].goal {
                self.waiting[agent] = 0;
            } else {
                self.waiting[agent] += 1;
            }
        }

        let mut order: Vec<AgentId> = (0..self.here.len()).collect();
        order.sort_unstable_by_key(|&agent| core::cmp::Reverse((self.waiting[agent], agent)));
        for agent in order {
            if self.decided[agent].is_none() {
                self.chain.clear();
                self.plan(agent, None);
            }
        }

        self.advance();
    }

    /// Carries out every decision and clears the step's bookkeeping.
    fn advance(&mut self) {
        for agent in 0..self.here.len() {
            let (vertex, heading) = self.decided[agent].take().expect("every agent decided");
            self.occupant[self.here[agent]] = None;
            self.reserved[vertex] = None;
            self.claim[agent] = None;
            self.here[agent] = vertex;
            self.heading[agent] = heading;
            self.paths[agent].push(vertex);
            self.heading_paths[agent].push(heading);
        }
    }

    /// The recursive procedure: decides `agent`, which inherited its priority from
    /// `pusher` if any. Returns `false` if it is stuck and stays put.
    fn plan(&mut self, agent: AgentId, pusher: Option<AgentId>) -> bool {
        self.stats.generated += 1;

        let mut candidates = self.candidates(agent);
        if let Some(pusher) = pusher {
            self.escape(&mut candidates, agent, pusher);
        }

        let partner = self.swap_partner(agent, &candidates);
        if partner.is_some() {
            candidates.reverse();
        }

        self.request_first(agent, &mut candidates);

        for (skipped, &target) in candidates.iter().enumerate() {
            if self.reserved[target].is_some() || pusher.is_some_and(|p| self.here[p] == target) {
                continue;
            }

            self.claim(agent, target);
            if self
                .chain
                .first()
                .is_some_and(|&(first, _)| self.here[first] == target)
            {
                self.chain.push((agent, target));
                self.resolve_cycle();
                self.chain.pop();
                return true;
            }

            if !self.clear_way(agent, target) {
                self.unclaim(agent, target);
                continue;
            }

            if self.decided[agent].is_none() {
                self.decide(agent, target, pusher);
                if skipped == 0 {
                    if let Some(partner) = partner {
                        self.pull(agent, partner);
                    }
                }
            }

            return true;
        }

        self.stay(agent, self.heading[agent]);

        false
    }

    /// Lets the agent standing at `target`, if it is still undecided, decide first with the
    /// priority of `agent`. Returns `false` if that agent is stuck.
    fn clear_way(&mut self, agent: AgentId, target: Vertex) -> bool {
        let Some(other) = self.occupant[target] else {
            return true;
        };

        if other == agent || self.claim[other].is_some() || self.decided[other].is_some() {
            return true;
        }

        self.chain.push((agent, target));
        let moved = self.plan(other, Some(agent));
        self.chain.pop();

        moved
    }

    /// Decides `agent`, which holds the claim on `target`: `Compute_action` towards it,
    /// except that it waits if the vertex stays occupied by an agent that only turns.
    fn decide(&mut self, agent: AgentId, target: Vertex, pusher: Option<AgentId>) {
        let (here, heading) = (self.here[agent], self.heading[agent]);
        let (vertex, next_heading) = self.costs.action(here, heading, target);

        if vertex != here && self.stays_occupied(agent, target) {
            self.unclaim(agent, target);
            self.intent[agent] = Some(target);
            self.stay(agent, heading);
            return;
        }

        if vertex == here {
            self.unclaim(agent, target);
            self.intent[agent] = (next_heading != heading).then_some(target);
            self.stay(agent, next_heading);
            return;
        }

        self.decided[agent] = Some((vertex, next_heading));
        self.reserved[vertex] = Some(agent);
        self.intent[agent] = None;
        if let Some(pusher) = pusher {
            *self.pushes.entry((agent, pusher)).or_default() += 1;
        }
    }

    /// Returns `true` if another agent stands at `vertex` and stays there at the next step.
    fn stays_occupied(&self, agent: AgentId, vertex: Vertex) -> bool {
        self.occupant[vertex].is_some_and(|other| {
            other != agent && self.decided[other].is_some_and(|(next, _)| next == vertex)
        })
    }

    /// The paper's cycle handling for the chain, which closes on the agent that started it:
    /// everyone moves into their request if all face it, else the misaligned agents turn
    /// towards it and the others wait.
    fn resolve_cycle(&mut self) {
        let cycle: Vec<(AgentId, Vertex)> = self.chain.clone();
        let aligned = cycle.iter().all(|&(agent, target)| {
            self.instance
                .graph()
                .forward(self.here[agent], self.heading[agent])
                == Some(target)
        });

        for &(agent, target) in &cycle {
            self.unclaim(agent, target);
        }

        for &(agent, target) in &cycle {
            let heading = self.heading[agent];
            if aligned {
                self.decided[agent] = Some((target, heading));
                self.reserved[target] = Some(agent);
                self.intent[agent] = None;
            } else {
                let (_, turned) = self.costs.action(self.here[agent], heading, target);
                self.intent[agent] = (turned != heading).then_some(target);
                self.stay(agent, turned);
            }
        }
    }

    /// Pulls the swap `partner` of `agent` into the vertex `agent` leaves, turning it first
    /// if it does not face it; neither moves if `agent` stays.
    fn pull(&mut self, agent: AgentId, partner: AgentId) {
        let vacated = self.here[agent];
        let free = self.reserved[vacated].is_none_or(|holder| holder == agent);
        if self.claim[partner].is_some() || self.decided[partner].is_some() || !free {
            return;
        }

        let (here, heading) = (self.here[partner], self.heading[partner]);
        let (vertex, next_heading) = self.costs.action(here, heading, vacated);
        let agent_stays = self.decided[agent].is_some_and(|(next, _)| next == vacated);

        if vertex == here || agent_stays {
            self.intent[partner] = Some(vacated);
            self.stay(partner, if agent_stays { heading } else { next_heading });
        } else {
            self.decided[partner] = Some((vertex, next_heading));
            self.reserved[vertex] = Some(partner);
            self.intent[partner] = None;
        }
    }

    /// Decides that `agent` stays at its vertex, facing `heading`.
    fn stay(&mut self, agent: AgentId, heading: Heading) {
        let here = self.here[agent];
        self.decided[agent] = Some((here, heading));
        self.reserved[here] = Some(agent);
    }

    /// Claims `vertex` for `agent` while its decision is in progress.
    fn claim(&mut self, agent: AgentId, vertex: Vertex) {
        self.claim[agent] = Some(vertex);
        self.reserved[vertex] = Some(agent);
    }

    /// Releases the claim of `agent` on `vertex`, if it still holds it.
    fn unclaim(&mut self, agent: AgentId, vertex: Vertex) {
        if self.reserved[vertex] == Some(agent) {
            self.reserved[vertex] = None;
        }
    }

    /// The vertices `agent` may request, best first: its own and its neighbours from which
    /// the goal is reachable, by `TC`, unoccupied ones first among equals, random otherwise.
    fn candidates(&mut self, agent: AgentId) -> Vec<Vertex> {
        let here = self.here[agent];
        let heading = self.heading[agent];
        let mut candidates = vec![here];
        self.instance
            .graph()
            .neighbors(here, |next| candidates.push(next));
        self.random.shuffle(&mut candidates);

        let mut ranked: Vec<(u32, bool, Vertex)> = candidates
            .into_iter()
            .filter_map(|vertex| {
                let rank = self.costs.rank(agent, here, heading, vertex)?;
                Some((rank, self.occupant[vertex].is_some(), vertex))
            })
            .collect();
        ranked.sort_by_key(|&(rank, occupied, _)| (rank, occupied));

        ranked.into_iter().map(|(_, _, vertex)| vertex).collect()
    }

    /// The k-push escape trigger: if `pusher` has pushed `agent` [`PUSH_LIMIT`] times,
    /// shuffles its preferences and starts counting again.
    fn escape(&mut self, candidates: &mut [Vertex], agent: AgentId, pusher: AgentId) {
        let count = self.pushes.entry((agent, pusher)).or_default();
        if *count >= PUSH_LIMIT && candidates.len() > 1 {
            *count = 0;
            self.random.shuffle(candidates);
        }
    }

    /// Moves the vertex `agent` turned or waited towards last to the front, if still a
    /// candidate.
    fn request_first(&self, agent: AgentId, candidates: &mut Vec<Vertex>) {
        let Some(intent) = self.intent[agent] else {
            return;
        };

        if let Some(index) = candidates.iter().position(|&vertex| vertex == intent) {
            let vertex = candidates.remove(index);
            candidates.insert(0, vertex);
        }
    }
}

#[cfg(test)]
mod tests {
    //! PIBT*_T plans checked by a validator that knows only the graph's moves and turns.

    use super::*;

    use heuristic_search::Limits;
    use proptest::prelude::*;

    use crate::conflict::first_conflict;
    use crate::graph::{Graph, Grid};
    use crate::instance::Agent;

    /// Checks that a solved `run` is a valid plan for `instance` from `headings`, ending at
    /// the goals.
    fn assert_valid<G: OrientedGraph>(
        instance: &Instance<G>,
        headings: &[Heading],
        run: &TurnResult,
    ) {
        let solution = run.result.solution().expect("solved");
        assert_steps(instance, headings, &solution.paths, &run.headings);
        for (path, agent) in solution.paths.iter().zip(instance.agents()) {
            assert_eq!(path.last(), Some(&agent.goal));
        }
    }

    /// Checks the steps taken so far, `paths` and `turns` per agent, from the starts of
    /// `instance` facing `headings`: every step of every agent waits, turns once, or moves
    /// forward without turning; all lists have the same length; no two agents conflict.
    fn assert_steps<G: OrientedGraph>(
        instance: &Instance<G>,
        headings: &[Heading],
        paths: &[Path],
        turns: &[Vec<Heading>],
    ) {
        let graph = instance.graph();
        let steps = paths[0].len();

        assert_eq!(first_conflict(paths), None);
        for (agent, ((path, turns), start)) in
            paths.iter().zip(turns).zip(instance.agents()).enumerate()
        {
            assert_eq!(
                (path.len(), turns.len()),
                (steps, steps),
                "agent {agent}: lengths"
            );
            assert_eq!(path.first(), Some(&start.start));
            assert_eq!(turns[0], headings[agent]);
            for t in 1..steps {
                let (from, to, before, after) = (path[t - 1], path[t], turns[t - 1], turns[t]);
                if from == to {
                    let mut one_turn = false;
                    graph.turns(before, |next| one_turn |= next == after);
                    assert!(
                        after == before || one_turn,
                        "agent {agent} at {t}: not a single turn"
                    );
                } else {
                    assert_eq!(after, before, "agent {agent} at {t}: turned while moving");
                    assert_eq!(
                        graph.forward(from, before),
                        Some(to),
                        "agent {agent} at {t}: not forward"
                    );
                }
            }
        }
    }

    /// The four cells of a free 2x2 grid in clockwise order, each agent's goal the next one.
    fn carousel() -> Instance<Grid> {
        let agents = [(0, 1), (1, 3), (3, 2), (2, 0)]
            .map(|(start, goal)| Agent { start, goal })
            .to_vec();

        Instance::new(Grid::new(2, 2), agents).expect("valid")
    }

    /// Facing their goals, the four agents of a full carousel move at once: a cycle.
    #[test]
    fn aligned_cycle_moves_at_once() {
        let instance = carousel();
        let headings = [Grid::EAST, Grid::SOUTH, Grid::WEST, Grid::NORTH];

        let run = pibt_turns(&instance, &headings, 0, Limits::expansions(10));

        assert_valid(&instance, &headings, &run);
        assert_eq!(run.result.stats.expanded, 1);
    }

    /// All facing east, the carousel agents first turn towards their goals, then move.
    #[test]
    fn misaligned_cycle_turns_first() {
        let instance = carousel();
        let headings = [Grid::EAST; 4];

        let run = pibt_turns(&instance, &headings, 0, Limits::expansions(50));

        assert_valid(&instance, &headings, &run);
        assert!(run.result.stats.expanded > 1);
    }

    /// An agent facing away from its goal in a corridor must turn twice before moving.
    #[test]
    fn turns_before_moving() {
        let instance =
            Instance::new(Grid::new(3, 1), vec![Agent { start: 0, goal: 2 }]).expect("valid");

        let run = pibt_turns(&instance, &[Grid::WEST], 0, Limits::expansions(50));

        assert_valid(&instance, &[Grid::WEST], &run);
        assert_eq!(
            run.headings[0],
            [Grid::WEST, Grid::SOUTH, Grid::EAST, Grid::EAST, Grid::EAST]
        );
    }

    /// An unreachable goal is proven at once.
    #[test]
    fn proves_an_unreachable_goal() {
        let instance =
            Instance::new(Grid::parse(&[".@."]), vec![Agent { start: 0, goal: 2 }]).expect("valid");

        let run = pibt_turns(&instance, &[Grid::EAST], 0, Limits::NONE);

        assert_eq!(run.result.outcome, Outcome::Unsolvable);
        assert_eq!(run.headings, Vec::<Vec<Heading>>::new());
    }

    /// A missing heading is a caller error.
    #[test]
    #[should_panic(expected = "one heading per agent")]
    fn needs_a_heading_per_agent() {
        let _ = pibt_turns(&carousel(), &[Grid::EAST], 0, Limits::NONE);
    }

    /// A one-way ring with a single heading: an oriented graph that is not a grid.
    struct Ring(usize);

    impl Graph for Ring {
        fn vertex_count(&self) -> usize {
            self.0
        }

        fn neighbors(&self, vertex: Vertex, mut emit: impl FnMut(Vertex)) {
            emit((vertex + 1) % self.0);
        }
    }

    impl OrientedGraph for Ring {
        fn heading_count(&self) -> usize {
            1
        }

        fn forward(&self, vertex: Vertex, _: Heading) -> Option<Vertex> {
            Some((vertex + 1) % self.0)
        }

        fn turns(&self, _: Heading, _: impl FnMut(Heading)) {}
    }

    /// On a one-way ring, agents follow each other round to their goals.
    #[test]
    fn works_on_other_oriented_graphs() {
        let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 1, goal: 3 }];
        let instance = Instance::new(Ring(5), agents).expect("valid");

        let run = pibt_turns(&instance, &[0, 0], 0, Limits::expansions(50));

        assert_valid(&instance, &[0, 0], &run);
    }

    /// A grid of 3-4 by 3-4 cells with walls, 2-5 agents on distinct free starts and goals,
    /// random headings and a seed.
    fn case() -> impl Strategy<Value = (Instance<Grid>, Vec<Heading>, u64)> {
        (
            3_usize..=4,
            3_usize..=4,
            prop::collection::vec(prop::bool::weighted(0.15), 16),
            2_usize..=5,
            any::<u64>(),
        )
            .prop_flat_map(|(width, height, blocked, agents, seed)| {
                let free: Vec<Vertex> = (0..width * height).filter(|&v| !blocked[v]).collect();
                let headings = prop::collection::vec(0_usize..4, agents);
                (
                    Just((width, height, blocked, agents, seed)),
                    Just(free.clone()).prop_shuffle(),
                    Just(free).prop_shuffle(),
                    headings,
                )
            })
            .prop_filter_map(
                "enough free cells",
                |((width, height, blocked, agents, seed), starts, goals, headings)| {
                    if starts.len() < agents {
                        return None;
                    }

                    let mut grid = Grid::new(width, height);
                    for vertex in (0..width * height).filter(|&v| blocked[v]) {
                        let (x, y) = grid.cell(vertex);
                        grid.block(x, y);
                    }

                    let agents = (0..agents)
                        .map(|i| Agent {
                            start: starts[i],
                            goal: goals[i],
                        })
                        .collect();

                    Some((Instance::new(grid, agents).ok()?, headings, seed))
                },
            )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// Every step PIBT*_T takes is valid and conflict-free, solved or not; every plan
        /// it returns ends at the goals; and the same seed gives the same run.
        #[test]
        fn plans_are_valid((instance, headings, seed) in case()) {
            let costs = Costs::new(instance.graph(), instance.agents());
            let reachable = instance.agents().iter().enumerate().all(|(i, a)| costs.least(i, a.start).is_some());
            if reachable {
                let mut run = Run::new(&instance, costs, &headings, seed);
                for _ in 0..100 {
                    run.step();
                }

                assert_steps(&instance, &headings, &run.paths, &run.heading_paths);
            }

            let run = pibt_turns(&instance, &headings, seed, Limits::expansions(300));
            if run.result.solution().is_some() {
                assert_valid(&instance, &headings, &run);
            }

            prop_assert_eq!(&run, &pibt_turns(&instance, &headings, seed, Limits::expansions(300)));
        }
    }
}
