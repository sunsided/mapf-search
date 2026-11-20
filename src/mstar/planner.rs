//! One M* search over the joint space of a group of agents ([`Planner`]), and what all
//! searches of one run share ([`Context`]).
//!
//! The top level plans every agent. In recursive M*, a state whose collision set holds a
//! group smaller than the planner's own agents moves that group along the group's optimal
//! joint path, which a [`Planner`] for that group computes on demand. That planner stays
//! alive for the whole run: its collision sets, predecessor sets and the optimal next steps
//! it found ([`Planner::forward`], with their exact costs to go) carry over to later queries from other start states.

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use heuristic_search::{Budget, Stats};

use super::collision::CollisionSet;
use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::path::Objective;

/// A joint state of a planner's agents, in the planner's agent order: every agent's vertex
/// and whether it has finished (stays there for good).
pub(super) type JointState = Vec<(Vertex, bool)>;

/// The budget stopped the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LimitReached;

/// One agent's choice for one joint step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Choice {
    /// Where the agent is after the step.
    next: Vertex,
    /// Whether the agent is finished after the step.
    finished: bool,
    /// What the step costs the agent: 1 unless it is finished after it.
    cost: u32,
}

impl Choice {
    /// Finishing, or staying finished, at `vertex`; free.
    fn finish(vertex: Vertex) -> Self {
        Self {
            next: vertex,
            finished: true,
            cost: 0,
        }
    }

    /// A move to `next`, or a wait if it is the current vertex; costs 1.
    fn move_to(next: Vertex) -> Self {
        Self {
            next,
            finished: false,
            cost: 1,
        }
    }

    /// The step from `from` to `to`, finished or not.
    fn between(from: (Vertex, bool), to: (Vertex, bool)) -> Self {
        debug_assert!(!from.1 || to == from, "finished agents stay finished");

        if to.1 {
            Self::finish(to.0)
        } else {
            Self::move_to(to.0)
        }
    }
}

/// What every planner of one run shares: the instance, the individual policies, the
/// planners of the subgroups and the counters.
pub(super) struct Context<'a, G> {
    /// The instance being solved.
    pub(super) instance: &'a Instance<G>,
    /// What the search minimizes: how agents' step costs combine, and the heuristic.
    pub(super) objective: Objective,
    /// Recursive M*: keep collision groups apart and plan each one on its own. `false` is
    /// basic M*.
    pub(super) recursive: bool,
    /// Distances to each agent's goal, indexed by agent.
    pub(super) distances: Vec<Vec<Option<u32>>>,
    /// `policy[agent][vertex]`: the agent's next vertex on its individual optimal path;
    /// `None` at its goal and where the goal is unreachable.
    pub(super) policy: Vec<Vec<Option<Vertex>>>,
    /// The planner of every subgroup queried so far, by its sorted agents. A planner is
    /// taken out while it runs.
    pub(super) planners: BTreeMap<Vec<AgentId>, Planner>,
    /// Counters over all planners, reported in the result.
    pub(super) stats: Stats,
}

impl<G: Graph> Context<'_, G> {
    /// The next joint state of `group` from `state` on an optimal joint path of the group
    /// alone, `None` if the group cannot reach its goals from there.
    ///
    /// # Errors
    ///
    /// [`LimitReached`] if the budget stopped the group's search.
    fn group_step(
        &mut self,
        group: &[AgentId],
        state: JointState,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<JointState>, LimitReached> {
        if state.iter().all(|&(_, finished)| finished) {
            return Ok(Some(state));
        }

        let mut planner = self
            .planners
            .remove(group)
            .unwrap_or_else(|| Planner::new(group.to_vec()));
        let step = planner.policy(self, state, budget);
        self.planners.insert(group.to_vec(), planner);

        step
    }

    /// The choices of `agent` in `state`: every move and wait towards a vertex that can still
    /// reach its goal, plus finishing at the goal, if `branch`; else its policy move.
    fn choices(
        &self,
        (here, finished): (Vertex, bool),
        agent: AgentId,
        branch: bool,
    ) -> Vec<Choice> {
        let at_goal = here == self.instance.agents()[agent].goal;
        if finished || (at_goal && !branch) {
            return vec![Choice::finish(here)];
        }

        if !branch {
            let next = self.policy[agent][here].expect("agents stay where the goal is reachable");
            return vec![Choice::move_to(next)];
        }

        let mut choices = vec![Choice::move_to(here)];
        self.instance.graph().neighbors(here, |next| {
            if self.distances[agent][next].is_some() {
                choices.push(Choice::move_to(next));
            }
        });
        if at_goal {
            choices.push(Choice::finish(here));
        }

        choices
    }

    /// The cost of a joint step whose agents pay `costs`: their sum for sum of costs, 1 if
    /// any agent pays for makespan.
    fn step_cost(&self, mut costs: impl Iterator<Item = u32>) -> u32 {
        match self.objective {
            Objective::SumOfCosts => costs.sum(),
            Objective::Makespan => u32::from(costs.any(|cost| cost > 0)),
        }
    }

    /// The heuristic of `state` (over `agents`) and its sum-of-costs estimate: the sum of
    /// the unfinished agents' distances to their goals is both for sum of costs; for
    /// makespan the heuristic is their maximum. `None` if one cannot reach its goal.
    fn heuristic(&self, agents: &[AgentId], state: &JointState) -> Option<(u32, u32)> {
        let (sum, max) = agents
            .iter()
            .zip(state)
            .filter(|(_, (_, finished))| !finished)
            .try_fold((0, 0), |(sum, max): (u32, u32), (&agent, &(vertex, _))| {
                let distance = self.distances[agent][vertex]?;
                Some((sum + distance, max.max(distance)))
            })?;

        match self.objective {
            Objective::SumOfCosts => Some((sum, sum)),
            Objective::Makespan => Some((max, sum)),
        }
    }
}

/// One generated joint state of a [`Planner`].
#[derive(Debug, Clone)]
struct Node {
    /// The joint state.
    state: JointState,
    /// The heuristic; the exact cost to the goals where a query found it.
    h: u32,
    /// The sum of the unfinished agents' distances to their goals; with
    /// [`Node::soc_g`] a tie-breaker that steers makespan searches through their wide
    /// plateaus of equal `f`.
    soc_h: u32,
    /// The sum of the agents' step costs along the path that set [`Node::g`].
    soc_g: u32,
    /// The agents that are planned together here.
    collision: CollisionSet,
    /// Bumped whenever [`Node::collision`] changes; part of the stale-pop check.
    version: u32,
    /// Every node that generated this one, so collision sets can be passed back.
    back: BTreeSet<usize>,
    /// The query that [`Node::g`], [`Node::parent`] and [`Node::expanded_at`] belong to;
    /// they are void for any other.
    query: u32,
    /// The cheapest known cost from the query's start.
    g: u32,
    /// The predecessor on the cheapest known path; `None` for the start.
    parent: Option<usize>,
    /// The `g` and collision-set version of the last expansion in the query; a pop with the
    /// same values is a stale duplicate.
    expanded_at: Option<(u32, u32)>,
}

/// An open-list entry: smallest `f` first, then the smallest sum-of-costs estimate
/// ([`Node::soc_g`] plus [`Node::soc_h`]), then larger `g`, then first in.
type Entry = Reverse<(u32, u32, Reverse<u32>, u64, usize)>;

/// An M* search over the joint space of a group of agents, reusable across queries.
#[derive(Debug, Clone)]
pub(super) struct Planner {
    /// The planned agents, sorted; the order of every [`JointState`] here.
    agents: Vec<AgentId>,
    /// Every generated node, over all queries.
    nodes: Vec<Node>,
    /// The node of each generated joint state.
    index: BTreeMap<JointState, usize>,
    /// The open nodes of the current query; may hold stale duplicates.
    open: BinaryHeap<Entry>,
    /// The current query; starts at 0, so no node belongs to a query before the first.
    query: u32,
    /// Push counter for first-in tie-breaking.
    pushes: u64,
    /// Every state's next state on an optimal path to the goals found so far, with the exact
    /// cost from the state to the goals; `None` where the goals were proven unreachable.
    forward: BTreeMap<JointState, Option<(JointState, u32)>>,
}

impl Planner {
    /// A planner for the sorted `agents`.
    pub(super) fn new(agents: Vec<AgentId>) -> Self {
        Self {
            agents,
            nodes: Vec::new(),
            index: BTreeMap::new(),
            open: BinaryHeap::new(),
            query: 0,
            pushes: 0,
            forward: BTreeMap::new(),
        }
    }

    /// The next state after `state` on an optimal path to the goals, from the cache or a new
    /// search; `None` if the goals are unreachable from `state`.
    ///
    /// # Errors
    ///
    /// [`LimitReached`] if the budget stopped the search.
    fn policy<G: Graph>(
        &mut self,
        cx: &mut Context<'_, G>,
        state: JointState,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<JointState>, LimitReached> {
        if let Some(next) = self.forward.get(&state) {
            return Ok(next.as_ref().map(|(next, _)| next.clone()));
        }

        let Some(path) = self.search(cx, state.clone(), budget)? else {
            self.forward.insert(state, None);
            return Ok(None);
        };

        Ok(path.get(1).cloned())
    }

    /// The exact optimal cost from `state` to the goals, if known: `0` once every agent
    /// finished, else from an earlier query's optimal path through `state`.
    fn exact(&self, state: &JointState) -> Option<u32> {
        if state.iter().all(|&(_, finished)| finished) {
            return Some(0);
        }

        self.forward.get(state)?.as_ref().map(|&(_, cost)| cost)
    }

    /// Ends the query at node `id`, whose exact cost to the goals `rest` is known: records
    /// the optimal path from the query's start to it in [`Planner::forward`] and returns
    /// the whole path to the goals.
    fn finish(&mut self, id: usize, rest: u32) -> Vec<JointState> {
        let chain: Vec<usize> =
            core::iter::successors(Some(id), |&id| self.nodes[id].parent).collect();
        let total = self.nodes[id].g + rest;
        for pair in chain.windows(2) {
            let (next, here) = (pair[0], pair[1]);
            let to_go = total - self.nodes[here].g;
            let entry = (self.nodes[next].state.clone(), to_go);
            self.forward
                .insert(self.nodes[here].state.clone(), Some(entry));
            self.nodes[here].h = to_go;
        }

        let mut path: Vec<JointState> = chain
            .iter()
            .rev()
            .map(|&id| self.nodes[id].state.clone())
            .collect();
        while let Some(Some((next, _))) = path.last().and_then(|last| self.forward.get(last)) {
            path.push(next.clone());
        }

        path
    }

    /// An optimal path of joint states from `start` until every agent finished; `None` if
    /// there is none.
    ///
    /// The search ends early at a state whose exact cost to the goals an earlier query
    /// found: its `f` is exact, and it is the lowest on the open list, so no cheaper path
    /// exists. Such states carry the exact cost as their heuristic, which keeps it
    /// admissible.
    ///
    /// # Errors
    ///
    /// [`LimitReached`] if the budget stopped the search.
    pub(super) fn search<G: Graph>(
        &mut self,
        cx: &mut Context<'_, G>,
        start: JointState,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<Vec<JointState>>, LimitReached> {
        self.query += 1;
        self.open.clear();
        let Some(root) = self.node(cx, start) else {
            return Ok(None);
        };

        self.touch(root);
        self.nodes[root].g = 0;
        self.nodes[root].soc_g = 0;
        self.push(root);
        while let Some(Reverse((_, _, Reverse(g), _, id))) = self.open.pop() {
            if self.is_stale(id, g) {
                continue;
            }

            if let Some(rest) = self.exact(&self.nodes[id].state) {
                return Ok(Some(self.finish(id, rest)));
            }

            if budget.exhausted(&cx.stats) {
                return Err(LimitReached);
            }

            self.expand(cx, id, budget)?;
        }

        Ok(None)
    }

    /// Returns `true` if the pop of node `id` at `g` is outdated: the node has a cheaper
    /// path now, or was expanded with this `g` and collision set already.
    fn is_stale(&self, id: usize, g: u32) -> bool {
        let node = &self.nodes[id];

        g != node.g || node.expanded_at == Some((node.g, node.version))
    }

    /// Generates the successors of node `id` under its collision set.
    fn expand<G: Graph>(
        &mut self,
        cx: &mut Context<'_, G>,
        id: usize,
        budget: &mut impl Budget<Stats>,
    ) -> Result<(), LimitReached> {
        cx.stats.expanded += 1;
        let node = &mut self.nodes[id];
        if node.expanded_at.is_some() {
            cx.stats.reopened += 1;
        }

        node.expanded_at = Some((node.g, node.version));
        let Some(options) = self.options(cx, id, budget)? else {
            return Ok(());
        };

        let mut steps = Vec::new();
        cartesian(&options, &mut Vec::with_capacity(options.len()), &mut steps);
        for step in steps {
            self.step(cx, id, &step);
        }

        Ok(())
    }

    /// Every agent's choices at node `id`, in planner order: full branching for the agents
    /// of a group that holds every agent of this planner (in basic M*: of any group), the
    /// group's optimal joint step for a smaller group, the individual policy for the rest.
    /// `None` if a group cannot reach its goals, so the node has no successors.
    ///
    /// # Errors
    ///
    /// [`LimitReached`] if the budget stopped a group's search.
    fn options<G: Graph>(
        &self,
        cx: &mut Context<'_, G>,
        id: usize,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<Vec<Vec<Choice>>>, LimitReached> {
        let node = &self.nodes[id];
        let mut options: Vec<Vec<Choice>> = self
            .agents
            .iter()
            .zip(&node.state)
            .map(|(&agent, &at)| {
                let branch = !cx.recursive && node.collision.contains(agent);
                cx.choices(at, agent, branch)
            })
            .collect();
        if !cx.recursive {
            return Ok(Some(options));
        }

        for group in node.collision.groups() {
            let locals: Vec<usize> = group.iter().map(|&agent| self.local(agent)).collect();
            if group.len() == self.agents.len() {
                for (&agent, &local) in group.iter().zip(&locals) {
                    options[local] = cx.choices(node.state[local], agent, true);
                }

                continue;
            }

            let from: JointState = locals.iter().map(|&local| node.state[local]).collect();
            let Some(to) = cx.group_step(group, from.clone(), budget)? else {
                return Ok(None);
            };

            for ((&local, &before), &after) in locals.iter().zip(&from).zip(&to) {
                options[local] = vec![Choice::between(before, after)];
            }
        }

        Ok(Some(options))
    }

    /// Handles the joint step `step` out of node `id`: passes collisions back, or records
    /// `id` as a predecessor of the target and relaxes it.
    fn step<G: Graph>(&mut self, cx: &mut Context<'_, G>, id: usize, step: &[Choice]) {
        let collisions = self.collisions(&self.nodes[id].state, step);
        if !collisions.is_empty() {
            for pair in collisions {
                self.pass_back(cx, id, &CollisionSet::of(pair));
            }

            return;
        }

        let state: JointState = step.iter().map(|c| (c.next, c.finished)).collect();
        let target = self
            .node(cx, state)
            .expect("choices keep the goals reachable");

        self.nodes[target].back.insert(id);
        let inherited = self.nodes[target].collision.clone();
        self.pass_back(cx, id, &inherited);

        self.touch(target);
        let g = self.nodes[id].g + cx.step_cost(step.iter().map(|c| c.cost));
        if g < self.nodes[target].g {
            let soc_g = self.nodes[id].soc_g + step.iter().map(|c| c.cost).sum::<u32>();
            let node = &mut self.nodes[target];
            node.g = g;
            node.soc_g = soc_g;
            node.parent = Some(id);
            self.push(target);
        }
    }

    /// Merges `set` into the collision set of node `id` and, wherever a set grows, reopens
    /// that node (if the current query reached it) and passes its set on to the node's
    /// predecessors.
    fn pass_back<G: Graph>(&mut self, cx: &Context<'_, G>, id: usize, set: &CollisionSet) {
        let single = !cx.recursive;
        let mut work = vec![(id, set.clone())];
        while let Some((id, set)) = work.pop() {
            let node = &mut self.nodes[id];
            if !node.collision.merge_all(&set, single) {
                continue;
            }

            node.version += 1;
            let merged = node.collision.clone();
            work.extend(node.back.iter().map(|&back| (back, merged.clone())));
            if node.query == self.query && node.g != u32::MAX {
                self.push(id);
            }
        }
    }

    /// The pairs of agents (global ids, smaller first) that collide in the joint step `step`
    /// from `state`: at the same vertex afterwards, or swapping along an edge.
    fn collisions(&self, state: &JointState, step: &[Choice]) -> Vec<Vec<AgentId>> {
        let mut pairs = Vec::new();
        for (a, choice_a) in step.iter().enumerate() {
            for (b, choice_b) in step.iter().enumerate().skip(a + 1) {
                let (from_a, from_b) = (state[a].0, state[b].0);
                let same = choice_a.next == choice_b.next;
                let swap =
                    from_a != choice_a.next && from_a == choice_b.next && from_b == choice_a.next;
                if same || swap {
                    pairs.push(vec![self.agents[a], self.agents[b]]);
                }
            }
        }

        pairs
    }

    /// The node of `state`, created if new; `None` if an agent cannot reach its goal from
    /// there.
    fn node<G: Graph>(&mut self, cx: &mut Context<'_, G>, state: JointState) -> Option<usize> {
        if let Some(&id) = self.index.get(&state) {
            return Some(id);
        }

        let (estimate, soc_h) = cx.heuristic(&self.agents, &state)?;
        let h = self.exact(&state).unwrap_or(estimate);
        let id = self.nodes.len();
        cx.stats.generated += 1;
        self.index.insert(state.clone(), id);
        self.nodes.push(Node {
            state,
            h,
            soc_h,
            soc_g: 0,
            collision: CollisionSet::default(),
            version: 0,
            back: BTreeSet::new(),
            query: 0,
            g: u32::MAX,
            parent: None,
            expanded_at: None,
        });

        Some(id)
    }

    /// Makes node `id` part of the current query, with no known path yet, if it is not.
    fn touch(&mut self, id: usize) {
        let node = &mut self.nodes[id];
        if node.query != self.query {
            node.query = self.query;
            node.g = u32::MAX;
            node.parent = None;
            node.expanded_at = None;
        }
    }

    /// Opens node `id` at its current `g`.
    fn push(&mut self, id: usize) {
        let node = &self.nodes[id];
        let key = (
            node.g + node.h,
            node.soc_g + node.soc_h,
            Reverse(node.g),
            self.pushes,
            id,
        );
        self.pushes += 1;
        self.open.push(Reverse(key));
    }

    /// The position of the global `agent` in this planner's joint states.
    fn local(&self, agent: AgentId) -> usize {
        self.agents
            .binary_search(&agent)
            .expect("collision groups hold the planner's agents")
    }
}

/// Every combination of one choice per agent from `options`, appended to `out`.
fn cartesian(options: &[Vec<Choice>], picked: &mut Vec<Choice>, out: &mut Vec<Vec<Choice>>) {
    let Some(choices) = options.get(picked.len()) else {
        out.push(picked.clone());
        return;
    };

    for &choice in choices {
        picked.push(choice);
        cartesian(options, picked, out);
        picked.pop();
    }
}
