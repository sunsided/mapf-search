//! One M* search over the joint space of a group of agents ([`Planner`]).
//!
//! The top level plans every agent. In recursive M*, a state whose collision set holds a
//! group smaller than the planner's own agents moves that group along the group's optimal
//! joint path, which a [`Planner`] for that group computes on demand. That planner stays
//! alive for the whole search: its collision sets, predecessor sets and the optimal next
//! steps it found (with their exact costs to go) carry over to later queries from other
//! start states.
//!
//! With operator decomposition, a standard state where agents branch is expanded one agent
//! at a time: agent `k` of an intermediate state picks its move (all of them if it is in
//! the intermediate state's own collision set, else its policy move), and the last agent's
//! move yields a standard state. Collisions with the agents that moved earlier in the same
//! step are found at the intermediate state and passed back like any other.

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use heuristic_search::{Budget, Stats};

use super::collision::CollisionSet;
use super::context::{Context, LimitReached};
use super::state::{Choice, JointState, Key, collide};
use crate::graph::Graph;
use crate::instance::AgentId;

/// One generated state of a [`Planner`].
#[derive(Debug, Clone)]
struct Node {
    /// The state.
    key: Key,
    /// The heuristic; the exact cost to the goals where a query found it.
    h: u32,
    /// The agents that branch here.
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

/// An open-list entry: smallest `f` first, then larger `g`, then first in.
type Entry = Reverse<(u32, Reverse<u32>, u64, usize)>;

/// An M* search over the joint space of a group of agents, reusable across queries.
#[derive(Debug, Clone)]
pub(super) struct Planner {
    /// The planned agents, sorted; the order of every [`JointState`] here.
    agents: Vec<AgentId>,
    /// Every generated node, over all queries.
    nodes: Vec<Node>,
    /// The node of each generated state.
    index: BTreeMap<Key, usize>,
    /// The open nodes of the current query; may hold stale duplicates.
    open: BinaryHeap<Entry>,
    /// The current query; starts at 0, so no node belongs to a query before the first.
    query: u32,
    /// Push counter for first-in tie-breaking.
    pushes: u64,
    /// Every standard state's next standard state on an optimal path to the goals found so
    /// far, with the exact cost from the state to the goals; `None` where the goals were
    /// proven unreachable.
    forward: BTreeMap<Key, Option<(Key, u32)>>,
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

    /// The next standard state after `key` on an optimal path to the goals, from the cache
    /// or a new search; `None` if the goals are unreachable from `key`.
    ///
    /// # Errors
    ///
    /// [`LimitReached`] if the budget stopped the search.
    pub(super) fn policy<G: Graph>(
        &mut self,
        cx: &mut Context<'_, G>,
        key: Key,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<Key>, LimitReached> {
        if let Some(next) = self.forward.get(&key) {
            return Ok(next.as_ref().map(|(next, _)| next.clone()));
        }

        let Some(path) = self.search(cx, key.clone(), budget)? else {
            self.forward.insert(key, None);
            return Ok(None);
        };

        Ok(path.get(1).cloned())
    }

    /// The exact optimal cost from `key` to the goals, if known: `0` at a goal, else from an
    /// earlier query's optimal path through `key`.
    fn exact(&self, key: &Key) -> Option<u32> {
        if key.is_goal() {
            return Some(0);
        }

        self.forward.get(key)?.as_ref().map(|&(_, cost)| cost)
    }

    /// Ends the query at node `id`, whose exact cost to the goals `rest` is known: records
    /// the optimal path of standard states from the query's start to it in
    /// [`Planner::forward`] and returns the whole path of standard states to the goals.
    fn finish(&mut self, id: usize, rest: u32) -> Vec<Key> {
        let chain: Vec<usize> = core::iter::successors(Some(id), |&id| self.nodes[id].parent)
            .filter(|&id| self.nodes[id].key.is_standard())
            .collect();
        let total = self.nodes[id].g + rest;
        for pair in chain.windows(2) {
            let (next, here) = (pair[0], pair[1]);
            let to_go = total - self.nodes[here].g;
            let entry = (self.nodes[next].key.clone(), to_go);
            self.forward
                .insert(self.nodes[here].key.clone(), Some(entry));
            self.nodes[here].h = to_go;
        }

        let mut path: Vec<Key> = chain
            .iter()
            .rev()
            .map(|&id| self.nodes[id].key.clone())
            .collect();
        while let Some(Some((next, _))) = path.last().and_then(|last| self.forward.get(last)) {
            path.push(next.clone());
        }

        path
    }

    /// An optimal path of standard states from `start` until every agent finished; `None`
    /// if there is none.
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
        start: Key,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<Vec<Key>>, LimitReached> {
        self.query += 1;
        self.open.clear();
        let Some(root) = self.node(cx, start) else {
            return Ok(None);
        };

        self.touch(root);
        self.nodes[root].g = 0;
        self.push(root);
        while let Some(Reverse((_, Reverse(g), _, id))) = self.open.pop() {
            if self.is_stale(id, g) {
                continue;
            }

            if let Some(rest) = self.exact(&self.nodes[id].key) {
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
        if self.decomposes(cx, id) {
            self.expand_one(cx, id);
            return Ok(());
        }

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

    /// Returns `true` if node `id` is expanded one agent at a time: every intermediate
    /// state, and, with operator decomposition, a standard state where agents branch (in
    /// basic M*: any agent of its collision set; in recursive M*: a group of all agents).
    fn decomposes<G>(&self, cx: &Context<'_, G>, id: usize) -> bool {
        let node = &self.nodes[id];
        if !node.key.is_standard() {
            return true;
        }

        let groups = node.collision.groups();
        cx.decompose
            && if cx.recursive {
                groups.iter().any(|group| group.len() == self.agents.len())
            } else {
                !groups.is_empty()
            }
    }

    /// Moves the next agent of node `id` (the first one not moved yet): over all its
    /// choices if its collision set holds it, else along its policy. Collisions with the
    /// agents moved earlier in the step are passed back from here.
    fn expand_one<G: Graph>(&mut self, cx: &mut Context<'_, G>, id: usize) {
        let key = self.nodes[id].key.clone();
        let k = key.moved.len();
        let agent = self.agents[k];
        let branch = self.nodes[id].collision.contains(agent);
        let from = key.base[k];

        for choice in cx.choices(from, agent, branch, key.time) {
            let colliders: Vec<AgentId> = (0..k)
                .filter(|&j| collide((key.base[j].0, key.moved[j].0), (from.0, choice.next)))
                .map(|j| self.agents[j])
                .collect();
            if !colliders.is_empty() {
                for other in colliders {
                    self.pass_back(cx, id, &CollisionSet::of(vec![other, agent]));
                }

                continue;
            }

            let mut moved = key.moved.clone();
            moved.push(choice.position());
            let target = if moved.len() == self.agents.len() {
                Key::standard(cx.next_time(key.time), moved)
            } else {
                Key {
                    time: key.time,
                    base: key.base.clone(),
                    moved,
                }
            };
            self.link(cx, id, target, choice.cost);
        }
    }

    /// Every agent's choices at the standard node `id`, in planner order: full branching
    /// for the agents of a group that holds every agent of this planner (in basic M*: of
    /// any group), the group's optimal joint step for a smaller group, the individual
    /// policy for the rest. `None` if a group cannot reach its goals, so the node has no
    /// successors.
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
        let (time, state) = (node.key.time, &node.key.base);
        let mut options: Vec<Vec<Choice>> = self
            .agents
            .iter()
            .zip(state)
            .map(|(&agent, &at)| {
                let branch = !cx.recursive && node.collision.contains(agent);
                cx.choices(at, agent, branch, time)
            })
            .collect();
        if !cx.recursive {
            return Ok(Some(options));
        }

        for group in node.collision.groups() {
            let locals: Vec<usize> = group.iter().map(|&agent| self.local(agent)).collect();
            if group.len() == self.agents.len() {
                for (&agent, &local) in group.iter().zip(&locals) {
                    options[local] = cx.choices(state[local], agent, true, time);
                }

                continue;
            }

            let from: JointState = locals.iter().map(|&local| state[local]).collect();
            let Some(to) = cx.group_step(group, Key::standard(time, from.clone()), budget)? else {
                return Ok(None);
            };

            for ((&local, &before), &after) in locals.iter().zip(&from).zip(&to.base) {
                options[local] = vec![Choice::between(before, after)];
            }
        }

        Ok(Some(options))
    }

    /// Handles the joint step `step` out of the standard node `id`: passes collisions back,
    /// or links the target.
    fn step<G: Graph>(&mut self, cx: &mut Context<'_, G>, id: usize, step: &[Choice]) {
        let key = &self.nodes[id].key;
        let collisions = self.collisions(&key.base, step);
        if !collisions.is_empty() {
            for pair in collisions {
                self.pass_back(cx, id, &CollisionSet::of(pair));
            }

            return;
        }

        let target = Key::standard(
            cx.next_time(key.time),
            step.iter().map(|c| c.position()).collect(),
        );
        let cost = step.iter().map(|c| c.cost).sum();
        self.link(cx, id, target, cost);
    }

    /// Records node `id` as a predecessor of the state `target`, reached at `cost`, passes
    /// the target's collision set back and relaxes it. Nothing if the target is a dead end.
    fn link<G: Graph>(&mut self, cx: &mut Context<'_, G>, id: usize, target: Key, cost: u32) {
        let Some(target) = self.node(cx, target) else {
            return;
        };

        self.nodes[target].back.insert(id);
        let inherited = self.nodes[target].collision.clone();
        self.pass_back(cx, id, &inherited);

        self.touch(target);
        let g = self.nodes[id].g + cost;
        if g < self.nodes[target].g {
            self.nodes[target].g = g;
            self.nodes[target].parent = Some(id);
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
                if collide((state[a].0, choice_a.next), (state[b].0, choice_b.next)) {
                    pairs.push(vec![self.agents[a], self.agents[b]]);
                }
            }
        }

        pairs
    }

    /// The node of `key`, created if new; `None` if an agent cannot reach its goal (in
    /// time) from there.
    fn node<G: Graph>(&mut self, cx: &mut Context<'_, G>, key: Key) -> Option<usize> {
        if let Some(&id) = self.index.get(&key) {
            return Some(id);
        }

        let estimate = cx.heuristic(&self.agents, &key)?;
        let h = self.exact(&key).unwrap_or(estimate);
        let id = self.nodes.len();
        cx.stats.generated += 1;
        self.index.insert(key.clone(), id);
        self.nodes.push(Node {
            key,
            h,
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
        let key = (node.g + node.h, Reverse(node.g), self.pushes, id);
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
