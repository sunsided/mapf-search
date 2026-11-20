//! M* ([`mstar`], Wagner and Choset 2015): A* in the joint space of all agents with
//! subdimensional expansion, optimal for sum of costs.
//!
//! Each agent follows its individual optimal policy (a fixed shortest path to its goal)
//! until it collides. Colliding agents join the *collision set* of the state where the
//! collision was found, and that set is passed back to every predecessor. Only agents in a
//! state's collision set branch over all their moves; the others keep following their
//! policy. So the search stays low-dimensional where agents do not interact.
//!
//! Goal-stay pricing, the same as an exact joint search: a joint state is every agent's
//! vertex plus a *finished* flag. An unfinished agent at its goal may finish; from then on
//! it stays there for free. Every other unfinished agent pays 1 per step, move or wait. So
//! waiting at the goal before leaving it again is paid for. The heuristic, the sum of the
//! unfinished agents' distances to their goals, is consistent.
//!
//! ```mermaid
//! flowchart TD
//!     pop[pop the state with the lowest f] -->|all agents finished| solved[Solved]
//!     pop -->|open list empty| unsolvable[Unsolvable]
//!     pop --> succ[successors: collision set branches, the rest follow their policy]
//!     succ -->|transition collides| back[add the colliders to the collision set, pass it back to predecessors, reopen them]
//!     succ -->|collision-free| relax[record predecessor, pass back its collision set, relax g]
//!     back --> pop
//!     relax --> pop
//! ```

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use heuristic_search::{Budget, Stats};

use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::low_level::distance_tables;
use crate::path::Path;
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds a conflict-free plan with the optimal sum of costs with M*.
///
/// The joint space is finite, so M* proves unsolvability by exhausting it
/// ([`Outcome::Unsolvable`]), also where CBS and ICTS only stop at their budget, such as
/// two agents swapping the ends of a corridor. `budget` is asked before every expansion;
/// [`MapfResult::stats`] counts expansions (re-expansions included) in
/// [`Stats::expanded`], generated joint states in [`Stats::generated`] and expansions after
/// a collision set grew in [`Stats::reopened`].
///
/// This is basic M*: collision sets are merged into one set per state, not recursive M*'s
/// independent subsets, so tightly coupled groups of more than a few agents are slow.
/// Makespan is not supported.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Outcome, mstar};
///
/// // Two agents swap the ends of a corridor with one side pocket.
/// let grid = Grid::parse(&["...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid.clone(), agents)?;
/// let result = mstar(&instance, Limits::NONE);
/// assert_eq!(result.solution().expect("solvable").sum_of_costs, 7);
///
/// // Without the pocket they cannot pass, and M* proves it.
/// let corridor = mapf_search::Grid::parse(&["..."]);
/// let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
/// let instance = Instance::new(corridor, agents)?;
/// assert_eq!(mstar(&instance, Limits::NONE).outcome, Outcome::Unsolvable);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn mstar<G: Graph>(instance: &Instance<G>, mut budget: impl Budget) -> MapfResult {
    let distances = distance_tables(instance.graph(), instance.agents());
    let mut search = Search {
        policy: policies(instance.graph(), &distances),
        instance,
        distances,
        nodes: Vec::new(),
        index: BTreeMap::new(),
        open: BinaryHeap::new(),
        stats: Stats::default(),
    };
    let outcome = search.run(&mut budget);

    MapfResult {
        outcome,
        stats: search.stats,
    }
}

/// A joint state: every agent's vertex and whether it has finished (stays there for good).
type JointState = Vec<(Vertex, bool)>;

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

/// One joint state the search has generated.
#[derive(Debug, Clone)]
struct Node {
    /// The joint state.
    state: JointState,
    /// The cheapest known cost from the start.
    g: u32,
    /// The heuristic: the unfinished agents' distances to their goals.
    h: u32,
    /// The predecessor on the cheapest known path; `None` for the start.
    parent: Option<usize>,
    /// The agents that branch over all moves here.
    collision: BTreeSet<AgentId>,
    /// Every node that generated this one, so collision sets can be passed back.
    back: BTreeSet<usize>,
    /// The `g` and collision-set size of the last expansion; a pop with the same values
    /// is a stale duplicate.
    expanded_at: Option<(u32, usize)>,
}

/// An open-list entry: smallest `f` first, then larger `g`, then first in.
type Entry = Reverse<(u32, Reverse<u32>, u64, usize)>;

/// The state of one M* run.
struct Search<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// `policy[agent][vertex]`: the agent's next vertex on its individual optimal path;
    /// `None` at its goal and where the goal is unreachable.
    policy: Vec<Vec<Option<Vertex>>>,
    /// Every generated node.
    nodes: Vec<Node>,
    /// The node of each generated joint state.
    index: BTreeMap<JointState, usize>,
    /// The open nodes; may hold stale duplicates.
    open: BinaryHeap<Entry>,
    /// Counters reported in [`MapfResult::stats`].
    stats: Stats,
}

impl<G: Graph> Search<'_, G> {
    /// Runs the search to its end.
    fn run(&mut self, budget: &mut impl Budget) -> Outcome {
        let start: JointState = self
            .instance
            .agents()
            .iter()
            .map(|a| (a.start, false))
            .collect();
        let Some(h) = self.heuristic(&start) else {
            return Outcome::Unsolvable;
        };

        let root = self.add_node(start, h);
        self.nodes[root].g = 0;
        self.push(root);
        while let Some(Reverse((_, Reverse(g), _, id))) = self.open.pop() {
            let node = &self.nodes[id];
            if g != node.g || node.expanded_at == Some((node.g, node.collision.len())) {
                continue;
            }

            if node.state.iter().all(|&(_, finished)| finished) {
                return Outcome::Solved(Solution::new(self.paths(id)));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.expand(id);
        }

        Outcome::Unsolvable
    }

    /// Generates the successors of node `id` under its collision set.
    fn expand(&mut self, id: usize) {
        self.stats.expanded += 1;
        if self.nodes[id].expanded_at.is_some() {
            self.stats.reopened += 1;
        }

        let node = &mut self.nodes[id];
        node.expanded_at = Some((node.g, node.collision.len()));
        let node = &self.nodes[id];
        let options: Vec<Vec<Choice>> = (0..node.state.len())
            .map(|agent| self.choices(&node.state, agent, node.collision.contains(&agent)))
            .collect();

        let mut picked = Vec::with_capacity(options.len());
        let mut steps = Vec::new();
        cartesian(&options, &mut picked, &mut steps);
        for step in steps {
            self.step(id, &step);
        }
    }

    /// Handles the joint step `step` out of node `id`: passes collisions back, or records
    /// `id` as a predecessor of the target and relaxes it.
    fn step(&mut self, id: usize, step: &[Choice]) {
        let colliders = colliders(&self.nodes[id].state, step);
        if !colliders.is_empty() {
            self.pass_back(id, &colliders);
            return;
        }

        let state: JointState = step.iter().map(|c| (c.next, c.finished)).collect();
        let target = if let Some(&target) = self.index.get(&state) {
            target
        } else {
            let h = self
                .heuristic(&state)
                .expect("choices keep the goal reachable");
            self.add_node(state, h)
        };

        self.nodes[target].back.insert(id);
        let inherited = self.nodes[target].collision.clone();
        self.pass_back(id, &inherited);

        let g = self.nodes[id].g + step.iter().map(|c| c.cost).sum::<u32>();
        if g < self.nodes[target].g {
            self.nodes[target].g = g;
            self.nodes[target].parent = Some(id);
            self.push(target);
        }
    }

    /// Merges `agents` into the collision set of node `id` and, wherever a set grows,
    /// reopens that node and passes its set on to the node's predecessors.
    fn pass_back(&mut self, id: usize, agents: &BTreeSet<AgentId>) {
        let mut work = vec![(id, agents.clone())];
        while let Some((id, agents)) = work.pop() {
            let node = &mut self.nodes[id];
            if agents.is_subset(&node.collision) {
                continue;
            }

            node.collision.extend(agents);
            let merged = node.collision.clone();
            work.extend(node.back.iter().map(|&back| (back, merged.clone())));
            self.push(id);
        }
    }

    /// The choices of `agent` in `state`: everything if it is in the collision set
    /// (`branch`), else its policy move.
    fn choices(&self, state: &JointState, agent: AgentId, branch: bool) -> Vec<Choice> {
        let (here, finished) = state[agent];
        let finish = Choice {
            next: here,
            finished: true,
            cost: 0,
        };
        let move_to = |next| Choice {
            next,
            finished: false,
            cost: 1,
        };
        let at_goal = here == self.instance.agents()[agent].goal;
        if finished {
            return vec![finish];
        }

        if !branch {
            if at_goal {
                return vec![finish];
            }

            let next = self.policy[agent][here].expect("agents stay where the goal is reachable");
            return vec![move_to(next)];
        }

        let mut choices = vec![move_to(here)];
        self.instance.graph().neighbors(here, |next| {
            if self.distances[agent][next].is_some() {
                choices.push(move_to(next));
            }
        });
        if at_goal {
            choices.push(finish);
        }

        choices
    }

    /// The sum of the unfinished agents' distances to their goals; `None` if one cannot
    /// reach its goal.
    fn heuristic(&self, state: &JointState) -> Option<u32> {
        state
            .iter()
            .zip(&self.distances)
            .filter(|((_, finished), _)| !finished)
            .map(|(&(vertex, _), distances)| distances[vertex])
            .sum()
    }

    /// Stores a new node for `state` with heuristic `h`; `g` is infinite until relaxed.
    fn add_node(&mut self, state: JointState, h: u32) -> usize {
        let id = self.nodes.len();
        self.stats.generated += 1;
        self.index.insert(state.clone(), id);
        self.nodes.push(Node {
            state,
            g: u32::MAX,
            h,
            parent: None,
            collision: BTreeSet::new(),
            back: BTreeSet::new(),
            expanded_at: None,
        });

        id
    }

    /// Opens node `id` at its current `g`.
    fn push(&mut self, id: usize) {
        let node = &self.nodes[id];
        let key = (node.g + node.h, Reverse(node.g), self.stats.generated, id);
        self.open.push(Reverse(key));
    }

    /// The agents' paths along the parent chain to node `id`, trimmed of trailing waits.
    fn paths(&self, id: usize) -> Vec<Path> {
        let mut states: Vec<&JointState> =
            core::iter::successors(Some(id), |&id| self.nodes[id].parent)
                .map(|id| &self.nodes[id].state)
                .collect();
        states.reverse();

        (0..self.instance.agents().len())
            .map(|agent| {
                let mut path: Path = states.iter().map(|state| state[agent].0).collect();
                while path.len() > 1 && path[path.len() - 2] == path[path.len() - 1] {
                    path.pop();
                }

                path
            })
            .collect()
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

/// The agents that collide in the joint step `step` from `state`: two agents at the same
/// vertex afterwards, or two agents swapping along an edge.
fn colliders(state: &JointState, step: &[Choice]) -> BTreeSet<AgentId> {
    let mut agents = BTreeSet::new();
    for (a, choice_a) in step.iter().enumerate() {
        for (b, choice_b) in step.iter().enumerate().skip(a + 1) {
            let (from_a, from_b) = (state[a].0, state[b].0);
            let same = choice_a.next == choice_b.next;
            let swap =
                from_a != choice_a.next && from_a == choice_b.next && from_b == choice_a.next;
            if same || swap {
                agents.extend([a, b]);
            }
        }
    }

    agents
}

/// Each agent's individual optimal policy: from every vertex, the smallest-numbered
/// neighbour one step closer to its goal.
fn policies<G: Graph>(graph: &G, distances: &[Vec<Option<u32>>]) -> Vec<Vec<Option<Vertex>>> {
    distances
        .iter()
        .map(|distances| {
            (0..graph.vertex_count())
                .map(|vertex| {
                    let closer = distances[vertex]?.checked_sub(1)?;
                    let mut best = None;
                    graph.neighbors(vertex, |next| {
                        if distances[next] == Some(closer) {
                            best = Some(best.map_or(next, |b: Vertex| b.min(next)));
                        }
                    });

                    best
                })
                .collect()
        })
        .collect()
}
