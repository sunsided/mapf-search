//! EPEA* ([`epea`], Goldenberg et al. 2014): enhanced partial expansion A* in the joint
//! space of all agents, optimal for sum of costs or makespan ([`Objective`]).
//!
//! A* over joint states generates every combination of the agents' moves, `5^k` on a grid
//! for `k` agents, and most of those children never leave the open list. EPEA* generates
//! only the children whose `f` equals the stored value `F` of the expanded node, which
//! starts at the node's own `f`. The node is then put back with `F` raised to the next
//! larger `f` among its children, or closed if there is none. The children with a given
//! `f` come from an operator selection function ([`Operators`]) that works per agent: the
//! heuristic is the sum of the agents' distances, so each agent's move raises `f` by a
//! small amount of its own (`0`, `1` or `2` on an undirected graph), and the joint moves
//! with a given rise are the combinations that add up to it.
//!
//! Joint states and pricing are those of [`mstar`](crate::mstar()): every agent's vertex
//! plus a *finished* flag. An unfinished agent at its goal may finish, and stays there for
//! free from then on; every other unfinished agent pays 1 per step, move or wait. The
//! joint space is finite, so EPEA* proves unsolvability by exhausting it.
//!
//! **Makespan** is solved as by M*: the sum-of-costs plan's makespan bounds the optimum
//! from above, the largest individual distance from below, and a sum-of-costs search over
//! timed states under each deadline in between, pruning states from which an agent cannot
//! reach its goal in time, finds the first feasible one.
//!
//! ```mermaid
//! flowchart TD
//!     pop[pop the node with the lowest F] -->|open list empty| unsolvable[Unsolvable]
//!     pop -->|every agent finished| solved[Solved]
//!     pop --> budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| children[generate the collision-free children with f = F]
//!     children --> next{a larger f among the children?}
//!     next -->|yes| reinsert[F = that f, put the node back]
//!     next -->|no| close[close the node]
//!     reinsert --> pop
//!     close --> pop
//! ```

mod osf;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use heuristic_search::{Budget, Stats};

use self::osf::{Move, Operators};
use crate::graph::{Graph, Vertex};
use crate::instance::Instance;
use crate::low_level::{distance_tables, makespan_lower_bound};
use crate::path::{Objective, Path, Time, trim_trailing_waits};
use crate::solution::{MapfResult, Outcome, Solution, makespan_by_deadlines};

/// Finds a conflict-free plan with the optimal `objective` value with enhanced partial
/// expansion A* in the joint space.
///
/// The joint space is finite, so EPEA* proves unsolvability by exhausting it
/// ([`Outcome::Unsolvable`]), also where CBS and ICTS only stop at their budget. `budget`
/// is asked before every expansion; [`MapfResult::stats`] counts expansions (each partial
/// expansion of a node once) in [`Stats::expanded`] and generated children in
/// [`Stats::generated`], summed over the searches of the module docs.
///
/// For [`Objective::Makespan`] the plan has the optimal makespan and, among such plans, the
/// least sum of costs.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Instance, Limits, Objective, Outcome, epea};
///
/// // Two agents swap the ends of a corridor with one side pocket.
/// let grid = Grid::parse(&["...", "@.@"]);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 0), goal: grid.vertex(0, 0) },
/// ];
/// let instance = Instance::new(grid, agents)?;
/// let result = epea(&instance, Objective::SumOfCosts, Limits::NONE);
/// assert_eq!(result.solution().expect("solvable").sum_of_costs, 7);
///
/// // Without the pocket they cannot pass, and EPEA* proves it.
/// let corridor = Grid::parse(&["..."]);
/// let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
/// let instance = Instance::new(corridor, agents)?;
/// let result = epea(&instance, Objective::Makespan, Limits::NONE);
/// assert_eq!(result.outcome, Outcome::Unsolvable);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn epea<G: Graph>(
    instance: &Instance<G>,
    objective: Objective,
    mut budget: impl Budget<Stats>,
) -> MapfResult {
    let mut run = Run {
        instance,
        distances: distance_tables(instance.graph(), instance.agents()),
        stats: Stats::default(),
    };

    let outcome = match objective {
        Objective::SumOfCosts => run.search(None, &mut budget),
        Objective::Makespan => {
            let lower_bound = makespan_lower_bound(instance.agents(), &run.distances);
            makespan_by_deadlines(lower_bound, |deadline| run.search(deadline, &mut budget))
        }
    };

    MapfResult {
        outcome,
        stats: run.stats,
    }
}

/// One [`epea`] call: what its searches share, and the counters over all of them.
struct Run<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// Distances to each agent's goal, indexed by agent.
    distances: Vec<Vec<Option<u32>>>,
    /// Counters over every search so far.
    stats: Stats,
}

impl<G: Graph> Run<'_, G> {
    /// One search for the plan with the optimal sum of costs among those in which every
    /// agent finishes by `deadline` (any time if `None`).
    fn search(&mut self, deadline: Option<Time>, budget: &mut impl Budget<Stats>) -> Outcome {
        let mut search = Search {
            instance: self.instance,
            distances: &self.distances,
            deadline,
            nodes: Vec::new(),
            index: BTreeMap::new(),
            open: BinaryHeap::new(),
            stats: self.stats,
        };
        let outcome = search.run(budget);
        self.stats = search.stats;

        outcome
    }
}

/// A joint state: the time (always `0` without a deadline, where time does not matter),
/// then every agent's vertex and whether it has finished.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    /// The time step of the state; counted only under a deadline.
    time: Time,
    /// Every agent's vertex and finished flag, indexed by agent.
    agents: Vec<(Vertex, bool)>,
}

impl Key {
    /// Returns `true` if every agent has finished.
    fn is_goal(&self) -> bool {
        self.agents.iter().all(|&(_, finished)| finished)
    }
}

/// A node of the search: a joint state reached by its best known path.
struct Node {
    /// The joint state.
    key: Key,
    /// The cost of the best known path to the state.
    g: u32,
    /// The heuristic: the sum of the unfinished agents' distances to their goals.
    h: u32,
    /// How much above `g + h` the children generated next lie: `F - f`.
    rise: u32,
    /// The node the best known path comes from; `None` for the start.
    parent: Option<usize>,
}

/// An open entry: `F`, then `h` (deeper first), then the node id, then the node's rise and
/// `g` when it was opened.
type Entry = (u32, u32, usize, u32, u32);

/// One EPEA* search, with or without a deadline.
struct Search<'a, G> {
    /// The instance being solved.
    instance: &'a Instance<G>,
    /// Distances to each agent's goal, indexed by agent.
    distances: &'a [Vec<Option<u32>>],
    /// The step by which every agent must have finished, if any.
    deadline: Option<Time>,
    /// Every node generated so far, indexed by id.
    nodes: Vec<Node>,
    /// The id of each generated state.
    index: BTreeMap<Key, usize>,
    /// Open entries, smallest first. An entry whose rise or `g` no longer matches its node
    /// is stale and skipped.
    open: BinaryHeap<Reverse<Entry>>,
    /// Counters, carried over from the earlier searches of the run.
    stats: Stats,
}

impl<G: Graph> Search<'_, G> {
    /// Runs the search to its end.
    fn run(&mut self, budget: &mut impl Budget<Stats>) -> Outcome {
        let start = Key {
            time: 0,
            agents: self
                .instance
                .agents()
                .iter()
                .map(|a| (a.start, false))
                .collect(),
        };
        let Some(h) = self.heuristic(&start) else {
            return Outcome::Unsolvable;
        };

        self.insert(start, 0, h, None);

        while let Some(Reverse((_, _, id, rise, g))) = self.open.pop() {
            let node = &self.nodes[id];
            if node.rise != rise || node.g != g {
                continue;
            }

            if rise == 0 && node.key.is_goal() {
                return Outcome::Solved(Solution::new(self.paths(id)));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            self.expand(id);
        }

        Outcome::Unsolvable
    }

    /// Generates the children of node `id` that lie at its stored `F`, then raises `F` to
    /// the next larger `f` among its children and puts it back, if there is one.
    fn expand(&mut self, id: usize) {
        self.stats.expanded += 1;

        let operators = self.operators(&self.nodes[id].key);
        let (g, rise) = (self.nodes[id].g, self.nodes[id].rise);
        for joint in operators.with_rise(rise) {
            self.generate(id, g, &joint);
        }

        if let Some(next) = operators.next_rise(rise) {
            self.nodes[id].rise = next;
            self.push(id);
        }
    }

    /// Adds the child that `joint` leads to from node `parent` at cost `g`, unless it is
    /// the parent itself or known with a path at most as cheap.
    fn generate(&mut self, parent: usize, g: u32, joint: &[Move]) {
        let time = self.nodes[parent].key.time;
        let key = Key {
            time: if self.deadline.is_some() { time + 1 } else { 0 },
            agents: joint.iter().map(|step| (step.to, step.finished)).collect(),
        };
        let g = g + joint.iter().map(|step| step.cost).sum::<u32>();
        if key == self.nodes[parent].key {
            return;
        }

        self.stats.generated += 1;
        let h = self
            .heuristic(&key)
            .expect("moves keep every agent in time");
        self.insert(key, g, h, Some(parent));
    }

    /// Records `key` reached at cost `g` from `parent` and opens it, unless it is known
    /// with a path at most as cheap.
    fn insert(&mut self, key: Key, g: u32, h: u32, parent: Option<usize>) {
        if let Some(&id) = self.index.get(&key) {
            let node = &mut self.nodes[id];
            if node.g <= g {
                return;
            }

            node.g = g;
            node.rise = 0;
            node.parent = parent;
            self.stats.reopened += 1;
            self.push(id);
            return;
        }

        let id = self.nodes.len();
        self.index.insert(key.clone(), id);
        self.nodes.push(Node {
            key,
            g,
            h,
            rise: 0,
            parent,
        });
        self.push(id);
    }

    /// Opens node `id` at its stored `F`.
    fn push(&mut self, id: usize) {
        let node = &self.nodes[id];
        let f = node.g + node.h + node.rise;
        self.open.push(Reverse((f, node.h, id, node.rise, node.g)));
    }

    /// The operators of the joint state `key`: each agent's moves with their rises.
    fn operators(&self, key: &Key) -> Operators {
        let moves = key
            .agents
            .iter()
            .enumerate()
            .map(|(agent, &position)| self.moves(agent, position, key.time))
            .collect();

        Operators::new(moves)
    }

    /// The moves of `agent` at `(here, finished)` and `time`: staying finished; or every
    /// move and wait towards a vertex from which the goal is reachable (in time), plus
    /// finishing at the goal.
    fn moves(&self, agent: usize, (here, finished): (Vertex, bool), time: Time) -> Vec<Move> {
        let stay = Move {
            from: here,
            to: here,
            finished: true,
            cost: 0,
            rise: 0,
        };
        if finished {
            return Vec::from([stay]);
        }

        let distances = &self.distances[agent];
        let current = distances[here].expect("agents stay where the goal is reachable");
        let mut moves = Vec::new();
        let mut consider = |next: Vertex| {
            let Some(distance) = distances[next] else {
                return;
            };

            if self
                .deadline
                .is_none_or(|deadline| time + 1 + distance <= deadline)
            {
                let rise = (1 + distance)
                    .checked_sub(current)
                    .expect("a step lowers the distance by at most 1");
                moves.push(Move {
                    from: here,
                    to: next,
                    finished: false,
                    cost: 1,
                    rise,
                });
            }
        };
        consider(here);
        self.instance.graph().neighbors(here, &mut consider);
        if here == self.instance.agents()[agent].goal {
            moves.push(stay);
        }

        moves
    }

    /// The sum of the unfinished agents' distances to their goals in `key`; `None` if one
    /// cannot reach its goal (in time).
    ///
    /// Consistent: a step costs an unfinished agent 1 and lowers its distance by at most 1.
    fn heuristic(&self, key: &Key) -> Option<u32> {
        key.agents
            .iter()
            .zip(self.distances)
            .filter(|((_, finished), _)| !finished)
            .map(|(&(vertex, _), distances)| {
                let distance = distances[vertex]?;
                let late = self
                    .deadline
                    .is_some_and(|deadline| key.time + distance > deadline);

                (!late).then_some(distance)
            })
            .sum()
    }

    /// Each agent's path to node `id`, trimmed of trailing waits.
    fn paths(&self, id: usize) -> Vec<Path> {
        let mut states = Vec::new();
        let mut at = Some(id);
        while let Some(id) = at {
            states.push(&self.nodes[id].key);
            at = self.nodes[id].parent;
        }

        states.reverse();

        (0..self.instance.agents().len())
            .map(|agent| {
                let mut path: Path = states.iter().map(|state| state.agents[agent].0).collect();
                trim_trailing_waits(&mut path);

                path
            })
            .collect()
    }
}
