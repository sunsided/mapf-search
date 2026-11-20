//! ECBS's low level: focal search over `(vertex, time)` states ([`focal_plan`]), with the
//! number of conflicts with other agents' paths ([`ConflictTable`]) as the focal key.
//!
//! The state includes the time, so a state's path cost `g` is its time and never improves;
//! no state is reopened, and `f_min` stays a valid lower bound on the agent's optimal cost.
//! The horizon is the one of the A* low level ([`crate::space_time_astar`]).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use heuristic_search::Weight;

use crate::constraint::ConstraintSet;
use crate::focal::{FocalList, Handle};
use crate::graph::{Graph, Vertex};
use crate::instance::{Agent, AgentId};
use crate::path::{Path, Time};

/// Where the other agents are, for counting the conflicts of one agent's moves.
#[derive(Debug, Clone, Default)]
pub(super) struct ConflictTable {
    /// How many other agents occupy a vertex at a time, before their final arrival.
    visits: BTreeMap<(Time, Vertex), u32>,
    /// The vertices other agents stay at for good, with the time they arrive there.
    parked: BTreeMap<Vertex, Time>,
    /// How many other agents move `from -> to` between a time and the next, as
    /// `(time, from, to)`; waits are not recorded.
    moves: BTreeMap<(Time, Vertex, Vertex), u32>,
}

impl ConflictTable {
    /// The table of every path in `paths`, indexed by agent, except that of `agent`.
    pub(super) fn without(paths: &[Path], agent: AgentId) -> Self {
        let mut table = Self::default();
        for (_, path) in paths
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != agent)
        {
            table.add(path);
        }

        table
    }

    /// Records `path`.
    fn add(&mut self, path: &[Vertex]) {
        let last = path.len() - 1;
        let time = |index: usize| Time::try_from(index).expect("path lengths fit in u32");

        for (index, &vertex) in path[..last].iter().enumerate() {
            *self.visits.entry((time(index), vertex)).or_default() += 1;
        }

        for (index, step) in path.windows(2).enumerate() {
            if step[0] != step[1] {
                *self
                    .moves
                    .entry((time(index), step[0], step[1]))
                    .or_default() += 1;
            }
        }

        self.parked.insert(path[last], time(last));
    }

    /// The number of conflicts of the move `from -> to` (a wait if equal) between `time`
    /// and `time + 1`: vertex conflicts at `to` and swaps with other agents.
    fn count(&self, from: Vertex, to: Vertex, time: Time) -> u32 {
        let next = time + 1;
        let visits = self.visits.get(&(next, to)).copied().unwrap_or(0);
        let parked = u32::from(self.parked.get(&to).is_some_and(|&since| since <= next));
        let swaps = if from == to {
            0
        } else {
            self.moves.get(&(time, to, from)).copied().unwrap_or(0)
        };

        visits + parked + swaps
    }
}

/// What [`focal_plan`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FocalPath {
    /// The path; it costs at most `w` times [`FocalPath::lower_bound`].
    pub(super) path: Path,
    /// A lower bound on the cost of the agent's cheapest path under its constraints.
    pub(super) lower_bound: u32,
}

/// One state of the focal search.
#[derive(Debug, Clone)]
struct Node {
    /// The vertex.
    vertex: Vertex,
    /// The time, which is also the path cost `g`.
    time: Time,
    /// The conflicts accumulated on the best known path here.
    conflicts: u32,
    /// The index of the predecessor in the node arena; `None` for the start.
    parent: Option<usize>,
    /// Where the node sits in the focal list; `None` once expanded.
    handle: Option<Handle>,
}

/// The single-agent problem [`focal_plan`] solves.
pub(super) struct Problem<'a, G> {
    /// The graph.
    pub(super) graph: &'a G,
    /// The agent.
    pub(super) agent: Agent,
    /// Its constraints.
    pub(super) constraints: &'a ConstraintSet,
    /// Distances to its goal (`distances_to`).
    pub(super) distances: &'a [Option<u32>],
    /// The other agents' paths.
    pub(super) table: &'a ConflictTable,
}

/// A path of the agent that satisfies its constraints and ends at its goal for good, at
/// most `weight` times as long as its cheapest such path, preferring few conflicts with
/// `problem.table`. `None` if the agent has no path.
///
/// Focal key: the accumulated conflicts, then `f`, then larger `g`.
pub(super) fn focal_plan<G: Graph>(problem: &Problem<'_, G>, weight: Weight) -> Option<FocalPath> {
    let Problem {
        graph,
        agent,
        constraints,
        distances,
        table,
    } = *problem;
    if !constraints.allows_vertex(agent.start, 0) {
        return None;
    }

    let earliest_finish = constraints.blocked_until(agent.goal).map_or(0, |t| t + 1);
    let vertices = Time::try_from(graph.vertex_count()).unwrap_or(Time::MAX);
    let horizon = constraints.latest_step().saturating_add(vertices);
    let heuristic = |vertex: Vertex, time: Time| {
        let distance = distances[vertex]?;
        Some(distance.max(earliest_finish.saturating_sub(time)))
    };

    let mut search = Search {
        nodes: Vec::new(),
        index: BTreeMap::new(),
        list: FocalList::new(weight),
    };
    search.open(agent.start, 0, 0, None, heuristic(agent.start, 0)?);

    while let Some((current, lower_bound)) = search.list.pop() {
        search.nodes[current].handle = None;
        let Node { vertex, time, .. } = search.nodes[current];
        if vertex == agent.goal && time >= earliest_finish {
            return Some(FocalPath {
                path: search.path(current),
                lower_bound,
            });
        }

        if time >= horizon {
            continue;
        }

        let mut step = |next: Vertex| {
            if !constraints.allows(vertex, next, time) {
                return;
            }

            if let Some(h) = heuristic(next, time + 1) {
                let conflicts = search.nodes[current].conflicts + table.count(vertex, next, time);
                search.relax(next, time + 1, conflicts, current, h);
            }
        };
        step(vertex);
        graph.neighbors(vertex, step);
    }

    None
}

/// The node arena and focal list of one [`focal_plan`] run.
struct Search {
    /// Every generated node.
    nodes: Vec<Node>,
    /// The arena index of each generated `(vertex, time)`.
    index: BTreeMap<(Vertex, Time), usize>,
    /// The open nodes, as arena indices. Focal key: conflicts, `f`, larger `g` first.
    list: FocalList<(u32, u32, Reverse<Time>), usize>,
}

impl Search {
    /// Opens `(vertex, time)` reached with `conflicts` via `parent`, with heuristic `h`.
    fn open(&mut self, vertex: Vertex, time: Time, conflicts: u32, parent: Option<usize>, h: u32) {
        let id = self.nodes.len();
        let f = time + h;
        let handle = self.list.push(f, f, (conflicts, f, Reverse(time)), id);

        self.nodes.push(Node {
            vertex,
            time,
            conflicts,
            parent,
            handle: Some(handle),
        });
        self.index.insert((vertex, time), id);
    }

    /// Opens `(vertex, time)`, or re-keys it if it is open with more conflicts.
    fn relax(&mut self, vertex: Vertex, time: Time, conflicts: u32, parent: usize, h: u32) {
        let Some(&id) = self.index.get(&(vertex, time)) else {
            self.open(vertex, time, conflicts, Some(parent), h);
            return;
        };

        let node = &mut self.nodes[id];
        let Some(handle) = node.handle.filter(|_| conflicts < node.conflicts) else {
            return;
        };

        self.list.remove(handle);
        let f = time + h;
        node.conflicts = conflicts;
        node.parent = Some(parent);
        node.handle = Some(self.list.push(f, f, (conflicts, f, Reverse(time)), id));
    }

    /// The vertices from the start to node `last`.
    fn path(&self, last: usize) -> Path {
        let mut path: Path = core::iter::successors(Some(last), |&id| self.nodes[id].parent)
            .map(|id| self.nodes[id].vertex)
            .collect();
        path.reverse();

        path
    }
}
