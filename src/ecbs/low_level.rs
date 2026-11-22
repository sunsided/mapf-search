//! ECBS's low level: focal search over `(vertex, time)` states ([`focal_plan`]), with the
//! number of conflicts with other agents' paths ([`ConflictTable`]) as the focal key.
//!
//! The state includes the time, so a state's path cost `g` is its time and never improves;
//! no state is reopened, and `f_min` stays a valid lower bound on the agent's optimal cost.
//! The horizon is the one of the A* low level ([`crate::space_time_astar`]). The open list
//! is `heuristic-search`'s [`FocalList`]: `f` orders it and decides focal membership,
//! which is what a single-agent search needs.
//!
//! Finishing is a step of its own: from the goal, once the agent may stay there for good,
//! a free step leads to a *finished* copy of the state, the only goal state. It costs the
//! conflicts the agent would cause by staying ([`ConflictTable::after`]): every later
//! visit of another agent. So a path that finishes early where others pass later competes
//! with one that waits elsewhere first, on the same focal key.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use heuristic_search::{FocalList, Weight};

use crate::conflict_table::ConflictTable;
use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex};
use crate::instance::Agent;
use crate::path::{Path, Time};

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
    /// Whether the agent stays at the goal for good from here; a goal state.
    finished: bool,
    /// The conflicts accumulated on the best known path here.
    conflicts: u32,
    /// The index of the predecessor in the node arena; `None` for the start.
    parent: Option<usize>,
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
/// `problem.table`, counting those after it finished. `None` if the agent has no path.
///
/// Focal key: the accumulated conflicts, then `f`, then larger `g`.
pub(super) fn focal_plan<G: Graph>(problem: &Problem<'_, G>, weight: Weight) -> Option<FocalPath> {
    let start = problem.agent.start;
    if !problem.constraints.allows_vertex(start, 0) {
        return None;
    }

    let bounds = Bounds::new(problem);
    let mut search = Search {
        nodes: Vec::new(),
        index: BTreeMap::new(),
        list: FocalList::new(weight),
    };
    let h = bounds.heuristic(problem, start, 0)?;
    search.open((start, 0, false), 0, None, h);

    while let Some(lower_bound) = search.list.min_f() {
        let current = search.list.pop().expect("the list is not empty");
        if search.nodes[current].finished {
            return Some(FocalPath {
                path: search.path(current),
                lower_bound,
            });
        }

        search.expand(problem, &bounds, current);
    }

    None
}

/// A search state: vertex, time and whether the agent finished there.
type Key = (Vertex, Time, bool);

/// The time bounds of one [`focal_plan`] run, from the agent's constraints.
struct Bounds {
    /// The agent may finish at its goal from this time on.
    earliest_finish: Time,
    /// No successor is generated at a later time than this.
    horizon: Time,
}

impl Bounds {
    /// The bounds of `problem`, as in the A* low level.
    fn new<G: Graph>(problem: &Problem<'_, G>) -> Self {
        let constraints = problem.constraints;

        Self {
            earliest_finish: constraints.earliest_finish(problem.agent.goal),
            horizon: constraints.horizon(problem.graph.vertex_count()),
        }
    }

    /// The admissible estimate at `vertex` and `time` of an unfinished agent; `None` if the
    /// goal is unreachable from there.
    fn heuristic<G>(&self, problem: &Problem<'_, G>, vertex: Vertex, time: Time) -> Option<u32> {
        let distance = problem.distances[vertex]?;

        Some(distance.max(self.earliest_finish.saturating_sub(time)))
    }
}

/// The node arena and focal list of one [`focal_plan`] run.
struct Search {
    /// Every generated node.
    nodes: Vec<Node>,
    /// The arena index of each generated state.
    index: BTreeMap<Key, usize>,
    /// The open nodes, as arena indices, with `f` as the order, the time as `g` and the
    /// conflicts as the focal key; ties go to the smaller `f`, then the larger `g`.
    list: FocalList<u32, u32>,
}

impl Search {
    /// Generates the successors of the unfinished node `current`: finishing, if it is at
    /// the goal and may stay, and every allowed move and wait within the horizon.
    fn expand<G: Graph>(&mut self, problem: &Problem<'_, G>, bounds: &Bounds, current: usize) {
        let Node {
            vertex,
            time,
            conflicts,
            ..
        } = self.nodes[current];
        if vertex == problem.agent.goal && time >= bounds.earliest_finish {
            let staying = conflicts + problem.table.after(vertex, time);
            self.relax((vertex, time, true), staying, current, 0);
        }

        if time >= bounds.horizon {
            return;
        }

        let mut step = |next: Vertex| {
            if !problem.constraints.allows(vertex, next, time) {
                return;
            }

            if let Some(h) = bounds.heuristic(problem, next, time + 1) {
                let conflicts = conflicts + problem.table.count(vertex, next, time);
                self.relax((next, time + 1, false), conflicts, current, h);
            }
        };
        step(vertex);
        problem.graph.neighbors(vertex, step);
    }

    /// Opens `key` reached with `conflicts` via `parent`, with heuristic `h`.
    fn open(&mut self, key: Key, conflicts: u32, parent: Option<usize>, h: u32) {
        let (vertex, time, finished) = key;
        let id = self.nodes.len();
        self.list.push(id, time + h, time, conflicts);

        self.nodes.push(Node {
            vertex,
            time,
            finished,
            conflicts,
            parent,
        });
        self.index.insert(key, id);
    }

    /// Opens `key`, or re-keys it if it is open with more conflicts.
    fn relax(&mut self, key: Key, conflicts: u32, parent: usize, h: u32) {
        let Some(&id) = self.index.get(&key) else {
            self.open(key, conflicts, Some(parent), h);
            return;
        };

        let node = &mut self.nodes[id];
        if !self.list.contains(id) || conflicts >= node.conflicts {
            return;
        }

        node.conflicts = conflicts;
        node.parent = Some(parent);
        self.list.push(id, node.time + h, node.time, conflicts);
    }

    /// The vertices from the start to node `last`, without the finishing step.
    fn path(&self, last: usize) -> Path {
        let mut path: Path = core::iter::successors(Some(last), |&id| self.nodes[id].parent)
            .filter(|&id| !self.nodes[id].finished)
            .map(|id| self.nodes[id].vertex)
            .collect();
        path.reverse();

        path
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the focal low level.

    use super::*;
    use alloc::vec;

    use crate::conflict::conflicts;
    use crate::graph::{Grid, distances_to};

    /// An agent whose goal another agent passes later does not finish there early when the
    /// weight leaves room to arrive after the other agent has passed.
    #[test]
    fn counts_conflicts_after_finishing() {
        let grid = Grid::new(4, 1);
        let agent = Agent { start: 0, goal: 1 };
        let passing = vec![3, 2, 1, 2, 3];
        let table = ConflictTable::without(&[vec![0], passing.clone()], 0);
        let distances = distances_to(&grid, agent.goal);
        let problem = Problem {
            graph: &grid,
            agent,
            constraints: &ConstraintSet::new(),
            distances: &distances,
            table: &table,
        };

        let tight = focal_plan(&problem, Weight::ONE).expect("reachable");
        assert_eq!(tight.path, [0, 1], "no slack: the conflict stays");

        let loose = focal_plan(&problem, Weight::integer(3)).expect("reachable");
        assert_eq!(loose.lower_bound, 1);
        assert!(loose.path.len() <= 4, "within the weight: {:?}", loose.path);
        assert_eq!(conflicts(&[loose.path, passing]), []);
    }
}
