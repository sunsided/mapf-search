//! SIPPS (Li et al. 2022): safe interval path planning with soft obstacles, the low level of
//! MAPF-LNS2. It finds a path for one agent with few collisions with a fixed set of other
//! paths, then the earliest arrival ([`sipps`]).
//!
//! SIPP (Phillips and Likhachev 2011) searches `(vertex, safe interval)` states instead of
//! `(vertex, time)` ones: an agent that can reach a vertex early in an interval can wait
//! there for the rest of it, so the earliest arrival per interval is all that matters. With
//! soft obstacles only, as here, a vertex's timeline `[0, inf)` is cut into maximal
//! intervals that are either *dirty* (another agent is there at every step of it) or
//! *clean* (nobody is). A node counts its collisions `c`: one for entering a dirty
//! interval, one for a swap along the edge it came by. Nodes are ordered by `c`, then by
//! `f`. Nodes of the same vertex, interval and goal flag prune each other by dominance and
//! trim overlapping intervals (the paper's Algorithm 3).
//!
//! The paper's guarantees: a path is found whenever one exists, and if a path without
//! collisions exists, the shortest such path is found. Otherwise the count is an
//! approximation: waiting through several dirty steps of one interval counts once. The
//! paper accepts that, as LNS2 only minimizes collisions as a proxy for colliding pairs.

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::graph::{Graph, Vertex};
use crate::instance::Agent;
use crate::path::{Path, Reservation, Time, reservations};

/// The end of an interval that never ends.
const FOREVER: Time = Time::MAX;

/// The other agents' paths as soft obstacles, by vertex and by move.
pub(super) struct Obstacles {
    /// The steps at which other agents stand at a vertex before their final arrival, by
    /// vertex; sorted, one entry per agent and step.
    visits: BTreeMap<Vertex, Vec<Time>>,
    /// The vertices other agents stay at for good, with the earliest step they are there.
    parked: BTreeMap<Vertex, Time>,
    /// Moves `(from, to, arrival)` that swap with another agent: it moves `to -> from`
    /// arriving at the same step.
    swaps: BTreeSet<(Vertex, Vertex, Time)>,
}

impl Obstacles {
    /// The obstacles of `paths`, each non-empty.
    pub(super) fn of<'p>(paths: impl IntoIterator<Item = &'p Path>) -> Self {
        let mut obstacles = Self {
            visits: BTreeMap::new(),
            parked: BTreeMap::new(),
            swaps: BTreeSet::new(),
        };
        for reservation in paths.into_iter().flat_map(|path| reservations(path)) {
            obstacles.add(reservation);
        }

        for steps in obstacles.visits.values_mut() {
            steps.sort_unstable();
        }

        obstacles
    }

    /// Records one reservation of another agent's path. A move is kept as the swap it
    /// forbids: the reverse move, arriving at the same step.
    fn add(&mut self, reservation: Reservation) {
        match reservation {
            Reservation::Visit { vertex, time } => {
                self.visits.entry(vertex).or_default().push(time);
            }
            Reservation::Move { from, to, time } => {
                self.swaps.insert((to, from, time + 1));
            }
            Reservation::Park { vertex, time } => {
                let since = self.parked.entry(vertex).or_insert(time);
                *since = (*since).min(time);
            }
        }
    }

    /// The safe intervals of `vertex`, in order, covering `[0, inf)`.
    fn intervals(&self, vertex: Vertex) -> Vec<Interval> {
        let parked = self.parked.get(&vertex).copied();
        let mut dirty: Vec<Time> = self.visits.get(&vertex).cloned().unwrap_or_default();
        dirty.retain(|&step| parked.is_none_or(|since| step < since));
        dirty.dedup();

        let mut intervals = Vec::new();
        let mut at = 0;
        let mut index = 0;
        while index < dirty.len() {
            let start = dirty[index];
            let mut end = start + 1;
            index += 1;
            while index < dirty.len() && dirty[index] == end {
                end += 1;
                index += 1;
            }

            if at < start {
                intervals.push(Interval::new(at, start, false));
            }

            intervals.push(Interval::new(start, end, true));
            at = end;
        }

        match parked {
            Some(since) if at < since => {
                intervals.push(Interval::new(at, since, false));
                intervals.push(Interval::new(since, FOREVER, true));
            }
            Some(_) => match intervals.last_mut() {
                Some(last) if last.dirty && last.high == at => last.high = FOREVER,
                _ => intervals.push(Interval::new(at, FOREVER, true)),
            },
            None => intervals.push(Interval::new(at, FOREVER, false)),
        }

        intervals
    }

    /// Returns `true` if another agent ever stands at `vertex`.
    fn touches(&self, vertex: Vertex) -> bool {
        self.visits.contains_key(&vertex) || self.parked.contains_key(&vertex)
    }

    /// The number of obstacles at `vertex` after step `time`: later visits, and another
    /// agent parking there later.
    fn after(&self, vertex: Vertex, time: Time) -> u32 {
        let visits = self
            .visits
            .get(&vertex)
            .map_or(0, |steps| steps.iter().filter(|&&step| step > time).count());
        let parked = self.parked.get(&vertex).is_some_and(|&since| since > time);

        u32::try_from(visits).unwrap_or(u32::MAX) + u32::from(parked)
    }

    /// One more than the last step at which a vertex obstacle is at `vertex`: a lower bound
    /// on the arrival of a path without collisions that ends there. `0` without any.
    fn clear_from(&self, vertex: Vertex) -> Time {
        self.visits
            .get(&vertex)
            .and_then(|steps| steps.last())
            .map_or(0, |&step| step + 1)
    }

    /// Returns `true` if moving `from -> to` and arriving at `arrival` swaps with another
    /// agent.
    fn swaps(&self, from: Vertex, to: Vertex, arrival: Time) -> bool {
        from != to && self.swaps.contains(&(from, to, arrival))
    }
}

/// A safe interval `[low, high)` of a vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    /// The first step of the interval.
    low: Time,
    /// The step after the last one; [`FOREVER`] if the interval never ends.
    high: Time,
    /// Whether another agent is at the vertex at every step of the interval.
    dirty: bool,
}

impl Interval {
    /// The interval `[low, high)`, dirty or clean.
    fn new(low: Time, high: Time, dirty: bool) -> Self {
        Self { low, high, dirty }
    }
}

/// A search node: a vertex, a part of one of its safe intervals, and the best path there.
#[derive(Debug, Clone)]
struct Node {
    /// The vertex.
    vertex: Vertex,
    /// The index of the safe interval in the vertex's list.
    interval: usize,
    /// The earliest arrival in the interval, also the node's `g`.
    low: Time,
    /// The end of the part of the interval the node stands for.
    high: Time,
    /// Whether this is a goal node: the agent stays at the goal for good from `low` on.
    goal: bool,
    /// The collisions of the path to the node.
    collisions: u32,
    /// The heuristic: a lower bound on the time still needed.
    h: Time,
    /// The node the path comes from; `None` for the root.
    parent: Option<usize>,
    /// Whether the node was pruned by a dominating node.
    pruned: bool,
}

impl Node {
    /// The nodes with the same identity prune each other.
    fn identity(&self) -> (Vertex, usize, bool) {
        (self.vertex, self.interval, self.goal)
    }
}

/// The path of `agent` on `graph` among the soft `obstacles` with the fewest collisions as
/// SIPPS counts them, the earliest arrival among those; `None` if the goal is unreachable.
/// `distances` holds the distances to the agent's goal.
pub(super) fn sipps<G: Graph>(
    graph: &G,
    agent: Agent,
    distances: &[Option<u32>],
    obstacles: &Obstacles,
) -> Option<Path> {
    distances[agent.start]?;

    let mut search = Search {
        graph,
        agent,
        distances,
        obstacles,
        clear_from: obstacles.clear_from(agent.goal),
        intervals: BTreeMap::new(),
        clean: Rc::from([Interval::new(0, FOREVER, false)]),
        nodes: Vec::new(),
        twins: BTreeMap::new(),
        open: BinaryHeap::new(),
    };

    search.run()
}

/// One SIPPS run.
struct Search<'a, G> {
    /// The graph the agent moves on.
    graph: &'a G,
    /// The agent's start and goal.
    agent: Agent,
    /// Distances to the goal.
    distances: &'a [Option<u32>],
    /// The other agents' paths.
    obstacles: &'a Obstacles,
    /// The earliest arrival at the goal of a path without collisions.
    clear_from: Time,
    /// The safe intervals of every vertex with obstacles looked at so far.
    intervals: BTreeMap<Vertex, Rc<[Interval]>>,
    /// The intervals of a vertex without obstacles: one clean interval, shared.
    clean: Rc<[Interval]>,
    /// Every node generated, indexed by id.
    nodes: Vec<Node>,
    /// The ids of the live nodes of each identity.
    twins: BTreeMap<(Vertex, usize, bool), Vec<usize>>,
    /// Open nodes by collisions, then `f`, then `h`, then id.
    open: BinaryHeap<Reverse<(u32, Time, Time, usize)>>,
}

impl<G: Graph> Search<'_, G> {
    /// The paper's Algorithm 1: pops nodes until a goal node.
    fn run(&mut self) -> Option<Path> {
        let first = self.interval(self.agent.start, 0);
        let root = Node {
            vertex: self.agent.start,
            interval: 0,
            low: 0,
            high: first.high,
            goal: false,
            collisions: u32::from(first.dirty),
            h: 0,
            parent: None,
            pruned: false,
        };
        self.insert(root);

        while let Some(Reverse((_, _, _, id))) = self.open.pop() {
            if self.nodes[id].pruned {
                continue;
            }

            if self.nodes[id].goal {
                return Some(self.path(id));
            }

            if self.nodes[id].vertex == self.agent.goal {
                let future = self.obstacles.after(self.agent.goal, self.nodes[id].low);
                if future == 0 {
                    return Some(self.path(id));
                }

                let mut finished = self.nodes[id].clone();
                finished.goal = true;
                finished.collisions += future;
                finished.parent = Some(id);
                self.insert(finished);
            }

            self.expand(id);
        }

        None
    }

    /// The paper's Algorithm 2: generates the successors of node `id` by moves into the
    /// overlapping safe intervals of its neighbours, and by waiting into its vertex's next
    /// interval.
    fn expand(&mut self, id: usize) {
        let Node {
            vertex,
            interval,
            low,
            high,
            ..
        } = self.nodes[id];

        let mut neighbors = Vec::new();
        self.graph.neighbors(vertex, |next| neighbors.push(next));
        for next in neighbors
            .into_iter()
            .filter(|&next| self.distances[next].is_some())
        {
            let earliest = low + 1;
            let latest = high.saturating_add(1);
            for (index, &safe) in self.intervals_of(next).iter().enumerate() {
                let arrival = earliest.max(safe.low);
                if arrival < safe.high.min(latest) {
                    self.arrive(id, next, index, arrival, safe, latest);
                }
            }
        }

        let next = interval + 1;
        let intervals = self.intervals_of(vertex);
        if let Some(&safe) = intervals.get(next) {
            if safe.low == high {
                self.arrive(id, vertex, next, safe.low, safe, FOREVER);
            }
        }
    }

    /// Generates the nodes of arriving at `vertex` in its interval `index` (`safe`) from
    /// node `parent`, the earliest at `arrival` and the latest before `latest`: one node,
    /// or two if arriving early swaps with another agent and arriving later does not.
    fn arrive(
        &mut self,
        parent: usize,
        vertex: Vertex,
        index: usize,
        arrival: Time,
        safe: Interval,
        latest: Time,
    ) {
        let from = self.nodes[parent].vertex;
        let collisions = self.nodes[parent].collisions + u32::from(safe.dirty);
        let end = safe.high.min(latest);
        let clean = (arrival..end).find(|&time| !self.obstacles.swaps(from, vertex, time));

        let node = |low: Time, high: Time, collisions: u32| Node {
            vertex,
            interval: index,
            low,
            high,
            goal: false,
            collisions,
            h: 0,
            parent: Some(parent),
            pruned: false,
        };

        match clean {
            Some(clean) if clean > arrival => {
                self.insert(node(arrival, clean, collisions + 1));
                self.insert(node(clean, safe.high, collisions));
            }
            Some(_) => self.insert(node(arrival, safe.high, collisions)),
            None => self.insert(node(arrival, safe.high, collisions + 1)),
        }
    }

    /// The paper's Algorithm 3: opens `node` unless a twin dominates it, prunes the twins
    /// it dominates, and trims overlapping intervals.
    fn insert(&mut self, mut node: Node) {
        node.h = self.heuristic(&node);
        let identity = node.identity();
        let mut twins = self.twins.remove(&identity).unwrap_or_default();

        for &twin in &twins {
            let other = &self.nodes[twin];
            if other.pruned {
                continue;
            }

            if other.low <= node.low && other.collisions <= node.collisions {
                self.twins.insert(identity, twins);
                return;
            }

            if node.low <= other.low && node.collisions <= other.collisions {
                self.nodes[twin].pruned = true;
            } else if node.low < other.high && other.low < node.high {
                if node.low < other.low {
                    node.high = other.low;
                } else {
                    self.nodes[twin].high = node.low;
                }
            }
        }

        let id = self.nodes.len();
        let key = (node.collisions, node.low + node.h, node.h, id);
        self.nodes.push(node);
        twins.retain(|&twin| !self.nodes[twin].pruned);
        twins.push(id);
        self.twins.insert(identity, twins);
        self.open.push(Reverse(key));
    }

    /// The paper's heuristic: the distance to the goal, and for a node without collisions
    /// at least the time until the goal is clear; `0` for a goal node.
    fn heuristic(&self, node: &Node) -> Time {
        if node.goal {
            return 0;
        }

        let distance = self.distances[node.vertex].expect("only vertices that reach the goal");
        if node.collisions == 0 {
            distance.max(self.clear_from.saturating_sub(node.low))
        } else {
            distance
        }
    }

    /// The safe interval `index` of `vertex`.
    fn interval(&mut self, vertex: Vertex, index: usize) -> Interval {
        self.intervals_of(vertex)[index]
    }

    /// The safe intervals of `vertex`, cached.
    fn intervals_of(&mut self, vertex: Vertex) -> Rc<[Interval]> {
        let obstacles = self.obstacles;
        if !obstacles.touches(vertex) {
            return Rc::clone(&self.clean);
        }

        Rc::clone(
            self.intervals
                .entry(vertex)
                .or_insert_with(|| obstacles.intervals(vertex).into()),
        )
    }

    /// The path to node `id`: the vertices of its ancestors at their arrival steps, with
    /// waits filled in.
    fn path(&self, id: usize) -> Path {
        let mut chain = Vec::new();
        let mut at = Some(id);
        while let Some(id) = at {
            chain.push(id);
            at = self.nodes[id].parent;
        }

        chain.reverse();

        let mut path = vec![self.agent.start];
        for &id in &chain[1..] {
            let Node { vertex, low, .. } = self.nodes[id];
            let arrival = usize::try_from(low).expect("steps fit in usize");
            while path.len() < arrival {
                path.push(*path.last().expect("non-empty"));
            }

            if path.len() == arrival {
                path.push(vertex);
            }
        }

        path
    }
}

#[cfg(test)]
mod tests {
    //! SIPPS against the space-time A* that minimizes collisions exactly.

    use super::*;

    use proptest::prelude::*;

    use crate::conflict_table::ConflictTable;
    use crate::graph::{Grid, distances_to};
    use crate::lns2::low_level::fewest_collisions;
    use crate::path::path_cost;

    /// The exact number of collisions of `path` with the paths in `table`, counted as the
    /// space-time A* counts them.
    fn collisions(path: &[Vertex], table: &ConflictTable) -> u32 {
        let moves: u32 = path
            .windows(2)
            .enumerate()
            .map(|(time, step)| table.count(step[0], step[1], Time::try_from(time).expect("short")))
            .sum();
        let goal = *path.last().expect("non-empty");

        moves + table.after(goal, path_cost(path))
    }

    /// A 4x4 grid with up to 3 walls, up to 3 other agents walking at random, and an agent
    /// whose start no other agent starts at. As in a MAPF instance, the walks end at
    /// distinct vertices other than the agent's goal.
    fn case() -> impl Strategy<Value = (Grid, Vec<Path>, Agent)> {
        let blocked = prop::collection::vec(0_usize..16, 0..=3);
        let walks = prop::collection::vec(
            (0_usize..16, prop::collection::vec(0_usize..5, 0..10)),
            0..=3,
        );

        (blocked, walks, 0_usize..16, 0_usize..16).prop_filter_map(
            "free, distinct starts and ends",
            |(walls, walks, start, goal)| {
                let mut grid = Grid::new(4, 4);
                for &wall in &walls {
                    let (x, y) = grid.cell(wall);
                    grid.block(x, y);
                }

                let free = |vertex: Vertex| !walls.contains(&vertex);
                if !free(start)
                    || !free(goal)
                    || walks.iter().any(|(from, _)| *from == start || !free(*from))
                {
                    return None;
                }

                let paths: Vec<Path> = walks
                    .iter()
                    .map(|(from, turns)| walk(&grid, *from, turns))
                    .collect();
                let ends: Vec<Vertex> = paths.iter().map(|path| path[path.len() - 1]).collect();
                let distinct = ends
                    .iter()
                    .enumerate()
                    .all(|(i, end)| *end != goal && !ends[..i].contains(end));

                distinct.then_some((grid, paths, Agent { start, goal }))
            },
        )
    }

    /// A walk on `grid` from `from`: each turn waits (`0`) or takes the `turn`-th neighbour
    /// if there is one.
    fn walk(grid: &Grid, from: Vertex, turns: &[usize]) -> Path {
        let mut path = vec![from];
        for &turn in turns {
            let here = *path.last().expect("non-empty");
            let mut neighbors = Vec::new();
            grid.neighbors(here, |next| neighbors.push(next));
            let next = turn
                .checked_sub(1)
                .and_then(|index| neighbors.get(index))
                .copied()
                .unwrap_or(here);
            path.push(next);
        }

        path
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// SIPPS finds a path exactly when space-time A* does; the path is valid; it never has
        /// fewer collisions than A*'s minimum; and where A* finds a path without collisions,
        /// SIPPS finds one as short (the paper's Theorem 2).
        #[test]
        fn agrees_with_space_time_astar((grid, others, agent) in case()) {
            let distances = distances_to(&grid, agent.goal);
            let table = ConflictTable::of(&others);
            let last = others.iter().map(Vec::len).max().unwrap_or(0);
            let exact = fewest_collisions(&grid, agent, &distances, &table, Time::try_from(last).expect("short"));
            let found = sipps(&grid, agent, &distances, &Obstacles::of(&others));

            prop_assert_eq!(found.is_some(), exact.is_some());
            let (Some(found), Some(exact)) = (found, exact) else {
                return Ok(());
            };

            prop_assert_eq!(found.first(), Some(&agent.start));
            prop_assert_eq!(found.last(), Some(&agent.goal));
            for step in found.windows(2) {
                let ((x0, y0), (x1, y1)) = (grid.cell(step[0]), grid.cell(step[1]));
                prop_assert!(x0.abs_diff(x1) + y0.abs_diff(y1) <= 1, "a jump in {found:?}");
                prop_assert!(step[0] == step[1] || !grid.is_blocked(step[1]));
            }

            let (best, got) = (collisions(&exact, &table), collisions(&found, &table));
            prop_assert!(got >= best);
            if best == 0 {
                prop_assert_eq!(got, 0, "{:?} against {:?}", found, exact);
                prop_assert_eq!(path_cost(&found), path_cost(&exact));
            }
        }
    }

    /// Safe intervals alternate between clean and dirty runs and end with the parked agent.
    #[test]
    fn builds_safe_intervals() {
        // Visits at 2, 3 and 6, then an agent parked from 8 on.
        let obstacles = Obstacles::of(&[vec![9, 9, 5, 5, 9], vec![9, 9, 9, 9, 9, 9, 5, 9, 5]]);
        let intervals = obstacles.intervals(5);

        assert_eq!(
            intervals,
            [
                Interval::new(0, 2, false),
                Interval::new(2, 4, true),
                Interval::new(4, 6, false),
                Interval::new(6, 7, true),
                Interval::new(7, 8, false),
                Interval::new(8, FOREVER, true),
            ]
        );
        assert_eq!(obstacles.intervals(4), [Interval::new(0, FOREVER, false)]);
    }
}
