//! The flow oracle of FOCBS ([`route`], Jin et al. 2026): routes agents without goals around
//! fixed paths of agents with goals by a min-cost flow in a time-expanded network, or names
//! the reservations that block every routing ([`Blocker`]).
//!
//! Each vertex `v` and step `t` up to the horizon `T` is a unit-capacity arc `v_t^in ->
//! v_t^out`, so two agents never share a vertex. Waiting is a free arc `v_t^out ->
//! v_{t+1}^in`. Moving along an edge `{v, w}` from `t` to `t + 1` passes a gadget whose
//! central arc has capacity 1 and cost 1, so one move counts once and two agents cannot swap
//! along the edge. The source feeds every start at step 0, and every vertex at step `T`
//! drains into the sink, so each agent ends wherever it likes.
//!
//! The vertex arcs the fixed paths occupy, their waits at the goal up to `T` included, and
//! the central arcs of their moves have capacity 0, labelled with the reserving agent.
//! Successive shortest paths with potentials then either routes every agent at the least
//! total number of moves, or stops at a maximum flow below that. In the second case the
//! nodes the source still reaches in the residual network form a minimum cut, and every
//! labelled arc leaving them is a blocker: no routing exists unless one of them is released.

use alloc::collections::{BinaryHeap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::graph::{Graph, Vertex};
use crate::path::{Path, Time, path_cost, position_at};

/// A reservation of a fixed path that blocks every routing of the other agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Blocker {
    /// Path `agent` occupies `vertex` at `time`.
    Vertex {
        /// The index of the fixed path.
        agent: usize,
        /// The occupied vertex.
        vertex: Vertex,
        /// The step.
        time: Time,
    },
    /// Path `agent` moves `from -> to` between `time` and `time + 1`.
    Move {
        /// The index of the fixed path.
        agent: usize,
        /// The vertex the move leaves.
        from: Vertex,
        /// The vertex the move enters.
        to: Vertex,
        /// The step the move starts at.
        time: Time,
    },
}

/// A routing of the agents without goals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Routing {
    /// Each agent's path, trimmed of its trailing waits; it stays at the last vertex.
    pub(crate) paths: Vec<Path>,
    /// The number of moves over all paths, the least possible.
    pub(crate) moves: u32,
}

/// Routes agents from `starts` around the `fixed` paths on `graph` until the latest final
/// arrival among them, at the fewest moves; or returns the blockers of a minimum cut if no
/// routing exists. An empty blocker list means that nothing can be released to help.
///
/// Starts must be distinct and differ from the fixed paths' starts.
pub(crate) fn route<G: Graph>(
    graph: &G,
    fixed: &[Path],
    starts: &[Vertex],
) -> Result<Routing, Vec<Blocker>> {
    let horizon = fixed.iter().map(|path| path_cost(path)).max().unwrap_or(0);
    let mut network = Network::build(graph, fixed, starts, horizon);
    let routed = network.min_cost_flow(starts.len());

    if routed < starts.len() {
        return Err(network.blockers());
    }

    Ok(network.decompose(starts))
}

/// One arc of the residual network.
#[derive(Debug, Clone)]
struct Arc {
    /// The head node.
    to: usize,
    /// The residual capacity.
    capacity: u32,
    /// The cost per unit; negative on reverse arcs.
    cost: i64,
    /// The index of the reverse arc in the head's list.
    reverse: usize,
    /// The reservation a capacity-0 forward arc stands for, if any.
    label: Option<Blocker>,
    /// Whether this is a forward arc of the network rather than a reverse one.
    forward: bool,
}

/// The time-expanded network of one oracle call.
struct Network {
    /// The arcs leaving each node.
    arcs: Vec<Vec<Arc>>,
    /// The number of vertices of the graph.
    vertices: usize,
    /// The last step.
    horizon: Time,
    /// The source node.
    source: usize,
    /// The sink node.
    sink: usize,
}

impl Network {
    /// The network of `graph` up to `horizon`, with the reservations of `fixed` at capacity
    /// 0 and the source feeding `starts`.
    fn build<G: Graph>(graph: &G, fixed: &[Path], starts: &[Vertex], horizon: Time) -> Self {
        let vertices = graph.vertex_count();
        let layers = usize::try_from(horizon).expect("steps fit in usize") + 1;
        let pairs = edge_pairs(graph);
        let gadget_base = 2 * vertices * layers;
        let source = gadget_base + 2 * pairs.len() * (layers - 1);
        let mut network = Self {
            arcs: vec![Vec::new(); source + 2],
            vertices,
            horizon,
            source,
            sink: source + 1,
        };

        network.add_vertices(fixed, layers);
        network.add_moves(fixed, &pairs, gadget_base, layers);
        for &start in starts {
            let node = network.vertex_in(start, 0);
            network.add(source, node, 1, 0, None);
        }

        for vertex in 0..vertices {
            let node = network.vertex_out(vertex, horizon);
            network.add(node, source + 1, 1, 0, None);
        }

        network
    }

    /// Adds the vertex arcs, reserved where a fixed path stands, and the wait arcs.
    fn add_vertices(&mut self, fixed: &[Path], layers: usize) {
        for time in 0..layers {
            let step = Time::try_from(time).expect("steps fit in u32");
            for vertex in 0..self.vertices {
                let holder = fixed
                    .iter()
                    .position(|path| position_at(path, step) == vertex);
                let label = holder.map(|agent| Blocker::Vertex {
                    agent,
                    vertex,
                    time: step,
                });
                let (from, to) = (self.vertex_in(vertex, step), self.vertex_out(vertex, step));
                self.add(from, to, u32::from(label.is_none()), 0, label);

                if time + 1 < layers {
                    let next = self.vertex_in(vertex, step + 1);
                    self.add(to, next, 1, 0, None);
                }
            }
        }
    }

    /// Adds a move gadget per vertex pair joined by an edge and per step: an entry arc
    /// from each end that has an edge to the other, a central arc of cost 1, reserved where
    /// a fixed path moves along the pair, and an exit arc into each end the other has an
    /// edge to.
    fn add_moves(
        &mut self,
        fixed: &[Path],
        pairs: &[(Vertex, Vertex, bool, bool)],
        base: usize,
        layers: usize,
    ) {
        for time in 0..layers - 1 {
            let step = Time::try_from(time).expect("steps fit in u32");
            for (index, &(a, b, a_to_b, b_to_a)) in pairs.iter().enumerate() {
                let entry = base + 2 * (time * pairs.len() + index);
                let exit = entry + 1;
                let label = mover(fixed, a, b, step);
                self.add(entry, exit, u32::from(label.is_none()), 1, label);

                if a_to_b {
                    self.add(self.vertex_out(a, step), entry, 1, 0, None);
                    self.add(exit, self.vertex_in(b, step + 1), 1, 0, None);
                }

                if b_to_a {
                    self.add(self.vertex_out(b, step), entry, 1, 0, None);
                    self.add(exit, self.vertex_in(a, step + 1), 1, 0, None);
                }
            }
        }
    }

    /// The node `v_t^in`.
    fn vertex_in(&self, vertex: Vertex, time: Time) -> usize {
        2 * (usize::try_from(time).expect("steps fit in usize") * self.vertices + vertex)
    }

    /// The node `v_t^out`.
    fn vertex_out(&self, vertex: Vertex, time: Time) -> usize {
        self.vertex_in(vertex, time) + 1
    }

    /// Adds an arc `from -> to` and its reverse.
    fn add(&mut self, from: usize, to: usize, capacity: u32, cost: i64, label: Option<Blocker>) {
        let (forward, reverse) = (self.arcs[from].len(), self.arcs[to].len());
        self.arcs[from].push(Arc {
            to,
            capacity,
            cost,
            reverse,
            label,
            forward: true,
        });
        self.arcs[to].push(Arc {
            to: from,
            capacity: 0,
            cost: -cost,
            reverse: forward,
            label: None,
            forward: false,
        });
    }

    /// Successive shortest paths with Dijkstra on reduced costs: sends up to `wanted` units
    /// from the source to the sink at the least cost and returns how many it sent.
    fn min_cost_flow(&mut self, wanted: usize) -> usize {
        let mut potential = vec![0_i64; self.arcs.len()];
        let mut sent = 0;
        while sent < wanted {
            let Some(path) = self.shortest_path(&mut potential) else {
                break;
            };

            for (node, arc) in path {
                let reverse = self.arcs[node][arc].reverse;
                let head = self.arcs[node][arc].to;
                self.arcs[node][arc].capacity -= 1;
                self.arcs[head][reverse].capacity += 1;
            }

            sent += 1;
        }

        sent
    }

    /// The cheapest augmenting path by reduced costs as `(node, arc index)` pairs from the
    /// source, or `None` if the sink is unreachable; updates the potentials.
    fn shortest_path(&self, potential: &mut [i64]) -> Option<Vec<(usize, usize)>> {
        let mut distance = vec![i64::MAX; self.arcs.len()];
        let mut previous: Vec<Option<(usize, usize)>> = vec![None; self.arcs.len()];
        let mut open = BinaryHeap::from([Reverse((0_i64, self.source))]);
        distance[self.source] = 0;
        while let Some(Reverse((here_distance, here))) = open.pop() {
            if here_distance > distance[here] {
                continue;
            }

            for (index, arc) in self.arcs[here].iter().enumerate() {
                if arc.capacity == 0 {
                    continue;
                }

                let reduced = arc.cost + potential[here] - potential[arc.to];
                let next = here_distance + reduced;
                if next < distance[arc.to] {
                    distance[arc.to] = next;
                    previous[arc.to] = Some((here, index));
                    open.push(Reverse((next, arc.to)));
                }
            }
        }

        if distance[self.sink] == i64::MAX {
            return None;
        }

        for (node, value) in potential.iter_mut().enumerate() {
            if distance[node] != i64::MAX {
                *value += distance[node];
            }
        }

        let mut path = Vec::new();
        let mut at = self.sink;
        while let Some((node, arc)) = previous[at] {
            path.push((node, arc));
            at = node;
        }

        path.reverse();

        Some(path)
    }

    /// The labelled arcs leaving the nodes the source reaches in the residual network,
    /// sorted and without duplicates.
    fn blockers(&self) -> Vec<Blocker> {
        let mut reached = vec![false; self.arcs.len()];
        let mut queue = VecDeque::from([self.source]);
        reached[self.source] = true;
        while let Some(node) = queue.pop_front() {
            for arc in &self.arcs[node] {
                if arc.capacity > 0 && !reached[arc.to] {
                    reached[arc.to] = true;
                    queue.push_back(arc.to);
                }
            }
        }

        let mut blockers: Vec<Blocker> = (0..self.arcs.len())
            .filter(|&node| reached[node])
            .flat_map(|node| self.arcs[node].iter())
            .filter(|arc| !reached[arc.to])
            .filter_map(|arc| arc.label)
            .collect();
        blockers.sort_unstable();
        blockers.dedup();

        blockers
    }

    /// The routing of the units of flow, one per start, followed arc by arc.
    fn decompose(&self, starts: &[Vertex]) -> Routing {
        let paths: Vec<Path> = starts.iter().map(|&start| self.follow(start)).collect();
        let moves = paths
            .iter()
            .map(|path| path.windows(2).filter(|step| step[0] != step[1]).count())
            .sum::<usize>();

        Routing {
            paths,
            moves: u32::try_from(moves).expect("move counts fit in u32"),
        }
    }

    /// The path of the unit of flow from `start`: its vertex at every step, trimmed of
    /// trailing waits.
    fn follow(&self, start: Vertex) -> Path {
        let mut path = vec![start];
        let mut node = self.vertex_out(start, 0);
        for time in 0..self.horizon {
            let next = self.next_vertex(node, time + 1);
            path.push(next);
            node = self.vertex_out(next, time + 1);
        }

        while path.len() > 1 && path[path.len() - 2] == path[path.len() - 1] {
            path.pop();
        }

        path
    }

    /// The vertex at step `time` that the flow leaving the out-node `node` (at `time - 1`)
    /// enters: by its wait arc or through a move gadget.
    fn next_vertex(&self, node: usize, time: Time) -> Vertex {
        let used = |arc: &&Arc| arc.forward && arc.capacity == 0 && self.carries(arc);
        let arc = self.arcs[node]
            .iter()
            .find(used)
            .expect("the flow leaves every node it enters");
        let mut head = arc.to;
        if head >= 2 * self.vertices * (usize::try_from(self.horizon).expect("fits") + 1) {
            let central = self.arcs[head]
                .iter()
                .find(used)
                .expect("through the gadget");
            let exit = self.arcs[central.to]
                .iter()
                .find(used)
                .expect("out of the gadget");
            head = exit.to;
        }

        let layer_start = 2 * usize::try_from(time).expect("fits") * self.vertices;

        (head - layer_start) / 2
    }

    /// Returns `true` if the forward `arc`, at residual capacity 0, carries flow rather than
    /// being reserved: its reverse arc has capacity left.
    fn carries(&self, arc: &Arc) -> bool {
        self.arcs[arc.to][arc.reverse].capacity > 0
    }
}

/// The fixed path that moves along the pair `{a, b}` between `time` and `time + 1`, as a
/// blocker, if any.
fn mover(fixed: &[Path], a: Vertex, b: Vertex, time: Time) -> Option<Blocker> {
    fixed.iter().enumerate().find_map(|(agent, path)| {
        let (from, to) = (position_at(path, time), position_at(path, time + 1));
        let along = (from, to) == (a, b) || (from, to) == (b, a);

        along.then_some(Blocker::Move {
            agent,
            from,
            to,
            time,
        })
    })
}

/// The vertex pairs `(a, b)`, `a < b`, joined by an edge in at least one direction, with
/// whether `a -> b` and `b -> a` exist.
fn edge_pairs<G: Graph>(graph: &G) -> Vec<(Vertex, Vertex, bool, bool)> {
    let mut directed = Vec::new();
    for vertex in 0..graph.vertex_count() {
        graph.neighbors(vertex, |next| {
            if next != vertex {
                directed.push((vertex, next));
            }
        });
    }

    directed.sort_unstable();
    let has = |from: Vertex, to: Vertex| directed.binary_search(&(from, to)).is_ok();
    let mut pairs: Vec<(Vertex, Vertex, bool, bool)> = directed
        .iter()
        .map(|&(from, to)| (from.min(to), from.max(to)))
        .map(|(a, b)| (a, b, has(a, b), has(b, a)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    pairs
}

#[cfg(test)]
mod tests {
    //! The oracle on small hand-made cases.

    use super::*;

    use crate::graph::{AdjacencyList, Grid};

    /// Without fixed paths, every agent stays: zero moves.
    #[test]
    fn idle_agents_stay() {
        let routing = route(&Grid::new(3, 3), &[], &[0, 4, 8]).expect("routable");

        assert_eq!(routing.paths, [vec![0], vec![4], vec![8]]);
        assert_eq!(routing.moves, 0);
    }

    /// An agent in the way of a fixed path steps aside into a side cell and stays there.
    #[test]
    fn steps_aside() {
        // `0 1 2` over `@ 4 @`; the fixed path runs 0 -> 1 -> 2, the agent waits at 1.
        let grid = Grid::parse(&["...", "@.@"]);
        let routing = route(&grid, &[vec![0, 1, 2]], &[1]).expect("routable");

        assert_eq!(routing.paths, [vec![1, 4]]);
        assert_eq!(routing.moves, 1);
    }

    /// In a corridor without room, the agent cannot get out of the way; the blockers are
    /// reservations of the fixed path.
    #[test]
    fn names_blockers_when_cornered() {
        let corridor = AdjacencyList::undirected(3, &[(0, 1), (1, 2)]);
        let blockers = route(&corridor, &[vec![0, 1, 2]], &[2]).expect_err("cornered");

        assert_ne!(blockers, []);
        assert!(blockers.iter().all(|blocker| matches!(
            blocker,
            Blocker::Vertex { agent: 0, .. } | Blocker::Move { agent: 0, .. }
        )));
    }
}
