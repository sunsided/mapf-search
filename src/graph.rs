//! The graph agents move on: the [`Graph`] trait over dense [`Vertex`] ids, explicit
//! [`AdjacencyList`]s and 4-connected [`Grid`]s.

mod grid;

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

pub use grid::Grid;

/// A vertex id. Graphs number their vertices densely as `0..vertex_count()`.
pub type Vertex = usize;

/// A finite directed graph over vertices `0..vertex_count()`.
///
/// Agents move along edges or wait in place; waits are implicit and must not be reported
/// as self-loops. For undirected graphs, report every edge in both directions.
pub trait Graph {
    /// The number of vertices. Vertex ids are `0..vertex_count()`.
    fn vertex_count(&self) -> usize;

    /// Calls `emit` once for every out-neighbour of `vertex`.
    ///
    /// `vertex` is always below [`Graph::vertex_count`].
    fn neighbors(&self, vertex: Vertex, emit: impl FnMut(Vertex));
}

/// The direction an agent faces on an [`OrientedGraph`], below
/// [`OrientedGraph::heading_count`].
pub type Heading = usize;

/// A [`Graph`] on which agents face a heading and only move forward, for solvers that model
/// turning in place as taking time ([`pibt_turns`](crate::pibt_turns())).
///
/// Each step an agent either moves to [`OrientedGraph::forward`] of its vertex and heading,
/// keeping the heading, or turns in place to one of [`OrientedGraph::turns`], or waits.
/// The out-neighbours of a vertex in [`Graph::neighbors`] must be exactly the vertices
/// some heading moves forward to, so that a vertex path alone still describes where an
/// agent is; a turn is a wait in it.
pub trait OrientedGraph: Graph {
    /// The number of headings. Headings are `0..heading_count()`, the same at every vertex.
    fn heading_count(&self) -> usize;

    /// The vertex an agent at `vertex` facing `heading` reaches by moving forward, or
    /// `None` if there is no edge that way.
    fn forward(&self, vertex: Vertex, heading: Heading) -> Option<Vertex>;

    /// Calls `emit` once for every heading one turn away from `heading`. Turning must be
    /// able to reach every heading from every other.
    fn turns(&self, heading: Heading, emit: impl FnMut(Heading));
}

/// A graph stored as one list of out-neighbours per vertex.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AdjacencyList {
    /// `adjacency[v]` lists the out-neighbours of `v`. Every entry is below
    /// `adjacency.len()`.
    adjacency: Vec<Vec<Vertex>>,
}

impl AdjacencyList {
    /// Creates a graph with `vertex_count` vertices and the given directed edges.
    ///
    /// # Panics
    ///
    /// Panics if an edge names a vertex at or above `vertex_count`.
    #[must_use]
    pub fn directed(vertex_count: usize, edges: &[(Vertex, Vertex)]) -> Self {
        let mut adjacency = vec![Vec::new(); vertex_count];
        for &(from, to) in edges {
            assert!(
                from < vertex_count && to < vertex_count,
                "edge ({from}, {to}) names a vertex outside 0..{vertex_count}"
            );
            adjacency[from].push(to);
        }

        Self { adjacency }
    }

    /// Creates a graph with `vertex_count` vertices where every edge goes both ways.
    ///
    /// # Panics
    ///
    /// Panics if an edge names a vertex at or above `vertex_count`.
    #[must_use]
    pub fn undirected(vertex_count: usize, edges: &[(Vertex, Vertex)]) -> Self {
        let both: Vec<_> = edges.iter().flat_map(|&(a, b)| [(a, b), (b, a)]).collect();

        Self::directed(vertex_count, &both)
    }
}

/// A shared reference to a graph is the same graph, so sub-instances can borrow it.
impl<G: Graph + ?Sized> Graph for &G {
    fn vertex_count(&self) -> usize {
        (**self).vertex_count()
    }

    fn neighbors(&self, vertex: Vertex, emit: impl FnMut(Vertex)) {
        (**self).neighbors(vertex, emit);
    }
}

impl Graph for AdjacencyList {
    fn vertex_count(&self) -> usize {
        self.adjacency.len()
    }

    fn neighbors(&self, vertex: Vertex, emit: impl FnMut(Vertex)) {
        self.adjacency[vertex].iter().copied().for_each(emit);
    }
}

/// Exact distances, in edges, from every vertex to `target`; `None` where `target` is
/// unreachable.
///
/// Breadth-first search backwards along reversed edges. Costs `O(V + E)` time and memory.
pub(crate) fn distances_to<G: Graph + ?Sized>(graph: &G, target: Vertex) -> Vec<Option<u32>> {
    let reverse = reverse_adjacency(graph);
    let mut distance = vec![None; graph.vertex_count()];
    let mut queue = VecDeque::from([target]);
    distance[target] = Some(0);

    while let Some(vertex) = queue.pop_front() {
        let next = distance[vertex].map(|d: u32| d + 1);
        for &previous in &reverse[vertex] {
            if distance[previous].is_none() {
                distance[previous] = next;
                queue.push_back(previous);
            }
        }
    }

    distance
}

/// The in-neighbours of every vertex.
pub(crate) fn reverse_adjacency<G: Graph + ?Sized>(graph: &G) -> Vec<Vec<Vertex>> {
    let mut reverse = vec![Vec::new(); graph.vertex_count()];
    for from in 0..graph.vertex_count() {
        graph.neighbors(from, |to| reverse[to].push(from));
    }

    reverse
}
