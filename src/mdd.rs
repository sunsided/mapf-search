//! Multi-value decision diagrams ([`Mdd`]): every path of one agent that reaches its goal at
//! exactly a given time under its constraints, layered by time.
//!
//! CBS uses them to classify conflicts as cardinal, semi-cardinal or non-cardinal
//! (Boyarski et al. 2015); ICTS searches their cross product (Sharon et al. 2013).
//!
//! Goal-stay semantics: after its last level the agent stays at its goal forever, so every
//! level past [`Mdd::depth`] is the singleton goal ([`Mdd::level`]).

use alloc::vec;
use alloc::vec::Vec;

use crate::constraint::ConstraintSet;
use crate::graph::{Graph, Vertex};
use crate::instance::Agent;
use crate::path::Time;

/// All paths of one agent from its start at time `0` to its goal at time [`Mdd::depth`] that
/// satisfy its constraints, as a layered graph.
///
/// A path may pass the goal, or wait there, before its last level. Every node lies on at
/// least one such path, so every path through [`Mdd::children`] is a valid full path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mdd {
    /// `levels[t]` lists the vertices some path occupies at time `t`, sorted ascending.
    /// `levels[0]` is the start, the last level is the goal; never empty.
    levels: Vec<Vec<Vertex>>,
    /// `children[t][i]` are the indices into `levels[t + 1]` reachable in one step from
    /// `levels[t][i]`. One entry fewer than `levels`.
    children: Vec<Vec<Vec<usize>>>,
}

impl Mdd {
    /// The diagram of the paths of `agent` that arrive at its goal for good at time `depth`
    /// under `constraints`, or `None` if there is no such path.
    ///
    /// `distances` is the table of distances to the agent's goal (`distances_to`). Costs
    /// `O(depth * (V + E))` time.
    pub(crate) fn build<G: Graph>(
        graph: &G,
        agent: Agent,
        constraints: &ConstraintSet,
        distances: &[Option<u32>],
        depth: Time,
    ) -> Option<Self> {
        let goal_free_after = constraints
            .blocked_until(agent.goal)
            .is_none_or(|t| t < depth);
        let start_fits = distances[agent.start].is_some_and(|d| d <= depth);
        if !goal_free_after || !start_fits || !constraints.allows_vertex(agent.start, 0) {
            return None;
        }

        let mut levels = forward_levels(graph, agent, constraints, distances, depth);
        if !levels.last().is_some_and(|last| last.contains(&agent.goal)) {
            return None;
        }

        *levels.last_mut().expect("depth + 1 levels") = vec![agent.goal];
        let children = prune_backwards(graph, constraints, &mut levels);

        Some(Self { levels, children })
    }

    /// The time of the last level: the arrival time, so the cost, of every path.
    pub(crate) fn depth(&self) -> Time {
        Time::try_from(self.levels.len() - 1).expect("depths fit in u32 time steps")
    }

    /// The vertices of level `time`; past the last level, the goal alone.
    pub(crate) fn level(&self, time: Time) -> &[Vertex] {
        let index = usize::try_from(time).unwrap_or(usize::MAX);

        self.levels
            .get(index)
            .or(self.levels.last())
            .expect("never empty")
    }

    /// The only vertex every path occupies at `time`, if the level has width 1.
    pub(crate) fn singleton(&self, time: Time) -> Option<Vertex> {
        match self.level(time) {
            &[vertex] => Some(vertex),
            _ => None,
        }
    }

    /// The indices into level `time + 1` of the successors of node `index` of level `time`.
    /// At or past the last level, the goal node waits in place: `[0]`.
    pub(crate) fn children(&self, time: Time, index: usize) -> &[usize] {
        let level = usize::try_from(time).unwrap_or(usize::MAX);

        self.children.get(level).map_or(&[0], |nodes| &nodes[index])
    }
}

/// Forward layers: the vertices reachable at each time `0..=depth` under `constraints` from
/// which the goal is still reachable by `depth`. Unpruned: some may lead nowhere.
fn forward_levels<G: Graph>(
    graph: &G,
    agent: Agent,
    constraints: &ConstraintSet,
    distances: &[Option<u32>],
    depth: Time,
) -> Vec<Vec<Vertex>> {
    let mut levels = vec![vec![agent.start]];
    for time in 0..depth {
        let remaining = depth - time - 1;
        let mut next = Vec::new();
        for &vertex in levels.last().expect("never empty") {
            successors(graph, constraints, vertex, time, |to| {
                if distances[to].is_some_and(|d| d <= remaining) {
                    next.push(to);
                }
            });
        }

        next.sort_unstable();
        next.dedup();
        levels.push(next);
    }

    levels
}

/// Removes the nodes that cannot reach the goal level, back to front, and returns the
/// children lists of the remaining nodes.
fn prune_backwards<G: Graph>(
    graph: &G,
    constraints: &ConstraintSet,
    levels: &mut [Vec<Vertex>],
) -> Vec<Vec<Vec<usize>>> {
    let mut children = vec![Vec::new(); levels.len() - 1];
    for level in (0..levels.len() - 1).rev() {
        let time = Time::try_from(level).expect("depths fit in u32 time steps");
        let (now, later) = levels.split_at_mut(level + 1);
        let next = &later[0];

        let mut kept = Vec::new();
        let mut lists = Vec::new();
        for &vertex in &now[level] {
            let mut list = Vec::new();
            successors(graph, constraints, vertex, time, |to| {
                if let Ok(index) = next.binary_search(&to) {
                    list.push(index);
                }
            });

            if !list.is_empty() {
                list.sort_unstable();
                kept.push(vertex);
                lists.push(list);
            }
        }

        now[level] = kept;
        children[level] = lists;
    }

    children
}

/// Calls `emit` for every vertex the agent may be at, at `time + 1`, coming from `vertex`
/// at `time`: the wait and every allowed move.
fn successors<G: Graph>(
    graph: &G,
    constraints: &ConstraintSet,
    vertex: Vertex,
    time: Time,
    mut emit: impl FnMut(Vertex),
) {
    let mut step = |to: Vertex| {
        if constraints.allows(vertex, to, time) {
            emit(to);
        }
    };

    step(vertex);
    graph.neighbors(vertex, step);
}

#[cfg(test)]
mod tests {
    //! Unit tests for MDD construction.

    use super::*;
    use crate::constraint::Constraint;
    use crate::graph::{Grid, distances_to};

    /// Builds the MDD of `agent` on `grid` at `depth` under `constraints`.
    fn mdd(grid: &Grid, agent: Agent, constraints: &ConstraintSet, depth: Time) -> Option<Mdd> {
        let distances = distances_to(grid, agent.goal);

        Mdd::build(grid, agent, constraints, &distances, depth)
    }

    /// On a 2x2 grid, the two shortest paths across split at time 1.
    #[test]
    fn diagonal_has_two_middle_vertices() {
        let grid = Grid::new(2, 2);
        let agent = Agent { start: 0, goal: 3 };
        let mdd = mdd(&grid, agent, &ConstraintSet::new(), 2).expect("reachable");

        assert_eq!(mdd.level(0), [0]);
        assert_eq!(mdd.level(1), [1, 2]);
        assert_eq!(mdd.level(2), [3]);
        assert_eq!(mdd.level(9), [3]);
        assert_eq!(mdd.singleton(1), None);
        assert_eq!(mdd.children(0, 0), [0, 1]);
    }

    /// Too shallow is `None`; one extra step admits waits and detours.
    #[test]
    fn depth_bounds_the_paths() {
        let grid = Grid::new(3, 1);
        let agent = Agent { start: 0, goal: 2 };

        assert_eq!(mdd(&grid, agent, &ConstraintSet::new(), 1), None);

        let wide = mdd(&grid, agent, &ConstraintSet::new(), 3).expect("reachable");
        assert_eq!(wide.level(1), [0, 1]);
        assert_eq!(wide.level(2), [1, 2]);
    }

    /// Constraints remove nodes, and a goal constraint after the depth rules it out.
    #[test]
    fn constraints_prune() {
        let grid = Grid::new(2, 2);
        let agent = Agent { start: 0, goal: 3 };
        let mut constraints = ConstraintSet::new();
        constraints.insert(&Constraint::Vertex {
            agent: 0,
            vertex: 1,
            time: 1,
        });

        let pruned = mdd(&grid, agent, &constraints, 2).expect("via 2");
        assert_eq!(pruned.singleton(1), Some(2));

        constraints.insert(&Constraint::Vertex {
            agent: 0,
            vertex: 3,
            time: 4,
        });
        assert_eq!(mdd(&grid, agent, &constraints, 2), None);
    }
}
