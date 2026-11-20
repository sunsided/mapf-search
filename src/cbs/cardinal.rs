//! Conflict classification by MDDs (Boyarski et al. 2015) and the CG heuristic built on it
//! (Felner et al. 2018).
//!
//! A conflict is *cardinal* for an agent if resolving it against that agent raises the
//! agent's cost: every one of its cheapest paths, so its MDD, has the conflicting vertex (or
//! edge) as the only option at that time. A conflict cardinal for both agents raises the
//! sum of costs of every child by at least 1.

use alloc::rc::Rc;
use alloc::vec::Vec;

use crate::conflict::Conflict;
use crate::instance::AgentId;
use crate::mdd::Mdd;

/// How much resolving a conflict must cost; ordered from least to most urgent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Cardinality {
    /// Neither child necessarily costs more.
    NonCardinal,
    /// One of the two children costs more.
    SemiCardinal,
    /// Both children cost more.
    Cardinal,
}

/// The cardinality of `conflict`, given the MDD of each of its two agents at its current
/// cost.
///
/// # Panics
///
/// Panics if an involved agent has no MDD in `mdds`.
pub(super) fn cardinality(conflict: &Conflict, mdds: &[Option<Rc<Mdd>>]) -> Cardinality {
    let (a, b) = conflict.agents();
    let mdd = |agent: AgentId| mdds[agent].as_deref().expect("MDDs of conflicting agents");
    let forced = match *conflict {
        Conflict::Vertex { vertex, time, .. } => {
            [a, b].map(|agent| mdd(agent).singleton(time) == Some(vertex))
        }
        Conflict::Edge { from, to, time, .. } => {
            [(a, from, to), (b, to, from)].map(|(agent, from, to)| {
                let mdd = mdd(agent);
                mdd.singleton(time) == Some(from) && mdd.singleton(time + 1) == Some(to)
            })
        }
    };

    match forced {
        [true, true] => Cardinality::Cardinal,
        [false, false] => Cardinality::NonCardinal,
        _ => Cardinality::SemiCardinal,
    }
}

/// A lower bound on the size of a minimum vertex cover of the graph with `edges`: exact up
/// to [`EXACT_COVER_LIMIT`], else the largest size proven necessary.
///
/// Each edge is a pair of agents with a cardinal conflict; each cover vertex is an agent
/// whose cost must rise by at least 1, so the result is an admissible bound on the
/// sum-of-costs increase.
pub(super) fn vertex_cover_bound(edges: &[(AgentId, AgentId)]) -> u32 {
    let mut size = matching_size(edges);
    while size < EXACT_COVER_LIMIT && !has_cover(edges, size) {
        size += 1;
    }

    size
}

/// Cover sizes up to which [`vertex_cover_bound`] searches exactly; the search is
/// exponential in the size.
const EXACT_COVER_LIMIT: u32 = 12;

/// The size of a greedy maximal matching: every vertex cover has at least that many
/// vertices.
fn matching_size(edges: &[(AgentId, AgentId)]) -> u32 {
    let mut matched: Vec<AgentId> = Vec::new();
    let mut size = 0;
    for &(a, b) in edges {
        if !matched.contains(&a) && !matched.contains(&b) {
            matched.extend([a, b]);
            size += 1;
        }
    }

    size
}

/// Returns `true` if at most `size` vertices cover `edges`: branch on the two ends of the
/// first edge.
fn has_cover(edges: &[(AgentId, AgentId)], size: u32) -> bool {
    let Some(&(a, b)) = edges.first() else {
        return true;
    };

    if size == 0 {
        return false;
    }

    [a, b].into_iter().any(|vertex| {
        let rest: Vec<_> = edges
            .iter()
            .copied()
            .filter(|&(x, y)| x != vertex && y != vertex)
            .collect();

        has_cover(&rest, size - 1)
    })
}

#[cfg(test)]
mod tests {
    //! Unit tests for the vertex cover bound.

    use super::*;

    /// A triangle needs two vertices, a star one, a path of four edges two.
    #[test]
    fn covers_small_graphs() {
        assert_eq!(vertex_cover_bound(&[]), 0);
        assert_eq!(vertex_cover_bound(&[(0, 1), (1, 2), (0, 2)]), 2);
        assert_eq!(vertex_cover_bound(&[(0, 1), (0, 2), (0, 3)]), 1);
        assert_eq!(vertex_cover_bound(&[(0, 1), (1, 2), (2, 3), (3, 4)]), 2);
    }
}
