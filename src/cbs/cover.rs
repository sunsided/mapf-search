//! Lower bounds on vertex covers of the graphs the CBS heuristics build: unweighted for CG
//! and DG ([`vertex_cover_bound`]), edge-weighted for WDG ([`weighted_cover_bound`]).
//!
//! Each vertex is an agent; an edge says that the two agents' costs must rise together by
//! at least its weight. Any cover is then a lower bound on the rise of the sum of costs, so
//! the minimum cover is admissible, and so is every lower bound on it.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::instance::AgentId;

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

/// A lower bound on the minimum *edge-weighted* vertex cover of `edges` (Li et al. 2019):
/// the least `sum x_i` over non-negative integers with `x_a + x_b >= w` for every edge
/// `(a, b, w)`. Exact per connected component while the search space of a component stays
/// below [`EXACT_WEIGHTED_LIMIT`] assignments, else the sum of the weights of a greedy set
/// of disjoint edges, which every cover must pay.
pub(super) fn weighted_cover_bound(edges: &[(AgentId, AgentId, u32)]) -> u32 {
    components(edges)
        .iter()
        .map(|component| {
            exact_weighted_cover(component).unwrap_or_else(|| disjoint_weight(component))
        })
        .sum()
}

/// Assignments up to which [`weighted_cover_bound`] searches a component exactly: the
/// product over its agents of one plus the largest incident weight.
const EXACT_WEIGHTED_LIMIT: u64 = 1 << 16;

/// The edges of `edges` with positive weight, grouped by connected component.
fn components(edges: &[(AgentId, AgentId, u32)]) -> Vec<Vec<(AgentId, AgentId, u32)>> {
    let mut parent: BTreeMap<AgentId, AgentId> = BTreeMap::new();
    let weighted: Vec<_> = edges.iter().copied().filter(|&(_, _, w)| w > 0).collect();
    for &(a, b, _) in &weighted {
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    let mut groups: BTreeMap<AgentId, Vec<(AgentId, AgentId, u32)>> = BTreeMap::new();
    for edge in weighted {
        groups
            .entry(root(&mut parent, edge.0))
            .or_default()
            .push(edge);
    }

    groups.into_values().collect()
}

/// The representative of `agent` in the union-find forest `parent`.
fn root(parent: &mut BTreeMap<AgentId, AgentId>, agent: AgentId) -> AgentId {
    let mut at = agent;
    while let Some(&up) = parent.get(&at) {
        at = up;
    }

    at
}

/// The sum of the weights of a greedy set of pairwise disjoint edges, heaviest first.
fn disjoint_weight(edges: &[(AgentId, AgentId, u32)]) -> u32 {
    let mut sorted = edges.to_vec();
    sorted.sort_by_key(|&(_, _, w)| core::cmp::Reverse(w));

    let mut used: Vec<AgentId> = Vec::new();
    let mut total = 0;
    for (a, b, w) in sorted {
        if !used.contains(&a) && !used.contains(&b) {
            used.extend([a, b]);
            total += w;
        }
    }

    total
}

/// The minimum weighted cover of one component, or `None` if its search space exceeds
/// [`EXACT_WEIGHTED_LIMIT`].
fn exact_weighted_cover(edges: &[(AgentId, AgentId, u32)]) -> Option<u32> {
    let mut agents: Vec<AgentId> = edges.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    agents.sort_unstable();
    agents.dedup();
    let index = |agent| agents.binary_search(&agent).expect("an end of an edge");
    let local: Vec<(usize, usize, u32)> = edges
        .iter()
        .map(|&(a, b, w)| (index(a), index(b), w))
        .collect();

    let mut most = alloc::vec![0; agents.len()];
    for &(a, b, w) in &local {
        most[a] = most[a].max(w);
        most[b] = most[b].max(w);
    }

    let space = most
        .iter()
        .try_fold(1_u64, |product, &m| product.checked_mul(u64::from(m) + 1))?;
    if space > EXACT_WEIGHTED_LIMIT {
        return None;
    }

    let mut search = WeightedCover {
        edges: &local,
        most: &most,
        values: alloc::vec![0; agents.len()],
        best: most.iter().sum(),
    };
    search.assign(0, 0);

    Some(search.best)
}

/// The branch-and-bound search of [`exact_weighted_cover`].
struct WeightedCover<'a> {
    /// The edges, by local agent index.
    edges: &'a [(usize, usize, u32)],
    /// The largest weight at each agent: no cover needs more there.
    most: &'a [u32],
    /// The values assigned so far, by agent; only the first `next` are meaningful.
    values: Vec<u32>,
    /// The cheapest cover found; starts at the trivial one.
    best: u32,
}

impl WeightedCover<'_> {
    /// Tries every value of agent `next` that, with the values of the agents before it,
    /// covers each edge between them, then recurses; `sum` is the total of those values.
    fn assign(&mut self, next: usize, sum: u32) {
        if sum >= self.best {
            return;
        }

        if next == self.values.len() {
            self.best = sum;
            return;
        }

        let least = self
            .edges
            .iter()
            .filter_map(|&(a, b, w)| match (a, b) {
                (a, b) if a == next && b < next => Some(w.saturating_sub(self.values[b])),
                (a, b) if b == next && a < next => Some(w.saturating_sub(self.values[a])),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        for value in least..=self.most[next] {
            self.values[next] = value;
            self.assign(next + 1, sum + value);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the cover bounds.

    use super::*;

    /// A triangle needs two vertices, a star one, a path of four edges two.
    #[test]
    fn covers_small_graphs() {
        assert_eq!(vertex_cover_bound(&[]), 0);
        assert_eq!(vertex_cover_bound(&[(0, 1), (1, 2), (0, 2)]), 2);
        assert_eq!(vertex_cover_bound(&[(0, 1), (0, 2), (0, 3)]), 1);
        assert_eq!(vertex_cover_bound(&[(0, 1), (1, 2), (2, 3), (3, 4)]), 2);
    }

    /// Weighted covers: a heavy star pays its centre once; a triangle with weights 2, 2, 2
    /// costs 3 (one each); separate components add up; zero weights are ignored.
    #[test]
    fn covers_weighted_graphs() {
        assert_eq!(weighted_cover_bound(&[]), 0);
        assert_eq!(weighted_cover_bound(&[(0, 1, 3), (0, 2, 3), (0, 3, 2)]), 3);
        assert_eq!(weighted_cover_bound(&[(0, 1, 2), (1, 2, 2), (0, 2, 2)]), 3);
        assert_eq!(weighted_cover_bound(&[(0, 1, 2), (5, 6, 1), (7, 8, 0)]), 3);
        assert_eq!(weighted_cover_bound(&[(0, 1, 1), (1, 2, 1), (0, 2, 1)]), 2);
    }

    /// Above the exact limit, the disjoint-edge bound stays below the true cover.
    #[test]
    fn large_components_fall_back_to_disjoint_edges() {
        let path: Vec<_> = (0..20).map(|a| (a, a + 1, 5)).collect();

        assert_eq!(
            weighted_cover_bound(&path),
            50,
            "ten disjoint edges of weight 5"
        );
    }
}
