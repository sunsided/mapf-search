//! Collision sets ([`CollisionSet`]): the agents of a joint state that must be planned
//! together, as disjoint groups.
//!
//! Basic M* keeps one group per state: every collision is merged into it. Recursive M*
//! (Wagner and Choset 2015) keeps the groups apart as long as no collision links them, and
//! plans each group as its own subproblem.

use alloc::vec::Vec;

use crate::instance::AgentId;

/// Disjoint groups of agents that collided, directly or through the states after this one.
///
/// Every group is sorted ascending and has at least two agents; the groups are sorted by
/// their first agent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CollisionSet {
    /// The groups; disjoint.
    groups: Vec<Vec<AgentId>>,
}

impl CollisionSet {
    /// The set with the single group `group`, which must be sorted and duplicate-free.
    pub(super) fn of(group: Vec<AgentId>) -> Self {
        debug_assert!(group.is_sorted(), "groups are sorted");

        Self {
            groups: Vec::from([group]),
        }
    }

    /// The groups, disjoint and each sorted.
    pub(super) fn groups(&self) -> &[Vec<AgentId>] {
        &self.groups
    }

    /// Returns `true` if `agent` is in a group.
    pub(super) fn contains(&self, agent: AgentId) -> bool {
        self.groups
            .iter()
            .any(|group| group.binary_search(&agent).is_ok())
    }

    /// Merges every group of `other` into this set (see [`CollisionSet::merge`]). Returns
    /// `true` if the set changed.
    pub(super) fn merge_all(&mut self, other: &Self, single: bool) -> bool {
        other
            .groups
            .iter()
            .fold(false, |changed, group| self.merge(group, single) | changed)
    }

    /// Adds the sorted `group`: it and every group it shares an agent with become one group.
    /// With `single` (basic M*), every group is merged into one. Returns `true` if the set
    /// changed, that is, if no existing group already holds all of `group`.
    pub(super) fn merge(&mut self, group: &[AgentId], single: bool) -> bool {
        let covered = self
            .groups
            .iter()
            .any(|existing| group.iter().all(|a| existing.binary_search(a).is_ok()));
        if covered {
            return false;
        }

        let mut merged = group.to_vec();
        self.groups.retain(|existing| {
            let joins = single || existing.iter().any(|a| group.binary_search(a).is_ok());
            if joins {
                merged.extend_from_slice(existing);
            }

            !joins
        });
        merged.sort_unstable();
        merged.dedup();

        let at = self
            .groups
            .partition_point(|existing| existing[0] < merged[0]);
        self.groups.insert(at, merged);

        true
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for merging collision sets.

    use super::*;
    use alloc::vec;

    /// Disjoint collisions stay apart; a linking one joins them.
    #[test]
    fn groups_merge_only_when_linked() {
        let mut set = CollisionSet::default();
        assert!(set.merge(&[3, 4], false));
        assert!(set.merge(&[0, 1], false));
        assert_eq!(set.groups(), [vec![0, 1], vec![3, 4]]);
        assert!(!set.merge(&[0, 1], false), "already covered");

        assert!(set.merge(&[1, 3], false));
        assert_eq!(set.groups(), [vec![0, 1, 3, 4]]);
        assert!(set.contains(4) && !set.contains(2));
    }

    /// Basic M* merges everything into one group.
    #[test]
    fn single_mode_merges_everything() {
        let mut set = CollisionSet::of(vec![0, 1]);
        let other = CollisionSet::of(vec![4, 5]);

        assert!(set.merge_all(&other, true));
        assert_eq!(set.groups(), [vec![0, 1, 4, 5]]);
        assert!(!set.merge_all(&other, true));
    }
}
