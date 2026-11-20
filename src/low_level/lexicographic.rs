//! A cost compared lexicographically ([`Lexicographic`]): path length first, then the
//! number of conflicts, for CBS's conflict-avoidance tie-break.
//!
//! `heuristic-search` runs A* over any [`Cost`], so a pair cost needs no engine change.
//! The type lives here until `heuristic-search` or `search-core` ships a generic one.

use core::ops::Add;

use heuristic_search::Cost;

/// A path cost: [`Lexicographic::primary`] first, ties broken by
/// [`Lexicographic::secondary`]. Both add up along a path.
///
/// A* over it with a heuristic `(h, 0)`, where `h` is consistent for the primary cost,
/// returns a path of minimal primary cost and, among those, of minimal secondary cost: the
/// heuristic stays consistent, since `(h(u), 0) <= (c + h(v), s)` whenever
/// `h(u) <= c + h(v)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Lexicographic {
    /// The cost that is minimized: here, the number of time steps.
    pub(crate) primary: u32,
    /// The tie-breaker: here, the number of conflicts with other agents' paths.
    pub(crate) secondary: u32,
}

impl Add for Lexicographic {
    type Output = Self;

    /// Component-wise.
    fn add(self, other: Self) -> Self {
        Self {
            primary: self.primary + other.primary,
            secondary: self.secondary + other.secondary,
        }
    }
}

impl Cost for Lexicographic {
    const ZERO: Self = Self {
        primary: 0,
        secondary: 0,
    };

    /// Component-wise, saturating.
    fn saturating_add(self, other: Self) -> Self {
        Self {
            primary: self.primary.saturating_add(other.primary),
            secondary: self.secondary.saturating_add(other.secondary),
        }
    }

    /// Component-wise, saturating.
    fn saturating_times(self, factor: u32) -> Self {
        Self {
            primary: self.primary.saturating_mul(factor),
            secondary: self.secondary.saturating_mul(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the lexicographic order.

    use super::*;

    /// Builds a cost.
    fn cost(primary: u32, secondary: u32) -> Lexicographic {
        Lexicographic { primary, secondary }
    }

    /// The primary cost decides; the secondary breaks ties.
    #[test]
    fn orders_lexicographically() {
        assert!(cost(1, 9) < cost(2, 0));
        assert!(cost(2, 0) < cost(2, 1));
        assert_eq!(cost(1, 2) + cost(3, 4), cost(4, 6));
        assert_eq!(
            cost(u32::MAX, 1).saturating_add(cost(1, 1)),
            cost(u32::MAX, 2)
        );
        assert_eq!(cost(2, 3).saturating_times(2), cost(4, 6));
    }
}
