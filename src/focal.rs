//! A focal list ([`FocalList`]) with a separate focal *value*: the open list of ECBS's
//! high level ([`ecbs`](crate::ecbs())).
//!
//! It holds items under two orders. The *open* order is by an admissible value `f`. The
//! *focal* sublist is every item whose *value* is at most `w * f_min`, ordered by a free
//! secondary key. Popping from the focal sublist keeps every pop's value within factor `w`
//! of the lower bound `f_min`. ECBS's high level uses the lower bound of a node as `f` and
//! its cost as the value.
//!
//! The low level uses `heuristic-search`'s `FocalList` instead. That one decides focal
//! membership by `f` itself (`f <= w * f_min`), which is right for a single-agent search,
//! where the value is `f`. The high level needs the two apart: a node's lower bound orders
//! the list and bounds the result, but its sum of costs, which may be larger, must be
//! within `w` of `f_min`. Ordering by the lower bound and admitting by the lower bound
//! would admit nodes whose cost exceeds the bound and break the `w` guarantee; ordering by
//! the cost would lose `f_min` as a lower bound. So this list stays until the upstream one
//! takes a separate value.

use alloc::collections::{BTreeMap, BTreeSet};

use heuristic_search::Weight;

/// A handle to an item in a [`FocalList`], for [`FocalList::remove`]. Unique per list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Handle(u64);

/// An open list with a focal sublist; see the module docs.
///
/// Every operation costs `O(log n)`, plus `O(k log n)` in [`FocalList::pop`] when `f_min`
/// changed and `k` items enter or leave the focal sublist.
#[derive(Debug, Clone)]
pub(crate) struct FocalList<S, T> {
    /// The suboptimality factor `w`; at least `1`.
    weight: Weight,
    /// Every item, by `f`, then push order.
    open: BTreeSet<(u32, Handle)>,
    /// Every item, by value, then push order.
    values: BTreeSet<(u32, Handle)>,
    /// The items with value at most `bound`, by secondary key, then value, then push order.
    focal: BTreeSet<(S, u32, Handle)>,
    /// The `f`, value, secondary key and payload of every item.
    items: BTreeMap<Handle, Item<S, T>>,
    /// The focal sublist holds exactly the items with value at most this; `None` while it
    /// is empty and unsynchronized (initially, and after the list ran empty).
    bound: Option<u32>,
    /// The handle the next push returns.
    next: u64,
}

impl<S: Ord + Copy, T> FocalList<S, T> {
    /// An empty list with suboptimality factor `weight`.
    pub(crate) fn new(weight: Weight) -> Self {
        Self {
            weight,
            open: BTreeSet::new(),
            values: BTreeSet::new(),
            focal: BTreeSet::new(),
            items: BTreeMap::new(),
            bound: None,
            next: 0,
        }
    }

    /// Adds `item` with admissible lower bound `f`, focal value `value` (at least `f`) and
    /// secondary key `secondary`.
    pub(crate) fn push(&mut self, f: u32, value: u32, secondary: S, item: T) -> Handle {
        debug_assert!(
            f <= value,
            "the value of an item is at least its lower bound"
        );
        let handle = Handle(self.next);
        self.next += 1;

        self.open.insert((f, handle));
        self.values.insert((value, handle));
        if self.bound.is_some_and(|bound| value <= bound) {
            self.focal.insert((secondary, value, handle));
        }

        self.items.insert(
            handle,
            Item {
                f,
                value,
                secondary,
                payload: item,
            },
        );

        handle
    }

    /// Removes the item of `handle`, if it is still in the list.
    pub(crate) fn remove(&mut self, handle: Handle) -> Option<T> {
        let item = self.items.remove(&handle)?;
        self.open.remove(&(item.f, handle));
        self.values.remove(&(item.value, handle));
        self.focal.remove(&(item.secondary, item.value, handle));

        Some(item.payload)
    }

    /// The smallest `f` in the list, if it is not empty.
    pub(crate) fn f_min(&self) -> Option<u32> {
        self.open.first().map(|&(f, _)| f)
    }

    /// Removes and returns the focal item with the smallest secondary key, with the `f_min`
    /// of the list just before the pop.
    ///
    /// Returns `None` if the list is empty or no item's value is within `w * f_min`; the
    /// latter cannot happen if every item's value is at most `w` times its `f`.
    pub(crate) fn pop(&mut self) -> Option<(T, u32)> {
        let f_min = self.f_min()?;
        self.synchronize(within(f_min, self.weight));

        let &(_, _, handle) = self.focal.first()?;
        let item = self.remove(handle).expect("focal items are in the list");
        if self.items.is_empty() {
            self.bound = None;
        }

        Some((item, f_min))
    }

    /// Makes the focal sublist hold exactly the items with value at most `bound`.
    fn synchronize(&mut self, bound: u32) {
        let old = self.bound.replace(bound);
        let (low, high, enter) = match old {
            None => (0, bound, true),
            Some(old) if old < bound => (old + 1, bound, true),
            Some(old) if old > bound => (bound + 1, old, false),
            Some(_) => return,
        };

        let range = (low, Handle(0))..=(high, Handle(u64::MAX));
        for &(value, handle) in self.values.range(range) {
            let secondary = self.items[&handle].secondary;
            if enter {
                self.focal.insert((secondary, value, handle));
            } else {
                self.focal.remove(&(secondary, value, handle));
            }
        }
    }
}

/// One entry of a [`FocalList`].
#[derive(Debug, Clone)]
struct Item<S, T> {
    /// The admissible lower bound that orders the open list.
    f: u32,
    /// The value compared with `w * f_min` for focal membership; at least [`Item::f`].
    value: u32,
    /// The focal order key.
    secondary: S,
    /// What the caller stored.
    payload: T,
}

/// `floor(weight * f_min)`, saturating: the largest `f` allowed in the focal sublist.
pub(crate) fn within(f_min: u32, weight: Weight) -> u32 {
    let scaled = u64::from(f_min) * u64::from(weight.numerator()) / u64::from(weight.denominator());

    u32::try_from(scaled).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the focal list.

    use super::*;
    use alloc::vec::Vec;

    /// Pops all items as `(item, f_min)`.
    fn drain(list: &mut FocalList<u32, char>) -> Vec<(char, u32)> {
        core::iter::from_fn(|| list.pop()).collect()
    }

    /// Pushes `item` with value equal to `f`, as a single-agent search does.
    fn push(list: &mut FocalList<u32, char>, f: u32, secondary: u32, item: char) -> Handle {
        list.push(f, f, secondary, item)
    }

    /// With weight 1, the list pops by `f`, ties by secondary key.
    #[test]
    fn weight_one_is_best_first() {
        let mut list = FocalList::new(Weight::ONE);
        push(&mut list, 5, 0, 'a');
        push(&mut list, 3, 9, 'b');
        push(&mut list, 3, 1, 'c');

        assert_eq!(drain(&mut list), [('c', 3), ('b', 3), ('a', 5)]);
    }

    /// With weight 2, anything up to twice `f_min` competes on the secondary key.
    #[test]
    fn weight_two_prefers_the_secondary_key() {
        let mut list = FocalList::new(Weight::integer(2));
        push(&mut list, 4, 5, 'a');
        push(&mut list, 7, 0, 'b');
        push(&mut list, 9, 0, 'c');

        assert_eq!(list.pop(), Some(('b', 4)));
        assert_eq!(list.pop(), Some(('a', 4)));
        assert_eq!(list.pop(), Some(('c', 9)));
        assert_eq!(list.pop(), None);
    }

    /// Removing the `f_min` item can shrink the focal sublist again.
    #[test]
    fn removal_and_falling_bounds() {
        let mut list = FocalList::new(Weight::new(3, 2));
        let low = push(&mut list, 2, 9, 'a');
        push(&mut list, 3, 0, 'b');
        push(&mut list, 10, 0, 'c');

        assert_eq!(list.pop(), Some(('b', 2)));
        assert_eq!(list.remove(low), Some('a'));
        assert_eq!(list.remove(low), None);

        push(&mut list, 4, 0, 'd');
        assert_eq!(list.pop(), Some(('d', 4)));
        assert_eq!(list.pop(), Some(('c', 10)));
    }

    /// ECBS's high level: lower bound `f`, focal membership by the larger cost.
    #[test]
    fn value_decides_membership() {
        let mut list = FocalList::new(Weight::integer(2));
        list.push(3, 6, 1, 'a');
        list.push(4, 5, 0, 'b');
        list.push(5, 7, 0, 'c');

        assert_eq!(list.pop(), Some(('b', 3)));
        assert_eq!(list.pop(), Some(('a', 3)));
        assert_eq!(list.pop(), Some(('c', 5)));
    }

    /// The bound is the floor of the scaled `f_min`.
    #[test]
    fn bound_rounds_down() {
        assert_eq!(within(10, Weight::new(11, 10)), 11);
        assert_eq!(within(5, Weight::new(3, 2)), 7);
        assert_eq!(within(u32::MAX, Weight::integer(2)), u32::MAX);
    }
}
