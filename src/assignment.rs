//! Agent-to-goal assignments in order of cost ([`Ranked`]), the root generator of
//! [`cbs_ta`](crate::cbs_ta()).
//!
//! An assignment gives each agent (row) a distinct goal (column) whose cost entry is
//! finite. The best one under an [`Objective`] comes from the Hungarian method
//! ([`best`]): for the sum of costs directly, for the makespan by the smallest bottleneck
//! threshold that still admits an assignment, then the cheapest sum below it. The next
//! best ones come from Murty's partitioning (Murty 1968): every popped assignment splits
//! its subproblem into children that each force a prefix of it and forbid the next pair.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use crate::path::Objective;

/// A cost matrix: `costs[agent][goal]`, `None` where the agent may not take the goal.
/// Every row has the same length.
pub(crate) type Costs = [Vec<Option<u32>>];

/// The order key of an assignment: the objective, then the sum of costs as a tie-breaker
/// (both equal for [`Objective::SumOfCosts`]).
pub(crate) type Key = (u64, u64);

/// The goal of each agent, indexed by agent.
pub(crate) type Assignment = Vec<usize>;

/// Every assignment of a cost matrix, cheapest first under an [`Objective`].
///
/// Each [`Ranked::next`] costs one solve per agent not yet forced in the popped subproblem.
pub(crate) struct Ranked<'a> {
    /// The cost matrix.
    costs: &'a Costs,
    /// What the assignments are ranked by.
    objective: Objective,
    /// Solved subproblems not yet returned, cheapest first.
    open: BinaryHeap<Candidate>,
    /// The number of subproblems pushed so far; breaks ties by creation order.
    pushed: u64,
}

impl<'a> Ranked<'a> {
    /// Ranks the assignments of `costs` by `objective`. Solves the unconstrained problem.
    pub(crate) fn new(costs: &'a Costs, objective: Objective) -> Self {
        let mut ranked = Self {
            costs,
            objective,
            open: BinaryHeap::new(),
            pushed: 0,
        };
        ranked.solve(Vec::new(), Vec::new());

        ranked
    }

    /// The key of the assignment [`Ranked::next`] returns, or `None` if none is left.
    pub(crate) fn peek(&self) -> Option<Key> {
        self.open.peek().map(|candidate| candidate.order.0.0)
    }

    /// The cheapest assignment not yet returned, or `None` if none is left.
    pub(crate) fn next(&mut self) -> Option<Assignment> {
        let Candidate {
            forced,
            forbidden,
            assignment,
            ..
        } = self.open.pop()?;

        self.partition(&forced, &forbidden, &assignment);

        Some(assignment)
    }

    /// Murty's partition of the subproblem (`forced`, `forbidden`) without `assignment`,
    /// its optimum: child `k` forces the pairs of `assignment` for the first `k` free
    /// agents and forbids the pair of the next one.
    fn partition(
        &mut self,
        forced: &[(usize, usize)],
        forbidden: &[(usize, usize)],
        assignment: &[usize],
    ) {
        let free: Vec<usize> = (0..assignment.len())
            .filter(|agent| forced.iter().all(|&(row, _)| row != *agent))
            .collect();

        let mut prefix = forced.to_vec();
        for &agent in &free {
            let pair = (agent, assignment[agent]);
            let mut excluded = forbidden.to_vec();
            excluded.push(pair);
            self.solve(prefix.clone(), excluded);
            prefix.push(pair);
        }
    }

    /// Solves the subproblem with the `forced` and `forbidden` pairs and opens it if it has
    /// an assignment.
    fn solve(&mut self, forced: Vec<(usize, usize)>, forbidden: Vec<(usize, usize)>) {
        let costs = constrained(self.costs, &forced, &forbidden);
        let Some((assignment, key)) = best(&costs, self.objective) else {
            return;
        };

        self.open.push(Candidate {
            order: Reverse((key, self.pushed)),
            forced,
            forbidden,
            assignment,
        });
        self.pushed += 1;
    }
}

/// A solved subproblem of [`Ranked`].
struct Candidate {
    /// The key and the creation order, reversed so that [`BinaryHeap`] pops the smallest.
    order: Reverse<(Key, u64)>,
    /// Pairs `(agent, goal)` every assignment of the subproblem contains.
    forced: Vec<(usize, usize)>,
    /// Pairs `(agent, goal)` no assignment of the subproblem contains.
    forbidden: Vec<(usize, usize)>,
    /// The subproblem's optimal assignment.
    assignment: Assignment,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order.cmp(&other.order)
    }
}

/// `costs` with every `forbidden` pair removed and, for each `forced` pair, every other
/// entry of its row and column.
fn constrained(
    costs: &Costs,
    forced: &[(usize, usize)],
    forbidden: &[(usize, usize)],
) -> Vec<Vec<Option<u32>>> {
    let mut costs = costs.to_vec();
    for &(agent, goal) in forbidden {
        costs[agent][goal] = None;
    }

    for &(agent, goal) in forced {
        for (row, entries) in costs.iter_mut().enumerate() {
            for (column, entry) in entries.iter_mut().enumerate() {
                if (row == agent) != (column == goal) {
                    *entry = None;
                }
            }
        }
    }

    costs
}

/// The best assignment of `costs` under `objective` and its key, or `None` if there is
/// none (more agents than goals, or no way to give every agent a distinct allowed goal).
fn best(costs: &Costs, objective: Objective) -> Option<(Assignment, Key)> {
    match objective {
        Objective::SumOfCosts => {
            let assignment = min_sum(costs)?;
            let sum = sum(costs, &assignment);
            Some((assignment, (sum, sum)))
        }
        Objective::Makespan => min_bottleneck(costs),
    }
}

/// The assignment with the smallest largest entry, the cheapest sum among those, and its
/// key: a binary search over the distinct entries for the smallest feasible threshold.
/// Without agents, the empty assignment with key `(0, 0)`.
fn min_bottleneck(costs: &Costs) -> Option<(Assignment, Key)> {
    if costs.is_empty() {
        return Some((Vec::new(), (0, 0)));
    }

    let mut thresholds: Vec<u32> = costs.iter().flatten().flatten().copied().collect();
    thresholds.sort_unstable();
    thresholds.dedup();

    let below = |threshold: u32| {
        let capped: Vec<Vec<Option<u32>>> = costs
            .iter()
            .map(|row| row.iter().map(|&c| c.filter(|&c| c <= threshold)).collect())
            .collect();
        min_sum(&capped)
    };

    let (mut low, mut high) = (0, thresholds.len());
    let mut found = None;
    while low < high {
        let middle = low + (high - low) / 2;
        if let Some(assignment) = below(thresholds[middle]) {
            found = Some((assignment, thresholds[middle]));
            high = middle;
        } else {
            low = middle + 1;
        }
    }

    let (assignment, bottleneck) = found?;
    let sum = sum(costs, &assignment);

    Some((assignment, (u64::from(bottleneck), sum)))
}

/// The sum of the entries `assignment` picks from `costs`, all finite.
fn sum(costs: &Costs, assignment: &[usize]) -> u64 {
    assignment
        .iter()
        .enumerate()
        .map(|(agent, &goal)| u64::from(costs[agent][goal].expect("an assigned pair is allowed")))
        .sum()
}

/// The assignment of `costs` with the smallest sum, by the Hungarian method with
/// potentials, `O(n^2 m)` for `n` agents and `m >= n` goals. `None` if there is none.
///
/// A forbidden entry costs more than any assignment of allowed entries, so the optimum
/// uses one only if no assignment avoids them.
fn min_sum(costs: &Costs) -> Option<Assignment> {
    let rows = costs.len();
    let columns = costs.first().map_or(0, Vec::len);
    if rows > columns {
        return None;
    }

    let largest = costs.iter().flatten().flatten().copied().max().unwrap_or(0);
    let forbidden =
        (i64::from(largest) + 1) * (i64::try_from(rows).expect("agent counts fit in i64") + 1);
    let entry = |row: usize, column: usize| costs[row][column].map_or(forbidden, i64::from);

    let column_of_row = hungarian(rows, columns, entry);
    let allowed = column_of_row
        .iter()
        .enumerate()
        .all(|(row, &column)| costs[row][column].is_some());

    allowed.then_some(column_of_row)
}

/// The minimum-sum assignment of `rows <= columns` with entries `entry(row, column)`: the
/// column of each row. The classic shortest-augmenting-path formulation with row and
/// column potentials; index `0` is a sentinel, so rows and columns are 1-based inside.
fn hungarian(rows: usize, columns: usize, entry: impl Fn(usize, usize) -> i64) -> Vec<usize> {
    let mut row_potential = vec![0_i64; rows + 1];
    let mut column_potential = vec![0_i64; columns + 1];
    let mut row_of_column = vec![0_usize; columns + 1];
    let mut previous = vec![0_usize; columns + 1];

    for row in 1..=rows {
        row_of_column[0] = row;
        let mut column = 0;
        let mut slack = vec![i64::MAX; columns + 1];
        let mut used = vec![false; columns + 1];

        while row_of_column[column] != 0 {
            used[column] = true;
            let current = row_of_column[column];
            let (mut delta, mut next) = (i64::MAX, 0);
            for candidate in 1..=columns {
                if used[candidate] {
                    continue;
                }

                let reduced = entry(current - 1, candidate - 1)
                    - row_potential[current]
                    - column_potential[candidate];
                if reduced < slack[candidate] {
                    slack[candidate] = reduced;
                    previous[candidate] = column;
                }

                if slack[candidate] < delta {
                    delta = slack[candidate];
                    next = candidate;
                }
            }

            for candidate in 0..=columns {
                if used[candidate] {
                    row_potential[row_of_column[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else {
                    slack[candidate] -= delta;
                }
            }

            column = next;
        }

        while column != 0 {
            let before = previous[column];
            row_of_column[column] = row_of_column[before];
            column = before;
        }
    }

    let mut column_of_row = vec![0; rows];
    for column in 1..=columns {
        if row_of_column[column] != 0 {
            column_of_row[row_of_column[column] - 1] = column - 1;
        }
    }

    column_of_row
}

#[cfg(test)]
mod tests {
    //! The ranked assignments against brute-force enumeration.

    use super::*;

    use proptest::prelude::*;

    /// Every assignment of `costs` with its key, by enumerating injective maps.
    fn all(costs: &Costs, objective: Objective) -> Vec<Key> {
        /// Extends `partial` by one agent at a time.
        fn extend(costs: &Costs, partial: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if partial.len() == costs.len() {
                out.push(partial.clone());
                return;
            }

            let agent = partial.len();
            for goal in 0..costs[agent].len() {
                if costs[agent][goal].is_some() && !partial.contains(&goal) {
                    partial.push(goal);
                    extend(costs, partial, out);
                    partial.pop();
                }
            }
        }

        let mut assignments = Vec::new();
        extend(costs, &mut Vec::new(), &mut assignments);

        let mut keys: Vec<Key> = assignments
            .iter()
            .map(|assignment| key_of(costs, assignment, objective))
            .collect();
        keys.sort_unstable();

        keys
    }

    /// Cost matrices with up to 4 agents and 5 goals, entries `0..6` or forbidden.
    fn matrix() -> impl Strategy<Value = Vec<Vec<Option<u32>>>> {
        (1_usize..=4, 1_usize..=5).prop_flat_map(|(rows, columns)| {
            prop::collection::vec(
                prop::collection::vec(prop::option::weighted(0.8, 0_u32..6), columns),
                rows,
            )
        })
    }

    proptest! {
        /// [`Ranked`] returns every assignment exactly once, in order of key, and each
        /// assignment is valid and has the key it was ranked by.
        #[test]
        fn ranks_every_assignment_in_order(costs in matrix(), makespan in any::<bool>()) {
            let objective = if makespan { Objective::Makespan } else { Objective::SumOfCosts };
            let expected = all(&costs, objective);

            let mut ranked = Ranked::new(&costs, objective);
            let mut seen = Vec::new();
            let mut keys = Vec::new();
            while let Some(key) = ranked.peek() {
                let assignment = ranked.next().expect("peeked");
                prop_assert!(!seen.contains(&assignment), "returned twice");
                for (agent, &goal) in assignment.iter().enumerate() {
                    prop_assert!(costs[agent][goal].is_some());
                }

                prop_assert_eq!(key, key_of(&costs, &assignment, objective));
                seen.push(assignment);
                keys.push(key);
            }

            prop_assert_eq!(keys, expected);
        }
    }

    /// The key of one `assignment` of `costs` under `objective`.
    fn key_of(costs: &Costs, assignment: &[usize], objective: Objective) -> Key {
        let total = sum(costs, assignment);
        let largest = assignment
            .iter()
            .enumerate()
            .map(|(agent, &goal)| u64::from(costs[agent][goal].expect("allowed")))
            .max()
            .unwrap_or(0);

        match objective {
            Objective::SumOfCosts => (total, total),
            Objective::Makespan => (largest, total),
        }
    }

    /// Without agents, both objectives rank exactly the empty assignment.
    #[test]
    fn no_agents_have_the_empty_assignment() {
        for objective in [Objective::SumOfCosts, Objective::Makespan] {
            let mut ranked = Ranked::new(&[], objective);

            assert_eq!(ranked.peek(), Some((0, 0)));
            assert_eq!(ranked.next(), Some(Vec::new()));
            assert_eq!(ranked.peek(), None);
        }
    }

    /// More agents than goals has no assignment.
    #[test]
    fn more_agents_than_goals_has_none() {
        let costs = vec![vec![Some(1)], vec![Some(2)]];

        assert!(best(&costs, Objective::SumOfCosts).is_none());
        assert!(Ranked::new(&costs, Objective::Makespan).peek().is_none());
    }

    /// A forbidden entry is avoided even when it would be cheaper to use it.
    #[test]
    fn forbidden_entries_are_never_used() {
        let costs = vec![vec![None, Some(9)], vec![Some(9), Some(0)]];

        let (assignment, key) = best(&costs, Objective::SumOfCosts).expect("feasible");

        assert_eq!(assignment, vec![1, 0]);
        assert_eq!(key, (18, 18));
    }
}
