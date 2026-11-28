//! Independence detection (Standley 2010) over CBS: agents that do not interact are solved
//! as separate, smaller problems ([`cbs_id`]), and those subproblems can be solved on
//! several threads ([`cbs_id_par`], feature `parallel`).
//!
//! Every agent starts in a group of its own. Each round solves the groups that are new
//! with CBS, then looks for conflicts between the paths of different groups. Groups in
//! conflict merge, every pair of them in the same round, and the merged groups are solved
//! again. The run ends when no two groups conflict.
//!
//! ```mermaid
//! flowchart TD
//!     start[one group per agent] --> solve[solve the new groups with CBS]
//!     solve -->|a group is unsolvable| unsolvable[Unsolvable]
//!     solve -->|a group hits its limits| limit[LimitReached]
//!     solve --> detect{paths of two groups conflict?}
//!     detect -->|no| solved[Solved]
//!     detect -->|yes| merge[merge every conflicting pair of groups]
//!     merge --> solve
//! ```
//!
//! The groups solved in one round share nothing, so they can be solved in parallel; the
//! result, statistics included, does not depend on how. The plan is optimal for the
//! objective: each group's plan is optimal for its agents, and restricting an optimal plan
//! of all agents to a group gives a plan of the group, so each group's optimum is at most
//! its share of the overall optimum, for the sum of costs and for the makespan alike. The
//! final groups do not conflict, so their plans together are a plan of that cost.

use alloc::vec::Vec;

use heuristic_search::{Limits, Stats};

use crate::cbs::{CbsConfig, cbs_with};
use crate::conflict::{Conflict, conflicts};
use crate::graph::Graph;
use crate::instance::{AgentId, Instance};
use crate::path::Path;
use crate::solution::{MapfResult, Outcome, Solution};

/// Finds an optimal conflict-free plan with CBS and independence detection: groups of
/// agents whose paths do not conflict are solved separately, and conflicting groups merge
/// and are solved together.
///
/// `config` is the CBS configuration of every group, `config.objective` the objective of
/// the whole plan. `limits` bound each CBS run on its own, so the whole run may expand up
/// to their bound times the number of group solves. [`MapfResult::stats`] adds up the
/// statistics of every CBS run.
///
/// The result is [`Outcome::Unsolvable`] if a group is unsolvable, which makes the whole
/// instance unsolvable, and [`Outcome::LimitReached`] if a group's CBS run hits `limits`.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, CbsConfig, Grid, Instance, Limits, Objective, cbs_id};
///
/// // Two pairs of agents far apart: each pair is one group, solved on its own.
/// let grid = Grid::new(8, 2);
/// let agents = vec![
///     Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) },
///     Agent { start: grid.vertex(2, 1), goal: grid.vertex(0, 1) },
///     Agent { start: grid.vertex(5, 0), goal: grid.vertex(7, 0) },
///     Agent { start: grid.vertex(7, 1), goal: grid.vertex(5, 1) },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let config = CbsConfig::improved(Objective::SumOfCosts);
/// let result = cbs_id(&instance, config, Limits::expansions(1_000));
/// assert_eq!(result.solution().map(|s| s.sum_of_costs), Some(8));
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
#[must_use]
pub fn cbs_id<G: Graph>(instance: &Instance<G>, config: CbsConfig, limits: Limits) -> MapfResult {
    detect(instance, |groups| {
        groups
            .iter()
            .map(|group| solve_group(instance, config, limits, group))
            .collect()
    })
}

/// [`cbs_id`] with the groups of each round solved on up to `threads` threads
/// (heuristic-search's `solve_batch`). The result, statistics included, is the same as
/// that of [`cbs_id`].
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, CbsConfig, Grid, Instance, Limits, Objective, cbs_id, cbs_id_par};
///
/// let grid = Grid::new(4, 4);
/// let agents = vec![
///     Agent { start: 0, goal: 15 },
///     Agent { start: 15, goal: 0 },
///     Agent { start: 3, goal: 12 },
/// ];
/// let instance = Instance::new(grid, agents)?;
///
/// let config = CbsConfig::improved(Objective::SumOfCosts);
/// let limits = Limits::expansions(1_000);
/// assert_eq!(cbs_id_par(&instance, config, limits, 4), cbs_id(&instance, config, limits));
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
#[cfg(feature = "parallel")]
#[must_use]
pub fn cbs_id_par<G: Graph + Sync>(
    instance: &Instance<G>,
    config: CbsConfig,
    limits: Limits,
    threads: usize,
) -> MapfResult {
    detect(instance, |groups| {
        heuristic_search::solve_batch(
            groups,
            threads,
            || (),
            |(), group| solve_group(instance, config, limits, group),
        )
    })
}

/// CBS on the agents of `group`, in that order, on the graph of `instance`.
fn solve_group<G: Graph>(
    instance: &Instance<G>,
    config: CbsConfig,
    limits: Limits,
    group: &[AgentId],
) -> MapfResult {
    let agents = group
        .iter()
        .map(|&agent| instance.agents()[agent])
        .collect();
    let sub = Instance::new(instance.graph(), agents).expect("a subset of a valid instance");

    cbs_with(&sub, config, limits)
}

/// Runs independence detection, solving each round's new groups with `solve`, which
/// returns one result per group, in order.
fn detect<G: Graph>(
    instance: &Instance<G>,
    mut solve: impl FnMut(&[Vec<AgentId>]) -> Vec<MapfResult>,
) -> MapfResult {
    let count = instance.agents().len();
    let mut groups: Vec<Vec<AgentId>> = (0..count).map(|agent| Vec::from([agent])).collect();
    let mut fresh: Vec<usize> = (0..count).collect();
    let mut paths: Vec<Path> = alloc::vec![Vec::new(); count];
    let mut stats = Stats::default();

    loop {
        let batch: Vec<Vec<AgentId>> = fresh.iter().map(|&g| groups[g].clone()).collect();
        for (group, result) in batch.iter().zip(solve(&batch)) {
            add(&mut stats, &result.stats);
            let Outcome::Solved(solution) = result.outcome else {
                return MapfResult {
                    outcome: result.outcome,
                    stats,
                };
            };

            for (&agent, path) in group.iter().zip(solution.paths) {
                paths[agent] = path;
            }
        }

        let found = conflicts(&paths);
        if found.is_empty() {
            return MapfResult {
                outcome: Outcome::Solved(Solution::new(paths)),
                stats,
            };
        }

        (groups, fresh) = merge(&groups, &found);
    }
}

/// The groups after merging every pair of `groups` that one of `found` joins: each sorted,
/// ordered by their first agent. Also the indices of the merged groups, which are new.
fn merge(groups: &[Vec<AgentId>], found: &[Conflict]) -> (Vec<Vec<AgentId>>, Vec<usize>) {
    let count: usize = groups.iter().map(Vec::len).sum();
    let mut group_of = alloc::vec![0; count];
    for (index, group) in groups.iter().enumerate() {
        for &agent in group {
            group_of[agent] = index;
        }
    }

    let mut root: Vec<usize> = (0..groups.len()).collect();
    for conflict in found {
        let (a, b) = conflict.agents();
        let (a, b) = (find(&mut root, group_of[a]), find(&mut root, group_of[b]));
        root[a.max(b)] = a.min(b);
    }

    let mut merged: Vec<Vec<AgentId>> = alloc::vec![Vec::new(); groups.len()];
    let mut parts = alloc::vec![0_usize; groups.len()];
    for (index, group) in groups.iter().enumerate() {
        let top = find(&mut root, index);
        merged[top].extend_from_slice(group);
        parts[top] += 1;
    }

    let mut next = Vec::new();
    let mut fresh = Vec::new();
    for (group, parts) in merged.into_iter().zip(parts) {
        if parts == 0 {
            continue;
        }

        let mut group = group;
        group.sort_unstable();
        if parts > 1 {
            fresh.push(next.len());
        }

        next.push(group);
    }

    (next, fresh)
}

/// The representative of `group` in the union-find forest `root`, halving paths on the
/// way.
fn find(root: &mut [usize], mut group: usize) -> usize {
    while root[group] != group {
        root[group] = root[root[group]];
        group = root[group];
    }

    group
}

/// Adds the counters of `more` to `total`.
fn add(total: &mut Stats, more: &Stats) {
    total.expanded += more.expanded;
    total.generated += more.generated;
    total.reopened += more.reopened;
    total.evaluations += more.evaluations;
}

#[cfg(test)]
mod tests {
    //! Unit tests for independence detection.

    use super::*;
    use crate::cbs::cbs;
    use crate::graph::Grid;
    use crate::instance::Agent;
    use crate::path::Objective;

    /// Two pairs of agents that each swap through a pocket, far apart.
    fn two_pairs() -> Instance<Grid> {
        let grid = Grid::parse(&["...@...", "@.@@@.@"]);
        let agents = Vec::from([
            Agent { start: 0, goal: 2 },
            Agent { start: 2, goal: 0 },
            Agent { start: 4, goal: 6 },
            Agent { start: 6, goal: 4 },
        ]);

        Instance::new(grid, agents).expect("valid")
    }

    /// The pairs are solved apart and cost what CBS on all agents costs.
    #[test]
    fn independent_pairs_match_cbs() {
        let instance = two_pairs();
        for objective in [Objective::SumOfCosts, Objective::Makespan] {
            let config = CbsConfig::improved(objective);
            let joint = cbs(&instance, objective, Limits::NONE);
            let split = cbs_id(&instance, config, Limits::NONE);

            let cost = |result: &MapfResult| result.solution().map(|s| s.cost(objective));
            assert_eq!(cost(&split), cost(&joint));
            assert!(cost(&split).is_some());
        }
    }

    /// Merging joins every conflicting pair at once and marks only merged groups new.
    #[test]
    fn merge_joins_conflicting_groups() {
        let groups = Vec::from([
            Vec::from([0]),
            Vec::from([1]),
            Vec::from([2, 3]),
            Vec::from([4]),
        ]);
        let found = [
            Conflict::Vertex {
                agents: (4, 0),
                vertex: 0,
                time: 0,
            },
            Conflict::Vertex {
                agents: (3, 0),
                vertex: 0,
                time: 0,
            },
        ];
        let (next, fresh) = merge(&groups, &found);

        assert_eq!(next, [Vec::from([0, 2, 3, 4]), Vec::from([1])]);
        assert_eq!(fresh, [0]);
    }

    /// An unreachable goal makes the instance unsolvable; tight limits stop it.
    #[test]
    fn unsolvable_and_limits() {
        let grid = Grid::parse(&[".@."]);
        let agents = Vec::from([Agent { start: 0, goal: 2 }]);
        let instance = Instance::new(grid, agents).expect("valid");
        let config = CbsConfig::improved(Objective::SumOfCosts);
        assert_eq!(
            cbs_id(&instance, config, Limits::NONE).outcome,
            Outcome::Unsolvable
        );

        let corridor = Grid::new(3, 1);
        let swap = Vec::from([Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }]);
        let instance = Instance::new(corridor, swap).expect("valid");
        let result = cbs_id(&instance, config, Limits::expansions(20));
        assert_eq!(result.outcome, Outcome::LimitReached);
    }

    /// No agents: an empty plan.
    #[test]
    fn no_agents() {
        let instance = Instance::new(Grid::new(2, 2), Vec::new()).expect("valid");
        let config = CbsConfig::improved(Objective::SumOfCosts);
        let result = cbs_id(&instance, config, Limits::NONE);

        assert_eq!(result.solution().map(|s| s.paths.len()), Some(0));
    }

    /// The parallel run gives the same result as the sequential one, statistics included.
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_sequential() {
        let instance = two_pairs();
        let config = CbsConfig::improved(Objective::SumOfCosts);
        for threads in 1..=4 {
            assert_eq!(
                cbs_id_par(&instance, config, Limits::NONE, threads),
                cbs_id(&instance, config, Limits::NONE)
            );
        }
    }
}
