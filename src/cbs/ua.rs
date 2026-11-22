//! FOCBS (Jin et al. 2026): flow-oracle conflict-based search for MAPF with unassigned
//! agents, over a [`UaInstance`] whose agents either have a goal (*assigned*) or only a
//! start (*unassigned*).
//!
//! CBS searches the paths of the assigned agents alone, ordered by their sum of costs. At
//! every conflict-free node a flow oracle ([`route`]) routes the unassigned agents around
//! those paths at the fewest moves, up to the latest arrival. If no routing exists, the
//! oracle's minimum cut names the reservations of assigned agents that block it, and the
//! node gets one child per blocker that forbids it to its agent (*flow-conflict
//! branching*): every plan below the node must release one of them.

use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use super::{CbsConfig, Node, Search};
use crate::constraint::Constraint;
use crate::flow::{Blocker, Routing, route};
use crate::graph::{Graph, Vertex};
use crate::instance::UaInstance;
use crate::low_level::distance_tables;
use crate::path::{Objective, Path};
use crate::solution::{Outcome, Solution};

/// A plan for a [`UaInstance`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UaSolution {
    /// The paths of the assigned agents, indexed like [`UaInstance::assigned`]; its sum of
    /// costs is the objective FOCBS minimizes.
    pub assigned: Solution,
    /// The paths of the unassigned agents, indexed like [`UaInstance::unassigned`], each up
    /// to the latest arrival of an assigned agent, trimmed of trailing waits; the agent
    /// stays at its last vertex.
    pub unassigned: Vec<Path>,
    /// The number of moves of the unassigned agents, the least possible for these assigned
    /// paths.
    pub unassigned_moves: u32,
}

/// How a [`focbs`] run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UaOutcome {
    /// A conflict-free plan for every agent.
    Solved(UaSolution),
    /// Proven unsolvable: an assigned agent cannot reach its goal even alone, or the search
    /// ran out of options.
    Unsolvable,
    /// The [`Budget`] stopped the search; the instance may be unsolvable.
    LimitReached,
}

/// The result of a [`focbs`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UaResult {
    /// How the run ended.
    pub outcome: UaOutcome,
    /// High-level counters, as for [`cbs_with`](crate::cbs_with); blocker splits count as
    /// expansions.
    pub stats: Stats,
}

impl UaResult {
    /// The plan, if one was found.
    #[must_use]
    pub fn solution(&self) -> Option<&UaSolution> {
        match &self.outcome {
            UaOutcome::Solved(solution) => Some(solution),
            UaOutcome::Unsolvable | UaOutcome::LimitReached => None,
        }
    }
}

/// Plans the assigned agents of `instance` with the least sum of costs that still lets the
/// unassigned agents get out of the way, and routes those at the fewest moves, with FOCBS
/// and every improvement of [`CbsConfig::improved`].
///
/// See [`focbs_with`] for the guarantee and the budget.
///
/// # Examples
///
/// ```
/// use mapf_search::{Agent, Grid, Limits, UaInstance, focbs};
///
/// // An idle robot waits in a corridor with a side pocket; the other one must pass.
/// let grid = Grid::parse(&["...", "@.@"]);
/// let crossing = Agent { start: grid.vertex(0, 0), goal: grid.vertex(2, 0) };
/// let instance = UaInstance::new(grid.clone(), vec![crossing], vec![grid.vertex(1, 0)])?;
///
/// let result = focbs(&instance, Limits::expansions(1_000));
/// let solution = result.solution().expect("the idle robot steps into the pocket");
/// assert_eq!(solution.assigned.sum_of_costs, 2);
/// assert_eq!(solution.unassigned, [vec![grid.vertex(1, 0), grid.vertex(1, 1)]]);
/// assert_eq!(solution.unassigned_moves, 1);
/// # Ok::<(), mapf_search::InstanceError>(())
/// ```
pub fn focbs<G: Graph>(instance: &UaInstance<G>, budget: impl Budget<Stats>) -> UaResult {
    focbs_with(instance, CbsConfig::improved(Objective::SumOfCosts), budget)
}

/// Plans the assigned agents of `instance` with the least sum of costs that still lets the
/// unassigned agents get out of the way, and routes those at the fewest moves, with FOCBS
/// and the improvements `config` selects.
///
/// The guarantee is the paper's *sequential* optimality: the assigned paths have the least
/// sum of costs among all that leave a conflict-free routing for the unassigned agents, and
/// the routing has the fewest moves *for those paths* (not necessarily over all plans with
/// that sum of costs). The unassigned agents only need to be conflict-free until the
/// latest arrival of an assigned agent; after it nobody moves.
///
/// `budget` is asked before every constraint-tree expansion and before every oracle call.
/// On an unsolvable instance the search may run until its budget, as CBS does.
///
/// # Panics
///
/// Panics if `config.objective` is [`Objective::Makespan`]: FOCBS minimizes the sum of
/// costs.
pub fn focbs_with<G: Graph>(
    instance: &UaInstance<G>,
    config: CbsConfig,
    mut budget: impl Budget<Stats>,
) -> UaResult {
    assert_eq!(
        config.objective,
        Objective::SumOfCosts,
        "FOCBS minimizes the sum of costs"
    );

    let assigned = instance.assigned();
    let mut search = Search::forest(
        instance.graph(),
        assigned.iter().map(|agent| agent.start).collect(),
        assigned.iter().map(|agent| agent.goal).collect(),
        distance_tables(instance.graph(), assigned),
        None,
        config,
    );
    search.trees.push((0..assigned.len()).collect());
    search.unassigned = Some(Unassigned {
        starts: instance.unassigned().to_vec(),
        routing: None,
    });

    let outcome = match search.run(&mut budget) {
        Outcome::Solved(solution) => {
            let routing = search
                .unassigned
                .and_then(|unassigned| unassigned.routing)
                .expect("a solved node was routed");
            UaOutcome::Solved(UaSolution {
                assigned: solution,
                unassigned: routing.paths,
                unassigned_moves: routing.moves,
            })
        }
        Outcome::Unsolvable => UaOutcome::Unsolvable,
        Outcome::LimitReached | Outcome::Exhausted => UaOutcome::LimitReached,
    };

    UaResult {
        outcome,
        stats: search.stats,
    }
}

/// The unassigned agents of a FOCBS run.
pub(super) struct Unassigned {
    /// Their starts.
    starts: Vec<Vertex>,
    /// The routing of the node that solved the instance, once found.
    routing: Option<Routing>,
}

impl<G: Graph> Search<'_, G> {
    /// Returns `true` if popping `node` ends the search without further work: it is
    /// conflict-free and there are no unassigned agents to route.
    pub(super) fn is_final(&self, node: &Node) -> bool {
        node.conflict.is_none() && self.unassigned.is_none()
    }

    /// Routes the unassigned agents around the paths of the conflict-free `node`: `None`
    /// if that works (or there are none), else the blockers of the oracle's cut.
    pub(super) fn route_unassigned(&mut self, node: &Node) -> Option<Vec<Blocker>> {
        let unassigned = self.unassigned.as_mut()?;
        match route(self.graph, &node.paths, &unassigned.starts) {
            Ok(routing) => {
                unassigned.routing = Some(routing);
                None
            }
            Err(blockers) => Some(blockers),
        }
    }

    /// Flow-conflict branching: one child of `parent` per blocker that can be released, each
    /// forbidding the blocker to its agent. Blockers at step 0 (fixed by the starts) and
    /// those the node already forbids are skipped; without any other, the node is dropped.
    pub(super) fn expand_blockers(&mut self, parent: &Node, blockers: &[Blocker]) {
        self.stats.expanded += 1;

        for constraint in blockers.iter().filter_map(|&blocker| releasable(blocker)) {
            if parent.constraints.contains(&constraint) {
                continue;
            }

            if let Some(child) = self.child(parent, constraint) {
                self.push(child.node);
            }
        }
    }
}

/// The constraint that releases `blocker`; `None` for a vertex blocker at step 0, which the
/// starts fix.
fn releasable(blocker: Blocker) -> Option<Constraint> {
    match blocker {
        Blocker::Vertex { time: 0, .. } => None,
        Blocker::Vertex {
            agent,
            vertex,
            time,
        } => Some(Constraint::Vertex {
            agent,
            vertex,
            time,
        }),
        Blocker::Move {
            agent,
            from,
            to,
            time,
        } => Some(Constraint::Edge {
            agent,
            from,
            to,
            time,
        }),
    }
}

#[cfg(test)]
mod tests {
    //! FOCBS against a brute-force joint search over all agents.

    use super::*;
    use alloc::collections::{BTreeMap, BinaryHeap};
    use alloc::vec;
    use core::cmp::Reverse;

    use heuristic_search::Limits;
    use proptest::prelude::*;

    use crate::cbs::cbs_with;
    use crate::conflict::first_conflict;
    use crate::graph::Grid;
    use crate::instance::{Agent, Instance};
    use crate::path::{Time, path_cost, position_at};

    /// The expansion budget of the runs in these tests.
    const BUDGET: u64 = 5_000;

    /// A joint state: each assigned agent's vertex and finished flag, then each unassigned
    /// agent's vertex.
    type Joint = (Vec<(Vertex, bool)>, Vec<Vertex>);

    /// The least sum of costs of the assigned agents over all conflict-free joint plans, by
    /// Dijkstra over joint states with goal-stay pricing (an unfinished assigned agent pays
    /// 1 per step, finishing at its goal is free, unassigned agents move for free); `None`
    /// if no plan exists.
    fn optimal_sst(instance: &UaInstance<Grid>) -> Option<u32> {
        let start: Joint = (
            instance
                .assigned()
                .iter()
                .map(|a| (a.start, false))
                .collect(),
            instance.unassigned().to_vec(),
        );
        let mut best = BTreeMap::from([(start.clone(), 0_u32)]);
        let mut open = BinaryHeap::from([Reverse((0_u32, start))]);
        while let Some(Reverse((cost, state))) = open.pop() {
            if best.get(&state).is_some_and(|&known| known < cost) {
                continue;
            }

            if state.0.iter().all(|&(_, finished)| finished) {
                return Some(cost);
            }

            for (next, step) in successors(instance, &state) {
                let total = cost + step;
                if best.get(&next).is_none_or(|&known| total < known) {
                    best.insert(next.clone(), total);
                    open.push(Reverse((total, next)));
                }
            }
        }

        None
    }

    /// The conflict-free joint steps from `state`, each with its cost.
    fn successors(instance: &UaInstance<Grid>, state: &Joint) -> Vec<(Joint, u32)> {
        let grid = instance.graph();
        let options = |vertex: Vertex| {
            let mut next = vec![vertex];
            grid.neighbors(vertex, |n| next.push(n));
            next
        };

        let mut choices: Vec<Vec<(Vertex, bool, u32)>> = Vec::new();
        for (&(vertex, finished), agent) in state.0.iter().zip(instance.assigned()) {
            choices.push(if finished {
                vec![(vertex, true, 0)]
            } else {
                let mut list: Vec<_> = options(vertex).into_iter().map(|n| (n, false, 1)).collect();
                if vertex == agent.goal {
                    list.push((vertex, true, 0));
                }
                list
            });
        }

        for &vertex in &state.1 {
            choices.push(options(vertex).into_iter().map(|n| (n, false, 0)).collect());
        }

        let from: Vec<Vertex> = state
            .0
            .iter()
            .map(|&(v, _)| v)
            .chain(state.1.iter().copied())
            .collect();
        let mut found = Vec::new();
        let mut pick = vec![0; choices.len()];
        loop {
            let to: Vec<Vertex> = pick
                .iter()
                .zip(&choices)
                .map(|(&i, list)| list[i].0)
                .collect();
            if is_free(&from, &to) {
                let assigned = instance.assigned().len();
                let next = (
                    (0..assigned)
                        .map(|a| (to[a], choices[a][pick[a]].1))
                        .collect(),
                    to[assigned..].to_vec(),
                );
                let cost = (0..assigned).map(|a| choices[a][pick[a]].2).sum();
                found.push((next, cost));
            }

            let Some(position) = (0..pick.len()).find(|&k| pick[k] + 1 < choices[k].len()) else {
                break;
            };
            pick[position] += 1;
            pick[..position].fill(0);
        }

        found
    }

    /// Returns `true` if moving every agent `from -> to` has no vertex conflict and no swap.
    fn is_free(from: &[Vertex], to: &[Vertex]) -> bool {
        (0..to.len()).all(|a| {
            (a + 1..to.len()).all(|b| {
                to[a] != to[b] && !(from[a] == to[b] && from[b] == to[a] && from[a] != to[a])
            })
        })
    }

    /// The fewest moves the unassigned agents need around the fixed `assigned` paths until
    /// the latest arrival, by Dijkstra over their joint positions and the step.
    fn fewest_moves(instance: &UaInstance<Grid>, assigned: &[Path]) -> Option<u32> {
        let horizon: Time = assigned.iter().map(|p| path_cost(p)).max().unwrap_or(0);
        let idle = |here: &[Vertex]| {
            UaInstance::new(instance.graph().clone(), Vec::new(), here.to_vec()).expect("valid")
        };
        let start = (0, instance.unassigned().to_vec());
        let mut best = BTreeMap::from([(start.clone(), 0_u32)]);
        let mut open = BinaryHeap::from([Reverse((0_u32, start))]);
        while let Some(Reverse((cost, (time, here)))) = open.pop() {
            if time == horizon {
                return Some(cost);
            }

            let fixed = |at: Time| assigned.iter().map(move |p| position_at(p, at));
            let from: Vec<Vertex> = fixed(time).chain(here.iter().copied()).collect();
            for ((_, next), _) in successors(&idle(&here), &(Vec::new(), here.clone())) {
                let to: Vec<Vertex> = fixed(time + 1).chain(next.iter().copied()).collect();
                if !is_free(&from, &to) {
                    continue;
                }

                let moves = here.iter().zip(&next).filter(|(a, b)| a != b).count();
                let total = cost + u32::try_from(moves).expect("few agents");
                let key = (time + 1, next);
                if best.get(&key).is_none_or(|&known| total < known) {
                    best.insert(key.clone(), total);
                    open.push(Reverse((total, key)));
                }
            }
        }

        None
    }

    /// Up to 3 agents on a grid of 2-3 by 2-3 cells with walls, 1-2 of them assigned.
    fn instance() -> impl Strategy<Value = UaInstance<Grid>> {
        (
            2_usize..=3,
            2_usize..=3,
            prop::collection::vec(prop::bool::weighted(0.15), 9),
            1_usize..=2,
            0_usize..=2,
        )
            .prop_flat_map(|(width, height, blocked, assigned, unassigned)| {
                let cells = width * height;
                let free: Vec<Vertex> = (0..cells).filter(|&v| !blocked[v]).collect();
                (
                    Just((width, height, blocked)),
                    Just(free.clone()).prop_shuffle(),
                    Just(free).prop_shuffle(),
                    Just((assigned, unassigned)),
                )
            })
            .prop_filter_map(
                "enough free cells",
                |((width, height, blocked), starts, goals, (assigned, unassigned))| {
                    if starts.len() < assigned + unassigned || assigned + unassigned > 3 {
                        return None;
                    }

                    let mut grid = Grid::new(width, height);
                    for vertex in (0..width * height).filter(|&v| blocked[v]) {
                        let (x, y) = grid.cell(vertex);
                        grid.block(x, y);
                    }

                    let agents = (0..assigned)
                        .map(|i| Agent {
                            start: starts[i],
                            goal: goals[i],
                        })
                        .collect();
                    let idle = starts[assigned..assigned + unassigned].to_vec();

                    UaInstance::new(grid, agents, idle).ok()
                },
            )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// FOCBS finds the least assigned sum of costs over all joint plans, with plain and
        /// improved CBS; its plan is conflict-free; and its unassigned routing has the
        /// fewest moves for its assigned paths.
        #[test]
        fn matches_a_joint_search(instance in instance(), improved in any::<bool>()) {
            let soc = Objective::SumOfCosts;
            let config = if improved { CbsConfig::improved(soc) } else { CbsConfig::plain(soc) };
            let expected = optimal_sst(&instance);
            let result = focbs_with(&instance, config, Limits::expansions(BUDGET));

            match (&result.outcome, expected) {
                (UaOutcome::Solved(solution), Some(cost)) => {
                    prop_assert_eq!(solution.assigned.sum_of_costs, cost);
                    let mut all = solution.assigned.paths.clone();
                    all.extend(solution.unassigned.iter().cloned());
                    prop_assert_eq!(first_conflict(&all), None);
                    for (path, agent) in solution.assigned.paths.iter().zip(instance.assigned()) {
                        prop_assert_eq!((path.first(), path.last()), (Some(&agent.start), Some(&agent.goal)));
                    }

                    for (path, &start) in solution.unassigned.iter().zip(instance.unassigned()) {
                        prop_assert_eq!(path.first(), Some(&start));
                    }

                    prop_assert_eq!(Some(solution.unassigned_moves), fewest_moves(&instance, &solution.assigned.paths));
                }
                (UaOutcome::Unsolvable | UaOutcome::LimitReached, None) | (UaOutcome::LimitReached, Some(_)) => {}
                (outcome, cost) => prop_assert!(false, "{outcome:?}, expected {cost:?}"),
            }
        }
    }

    /// The idle robot sits on the goal at the end of a corridor with a pocket. The
    /// conflict-free path alone (sum of costs 2) leaves it no way out, so the oracle's
    /// blockers force the assigned agent to wait once while the robot slips into the pocket.
    #[test]
    fn branches_on_blockers() {
        // `0 1 2` over `@ 4 @`; the assigned agent runs 0 -> 2, the robot sits at 2.
        let grid = Grid::parse(&["...", "@.@"]);
        let instance =
            UaInstance::new(grid, vec![Agent { start: 0, goal: 2 }], vec![2]).expect("valid");

        for config in [
            CbsConfig::plain(Objective::SumOfCosts),
            CbsConfig::improved(Objective::SumOfCosts),
        ] {
            let result = focbs_with(&instance, config, Limits::expansions(BUDGET));
            let solution = result.solution().expect("solvable");

            assert_eq!(Some(solution.assigned.sum_of_costs), optimal_sst(&instance));
            assert_eq!(solution.assigned.sum_of_costs, 3);
            assert_eq!(solution.unassigned, [vec![2, 1, 4]]);
            assert_eq!(solution.unassigned_moves, 2);
        }
    }

    /// On a one-way ring the robot can only move along; it leaves as the assigned agent
    /// arrives behind it.
    #[test]
    fn follows_directed_edges() {
        let ring = crate::graph::AdjacencyList::directed(3, &[(0, 1), (1, 2), (2, 0)]);
        let instance =
            UaInstance::new(ring, vec![Agent { start: 0, goal: 1 }], vec![1]).expect("valid");

        let result = focbs(&instance, Limits::expansions(BUDGET));
        let solution = result.solution().expect("solvable");

        assert_eq!(solution.assigned.paths, [vec![0, 1]]);
        assert_eq!(solution.unassigned, [vec![1, 2]]);
    }

    /// Without unassigned agents, FOCBS is CBS, with the same statistics.
    #[test]
    fn without_unassigned_agents_is_cbs() {
        let grid = Grid::parse(&["...", "@.@"]);
        let agents = vec![Agent { start: 0, goal: 2 }, Agent { start: 2, goal: 0 }];
        let ua = UaInstance::new(grid.clone(), agents.clone(), Vec::new()).expect("valid");
        let plain = Instance::new(grid, agents).expect("valid");

        for config in [
            CbsConfig::plain(Objective::SumOfCosts),
            CbsConfig::improved(Objective::SumOfCosts),
        ] {
            let focbs = focbs_with(&ua, config, Limits::expansions(BUDGET));
            let cbs = cbs_with(&plain, config, Limits::expansions(BUDGET));

            assert_eq!(focbs.solution().map(|s| &s.assigned), cbs.solution());
            assert_eq!(focbs.stats, cbs.stats);
        }
    }

    /// No agents at all, and only unassigned ones, are solved without moves.
    #[test]
    fn trivial_instances() {
        let empty = UaInstance::new(Grid::new(2, 2), Vec::new(), Vec::new()).expect("valid");
        let idle = UaInstance::new(Grid::new(2, 2), Vec::new(), vec![0, 3]).expect("valid");

        let none = focbs(&empty, Limits::NONE);
        let parked = focbs(&idle, Limits::NONE);

        assert_eq!(none.solution().map(|s| s.unassigned_moves), Some(0));
        assert_eq!(
            parked.solution().map(|s| s.unassigned.clone()),
            Some(vec![vec![0], vec![3]])
        );
    }

    /// `Objective::Makespan` is rejected.
    #[test]
    #[should_panic(expected = "FOCBS minimizes the sum of costs")]
    fn rejects_makespan() {
        let instance = UaInstance::new(Grid::new(2, 1), Vec::new(), Vec::new()).expect("valid");

        let _ = focbs_with(
            &instance,
            CbsConfig::plain(Objective::Makespan),
            Limits::NONE,
        );
    }
}
