//! What all planners of one M* search share ([`Context`]): the instance, the individual
//! policies, the subgroup planners, the deadline and the counters.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use heuristic_search::{Budget, Stats};

use super::planner::Planner;
use super::state::{Choice, Key};
use crate::graph::{Graph, Vertex};
use crate::instance::{AgentId, Instance};
use crate::path::Time;

/// The budget stopped the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LimitReached;

/// What every planner of one search shares.
pub(super) struct Context<'a, G> {
    /// The instance being solved.
    pub(super) instance: &'a Instance<G>,
    /// Recursive M*: keep collision groups apart and plan each one on its own. `false` is
    /// basic M*.
    pub(super) recursive: bool,
    /// Operator decomposition: where agents branch, take the joint step one agent at a
    /// time through intermediate states.
    pub(super) decompose: bool,
    /// Every agent must have finished by this time; states from which an unfinished agent
    /// cannot reach its goal in time are pruned. `None`: no deadline, and time is not part
    /// of the state.
    pub(super) deadline: Option<Time>,
    /// Distances to each agent's goal, indexed by agent.
    pub(super) distances: &'a [Vec<Option<u32>>],
    /// `policy[agent][vertex]`: the agent's next vertex on its individual optimal path;
    /// `None` at its goal and where the goal is unreachable.
    pub(super) policy: &'a [Vec<Option<Vertex>>],
    /// The planner of every subgroup queried so far, by its sorted agents. A planner is
    /// taken out while it runs.
    pub(super) planners: BTreeMap<Vec<AgentId>, Planner>,
    /// Counters over all planners, reported in the result.
    pub(super) stats: Stats,
}

impl<G: Graph> Context<'_, G> {
    /// The next standard state of `group` after the standard state `key` on an optimal
    /// joint path of the group alone, `None` if the group cannot reach its goals from
    /// there (in time).
    ///
    /// # Errors
    ///
    /// [`LimitReached`] if the budget stopped the group's search.
    pub(super) fn group_step(
        &mut self,
        group: &[AgentId],
        key: Key,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<Key>, LimitReached> {
        if key.is_goal() {
            return Ok(Some(Key::standard(self.next_time(key.time), key.base)));
        }

        let mut planner = self
            .planners
            .remove(group)
            .unwrap_or_else(|| Planner::new(group.to_vec()));
        let step = planner.policy(self, key, budget);
        self.planners.insert(group.to_vec(), planner);

        step
    }

    /// The time after `time`: the next step with a deadline, `0` without.
    pub(super) fn next_time(&self, time: Time) -> Time {
        if self.deadline.is_some() { time + 1 } else { 0 }
    }

    /// The choices of `agent` at `at` and `time`: every move and wait towards a vertex from
    /// which it can still reach its goal (in time), plus finishing at the goal, if
    /// `branch`; else its policy move.
    pub(super) fn choices(
        &self,
        (here, finished): (Vertex, bool),
        agent: AgentId,
        branch: bool,
        time: Time,
    ) -> Vec<Choice> {
        let at_goal = here == self.instance.agents()[agent].goal;
        if finished || (at_goal && !branch) {
            return vec![Choice::finish(here)];
        }

        if !branch {
            let next = self.policy[agent][here].expect("agents stay where the goal is reachable");
            return vec![Choice::move_to(next)];
        }

        let next_time = self.next_time(time);
        let mut choices = Vec::new();
        let mut consider = |next: Vertex| {
            if self.in_time(agent, (next, false), next_time) {
                choices.push(Choice::move_to(next));
            }
        };
        consider(here);
        self.instance.graph().neighbors(here, consider);
        if at_goal {
            choices.push(Choice::finish(here));
        }

        choices
    }

    /// Returns `true` if `agent` at `at` and `time` can still finish by the deadline: it
    /// has finished, or its goal is reachable and, with a deadline, close enough.
    pub(super) fn in_time(
        &self,
        agent: AgentId,
        (vertex, finished): (Vertex, bool),
        time: Time,
    ) -> bool {
        if finished {
            return true;
        }

        self.distances[agent][vertex].is_some_and(|distance| {
            self.deadline
                .is_none_or(|deadline| time + distance <= deadline)
        })
    }

    /// The sum of the distances of the unfinished agents among `agents` at the current
    /// positions of `key` to their goals; `None` if one cannot reach its goal in time.
    ///
    /// Consistent: a move costs 1 and lowers the mover's distance by at most 1.
    pub(super) fn heuristic(&self, agents: &[AgentId], key: &Key) -> Option<u32> {
        let moved = key.moved.len();
        let next_time = self.next_time(key.time);

        agents
            .iter()
            .zip(key.positions())
            .enumerate()
            .try_fold(0, |sum, (index, (&agent, at))| {
                let time = if index < moved { next_time } else { key.time };
                if !self.in_time(agent, at, time) {
                    return None;
                }

                Some(
                    sum + if at.1 {
                        0
                    } else {
                        self.distances[agent][at.0]?
                    },
                )
            })
    }
}
