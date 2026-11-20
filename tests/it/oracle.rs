//! An independent reference solver and plan checker that share no code with the crate
//! beyond the [`Graph`] trait.
//!
//! The solver runs Dijkstra (from `heuristic-search`) over the joint state of all agents:
//! their positions plus one "finished" flag each. An unfinished agent at its goal may
//! finish; from then on it stays there and pays nothing. Every other unfinished agent pays
//! 1 per step, move or wait. This prices sum of costs under goal-stay exactly, including
//! waits at the goal before leaving it again. For makespan, a joint step costs 1 if any
//! agent pays, else 0. The joint space is finite, so the search also proves
//! unsolvability.

use heuristic_search::{Limits, Outcome, Successors, dijkstra};
use mapf_search::{Agent, Graph, Objective, Vertex};

/// A joint state: every agent's vertex and a bit mask of finished agents.
pub(crate) type JointState = (Vec<Vertex>, u32);

/// The joint state space of `agents` on `graph`.
struct Joint<'a, G> {
    /// The graph.
    graph: &'a G,
    /// The agents; at most 32.
    agents: &'a [Agent],
    /// How the agents' step costs combine into the cost of a joint step.
    objective: Objective,
}

/// One agent's choice for one joint step.
#[derive(Clone, Copy)]
pub(crate) struct Choice {
    /// Where the agent is after the step.
    pub(crate) next: Vertex,
    /// Whether the agent finishes with this step (it stays at its goal for good).
    pub(crate) finish: bool,
}

impl<G: Graph> Joint<'_, G> {
    /// Every choice of agent `index` in `state`, with its cost.
    fn choices(&self, state: &JointState, index: usize) -> Vec<(Choice, u32)> {
        let here = state.0[index];
        let stay = |finish| Choice { next: here, finish };
        if state.1 & (1 << index) != 0 {
            return vec![(stay(false), 0)];
        }

        let mut targets = vec![here];
        self.graph.neighbors(here, |next| targets.push(next));

        let mut choices: Vec<_> = targets
            .into_iter()
            .map(|next| {
                (
                    Choice {
                        next,
                        finish: false,
                    },
                    1,
                )
            })
            .collect();
        if here == self.agents[index].goal {
            choices.push((stay(true), 0));
        }

        choices
    }

    /// Extends the partial joint step `picked` by one agent at a time and emits every
    /// collision-free completion.
    fn combine(
        &self,
        state: &JointState,
        picked: &mut Vec<Choice>,
        cost: u32,
        emit: &mut impl FnMut((), JointState, u32),
    ) {
        let index = picked.len();
        if index == self.agents.len() {
            let joint_cost = match self.objective {
                Objective::SumOfCosts => cost,
                Objective::Makespan => u32::from(cost > 0),
            };
            emit((), apply(state, picked), joint_cost);
            return;
        }

        for (choice, step_cost) in self.choices(state, index) {
            if collides(&state.0, picked, choice) {
                continue;
            }

            picked.push(choice);
            self.combine(state, picked, cost + step_cost, emit);
            picked.pop();
        }
    }
}

impl<G: Graph> Successors for Joint<'_, G> {
    type State = JointState;
    type Action = ();
    type Cost = u32;

    fn successors(&self, state: &JointState, mut emit: impl FnMut((), JointState, u32)) {
        self.combine(state, &mut Vec::new(), 0, &mut emit);
    }
}

/// Returns `true` if agent `picked.len()` taking `choice` collides with the agents in
/// `picked`: same target vertex, or a swap along an edge.
pub(crate) fn collides(positions: &[Vertex], picked: &[Choice], choice: Choice) -> bool {
    let me = picked.len();

    picked.iter().enumerate().any(|(other, earlier)| {
        let same_target = earlier.next == choice.next;
        let swap = positions[me] != choice.next
            && earlier.next == positions[me]
            && choice.next == positions[other];

        same_target || swap
    })
}

/// The joint state after every agent took its choice.
pub(crate) fn apply(state: &JointState, picked: &[Choice]) -> JointState {
    let positions = picked.iter().map(|choice| choice.next).collect();
    let finished = picked
        .iter()
        .enumerate()
        .filter(|(_, choice)| choice.finish)
        .fold(state.1, |mask, (index, _)| mask | (1 << index));

    (positions, finished)
}

/// The optimal value of `objective` for `agents` on `graph`, or `None` if the instance is
/// unsolvable.
///
/// # Panics
///
/// Panics for more than 32 agents.
pub fn optimal_cost<G: Graph>(graph: &G, agents: &[Agent], objective: Objective) -> Option<u32> {
    assert!(agents.len() <= 32, "the oracle handles at most 32 agents");

    let all = if agents.len() == 32 {
        u32::MAX
    } else {
        (1 << agents.len()) - 1
    };
    let start = (agents.iter().map(|agent| agent.start).collect(), 0);
    let joint = Joint {
        graph,
        agents,
        objective,
    };
    let goal = |state: &JointState| state.1 == all;

    match dijkstra(&joint, start, &goal, Limits::NONE).outcome {
        Outcome::Found(solution) => Some(solution.cost),
        Outcome::Unsolvable => None,
        Outcome::LimitReached | Outcome::Cutoff => unreachable!("the oracle runs without limits"),
    }
}

/// Checks a plan without the crate's conflict detector: every path starts at its start,
/// ends at its goal, moves only along edges or waits, and no two agents share a vertex or
/// swap along an edge at any time, counting agents that wait at their goal.
///
/// Returns the plan's sum of costs and makespan, from each agent's arrival: the first time
/// from which it stays at its goal.
///
/// # Panics
///
/// Panics with a description of the first violation.
pub fn assert_valid_plan<G: Graph>(
    graph: &G,
    agents: &[Agent],
    paths: &[Vec<Vertex>],
) -> (u32, u32) {
    assert_eq!(paths.len(), agents.len(), "one path per agent");

    for (index, (agent, path)) in agents.iter().zip(paths).enumerate() {
        assert_eq!(path.first(), Some(&agent.start), "agent {index} start");
        assert_eq!(path.last(), Some(&agent.goal), "agent {index} goal");
        for (time, step) in path.windows(2).enumerate() {
            assert!(
                is_step(graph, step[0], step[1]),
                "agent {index} jumps at {time}"
            );
        }
    }

    let horizon = paths.iter().map(Vec::len).max().unwrap_or(0);
    let at = |path: &Vec<Vertex>, time: usize| path[time.min(path.len() - 1)];
    for time in 0..horizon {
        for (i, a) in paths.iter().enumerate() {
            for (j, b) in paths.iter().enumerate().skip(i + 1) {
                assert_ne!(
                    at(a, time),
                    at(b, time),
                    "agents {i} and {j} meet at {time}"
                );
                let swap = at(a, time) != at(a, time + 1)
                    && at(a, time) == at(b, time + 1)
                    && at(b, time) == at(a, time + 1);
                assert!(!swap, "agents {i} and {j} swap at {time}");
            }
        }
    }

    let arrivals: Vec<u32> = agents
        .iter()
        .zip(paths)
        .map(|(agent, path)| arrival(path, agent.goal))
        .collect();

    (
        arrivals.iter().sum(),
        arrivals.iter().copied().max().unwrap_or(0),
    )
}

/// The first time from which `path` stays at `goal`.
pub(crate) fn arrival(path: &[Vertex], goal: Vertex) -> u32 {
    let away = path.iter().rposition(|&vertex| vertex != goal);

    away.map_or(0, |time| u32::try_from(time + 1).expect("short paths"))
}

/// Returns `true` if moving from `from` to `to` is a wait or follows an edge.
fn is_step<G: Graph>(graph: &G, from: Vertex, to: Vertex) -> bool {
    let mut edge = false;
    graph.neighbors(from, |next| edge |= next == to);

    from == to || edge
}
