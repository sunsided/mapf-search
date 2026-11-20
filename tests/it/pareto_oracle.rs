//! An independent multi-objective reference: the exact Pareto front of a MAPF instance with
//! vector costs, by a label-setting search (Martins 1984) over the joint state of all
//! agents. It shares no code with the crate beyond the [`Graph`] trait, and none with
//! `multiobjective-search`.
//!
//! Joint states are those of [`crate::oracle`]: positions plus a "finished" mask. An
//! unfinished agent pays `costs(agent, from, to)` per step, a wait `costs(agent, v, v)`; finishing at its
//! goal and every later step are free. Labels are popped in lexicographic cost order; a
//! label that a permanent label at its state, or a found solution, weakly dominates is
//! dropped. The joint space is finite and every antichain of cost vectors is finite, so
//! the search ends, and without a horizon.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use mapf_search::{Agent, Graph, Vertex};

use crate::oracle::{Choice, JointState, apply, arrival, collides};

/// Returns `true` if `a` is no worse than `b` in every component.
fn weakly_dominates<const N: usize>(a: &[u32; N], b: &[u32; N]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

/// Component-wise sum.
fn plus<const N: usize>(a: [u32; N], b: [u32; N]) -> [u32; N] {
    std::array::from_fn(|i| a[i] + b[i])
}

/// The cost-unique Pareto-optimal front of `agents` on `graph` under `costs`, sorted
/// lexicographically; empty if the instance is unsolvable. `costs(agent, from, to)` is
/// what a move or wait costs `agent`.
///
/// # Panics
///
/// Panics for more than 31 agents.
pub fn pareto_front<G: Graph, const N: usize>(
    graph: &G,
    agents: &[Agent],
    costs: &impl Fn(usize, Vertex, Vertex) -> [u32; N],
) -> Vec<[u32; N]> {
    assert!(agents.len() < 32, "the oracle handles at most 31 agents");

    let all = (1_u32 << agents.len()) - 1;
    let start: JointState = (agents.iter().map(|agent| agent.start).collect(), 0);
    let mut open = BinaryHeap::from([Reverse(([0; N], start))]);
    let mut permanent: BTreeMap<JointState, Vec<[u32; N]>> = BTreeMap::new();
    let mut front: Vec<[u32; N]> = Vec::new();

    while let Some(Reverse((cost, state))) = open.pop() {
        let labels = permanent.entry(state.clone()).or_default();
        let dominated = |c: &[u32; N], labels: &[[u32; N]]| {
            labels.iter().chain(&front).any(|l| weakly_dominates(l, c))
        };
        if dominated(&cost, labels) {
            continue;
        }

        labels.push(cost);
        if state.1 == all {
            front.push(cost);
            continue;
        }

        for (next, step) in successors(graph, agents, costs, &state) {
            let next_cost = plus(cost, step);
            let known = permanent.get(&next).map_or(&[][..], Vec::as_slice);
            if !dominated(&next_cost, known) {
                open.push(Reverse((next_cost, next)));
            }
        }
    }

    front.sort_unstable();

    front
}

/// Every collision-free joint step from `state`, with its cost vector.
fn successors<G: Graph, const N: usize>(
    graph: &G,
    agents: &[Agent],
    costs: &impl Fn(usize, Vertex, Vertex) -> [u32; N],
    state: &JointState,
) -> Vec<(JointState, [u32; N])> {
    let options: Vec<Vec<(Choice, [u32; N])>> = (0..agents.len())
        .map(|index| {
            choices(
                graph,
                agents[index],
                &|u, v| costs(index, u, v),
                state,
                index,
            )
        })
        .collect();
    let mut out = Vec::new();
    combine(state, &options, &mut Vec::new(), [0; N], &mut out);

    out
}

/// The choices of agent `index` in `state` with their costs.
fn choices<G: Graph, const N: usize>(
    graph: &G,
    agent: Agent,
    costs: &impl Fn(Vertex, Vertex) -> [u32; N],
    state: &JointState,
    index: usize,
) -> Vec<(Choice, [u32; N])> {
    let here = state.0[index];
    let stay = |finish| Choice { next: here, finish };
    if state.1 & (1 << index) != 0 {
        return vec![(stay(false), [0; N])];
    }

    let mut targets = vec![here];
    graph.neighbors(here, |next| targets.push(next));
    let mut out: Vec<_> = targets
        .into_iter()
        .map(|next| {
            let choice = Choice {
                next,
                finish: false,
            };

            (choice, costs(here, next))
        })
        .collect();
    if here == agent.goal {
        out.push((stay(true), [0; N]));
    }

    out
}

/// Extends `picked` one agent at a time and collects every collision-free joint step.
fn combine<const N: usize>(
    state: &JointState,
    options: &[Vec<(Choice, [u32; N])>],
    picked: &mut Vec<Choice>,
    cost: [u32; N],
    out: &mut Vec<(JointState, [u32; N])>,
) {
    let index = picked.len();
    if index == options.len() {
        out.push((apply(state, picked), cost));
        return;
    }

    for &(choice, step) in &options[index] {
        if collides(&state.0, picked, choice) {
            continue;
        }

        picked.push(choice);
        combine(state, options, picked, plus(cost, step), out);
        picked.pop();
    }
}

/// The cost vector of a plan, recomputed from its paths: each agent pays for its steps up
/// to its arrival, the first time from which it stays at its goal.
pub fn plan_cost<const N: usize>(
    agents: &[Agent],
    paths: &[Vec<Vertex>],
    costs: &impl Fn(usize, Vertex, Vertex) -> [u32; N],
) -> [u32; N] {
    agents
        .iter()
        .zip(paths)
        .enumerate()
        .flat_map(|(index, (agent, path))| {
            let arrived = usize::try_from(arrival(path, agent.goal)).expect("short paths");

            path.windows(2)
                .take(arrived)
                .map(move |step| costs(index, step[0], step[1]))
        })
        .fold([0; N], plus)
}
