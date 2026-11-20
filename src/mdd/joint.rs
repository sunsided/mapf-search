//! A depth-first search of the cross product of some agents' MDDs for a
//! conflict-free combination ([`joint_paths`]).
//!
//! Every agent's MDD may have a different depth; an agent past its depth stays at its goal,
//! where it still blocks the others. The search ends when every agent is past its depth.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::graph::Vertex;
use crate::mdd::Mdd;
use crate::path::{Path, Time};

/// One path per MDD in `mdds`, each from its MDD, that together have no vertex conflict and
/// no swap, or `None` if no such combination exists. The paths end at their goals without
/// trailing waits.
pub(crate) fn joint_paths(mdds: &[&Mdd]) -> Option<Vec<Path>> {
    let depth = mdds.iter().map(|mdd| mdd.depth()).max().unwrap_or(0);
    let mut search = Search {
        mdds,
        depth,
        visited: BTreeSet::new(),
        trail: Vec::new(),
    };

    let start = alloc::vec![0; mdds.len()];
    if !search.reach_goal(0, start) {
        return None;
    }

    Some(search.paths())
}

/// The state of one [`joint_paths`] run.
struct Search<'a> {
    /// The MDDs, one per agent.
    mdds: &'a [&'a Mdd],
    /// The largest depth; at this level every agent is at its goal for good.
    depth: Time,
    /// The `(time, node indices)` already searched without success.
    visited: BTreeSet<(Time, Vec<usize>)>,
    /// The node indices of each time step on the current branch, from time `0`.
    trail: Vec<Vec<usize>>,
}

impl Search<'_> {
    /// Searches on from `nodes` (an index into each agent's MDD level) at `time`; on
    /// success the trail holds the joint path from time `0`.
    fn reach_goal(&mut self, time: Time, nodes: Vec<usize>) -> bool {
        if !self.visited.insert((time, nodes.clone())) {
            return false;
        }

        self.trail.push(nodes);
        if time == self.depth {
            return true;
        }

        let current = self.trail.last().expect("just pushed").clone();
        let mut successors = Vec::new();
        self.combine(time, &current, &mut Vec::new(), &mut successors);
        for next in successors {
            if self.reach_goal(time + 1, next) {
                return true;
            }
        }

        self.trail.pop();

        false
    }

    /// Extends the partial joint step `picked` agent by agent and collects every
    /// conflict-free completion from `nodes` at `time` into `out`.
    fn combine(
        &self,
        time: Time,
        nodes: &[usize],
        picked: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        let agent = picked.len();
        if agent == self.mdds.len() {
            out.push(picked.clone());
            return;
        }

        for &child in self.mdds[agent].children(time, nodes[agent]) {
            if !self.collides(time, nodes, picked, child) {
                picked.push(child);
                self.combine(time, nodes, picked, out);
                picked.pop();
            }
        }
    }

    /// Returns `true` if agent `picked.len()` moving to node `child` collides with an agent
    /// in `picked`: the same vertex at `time + 1`, or a swap.
    fn collides(&self, time: Time, nodes: &[usize], picked: &[usize], child: usize) -> bool {
        let me = picked.len();
        let (from, to) = (
            self.vertex(me, time, nodes[me]),
            self.vertex(me, time + 1, child),
        );

        picked.iter().enumerate().any(|(other, &their_child)| {
            let their_from = self.vertex(other, time, nodes[other]);
            let their_to = self.vertex(other, time + 1, their_child);

            their_to == to || (from != to && their_from == to && their_to == from)
        })
    }

    /// The vertex of node `index` of agent `agent`'s MDD at `time`.
    fn vertex(&self, agent: usize, time: Time, index: usize) -> Vertex {
        self.mdds[agent].level(time)[index]
    }

    /// The paths along the trail, trimmed of trailing waits at the goal.
    fn paths(&self) -> Vec<Path> {
        (0..self.mdds.len())
            .map(|agent| {
                let mut path: Path = (0..)
                    .zip(&self.trail)
                    .map(|(time, nodes)| self.vertex(agent, time, nodes[agent]))
                    .collect();
                while path.len() > 1 && path[path.len() - 2] == path[path.len() - 1] {
                    path.pop();
                }

                path
            })
            .collect()
    }
}
