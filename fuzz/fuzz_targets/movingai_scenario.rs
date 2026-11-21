//! Fuzzes the Moving AI `.scen` parser and `scenario_instance` with arbitrary text: neither
//! may panic, every accepted entry lies on the map it names, and an instance built from the
//! entries has one agent per requested entry.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;
use mapf_search::{Grid, parse_scenario, scenario_instance};

fuzz_target!(|data: &[u8]| {
    let Some((&agents, rest)) = data.split_first() else {
        return;
    };
    let Ok(text) = std::str::from_utf8(rest) else {
        return;
    };
    let Ok(entries) = parse_scenario(text) else {
        return;
    };

    for entry in &entries {
        let inside = |(x, y): (usize, usize)| x < entry.map_width && y < entry.map_height;
        assert!(
            inside(entry.start) && inside(entry.goal),
            "entry outside its map"
        );
    }

    // A small map with one blocked cell, so that size mismatches, blocked cells and the
    // instance checks are all reachable.
    let mut grid = Grid::new(4, 3);
    grid.block(1, 1);
    let agents = usize::from(agents % 4);
    if let Ok(instance) = scenario_instance(grid, &entries, agents) {
        assert_eq!(instance.agents().len(), agents, "agent count");
    }
});
