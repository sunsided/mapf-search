//! Moving AI parsers: well-formed files, every error kind, round trips of random maps, and
//! CBS on a parsed instance.

use std::fmt::Write;

use mapf_search::{
    Agent, ConstraintSet, Graph, Grid, Limits, Objective, ParseError, ParseErrorKind,
    ScenarioError, cbs, parse_map, parse_scenario, path_cost, scenario_instance, space_time_astar,
};
use proptest::prelude::*;

use crate::oracle::assert_valid_plan;

/// An 8 by 8 map with a wall that has two gaps, in the Moving AI format with CRLF line ends.
const MAP: &str = "type octile\r\nheight 8\r\nwidth 8\r\nmap\r\n\
    ........\r\n\
    ........\r\n\
    ...@@...\r\n\
    @@.@@.@@\r\n\
    ...@@..T\r\n\
    ........\r\n\
    ..S..G..\r\n\
    ........\r\n";

/// Six agents crossing the wall in both directions.
const SCEN: &str = "version 1\n\
    0\tcross.map\t8\t8\t0\t0\t7\t7\t9.9\n\
    0\tcross.map\t8\t8\t7\t7\t0\t0\t9.9\n\
    1\tcross.map\t8\t8\t2\t0\t5\t7\t8.2\n\
    1\tcross.map\t8\t8\t5\t7\t2\t1\t7.2\n\
    2\tcross.map\t8\t8\t7\t0\t0\t6\t9.4\n\
    2\tcross.map\t8\t8\t0\t5\t6\t1\t7.7\n";

/// The header, cell symbols and line ends of a real map file are understood.
#[test]
fn map_is_parsed() {
    let grid = parse_map(MAP).expect("valid map");

    assert_eq!((grid.width(), grid.height()), (8, 8));
    assert!(grid.is_blocked(grid.vertex(0, 3)));
    assert!(grid.is_blocked(grid.vertex(7, 4)), "trees block");
    assert!(!grid.is_blocked(grid.vertex(5, 3)), "the gap is free");
    assert!(!grid.is_blocked(grid.vertex(2, 6)), "S is free");
    assert_eq!(grid.vertex_count(), 64);
}

/// Every scenario line becomes one entry with all nine fields.
#[test]
fn scenario_is_parsed() {
    let entries = parse_scenario(SCEN).expect("valid scenario");

    assert_eq!(entries.len(), 6);
    assert_eq!(entries[2].bucket, 1);
    assert_eq!(entries[2].map, "cross.map");
    assert_eq!((entries[2].map_width, entries[2].map_height), (8, 8));
    assert_eq!((entries[2].start, entries[2].goal), ((2, 0), (5, 7)));
    assert!((entries[2].optimal_length - 8.2).abs() < 1e-9);
}

/// CBS solves the parsed instance for both objectives with valid plans, no cheaper than the
/// agents planned alone.
#[test]
fn cbs_solves_parsed_instance() {
    let grid = parse_map(MAP).expect("valid map");
    let entries = parse_scenario(SCEN).expect("valid scenario");
    let instance = scenario_instance(grid.clone(), &entries, 6).expect("valid instance");
    let alone: Vec<u32> = instance
        .agents()
        .iter()
        .map(|&agent| solo_cost(&grid, agent))
        .collect();

    for objective in [Objective::SumOfCosts, Objective::Makespan] {
        let result = cbs(&instance, objective, Limits::expansions(10_000));
        let solution = result.solution().expect("solvable");
        let (soc, makespan) = assert_valid_plan(&grid, instance.agents(), &solution.paths);

        assert_eq!((solution.sum_of_costs, solution.makespan), (soc, makespan));
        assert!(soc >= alone.iter().sum());
        assert!(makespan >= alone.iter().copied().max().unwrap_or(0));
    }
}

/// The cost of `agent`'s cheapest path when it is alone on `grid`.
fn solo_cost<G: Graph>(grid: &G, agent: Agent) -> u32 {
    let path = space_time_astar(grid, agent, &ConstraintSet::new()).expect("reachable");

    path_cost(&path)
}

/// Each malformed map is reported with its line and kind.
#[test]
fn map_errors_are_reported() {
    let error = |text: &str| parse_map(text).expect_err("malformed");
    let at = |line, kind| ParseError { line, kind };
    let header = "type octile\nheight 2\nwidth 2\nmap\n";

    assert_eq!(
        error("type octile\nheight 2\nmap\n..\n..\n"),
        at(3, ParseErrorKind::MissingHeader("width"))
    );
    assert_eq!(
        error("type octile\nheight two\n"),
        at(2, ParseErrorKind::BadNumber)
    );
    assert_eq!(
        error("type octile\nheight 2\nwidth 2\n"),
        at(4, ParseErrorKind::MissingHeader("map"))
    );
    assert_eq!(
        error("type octile\ncolour red\n"),
        at(2, ParseErrorKind::UnknownHeader)
    );
    assert_eq!(
        error("type octile\nheight 0\nwidth 2\nmap\n"),
        at(4, ParseErrorKind::EmptyMap)
    );
    assert_eq!(
        error(&format!("{header}..\n.\n")),
        at(
            6,
            ParseErrorKind::RowLength {
                expected: 2,
                found: 1
            }
        )
    );
    assert_eq!(
        error(&format!("{header}..\n.x\n")),
        at(6, ParseErrorKind::UnknownCell('x'))
    );
    assert_eq!(
        error(&format!("{header}..\n")),
        at(6, ParseErrorKind::MissingRows)
    );
    assert_eq!(
        error(&format!("{header}..\n..\n..\n")),
        at(7, ParseErrorKind::ExtraRows)
    );
    assert!(
        parse_map(&format!("{header}..\n..\n\n")).is_ok(),
        "trailing empty lines"
    );
}

/// A header that declares far more cells than the text holds is rejected before the grid
/// is allocated, instead of overflowing or exhausting memory.
#[test]
fn oversized_map_header_is_rejected() {
    let side = usize::MAX / 2;
    let text = format!("type octile\nheight {side}\nwidth {side}\nmap\n.\n");

    assert_eq!(
        parse_map(&text),
        Err(ParseError {
            line: 5,
            kind: ParseErrorKind::RowLength {
                expected: side,
                found: 1
            }
        })
    );
    assert_eq!(
        parse_map(&format!("type octile\nheight {side}\nwidth 1\nmap\n.\n")),
        Err(ParseError {
            line: 6,
            kind: ParseErrorKind::MissingRows
        })
    );
}

/// Each malformed scenario is reported with its line and kind.
#[test]
fn scenario_errors_are_reported() {
    let error = |text: &str| parse_scenario(text).expect_err("malformed");
    let at = |line, kind| ParseError { line, kind };

    assert_eq!(
        error("0\tm.map\t2\t2\t0\t0\t1\t1\t1.4\n"),
        at(1, ParseErrorKind::MissingHeader("version"))
    );
    assert_eq!(
        error("version 1\n0\tm.map\t2\t2\t0\t0\t1\t1\n"),
        at(2, ParseErrorKind::FieldCount { found: 8 })
    );
    assert_eq!(
        error("version 1\n0\tm.map\t2\t2\t0\t0\t1\tx\t1.4\n"),
        at(2, ParseErrorKind::BadNumber)
    );
    assert_eq!(
        error("version 1\n0\tm.map\t2\t2\t0\t0\t2\t1\t1.4\n"),
        at(2, ParseErrorKind::OutsideMap)
    );
}

/// Scenarios that do not fit the map or each other are rejected.
#[test]
fn scenario_instance_errors_are_reported() {
    let grid = parse_map("type octile\nheight 2\nwidth 2\nmap\n.@\n..\n").expect("valid map");
    let build = |scen: &str, agents| {
        let entries = parse_scenario(scen).expect("valid scenario");
        scenario_instance(grid.clone(), &entries, agents).expect_err("rejected")
    };

    assert_eq!(
        build("version 1\n0\tm\t3\t2\t0\t0\t0\t1\t1\n", 1),
        ScenarioError::MapSize { entry: 0 }
    );
    assert_eq!(
        build("version 1\n0\tm\t2\t2\t0\t0\t0\t1\t1\n", 2),
        ScenarioError::TooFewEntries { available: 1 }
    );
    assert_eq!(
        build("version 1\n0\tm\t2\t2\t0\t0\t1\t0\t1\n", 1),
        ScenarioError::BlockedCell { entry: 0 }
    );
    assert!(matches!(
        build(
            "version 1\n0\tm\t2\t2\t0\t0\t1\t1\t1\n0\tm\t2\t2\t0\t1\t1\t1\t1\n",
            2
        ),
        ScenarioError::Instance(_)
    ));
}

/// Renders `grid` as a `.map` file. Cell `v` is written as the free or the blocked symbol of
/// `symbols[v % symbols.len()]`; lines end in CRLF if `crlf` is set.
fn render(grid: &Grid, symbols: &[(char, char)], crlf: bool) -> String {
    let end = if crlf { "\r\n" } else { "\n" };
    let mut text = format!(
        "type octile{end}height {}{end}width {}{end}map{end}",
        grid.height(),
        grid.width()
    );
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let vertex = grid.vertex(x, y);
            let (free, blocked) = symbols[vertex % symbols.len()];
            text.push(if grid.is_blocked(vertex) {
                blocked
            } else {
                free
            });
        }

        write!(text, "{end}").expect("writing to a String cannot fail");
    }

    text
}

/// A random grid of up to 12 by 12 cells, about one in four blocked.
fn grid() -> impl Strategy<Value = Grid> {
    (1..=12_usize, 1..=12_usize)
        .prop_flat_map(|(width, height)| {
            let blocked = prop::collection::vec(prop::bool::weighted(0.25), width * height);
            (Just(width), Just(height), blocked)
        })
        .prop_map(|(width, height, blocked)| {
            let mut grid = Grid::new(width, height);
            for vertex in (0..blocked.len()).filter(|&vertex| blocked[vertex]) {
                let (x, y) = grid.cell(vertex);
                grid.block(x, y);
            }

            grid
        })
}

/// Pairs of a free and a blocked map symbol, cycled over the cells.
fn symbols() -> impl Strategy<Value = Vec<(char, char)>> {
    let free = prop::sample::select(vec!['.', 'G', 'S']);
    let blocked = prop::sample::select(vec!['@', 'O', 'T', 'W']);
    prop::collection::vec((free, blocked), 1..=4)
}

proptest! {
    /// Every grid survives rendering to a `.map` file, with any mix of the free and blocked
    /// symbols and either line end, and parsing it back.
    #[test]
    fn map_round_trips(grid in grid(), symbols in symbols(), crlf in any::<bool>()) {
        prop_assert_eq!(parse_map(&render(&grid, &symbols, crlf)), Ok(grid));
    }
}
