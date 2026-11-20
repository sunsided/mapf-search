//! Parsers for the Moving AI benchmark formats (Stern et al. 2019,
//! <https://movingai.com/benchmarks/mapf/>): grid maps ([`parse_map`]), scenarios
//! ([`parse_scenario`]) and instances built from both ([`scenario_instance`]).
//!
//! The parsers read strings, so they stay `no_std`; reading files is the caller's job. Maps
//! declare `type octile`, but MAPF benchmarks move agents 4-connected, as [`Grid`] does.
//!
//! # Examples
//!
//! ```
//! use mapf_search::{Limits, Objective, cbs, parse_map, parse_scenario, scenario_instance};
//!
//! let map = "type octile\nheight 2\nwidth 3\nmap\n...\n@.@\n";
//! let scen = "version 1\n\
//!     0\ttiny.map\t3\t2\t0\t0\t2\t0\t2\n\
//!     0\ttiny.map\t3\t2\t2\t0\t0\t0\t2\n";
//!
//! let grid = parse_map(map)?;
//! let entries = parse_scenario(scen)?;
//! let instance = scenario_instance(grid, &entries, 2)?;
//!
//! let result = cbs(&instance, Objective::SumOfCosts, Limits::expansions(1_000));
//! assert_eq!(result.solution().map(|s| s.sum_of_costs), Some(7));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

use crate::graph::Grid;
use crate::instance::{Agent, Instance, InstanceError};

/// Parses a Moving AI `.map` file into a [`Grid`].
///
/// The header lists `type`, `height` and `width` (one `key value` pair per line, in any
/// order) and ends with the line `map`. Then follow `height` rows of `width` cells:
/// `.`, `G` and `S` are free; `@`, `O`, `T` and `W` are blocked. Trailing empty lines are
/// ignored, and so are carriage returns at line ends.
///
/// # Errors
///
/// Returns a [`ParseError`] with the 1-based line number and a [`ParseErrorKind`]:
/// a missing or unknown header field or a bad number, a zero size, a row of the wrong
/// length, an unknown cell, or too few or too many rows.
pub fn parse_map(text: &str) -> Result<Grid, ParseError> {
    let mut lines = numbered_lines(text);
    let (width, height) = parse_map_header(&mut lines)?;
    let mut grid = Grid::new(width, height);

    for y in 0..height {
        let end = text.lines().count() + 1;
        let (line, row) = lines
            .next()
            .ok_or(ParseError::new(end, ParseErrorKind::MissingRows))?;
        parse_map_row(&mut grid, y, line, row)?;
    }

    if let Some((line, _)) = lines.find(|(_, row)| !row.is_empty()) {
        return Err(ParseError {
            line,
            kind: ParseErrorKind::ExtraRows,
        });
    }

    Ok(grid)
}

/// One agent of a Moving AI `.scen` file.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioEntry {
    /// The bucket, which groups entries of similar difficulty.
    pub bucket: u32,
    /// The map file name, as written in the scenario.
    pub map: String,
    /// The width of the map the entry was made for.
    pub map_width: usize,
    /// The height of the map the entry was made for.
    pub map_height: usize,
    /// The start cell `(x, y)`; `(0, 0)` is the top left.
    pub start: (usize, usize),
    /// The goal cell `(x, y)`.
    pub goal: (usize, usize),
    /// The optimal single-agent path length the file reports. Computed with octile
    /// (8-connected) moves, so on a 4-connected [`Grid`] it is only a lower bound.
    pub optimal_length: f64,
}

/// Parses a Moving AI `.scen` file: a `version` line, then one entry per line with nine
/// whitespace-separated fields (see [`ScenarioEntry`]). Empty lines are skipped.
///
/// # Errors
///
/// Returns a [`ParseError`] if the version line is missing, a line has the wrong number of
/// fields or a field is not a number.
pub fn parse_scenario(text: &str) -> Result<Vec<ScenarioEntry>, ParseError> {
    let mut lines = numbered_lines(text).filter(|(_, line)| !line.is_empty());
    match lines.next() {
        Some((_, line)) if line.starts_with("version") => {}
        other => {
            return Err(ParseError {
                line: other.map_or(1, |(line, _)| line),
                kind: ParseErrorKind::MissingHeader("version"),
            });
        }
    }

    lines
        .map(|(line, text)| parse_scenario_entry(line, text))
        .collect()
}

/// Builds an instance from the first `agents` entries of a scenario on `grid`.
///
/// # Errors
///
/// - [`ScenarioError::MapSize`] if an entry was made for a map of another size;
/// - [`ScenarioError::TooFewEntries`] if the scenario has fewer than `agents` entries;
/// - [`ScenarioError::BlockedCell`] if an entry's start or goal is blocked;
/// - [`ScenarioError::Instance`] if [`Instance::new`] rejects the agents, for example
///   for a shared goal.
pub fn scenario_instance(
    grid: Grid,
    entries: &[ScenarioEntry],
    agents: usize,
) -> Result<Instance<Grid>, ScenarioError> {
    let chosen = entries.get(..agents).ok_or(ScenarioError::TooFewEntries {
        available: entries.len(),
    })?;
    let agents = chosen
        .iter()
        .enumerate()
        .map(|(index, entry)| entry_agent(&grid, index, entry))
        .collect::<Result<Vec<_>, _>>()?;

    Instance::new(grid, agents).map_err(ScenarioError::Instance)
}

/// The agent of scenario entry `index` on `grid`, after checking the map size and cells.
fn entry_agent(grid: &Grid, index: usize, entry: &ScenarioEntry) -> Result<Agent, ScenarioError> {
    if (entry.map_width, entry.map_height) != (grid.width(), grid.height()) {
        return Err(ScenarioError::MapSize { entry: index });
    }

    let start = grid.vertex(entry.start.0, entry.start.1);
    let goal = grid.vertex(entry.goal.0, entry.goal.1);
    if grid.is_blocked(start) || grid.is_blocked(goal) {
        return Err(ScenarioError::BlockedCell { entry: index });
    }

    Ok(Agent { start, goal })
}

/// The lines of `text`, numbered from 1, without trailing whitespace.
fn numbered_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line.trim_end()))
}

/// Reads the header of a `.map` file up to and including the `map` line and returns the
/// `(width, height)` it declares.
fn parse_map_header<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<(usize, usize), ParseError> {
    let (mut width, mut height, mut kind) = (None, None, None);
    let mut last = 0;

    for (line, text) in lines.by_ref() {
        last = line;
        let mut words = text.split_whitespace();
        match (words.next(), words.next()) {
            (Some("map"), None) => return header_size(width, height, kind, line),
            (Some("type"), Some(value)) => kind = Some(value),
            (Some("height"), Some(value)) => height = Some(number(line, value)?),
            (Some("width"), Some(value)) => width = Some(number(line, value)?),
            (None, _) => {}
            _ => return Err(ParseError::new(line, ParseErrorKind::UnknownHeader)),
        }
    }

    Err(ParseError::new(
        last + 1,
        ParseErrorKind::MissingHeader("map"),
    ))
}

/// Checks that the header declared a type and a non-zero size.
fn header_size(
    width: Option<usize>,
    height: Option<usize>,
    kind: Option<&str>,
    line: usize,
) -> Result<(usize, usize), ParseError> {
    let missing = |field| ParseError::new(line, ParseErrorKind::MissingHeader(field));
    kind.ok_or_else(|| missing("type"))?;
    let width = width.ok_or_else(|| missing("width"))?;
    let height = height.ok_or_else(|| missing("height"))?;

    if width == 0 || height == 0 {
        return Err(ParseError::new(line, ParseErrorKind::EmptyMap));
    }

    Ok((width, height))
}

/// Blocks the obstacles of map row `y`, read from `line`.
fn parse_map_row(grid: &mut Grid, y: usize, line: usize, row: &str) -> Result<(), ParseError> {
    let found = row.chars().count();
    if found != grid.width() {
        return Err(ParseError::new(
            line,
            ParseErrorKind::RowLength {
                expected: grid.width(),
                found,
            },
        ));
    }

    for (x, cell) in row.chars().enumerate() {
        match cell {
            '.' | 'G' | 'S' => {}
            '@' | 'O' | 'T' | 'W' => grid.block(x, y),
            other => return Err(ParseError::new(line, ParseErrorKind::UnknownCell(other))),
        }
    }

    Ok(())
}

/// Parses one scenario entry from the text of line `line`.
fn parse_scenario_entry(line: usize, text: &str) -> Result<ScenarioEntry, ParseError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    let [bucket, map, width, height, sx, sy, gx, gy, optimal] = fields[..] else {
        return Err(ParseError::new(
            line,
            ParseErrorKind::FieldCount {
                found: fields.len(),
            },
        ));
    };

    let entry = ScenarioEntry {
        bucket: number(line, bucket)?,
        map: map.to_string(),
        map_width: number(line, width)?,
        map_height: number(line, height)?,
        start: (number(line, sx)?, number(line, sy)?),
        goal: (number(line, gx)?, number(line, gy)?),
        optimal_length: number(line, optimal)?,
    };

    let inside = |(x, y): (usize, usize)| x < entry.map_width && y < entry.map_height;
    if !inside(entry.start) || !inside(entry.goal) {
        return Err(ParseError::new(line, ParseErrorKind::OutsideMap));
    }

    Ok(entry)
}

/// Parses `text` as a number, reporting failures at `line`.
fn number<T: FromStr>(line: usize, text: &str) -> Result<T, ParseError> {
    text.parse()
        .map_err(|_| ParseError::new(line, ParseErrorKind::BadNumber))
}

/// A syntax error in a Moving AI file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseError {
    /// The 1-based line the error was found on.
    pub line: usize,
    /// What is wrong.
    pub kind: ParseErrorKind,
}

impl ParseError {
    /// An error of `kind` at `line`.
    fn new(line: usize, kind: ParseErrorKind) -> Self {
        Self { line, kind }
    }
}

/// What is wrong in a [`ParseError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseErrorKind {
    /// A required header line or field is missing: `type`, `height`, `width` or `map` in a
    /// map, `version` in a scenario.
    MissingHeader(&'static str),
    /// A map header line is not one of `type`, `height`, `width` or `map`.
    UnknownHeader,
    /// The map declares a zero width or height.
    EmptyMap,
    /// A field that must be a number is not one.
    BadNumber,
    /// A map row has the wrong number of cells.
    RowLength {
        /// The declared width.
        expected: usize,
        /// The number of cells in the row.
        found: usize,
    },
    /// A map cell is not one of `.`, `G`, `S`, `@`, `O`, `T`, `W`.
    UnknownCell(char),
    /// The map has fewer rows than its height.
    MissingRows,
    /// The map has non-empty lines after its last row.
    ExtraRows,
    /// A scenario line does not have nine fields.
    FieldCount {
        /// The number of fields on the line.
        found: usize,
    },
    /// A scenario start or goal lies outside the map size the entry declares.
    OutsideMap,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: ", self.line)?;
        match self.kind {
            ParseErrorKind::MissingHeader(field) => write!(f, "missing `{field}` header"),
            ParseErrorKind::UnknownHeader => f.write_str("unknown header line"),
            ParseErrorKind::EmptyMap => f.write_str("map has no cells"),
            ParseErrorKind::BadNumber => f.write_str("expected a number"),
            ParseErrorKind::RowLength { expected, found } => {
                write!(f, "row has {found} cells, expected {expected}")
            }
            ParseErrorKind::UnknownCell(cell) => write!(f, "unknown map cell {cell:?}"),
            ParseErrorKind::MissingRows => f.write_str("map has too few rows"),
            ParseErrorKind::ExtraRows => f.write_str("map has too many rows"),
            ParseErrorKind::FieldCount { found } => {
                write!(f, "scenario line has {found} fields, expected 9")
            }
            ParseErrorKind::OutsideMap => f.write_str("start or goal outside the map"),
        }
    }
}

impl core::error::Error for ParseError {}

/// Why [`scenario_instance`] could not build an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScenarioError {
    /// Entry `entry` was made for a map of another size.
    MapSize {
        /// The 0-based index of the entry.
        entry: usize,
    },
    /// The scenario has fewer entries than agents were asked for.
    TooFewEntries {
        /// The number of entries in the scenario.
        available: usize,
    },
    /// The start or goal of entry `entry` is a blocked cell.
    BlockedCell {
        /// The 0-based index of the entry.
        entry: usize,
    },
    /// The agents do not form a valid instance.
    Instance(InstanceError),
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MapSize { entry } => write!(f, "scenario entry {entry} is for another map size"),
            Self::TooFewEntries { available } => {
                write!(f, "the scenario has only {available} entries")
            }
            Self::BlockedCell { entry } => {
                write!(f, "scenario entry {entry} starts or ends on a blocked cell")
            }
            Self::Instance(error) => write!(f, "invalid instance: {error}"),
        }
    }
}

impl core::error::Error for ScenarioError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Instance(error) => Some(error),
            _ => None,
        }
    }
}
