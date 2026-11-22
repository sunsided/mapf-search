//! A 4-connected grid with blocked cells ([`Grid`]).

use alloc::vec;
use alloc::vec::Vec;

use super::{Graph, Heading, OrientedGraph, Vertex};

/// A rectangular 4-connected grid. Cells are vertices in row-major order,
/// `y * width + x`; blocked cells are vertices without edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    /// Number of columns. Never zero.
    width: usize,
    /// Number of rows. Never zero.
    height: usize,
    /// `blocked[y * width + x]` is `true` for obstacles. Length `width * height`.
    blocked: Vec<bool>,
}

impl Grid {
    /// Creates a grid of `width` by `height` free cells.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is zero.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "a grid needs at least one cell");

        Self {
            width,
            height,
            blocked: vec![false; width * height],
        }
    }

    /// Parses a grid from rows of text: `.` is free, `@` is blocked.
    ///
    /// All rows must have the same length. Moving AI maps use more symbols; their parser
    /// is planned separately.
    ///
    /// # Panics
    ///
    /// Panics if there are no rows, rows differ in length, or a row has another character.
    #[must_use]
    pub fn parse(rows: &[&str]) -> Self {
        let width = rows.first().map_or(0, |row| row.len());
        let mut grid = Self::new(width, rows.len());

        for (y, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), width, "row {y} has a different length");
            for (x, cell) in row.chars().enumerate() {
                match cell {
                    '.' => {}
                    '@' => grid.block(x, y),
                    other => panic!("unknown grid cell {other:?} at ({x}, {y})"),
                }
            }
        }

        grid
    }

    /// Marks the cell at `(x, y)` as an obstacle.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y)` is outside the grid.
    pub fn block(&mut self, x: usize, y: usize) {
        let vertex = self.vertex(x, y);
        self.blocked[vertex] = true;
    }

    /// The number of columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The number of rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The vertex of cell `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y)` is outside the grid.
    #[must_use]
    pub fn vertex(&self, x: usize, y: usize) -> Vertex {
        assert!(
            x < self.width && y < self.height,
            "cell ({x}, {y}) is outside the grid"
        );

        y * self.width + x
    }

    /// The cell `(x, y)` of `vertex`.
    #[must_use]
    pub fn cell(&self, vertex: Vertex) -> (usize, usize) {
        (vertex % self.width, vertex / self.width)
    }

    /// Returns `true` if `vertex` is an obstacle.
    #[must_use]
    pub fn is_blocked(&self, vertex: Vertex) -> bool {
        self.blocked[vertex]
    }
}

impl Grid {
    /// The heading towards larger `x`.
    pub const EAST: Heading = 0;
    /// The heading towards smaller `y`, the row above.
    pub const NORTH: Heading = 1;
    /// The heading towards smaller `x`.
    pub const WEST: Heading = 2;
    /// The heading towards larger `y`, the row below.
    pub const SOUTH: Heading = 3;
}

/// A grid as an [`OrientedGraph`]: four headings ([`Grid::EAST`], [`Grid::NORTH`],
/// [`Grid::WEST`], [`Grid::SOUTH`], counter-clockwise), and a turn is a quarter turn either
/// way, counter-clockwise first.
///
/// # Examples
///
/// ```
/// use mapf_search::{Grid, OrientedGraph};
///
/// let grid = Grid::parse(&["..", "@."]);
/// assert_eq!(grid.forward(grid.vertex(0, 0), Grid::EAST), Some(grid.vertex(1, 0)));
/// assert_eq!(grid.forward(grid.vertex(0, 0), Grid::SOUTH), None);
///
/// let mut turns = Vec::new();
/// grid.turns(Grid::EAST, |heading| turns.push(heading));
/// assert_eq!(turns, [Grid::NORTH, Grid::SOUTH]);
/// ```
impl OrientedGraph for Grid {
    fn heading_count(&self) -> usize {
        4
    }

    fn forward(&self, vertex: Vertex, heading: Heading) -> Option<Vertex> {
        if self.is_blocked(vertex) {
            return None;
        }

        let (x, y) = self.cell(vertex);
        let (nx, ny) = match heading {
            Self::EAST => (x + 1, y),
            Self::NORTH => (x, y.checked_sub(1)?),
            Self::WEST => (x.checked_sub(1)?, y),
            Self::SOUTH => (x, y + 1),
            _ => return None,
        };
        if nx >= self.width || ny >= self.height {
            return None;
        }

        let next = self.vertex(nx, ny);

        (!self.is_blocked(next)).then_some(next)
    }

    fn turns(&self, heading: Heading, mut emit: impl FnMut(Heading)) {
        emit((heading + 1) % 4);
        emit((heading + 3) % 4);
    }
}

impl Graph for Grid {
    fn vertex_count(&self) -> usize {
        self.blocked.len()
    }

    fn neighbors(&self, vertex: Vertex, mut emit: impl FnMut(Vertex)) {
        if self.is_blocked(vertex) {
            return;
        }

        let (x, y) = self.cell(vertex);
        let candidates = [
            x.checked_sub(1).map(|left| (left, y)),
            (x + 1 < self.width).then_some((x + 1, y)),
            y.checked_sub(1).map(|up| (x, up)),
            (y + 1 < self.height).then_some((x, y + 1)),
        ];

        for (nx, ny) in candidates.into_iter().flatten() {
            let next = self.vertex(nx, ny);
            if !self.is_blocked(next) {
                emit(next);
            }
        }
    }
}
