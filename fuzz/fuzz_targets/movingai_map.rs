//! Fuzzes the Moving AI `.map` parser with arbitrary text: it must return an error instead of
//! panicking or allocating without bound, and a map it accepts must survive a round trip
//! through the canonical `.`/`@` rendering unchanged.

#![no_main]
#![forbid(unsafe_code)]

use std::fmt::Write;

use libfuzzer_sys::fuzz_target;
use mapf_search::{Grid, parse_map};

/// Renders `grid` as a `.map` file with `.` for free and `@` for blocked cells.
fn render(grid: &Grid) -> String {
    let mut text = format!(
        "type octile\nheight {}\nwidth {}\nmap\n",
        grid.height(),
        grid.width()
    );
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let blocked = grid.is_blocked(grid.vertex(x, y));
            text.push(if blocked { '@' } else { '.' });
        }

        writeln!(text).expect("writing to a String cannot fail");
    }

    text
}

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(grid) = parse_map(text) else {
        return;
    };

    assert!(
        grid.width() > 0 && grid.height() > 0,
        "an accepted map is empty"
    );
    assert_eq!(parse_map(&render(&grid)), Ok(grid), "round trip");
});
