# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/sunsided/mapf-search/releases/tag/v0.1.0) - 2026-11-22

### Added

- Add the GitHub Actions lint workflow
- Add dormant release automation with release-plz
- Add CONTRIBUTING.md
- Add git hooks and Copilot review instructions
- Add README badges, a usage example and absolute links
- Add cargo-fuzz targets for the Moving AI parsers and the solvers
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos configuration
- Add the Codecov configuration
- Add the Taskfile and CI workflows
- Add feasibility check, ICBS improvements, low-level budget and per-agent costs to MO-CBS
- Add MO-CBS for multi-objective MAPF behind the multi-objective feature
- Add operator decomposition to M* and solve makespan by deadlines
- Add the DG and WDG heuristics to CBS
- Add disjoint splitting and positive constraints to CBS
- Add recursive M* and select it with MStarConfig
- Add a dependency-free solver benchmark
- Add basic M*
- Add ICTS with pairwise pruning
- Add PBS and permanent vertex blocks
- Add ECBS with a local focal list
- Add MDDs and improved CBS: prioritized conflicts, bypass, CG
- Add makespan-optimal CBS
- Add instance types, conflict detection, space-time A* and CBS
- Add design document

### Changed

- Set the version to 0.1.0 for the first release
- Report dependencies and workflow findings to GitHub code security
- Update criterion requirement
- Depend on the released multiobjective-search and drop the sibling checkout
- Depend on heuristic-search 0.2
- Depend on the released heuristic-search 0.1
- Describe the released scope and audience in the README
- Raise the MSRV to 1.86 and check every feature on it
- Extract map row reading and fuzz grid building into helpers
- Let Dependabot update the sibling checkout action
- Adopt the full lint table, clippy.toml and rustfmt.toml
- Render mermaid diagrams in the rustdoc output
- Test that random maps survive a render and parse round trip
- Check map rows before allocating the grid
- Check the MSRV with default features only
- Depend on search-core from crates.io
- Require search-core 0.3
- Use search-core's Lexicographic for the conflict-avoidance low level
- Count MO-CBS expansions after their children and refresh the docs
- Document the MO-CBS improvements, budget hook and per-agent costs
- Test MO-CBS's Update directly and default MoCbsConfig to tree-by-tree
- Document MO-CBS and compare it with the paper
- Make ODrM* the default of mstar
- Bound makespan ICTS in the cross-check and record the test runs
- Record test runs, benchmark measurements and open points in the design
- Cover WDG with disjoint splitting and document the new CBS options
- List ODrM* under the planned scope in the crate docs
- Use heuristic-search's FocalList in ECBS's low level
- Break CBS low-level ties with a conflict avoidance table
- Count conflicts after finishing in ECBS's low level
- Take an Objective in ECBS, PBS, ICTS and M*
- Bound budgets as Budget<Stats> after heuristic-search adopted search-core
- Benchmark with criterion, like heuristic-search
- Test the CBS improvements on single nodes
- Document the new solvers in the README and design
- Cross-check all solvers on larger random grids
- Parse Moving AI maps and scenarios
- Adapt to heuristic-search's by-value heuristics
- Re-export the budget types and bound discarded test cases
- Test CBS against a joint-space oracle
- Reserve the crate name with a 0.0.0 placeholder

### Fixed

- Fix ambiguous rustdoc links to ecbs, icts and mstar
