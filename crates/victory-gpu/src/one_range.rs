//! mac: stage 2's one-range rule, its plain-std half (`VICTORY_ALIAS_PHYSICAL=gpu`
//! and `=gpu-sections`; `notes/alias-physical-debug.md`, "One range per page").
//!
//! Two GPU ranges over one physical page do not agree within a submission,
//! whatever the barrier (`notes/tile-probe.md`, probes 5 and 8), and a host
//! copy is a range of its own. The rule: while a view reaches a heap tile, no
//! other range does. All views of a section reach a tile through the one
//! range of its section buffer; a piece view reaches it through its piece
//! tile, which is one range only while the tile is shown at one title address
//! (else the other address is another piece tile). So:
//!
//! - a heap tile shown at a title page under a live copy the GPU side made
//!   ([`Copies`]) is *copied*: no view may reach it ([`refuse`]);
//! - a piece view may not reach a tile shown at two or more title addresses;
//! - when a copy is made over pages, or the title maps a page so that a tile
//!   views reach becomes copied (or, for piece views, shown twice), the views
//!   reaching it leave the registries ([`leaving`]): their next lookups get
//!   the copy's range or copies of their own, as with the switch off.
//!
//! A view that left lives on in lists and heap slots made before (the gap
//! I4b has for remaps too). Copies beside copies are as before stage 2.
//!
//! Plain std, so its tests run everywhere; `crate::alias_physical` holds the
//! state and calls it under its lock.

// `crate::alias_physical` uses it on Windows; the tests everywhere.
#![allow(dead_code)]

use std::collections::HashMap;
use std::ops::Range;

use crate::tile_plan::{HeapTile, PAGE, Plan};

/// Title pages (64 KiB numbers) under live copies the GPU side made, by how
/// many, with the piece they were counted in: only pages in the
/// reservations' pieces count, and only while that piece stands (a
/// reservation made over it again has new pieces; copies of the old memory
/// no longer count there).
#[derive(Default)]
pub(crate) struct Copies {
    pages: HashMap<usize, (u32, usize)>,
}

/// Pages a copy counted: runs of page numbers, each in one piece.
pub(crate) type Counted = Vec<(Range<usize>, usize)>;

impl Copies {
    /// A copy of title memory `[from, to)` is made: its pages in the plan's
    /// pieces count, and are returned to give back.
    pub(crate) fn add(&mut self, plan: &Plan, from: usize, to: usize) -> Counted {
        let mut runs: Counted = Vec::new();
        if to <= from {
            return runs;
        }
        for page in from / PAGE..to.div_ceil(PAGE) {
            let Some((id, _)) = plan.tile_of(page * PAGE) else {
                continue;
            };
            let entry = self.pages.entry(page).or_insert((0, id));
            if entry.1 != id {
                *entry = (0, id);
            }
            entry.0 += 1;
            match runs.last_mut() {
                Some((run, piece)) if run.end == page && *piece == id => run.end += 1,
                _ => runs.push((page..page + 1, id)),
            }
        }
        runs
    }

    /// The copy whose pages [`Copies::add`] counted as `runs` is gone.
    pub(crate) fn remove(&mut self, runs: &Counted) {
        for (run, id) in runs {
            for page in run.clone() {
                if let Some(entry) = self.pages.get_mut(&page)
                    && entry.1 == *id
                {
                    entry.0 -= 1;
                    if entry.0 == 0 {
                        self.pages.remove(&page);
                    }
                }
            }
        }
    }

    /// Whether title page `at` (64 KiB aligned) lies under a live copy, in
    /// the piece it is in now.
    pub(crate) fn under(&self, plan: &Plan, at: usize) -> bool {
        self.pages
            .get(&(at / PAGE))
            .is_some_and(|&(_, id)| plan.tile_of(at / PAGE * PAGE).is_some_and(|(now, _)| now == id))
    }

    /// Title pages under live copies (those of reservations since made over
    /// again too, until their copies go).
    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether no page is under a live copy.
    pub(crate) fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

/// Whether heap tile `t` is reached through a copy: a title page showing it
/// now lies under a live copy.
pub(crate) fn copied(plan: &Plan, copies: &Copies, t: HeapTile) -> bool {
    !copies.is_empty() && plan.addresses(t).iter().any(|&a| copies.under(plan, a))
}

/// Why a view may not reach a heap tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// A copy reaches it too.
    Copied,
    /// A piece view's tile shown at two or more title addresses.
    TwoAddresses,
}

/// Why the one-range rule refuses a view over title memory `[from, to)` (of a
/// piece: `piece`), if it does: one of the heap tiles its pages show now is
/// copied, or (a piece view) shown at two or more addresses.
pub(crate) fn refuse(plan: &Plan, copies: &Copies, from: usize, to: usize, piece: bool) -> Option<Refusal> {
    if to <= from {
        return None;
    }
    let mut why = None;
    for page in from / PAGE..to.div_ceil(PAGE) {
        let Some(t) = plan.current_at(page * PAGE) else {
            continue;
        };
        if copied(plan, copies, t) {
            return Some(Refusal::Copied);
        }
        if piece && plan.addresses(t).len() >= 2 {
            why = Some(Refusal::TwoAddresses);
        }
    }
    why
}

/// What made the rule look at title memory: a copy made over it, or the
/// title mapping pages there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cause {
    Copy,
    Map,
}

/// The title ranges (whole pages, joined) whose views must leave after
/// `cause` over title memory `[from, to)`: every title address of each heap
/// tile its pages show now that views reach (`viewed(t)`: live views of
/// sections, of pieces) and that the rule no longer lets them reach: after a
/// copy, every such tile; after a map, a tile now copied, or (with `pieces`)
/// one shown at two or more addresses that piece views reach.
pub(crate) fn leaving(
    plan: &Plan,
    copies: &Copies,
    from: usize,
    to: usize,
    cause: Cause,
    pieces: bool,
    viewed: impl Fn(HeapTile) -> [u32; 2],
) -> Vec<Range<usize>> {
    let mut pages: Vec<usize> = Vec::new();
    if to <= from {
        return Vec::new();
    }
    let mut seen: Vec<HeapTile> = Vec::new();
    for page in from / PAGE..to.div_ceil(PAGE) {
        let Some(t) = plan.current_at(page * PAGE) else {
            continue;
        };
        let held = viewed(t);
        if held == [0, 0] || seen.contains(&t) {
            continue;
        }
        let gone = match cause {
            Cause::Copy => true,
            Cause::Map => {
                copied(plan, copies, t) || (pieces && held[1] > 0 && plan.addresses(t).len() >= 2)
            }
        };
        if gone {
            seen.push(t);
            pages.extend(plan.addresses(t).iter().map(|&a| a / PAGE));
        }
    }
    pages.sort_unstable();
    pages.dedup();
    let mut out: Vec<Range<usize>> = Vec::new();
    for page in pages {
        match out.last_mut() {
            Some(r) if r.end == page * PAGE => r.end += PAGE,
            _ => out.push(page * PAGE..(page + 1) * PAGE),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_plan::Never;

    const A: usize = 0x40_0000_0000;
    const B: usize = 0x50_0000_0000;

    fn tile(t: u32) -> HeapTile {
        HeapTile { heap: 0, tile: t }
    }

    /// Two reservations of 16 pages: A's pages 0..4 show heap tiles 10..14
    /// (a run), page 8 tile 20; B's page 2 shows tile 11 too (a page at two
    /// addresses), page 5 tile 20 (likewise).
    fn plan() -> Plan {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, 16 * PAGE);
        plan.reserve(B, 16 * PAGE);
        plan.map(A, 4 * PAGE, tile(10));
        plan.map(A + 8 * PAGE, PAGE, tile(20));
        plan.map(B + 2 * PAGE, PAGE, tile(11));
        plan.map(B + 5 * PAGE, PAGE, tile(20));
        plan
    }

    #[test]
    fn a_copy_makes_every_address_of_its_pages_tiles_copied() {
        let plan = plan();
        let mut copies = Copies::default();
        assert_eq!(refuse(&plan, &copies, A, A + 4 * PAGE, false), None, "nothing copied yet");
        // A copy of B's pages 2 and 3 (a title buffer that is not a run):
        // tile 11 is copied, and with it A's page 1, its other address.
        let runs = copies.add(&plan, B + 2 * PAGE + 8, B + 3 * PAGE + 8);
        assert_eq!(runs, vec![(B / PAGE + 2..B / PAGE + 4, 1)], "B's piece is piece 1");
        assert!(copied(&plan, &copies, tile(11)));
        assert!(!copied(&plan, &copies, tile(10)) && !copied(&plan, &copies, tile(20)));
        assert_eq!(
            refuse(&plan, &copies, A, A + 4 * PAGE, false),
            Some(Refusal::Copied),
            "a section view of A's run reaches tile 11"
        );
        assert_eq!(refuse(&plan, &copies, A + 2 * PAGE, A + 4 * PAGE, false), None, "A's pages 2 and 3 do not");
        // Pages outside the reservations are not counted.
        assert!(copies.add(&plan, 0x1000_0000, 0x1001_0000).is_empty());
        copies.remove(&runs);
        assert_eq!(copies.len(), 0);
        assert_eq!(refuse(&plan, &copies, A, A + 4 * PAGE, false), None, "the copy went");
    }

    #[test]
    fn copies_of_a_reservation_made_over_again_count_no_more() {
        let mut plan = plan();
        let mut copies = Copies::default();
        let old = copies.add(&plan, B, B + 4 * PAGE);
        assert!(copied(&plan, &copies, tile(11)));
        // B is reserved again (the title released it), and its page 2 shows
        // tile 11 again: the old copy is not over this memory.
        plan.reserve(B, 16 * PAGE);
        plan.map(B + 2 * PAGE, PAGE, tile(11));
        assert!(!copied(&plan, &copies, tile(11)), "a copy of the old memory");
        let new = copies.add(&plan, B + 2 * PAGE, B + 3 * PAGE);
        assert!(copied(&plan, &copies, tile(11)));
        // The old copy goes: the new one still counts.
        copies.remove(&old);
        assert!(copied(&plan, &copies, tile(11)), "the new copy's page stays counted");
        copies.remove(&new);
        assert!(!copied(&plan, &copies, tile(11)));
        assert_eq!(copies.len(), 0, "every page given back");
    }

    #[test]
    fn a_piece_view_may_not_reach_a_page_shown_twice() {
        let plan = plan();
        let copies = Copies::default();
        // A's pages 0..9 as a piece view reach tiles 11 and 20, each shown
        // at two addresses; a section view of the run 0..4 may reach 11 (its
        // two addresses are one range then).
        assert_eq!(refuse(&plan, &copies, A, A + 9 * PAGE, true), Some(Refusal::TwoAddresses));
        assert_eq!(refuse(&plan, &copies, A, A + 4 * PAGE, false), None);
        assert_eq!(refuse(&plan, &copies, A + 2 * PAGE, A + 4 * PAGE, true), None, "tiles 12 and 13: one address");
        // Copied comes first.
        let mut copies = Copies::default();
        copies.add(&plan, B + 5 * PAGE, B + 6 * PAGE);
        assert_eq!(refuse(&plan, &copies, A, A + 9 * PAGE, true), Some(Refusal::Copied));
    }

    #[test]
    fn views_leave_where_a_copy_or_a_map_brings_a_second_range() {
        let mut plan = plan();
        let mut copies = Copies::default();
        // Section views hold tiles 10..14 (A's run), a piece view tile 20.
        let viewed = |t: HeapTile| match t.tile {
            10..14 => [1, 0],
            20 => [0, 1],
            _ => [0, 0],
        };
        // A copy over B's page 2 (tile 11): both addresses of tile 11 leave,
        // A's page 1 and B's page 2.
        copies.add(&plan, B + 2 * PAGE, B + 3 * PAGE);
        let gone = leaving(&plan, &copies, B + 2 * PAGE, B + 3 * PAGE, Cause::Copy, true, viewed);
        assert_eq!(gone, vec![A + PAGE..A + 2 * PAGE, B + 2 * PAGE..B + 3 * PAGE]);
        // A copy over pages no view reaches: nothing leaves.
        assert!(leaving(&plan, &copies, B + 9 * PAGE, B + 10 * PAGE, Cause::Copy, true, viewed).is_empty());
        // The title maps tile 12 under the copy (B's page 2 shows 12 now):
        // tile 12 is copied, its views at A's page 2 leave.
        plan.map(B + 2 * PAGE, PAGE, tile(12));
        let gone = leaving(&plan, &copies, B + 2 * PAGE, B + 3 * PAGE, Cause::Map, true, viewed);
        assert_eq!(gone, vec![A + 2 * PAGE..A + 3 * PAGE, B + 2 * PAGE..B + 3 * PAGE]);
        // A map of tile 20 at a third address: piece views reach it, shown
        // at two or more addresses: all three addresses leave (with pieces);
        // views of sections alone have nothing to leave there.
        plan.map(B + 7 * PAGE, PAGE, tile(20));
        let gone = leaving(&plan, &copies, B + 7 * PAGE, B + 8 * PAGE, Cause::Map, true, viewed);
        assert_eq!(
            gone,
            vec![A + 8 * PAGE..A + 9 * PAGE, B + 5 * PAGE..B + 6 * PAGE, B + 7 * PAGE..B + 8 * PAGE]
        );
        assert!(leaving(&plan, &copies, B + 7 * PAGE, B + 8 * PAGE, Cause::Map, false, viewed).is_empty());
        // A map that brings no second range: nothing.
        plan.map(B + 9 * PAGE, PAGE, tile(13));
        assert!(leaving(&plan, &copies, B + 9 * PAGE, B + 10 * PAGE, Cause::Map, false, viewed).is_empty());
    }
}
