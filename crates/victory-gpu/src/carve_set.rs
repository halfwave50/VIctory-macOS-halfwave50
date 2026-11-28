//! mac: the title ranges carved out of zero-copy aliasing for withheld
//! writes (`VICTORY_WITHHELD_COPIES`, `notes/alias-physical-debug.md`,
//! "Withheld writes"): a set of disjoint ranges, joined where they meet.
//! `crate::shadow` serves a lookup inside one from a host copy, never from
//! an aliased region (`VICTORY_ALIAS_BUFFERS`) or a view of physical pages
//! (`VICTORY_ALIAS_PHYSICAL=gpu`).
//!
//! Plain std, so its tests run everywhere.

// `crate::shadow` uses it on Windows; the tests everywhere.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ops::Range;

/// Disjoint ranges `[start, end)` by start, none touching another.
#[derive(Debug, Default)]
pub(crate) struct Carved {
    ranges: BTreeMap<u64, u64>,
}

impl Carved {
    pub(crate) const fn new() -> Carved {
        Carved { ranges: BTreeMap::new() }
    }

    /// Adds `range`, joined with any it overlaps or touches. Whether
    /// anything new was carved.
    pub(crate) fn insert(&mut self, range: Range<u64>) -> bool {
        if range.is_empty() || self.contains(range.clone()) {
            return false;
        }
        let (mut start, mut end) = (range.start, range.end);
        let touching: Vec<u64> = self
            .ranges
            .range(..=end)
            .rev()
            .take_while(|&(_, &e)| e >= start)
            .map(|(&s, _)| s)
            .collect();
        for s in touching {
            let e = self.ranges.remove(&s).expect("listed above");
            start = start.min(s);
            end = end.max(e);
        }
        self.ranges.insert(start, end);
        true
    }

    /// Whether any byte of `range` is carved.
    pub(crate) fn overlaps(&self, range: Range<u64>) -> bool {
        !range.is_empty()
            && self
                .ranges
                .range(..range.end)
                .next_back()
                .is_some_and(|(_, &e)| e > range.start)
    }

    /// Whether all of `range` is carved (in one carved range: they never
    /// touch).
    pub(crate) fn contains(&self, range: Range<u64>) -> bool {
        !range.is_empty()
            && self
                .ranges
                .range(..=range.start)
                .next_back()
                .is_some_and(|(_, &e)| e >= range.end)
    }

    /// Carved ranges, and their bytes.
    pub(crate) fn totals(&self) -> (usize, u64) {
        (self.ranges.len(), self.ranges.iter().map(|(s, e)| e - s).sum())
    }
}

/// The range a write of `[address, address + len)` carves: the innermost
/// title buffer holding it (`buffer`, from `crate::shadow`'s registry), else
/// the `window`-aligned window around it, as `crate::shadow` copies loose
/// memory.
pub(crate) fn carve_range(address: u64, len: u64, buffer: Option<Range<u64>>, window: u64) -> Range<u64> {
    let end = address.saturating_add(len.max(1));
    if let Some(b) = buffer
        && b.start <= address
        && end <= b.end
    {
        return b;
    }
    let window = window.max(1);
    let start = address / window * window;
    start..end.div_ceil(window).saturating_mul(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carved_ranges_join_and_answer_overlaps() {
        let mut c = Carved::default();
        assert!(c.insert(0x1000..0x2000));
        assert!(!c.insert(0x1100..0x1200), "inside: nothing new");
        assert!(c.overlaps(0x1fff..0x3000) && !c.overlaps(0x2000..0x3000));
        assert!(!c.overlaps(0x0..0x1000) && c.overlaps(0x0..0x1001));
        assert!(c.contains(0x1000..0x2000) && !c.contains(0x1000..0x2001));
        // Touching joins; a range over two joins all three.
        assert!(c.insert(0x2000..0x2800));
        assert_eq!(c.totals(), (1, 0x1800));
        assert!(c.insert(0x4000..0x5000));
        assert_eq!(c.totals(), (2, 0x2800));
        assert!(c.insert(0x2700..0x4100));
        assert_eq!(c.totals(), (1, 0x4000));
        assert!(c.contains(0x1000..0x5000));
        assert!(!c.overlaps(0x10..0x10), "empty");
        assert!(!c.insert(0x9000..0x9000), "empty");
    }

    #[test]
    fn a_write_carves_its_title_buffer_or_window() {
        const W: u64 = 4 << 20;
        let buffer = Some(0x10_0100..0x10_9000);
        assert_eq!(carve_range(0x10_0200, 0x100, buffer.clone(), W), 0x10_0100..0x10_9000);
        // Past the buffer: the window around it.
        assert_eq!(carve_range(0x10_8f00, 0x200, buffer, W), 0..W);
        assert_eq!(carve_range(W + 5, 1, None, W), W..2 * W);
        assert_eq!(carve_range(W - 1, 2, None, W), 0..2 * W, "across two windows");
    }
}
