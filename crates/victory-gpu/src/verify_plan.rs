//! mac: stage 2's verify mode, its plain-std half (`VICTORY_ALIAS_PHYSICAL=verify`,
//! `notes/alias-physical-debug.md`): where a view of title memory reaches
//! its bytes, through the tile plan, and the CPU's compare of those bytes
//! with the title's own.
//!
//! A view (`crate::alias_physical::View`) is a buffer (a reservation's piece,
//! or a section buffer over a heap) and a byte offset in it. A binding turns
//! a title address into a byte of that buffer with its own arithmetic (root
//! views, table views' first element, vertex and index buffer addresses,
//! page-table pairs). [`Reach::tile`] says which heap tile that byte shows
//! now; the title shows a heap tile at the address too ([`Plan::current_at`]).
//! Both must be the same tile, and the bytes the same: [`compare`] checks
//! both over a few sampled stretches ([`samples`]), with the title's and the
//! heap's bytes read through closures (on Windows the title's address and
//! the heap's CPU mapping; in the tests, vectors).
//!
//! Plain std, so its tests run everywhere.

// `crate::alias_verify` uses it on Windows; the tests everywhere.
#![allow(dead_code)]

use std::ops::Range;

use crate::tile_plan::{HeapTile, PAGE, Plan};

/// What a view of title memory reaches its bytes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Reach {
    /// A reservation's piece: buffer byte `b` is the piece's tile
    /// `b / PAGE`, which shows what the plan maps there (it follows remaps).
    Piece { piece: usize },
    /// A section buffer over heap `heap`: buffer byte `b` is heap tile
    /// `tile0 + b / PAGE` (fixed: it does not follow remaps); the buffer
    /// has `tiles` tiles.
    Section { heap: u32, tile0: u32, tiles: u32 },
}

/// A byte past the end of the view's buffer (or of a piece whose
/// reservation is gone): D3D12 reads zero there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Past;

impl Reach {
    /// The heap tile buffer byte `byte` shows as xmem maps it now
    /// (`Ok(None)`: a tile unmapped or never mapped, which reads zero).
    pub(crate) fn tile(self, plan: &Plan, byte: u64) -> Result<Option<HeapTile>, Past> {
        let t = u32::try_from(byte / PAGE as u64).map_err(|_| Past)?;
        match self {
            Reach::Piece { piece } => {
                let range = plan.piece(piece).ok_or(Past)?;
                if byte >= range.len() as u64 {
                    return Err(Past);
                }
                Ok(plan.current(piece, t))
            }
            Reach::Section { heap, tile0, tiles } => {
                if t >= tiles {
                    return Err(Past);
                }
                Ok(Some(HeapTile {
                    heap,
                    tile: tile0.checked_add(t).ok_or(Past)?,
                }))
            }
        }
    }

    /// The same GPU range is reached through the same buffer at the same
    /// tile: `(buffer, tile in it)`, for telling two ranges over one page
    /// apart.
    pub(crate) fn range_id(self, byte: u64) -> (u64, u32) {
        let t = (byte / PAGE as u64) as u32;
        match self {
            Reach::Piece { piece } => (piece as u64, t),
            Reach::Section { heap, tile0, .. } => {
                (1 << 63 | u64::from(heap) << 32 | u64::from(tile0), t)
            }
        }
    }
}

/// A stretch of title memory to compare: `len` bytes at title address `at`,
/// inside one title page and one tile of the view's buffer, which the
/// binding puts at buffer byte `byte`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sample {
    pub(crate) at: usize,
    pub(crate) len: usize,
    pub(crate) byte: u64,
}

/// The stretches [`compare`] reads of a binding of `[from, to)`: `stretch`
/// bytes at its start and at its end, and at most `spread` more spread over
/// the pages between, each cut at title page and buffer tile edges.
/// `byte_of(a)` is the binding's buffer byte for title address `a`
/// (linear: `byte_of(a + k) = byte_of(a) + k`).
pub(crate) fn samples(
    from: usize,
    to: usize,
    byte_of: impl Fn(usize) -> u64,
    stretch: usize,
    spread: usize,
) -> Vec<Sample> {
    let mut out: Vec<Sample> = Vec::new();
    if to <= from || stretch == 0 {
        return out;
    }
    let mut starts = vec![from];
    let pages = (to - 1) / PAGE - from / PAGE;
    if spread > 0 && pages > 1 {
        let step = (pages / (spread + 1)).max(1);
        let mut page = from / PAGE + step;
        while page < (to - 1) / PAGE && starts.len() <= spread {
            starts.push(page * PAGE);
            page += step;
        }
    }
    starts.push(to.saturating_sub(stretch).max(from));
    starts.sort_unstable();
    starts.dedup();
    for start in starts {
        let end = start.saturating_add(stretch).min(to);
        let mut at = start;
        while at < end {
            // Already compared (the stretches at the ends may overlap).
            if let Some(s) = out.iter().find(|s| s.at <= at && at < s.at + s.len) {
                at = s.at + s.len;
                continue;
            }
            let byte = byte_of(at);
            let page_left = PAGE - at % PAGE;
            let tile_left = PAGE - (byte % PAGE as u64) as usize;
            let len = (end - at).min(page_left).min(tile_left);
            out.push(Sample { at, len, byte });
            at += len;
        }
    }
    out
}

/// The first difference [`compare`] found: the title address and the
/// buffer byte, the heap tile the title shows there and the one the view
/// reaches, and the first four words of each side from there (zeros past
/// what was read).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Found {
    pub(crate) at: usize,
    pub(crate) byte: u64,
    pub(crate) title_tile: Option<HeapTile>,
    pub(crate) view_tile: Result<Option<HeapTile>, Past>,
    pub(crate) expected: [u32; 4],
    pub(crate) found: [u32; 4],
}

/// What [`compare`] found: stretches and bytes compared, stretches whose
/// heap tile differs (a latent fault even where the bytes agree, as zeros
/// do), stretches whose bytes differ, title stretches the CPU could not read
/// (not mapped: taken as zeros), and the first difference of either kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Outcome {
    pub(crate) stretches: u32,
    pub(crate) bytes: u64,
    pub(crate) tiles_differ: u32,
    pub(crate) bytes_differ: u32,
    pub(crate) unreadable: u32,
    pub(crate) first: Option<Found>,
}

impl Outcome {
    /// Whether anything differs.
    pub(crate) fn differs(&self) -> bool {
        self.tiles_differ + self.bytes_differ > 0
    }
}

/// Compares, stretch by stretch, the bytes a view reaching through `reach`
/// shows at the samples' buffer bytes with the title's own at their
/// addresses. `title(at, out)` reads title memory (false: not readable,
/// taken as zeros); `heap(tile, offset, out)` reads a heap tile from
/// `offset` on. The heap tile the title shows is `plan.current_at`.
pub(crate) fn compare(
    plan: &Plan,
    reach: Reach,
    samples: &[Sample],
    title: &mut dyn FnMut(usize, &mut [u8]) -> bool,
    heap: &mut dyn FnMut(HeapTile, usize, &mut [u8]) -> bool,
) -> Outcome {
    let mut out = Outcome::default();
    let (mut want, mut got) = (Vec::new(), Vec::new());
    for s in samples {
        if s.len == 0 {
            continue;
        }
        out.stretches += 1;
        out.bytes += s.len as u64;
        let title_tile = plan.current_at(s.at / PAGE * PAGE);
        let view_tile = reach.tile(plan, s.byte);
        want.clear();
        want.resize(s.len, 0);
        got.clear();
        got.resize(s.len, 0);
        if !title(s.at, &mut want) {
            want.fill(0);
            out.unreadable += 1;
        }
        if let Ok(Some(t)) = view_tile
            && !heap(t, (s.byte % PAGE as u64) as usize, &mut got)
        {
            got.fill(0);
        }
        let tiles_differ =
            view_tile != Ok(title_tile) && !(title_tile.is_none() && view_tile == Err(Past));
        let first_byte = want.iter().zip(&got).position(|(a, b)| a != b);
        if tiles_differ {
            out.tiles_differ += 1;
        }
        if first_byte.is_some() {
            out.bytes_differ += 1;
        }
        if out.first.is_none() && (tiles_differ || first_byte.is_some()) {
            // From the word the first differing byte is in.
            let k = first_byte.map_or(0, |k| k / 4 * 4);
            out.first = Some(Found {
                at: s.at + k,
                byte: s.byte + k as u64,
                title_tile,
                view_tile,
                expected: words(&want[k..]),
                found: words(&got[k..]),
            });
        }
    }
    out
}

/// The first four little-endian words of `bytes` (zeros past its end).
pub(crate) fn words(bytes: &[u8]) -> [u32; 4] {
    let mut w = [0u32; 4];
    for (i, word) in w.iter_mut().enumerate() {
        let mut b = [0u8; 4];
        for (j, x) in b.iter_mut().enumerate() {
            *x = bytes.get(4 * i + j).copied().unwrap_or(0);
        }
        *word = u32::from_le_bytes(b);
    }
    w
}

/// The heap tile range `[first, first + n)` of a run's section view, as the
/// tiles of its section buffer: the buffer's first tile (`tile0`) is the
/// heap piece's, `j * per`.
pub(crate) fn section_tile0(first: u32, per: u32) -> u32 {
    if per == 0 { 0 } else { first / per * per }
}

/// Pages `[from, to)` reaches, for counting ranges over one page.
pub(crate) fn pages_of(range: &Range<usize>) -> Range<usize> {
    range.start / PAGE..range.end.div_ceil(PAGE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_plan::Never;
    use std::collections::HashMap;

    /// A reservation of `n` pages at `BASE`, its plan flushed after the
    /// maps; heaps of 64 tiles each, every word of heap `h` tile `t` at
    /// offset `o` being `h << 24 | t << 16 | o / 4`.
    const BASE: usize = 0x40_0000_0000;

    fn heap_word(t: HeapTile, offset: usize) -> u32 {
        t.heap << 24 | t.tile << 16 | (offset / 4) as u32
    }

    fn heap_bytes(t: HeapTile, offset: usize, out: &mut [u8]) -> bool {
        for (k, b) in out.iter_mut().enumerate() {
            let o = offset + k;
            *b = heap_word(t, o / 4 * 4).to_le_bytes()[o % 4];
        }
        true
    }

    /// The title's bytes are its pages' heap tiles' (one RAM), read
    /// through the plan as xmem maps it now; unmapped pages are not
    /// readable.
    fn title_bytes(plan: &Plan) -> impl FnMut(usize, &mut [u8]) -> bool + '_ {
        move |at, out| {
            let Some(t) = plan.current_at(at / PAGE * PAGE) else {
                return false;
            };
            heap_bytes(t, at % PAGE, out)
        }
    }

    fn plan_with(maps: &[(usize, HeapTile)], pages: usize) -> Plan {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(BASE, pages * PAGE);
        for &(page, t) in maps {
            plan.map(BASE + page * PAGE, PAGE, t);
        }
        plan
    }

    fn tile(heap: u32, tile: u32) -> HeapTile {
        HeapTile { heap, tile }
    }

    fn run(
        plan: &Plan,
        reach: Reach,
        from: usize,
        to: usize,
        byte_of: impl Fn(usize) -> u64,
    ) -> Outcome {
        let s = samples(from, to, byte_of, 64, 4);
        let mut title = title_bytes(plan);
        compare(plan, reach, &s, &mut title, &mut heap_bytes)
    }

    #[test]
    fn samples_cut_at_page_and_tile_edges() {
        // A buffer byte 32 bytes before a tile edge, a title stretch inside
        // one page: two stretches.
        let at = BASE + 0x100;
        let s = samples(at, at + 64, |a| (a - at) as u64 + PAGE as u64 - 32, 64, 0);
        assert_eq!(
            s,
            vec![
                Sample {
                    at,
                    len: 32,
                    byte: PAGE as u64 - 32
                },
                Sample {
                    at: at + 32,
                    len: 32,
                    byte: PAGE as u64
                },
            ]
        );
        // Start, end and spread stretches over 10 pages, none past `to`,
        // none twice.
        let (from, to) = (BASE + 8, BASE + 10 * PAGE - 8);
        let s = samples(from, to, |a| (a - BASE) as u64, 64, 3);
        assert!(
            s.iter()
                .all(|x| x.at >= from && x.at + x.len <= to && x.len > 0)
        );
        assert_eq!(s.first().map(|x| x.at), Some(from));
        assert_eq!(s.last().map(|x| x.at + x.len), Some(to));
        assert!(s.len() >= 5, "{s:?}");
        for (i, a) in s.iter().enumerate() {
            for b in &s[i + 1..] {
                assert!(a.at + a.len <= b.at || b.at + b.len <= a.at, "{a:?} {b:?}");
            }
        }
        // A short range: one stretch.
        assert_eq!(samples(BASE, BASE + 12, |a| a as u64, 64, 4).len(), 1);
        assert!(samples(BASE, BASE, |a| a as u64, 64, 4).is_empty());
    }

    #[test]
    fn a_piece_view_at_its_offset_reads_the_titles_bytes() {
        // Scattered heap tiles, flushed or not: a piece view follows the plan.
        let mut plan = plan_with(
            &[
                (0, tile(0, 7)),
                (1, tile(0, 3)),
                (2, tile(1, 9)),
                (5, tile(0, 7)),
            ],
            8,
        );
        let piece = Reach::Piece { piece: 0 };
        let check = |plan: &Plan| {
            run(plan, piece, BASE + 0x40, BASE + 6 * PAGE, |a| {
                (a - BASE) as u64
            })
        };
        let o = check(&plan);
        assert!(!o.differs(), "{o:?}");
        assert!(
            o.stretches >= 3 && o.unreadable > 0,
            "pages 3 and 4 are not mapped: {o:?}"
        );
        plan.flush(tile(0, 63));
        let o = check(&plan);
        assert!(!o.differs(), "{o:?}");
        // A remap: the piece follows it.
        plan.map(BASE + PAGE, PAGE, tile(1, 1));
        assert!(!check(&plan).differs());
    }

    #[test]
    fn a_view_one_page_off_is_reported_with_both_words() {
        let plan = plan_with(&[(0, tile(0, 7)), (1, tile(0, 3)), (2, tile(1, 9))], 4);
        let piece = Reach::Piece { piece: 0 };
        // The binding's offset is one page too far.
        let o = run(&plan, piece, BASE + 0x40, BASE + 2 * PAGE, |a| {
            (a - BASE + PAGE) as u64
        });
        assert!(
            o.differs() && o.tiles_differ > 0 && o.bytes_differ > 0,
            "{o:?}"
        );
        let f = o.first.expect("a difference");
        assert_eq!(f.at, BASE + 0x40);
        assert_eq!(f.byte, PAGE as u64 + 0x40);
        assert_eq!(f.title_tile, Some(tile(0, 7)));
        assert_eq!(f.view_tile, Ok(Some(tile(0, 3))));
        assert_eq!(f.expected[0], heap_word(tile(0, 7), 0x40));
        assert_eq!(f.found[0], heap_word(tile(0, 3), 0x40));
        // Four bytes off: the same tiles, other bytes.
        let o = run(&plan, piece, BASE + 0x40, BASE + 0x80, |a| {
            (a - BASE + 4) as u64
        });
        assert_eq!((o.tiles_differ, o.bytes_differ), (0, 1), "{o:?}");
        let f = o.first.unwrap();
        assert_eq!(
            (f.expected[0], f.found[0]),
            (heap_word(tile(0, 7), 0x40), heap_word(tile(0, 7), 0x44))
        );
    }

    #[test]
    fn a_section_view_reaches_heap_tiles_from_its_heap_piece() {
        // Pages 1..4 show heap 0's tiles 10..13: a run. The section buffer of
        // heap piece 1 (tiles 8..16, 8 a piece) starts at tile 8, so the
        // run's view starts at byte (10 - 8) * PAGE + its offset in the page.
        let plan = plan_with(&[(1, tile(0, 10)), (2, tile(0, 11)), (3, tile(0, 12))], 4);
        let tile0 = section_tile0(10, 8);
        assert_eq!(tile0, 8);
        let reach = Reach::Section {
            heap: 0,
            tile0,
            tiles: 8,
        };
        let from = BASE + PAGE + 0x30;
        let base = u64::from(10 - tile0) * PAGE as u64 + 0x30;
        let o = run(&plan, reach, from, BASE + 4 * PAGE, |a| {
            base + (a - from) as u64
        });
        assert!(!o.differs(), "{o:?}");
        // A base one tile too far.
        let wrong = base + PAGE as u64;
        let o = run(&plan, reach, from, BASE + 4 * PAGE, |a| {
            wrong + (a - from) as u64
        });
        assert!(o.differs(), "{o:?}");
        assert_eq!(o.first.unwrap().view_tile, Ok(Some(tile(0, 11))));
        // Counted from the heap's start (tile 0), not the heap piece's: past
        // the section buffer.
        let o = run(&plan, reach, from, from + 64, |a| {
            10 * PAGE as u64 + 0x30 + (a - from) as u64
        });
        assert_eq!(o.first.unwrap().view_tile, Err(Past));
        // Past the buffer: D3D12 reads zero; the title's bytes are not zero.
        let o = run(&plan, reach, from, from + 16, |_| 8 * PAGE as u64);
        assert_eq!(o.first.unwrap().view_tile, Err(Past));
    }

    #[test]
    fn a_remap_under_a_section_view_shows_it_stale_and_a_piece_view_not() {
        let mut plan = plan_with(&[(0, tile(0, 4)), (1, tile(0, 5)), (2, tile(0, 6))], 4);
        plan.flush(tile(0, 63));
        let section = Reach::Section {
            heap: 0,
            tile0: 0,
            tiles: 64,
        };
        let byte = |a: usize| 4 * PAGE as u64 + (a - BASE) as u64;
        assert!(!run(&plan, section, BASE, BASE + 3 * PAGE, byte).differs());
        // The title maps another page at page 1.
        plan.map(BASE + PAGE, PAGE, tile(0, 40));
        let o = run(&plan, section, BASE, BASE + 3 * PAGE, byte);
        assert!(o.tiles_differ > 0 && o.bytes_differ > 0, "{o:?}");
        let f = o.first.unwrap();
        assert_eq!(
            (f.at / PAGE * PAGE, f.title_tile, f.view_tile),
            (BASE + PAGE, Some(tile(0, 40)), Ok(Some(tile(0, 5))))
        );
        let piece = Reach::Piece { piece: 0 };
        assert!(!run(&plan, piece, BASE, BASE + 3 * PAGE, |a| (a - BASE) as u64).differs());
        // And unmapped: the title shows nothing there (zeros, not readable),
        // the section view still the old tile's bytes.
        plan.unmap(BASE + PAGE, PAGE);
        let o = run(&plan, section, BASE + PAGE, BASE + PAGE + 64, byte);
        assert!(o.differs() && o.unreadable == 1, "{o:?}");
        assert!(
            !run(&plan, piece, BASE + PAGE, BASE + PAGE + 64, |a| (a - BASE)
                as u64)
            .differs()
        );
    }

    #[test]
    fn range_ids_tell_two_ranges_over_one_page_apart() {
        let piece = Reach::Piece { piece: 3 };
        let section = Reach::Section {
            heap: 0,
            tile0: 8,
            tiles: 8,
        };
        let mut by_tile: HashMap<(u64, u32), u32> = HashMap::new();
        for (reach, byte) in [
            (piece, 0u64),
            (piece, 100),
            (section, 0),
            (piece, PAGE as u64),
        ] {
            *by_tile.entry(reach.range_id(byte)).or_default() += 1;
        }
        assert_eq!(by_tile[&piece.range_id(0)], 2, "one range");
        assert_eq!(by_tile.len(), 3);
        assert_ne!(piece.range_id(0), section.range_id(0));
    }
}
