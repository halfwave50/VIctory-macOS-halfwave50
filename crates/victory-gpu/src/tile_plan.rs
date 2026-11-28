//! mac: stage 2's tile plan (`VICTORY_ALIAS_PHYSICAL`,
//! `notes/alias-physical-design.md` §2, increment I3a): how the title's
//! mappings of physical pages become tile mappings of reserved buffers.
//!
//! Each placeholder reservation (`xmem`'s 64 KiB-page reservations) is covered
//! from its start by pieces of at most 4 GiB less one tile, one reserved
//! buffer each: descriptor-table views of a buffer of exactly 4 or 8 GiB read
//! zeros (probe 6, `notes/tile-probe.md`), and root views and clamp batches
//! take 32-bit offsets. Tile `t` of a piece is the
//! page address `piece base + t * 64 KiB`. `xmem` tells each view it maps
//! (a title range showing consecutive tiles of one heap: a section's pages)
//! and each it unmaps, after its CPU remap; the last change of a tile before
//! a flush wins. A flush drains the changes against what the tiles show
//! already, as one update per piece and heap (one `UpdateTileMappings` each)
//! with the ranges joined: consecutive tiles over consecutive heap tiles, or
//! over one heap tile (the sink, or one page shown at many addresses).
//!
//! A tile that shows a mapping and changes is first nulled ([`Flush::clears`],
//! one null `UpdateTileMappings` a piece, before its updates), then mapped:
//! mapping a tile mapped already costs 10 to 13 µs on D3DMetal's queue
//! thread, a null 0.8 and a fresh map 1.1 (probes 7 and 8). An unmapped tile
//! is then pointed at the sink, a zeroed tile, so a null that does not take
//! (probe 1's 64 tiles after remaps) is overwritten. A tile never mapped is
//! left as its buffer was made ([`Never::Untouched`]: such tiles read zero,
//! probe 7); [`Never::Sink`] points those at the sink too, at a piece's first
//! flush.
//!
//! Plain std, so its tests run everywhere; the D3D12 side (I3b) makes the
//! buffers and turns [`Update`]s into `UpdateTileMappings`.

// `crate::alias_physical` uses it on Windows (I3b); `Never::Sink` and parts of
// `Flush` are for tests and the counts.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

/// The 64 KiB page physical pages come in, and a D3D12 tile.
pub(crate) const PAGE: usize = 64 << 10;
/// The largest piece one reserved buffer covers: 4 GiB less one tile (a
/// table view of a buffer of 4 GiB or 8 GiB reads zeros; probe 6).
pub(crate) const PIECE: usize = (1 << 32) - PAGE;

/// A tile of a heap: the heap (one per physical section) and the tile's
/// number in it (a section's page index, or its sink's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct HeapTile {
    pub(crate) heap: u32,
    pub(crate) tile: u32,
}

/// What tiles never mapped get at their piece's first flush.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Never {
    /// Nothing: they stay as the buffer was made (D3D12's initial mapping).
    Untouched,
    /// The sink.
    Sink,
}

/// One range of an update: `count` tiles of the piece from `start` show heap
/// tiles from `heap_start` on, or all of them heap tile `heap_start` when
/// `one` (`D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE`; only with 2 or more).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileRange {
    pub(crate) start: u32,
    pub(crate) count: u32,
    pub(crate) heap_start: u32,
    pub(crate) one: bool,
}

/// One `UpdateTileMappings`: a piece's ranges over one heap, in tile order,
/// none overlapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Update {
    pub(crate) piece: usize,
    pub(crate) heap: u32,
    pub(crate) ranges: Vec<TileRange>,
}

/// A flush: per piece, the ranges of tiles to null first `(start, count)`
/// (every tile that showed a mapping and changes); its updates; and its
/// tiles by what they showed before: never mapped and now a heap tile
/// (`fresh`), a heap tile or the sink and now another heap tile
/// (`remapped`), and pointed at the sink (`sunk`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Flush {
    pub(crate) clears: Vec<(usize, Vec<(u32, u32)>)>,
    pub(crate) updates: Vec<Update>,
    pub(crate) fresh: usize,
    pub(crate) remapped: usize,
    pub(crate) sunk: usize,
}

/// A new reservation's pieces, and the pieces of the reservations it was
/// made over (their buffers go: the title released that memory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Reserved {
    pub(crate) pieces: Range<usize>,
    pub(crate) dropped: Vec<usize>,
}

/// What a tile shows, as flushed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Now {
    Heap(HeapTile),
    Sink,
}

/// A change waiting for the flush.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Change {
    Map(HeapTile),
    Unmap,
}

struct Piece {
    /// The title address of tile 0, and the tiles.
    base: usize,
    tiles: u32,
    /// What tiles show where it is not what an untouched tile shows (`floor`).
    now: HashMap<u32, Now>,
    /// What a tile not in `now` shows: never mapped (`None`), or the sink.
    floor: Option<Now>,
    /// Not flushed yet.
    new: bool,
    /// The changes since the last flush, the last one per tile.
    pending: BTreeMap<u32, Change>,
    /// Its reservation was made again over (its buffer goes).
    gone: bool,
}

/// The pieces of the title's reservations and their tiles' mappings.
pub(crate) struct Plan {
    never: Never,
    /// Reservations by start: their length and first piece.
    reservations: BTreeMap<usize, (usize, usize)>,
    pieces: Vec<Piece>,
    /// Pieces with changes since the last flush.
    dirty: BTreeSet<usize>,
}

impl Plan {
    pub(crate) fn new(never: Never) -> Self {
        Plan {
            never,
            reservations: BTreeMap::new(),
            pieces: Vec::new(),
            dirty: BTreeSet::new(),
        }
    }

    /// A placeholder reservation of `[at, at + len)` (whole 64 KiB pages):
    /// its pieces, of at most 4 GiB from `at` on. Reservations it overlaps
    /// are dropped first: the title released them through the host.
    pub(crate) fn reserve(&mut self, at: usize, len: usize) -> Reserved {
        let end = at.saturating_add(len);
        let over: Vec<usize> = self
            .reservations
            .range(..end)
            .filter(|&(&s, &(l, _))| s + l > at)
            .map(|(&s, _)| s)
            .collect();
        let mut dropped = Vec::new();
        for s in over {
            let (l, first) = self.reservations.remove(&s).expect("listed above");
            for id in first..first + l.div_ceil(PIECE) {
                let piece = &mut self.pieces[id];
                piece.gone = true;
                piece.pending.clear();
                piece.now.clear();
                self.dirty.remove(&id);
                dropped.push(id);
            }
        }
        let first = self.pieces.len();
        for base in (at..end).step_by(PIECE) {
            self.pieces.push(Piece {
                base,
                tiles: ((end - base).min(PIECE) / PAGE) as u32,
                now: HashMap::new(),
                floor: None,
                new: true,
                pending: BTreeMap::new(),
                gone: false,
            });
        }
        self.reservations.insert(at, (len, first));
        Reserved {
            pieces: first..self.pieces.len(),
            dropped,
        }
    }

    /// Piece `id`'s title range, while its reservation stands.
    pub(crate) fn piece(&self, id: usize) -> Option<Range<usize>> {
        let p = self.pieces.get(id).filter(|p| !p.gone)?;
        Some(p.base..p.base + p.tiles as usize * PAGE)
    }

    /// The piece title memory `[from, to)` lies in wholly, and the offset of
    /// `from` in it: where a view of the range starts in the piece's buffer.
    /// `None` for an empty range, one outside the reservations, or one that
    /// crosses the end of a piece (it keeps a copy).
    pub(crate) fn locate(&self, from: usize, to: usize) -> Option<(usize, u32)> {
        if to <= from {
            return None;
        }
        let (&at, &(len, first)) = self.reservations.range(..=from).next_back()?;
        if from - at >= len {
            return None;
        }
        let id = first + (from - at) / PIECE;
        let range = self.piece(id).filter(|r| r.contains(&from))?;
        (to <= range.end).then(|| (id, (from - range.start) as u32))
    }

    /// The pieces `[at, at + len)` reaches.
    pub(crate) fn pieces_of(&self, at: usize, len: usize) -> Vec<usize> {
        let (first_page, pages) = (at / PAGE, len / PAGE);
        let mut out: Vec<usize> = Vec::new();
        for k in 0..pages {
            let page_at = (first_page + k) * PAGE;
            if let Some((id, _)) = self.locate(page_at, page_at + PAGE)
                && out.last() != Some(&id)
            {
                out.push(id);
            }
        }
        out
    }

    /// `xmem` mapped `[at, at + len)` (whole pages) to the heap tiles from
    /// `first` on, one a page. The tiles of no reservation's piece are not
    /// recorded: their number is returned.
    pub(crate) fn map(&mut self, at: usize, len: usize, first: HeapTile) -> usize {
        self.record(at, len, |k| {
            Change::Map(HeapTile {
                heap: first.heap,
                tile: first.tile + k as u32,
            })
        })
    }

    /// `xmem` unmapped `[at, at + len)` (whole pages): those tiles go to the
    /// sink at the flush, if they show a page then. As [`Plan::map`], the
    /// tiles outside the pieces are returned.
    pub(crate) fn unmap(&mut self, at: usize, len: usize) -> usize {
        self.record(at, len, |_| Change::Unmap)
    }

    /// Records the change of each page of `[at, at + len)`, page `k` from
    /// `at` getting `change(k)`, split over the pieces it reaches.
    fn record(&mut self, at: usize, len: usize, change: impl Fn(usize) -> Change) -> usize {
        let (first_page, pages) = (at / PAGE, len / PAGE);
        let mut outside = 0;
        let mut k = 0;
        while k < pages {
            let page_at = (first_page + k) * PAGE;
            let Some((id, offset)) = self.locate(page_at, page_at + PAGE) else {
                outside += 1;
                k += 1;
                continue;
            };
            let piece = &mut self.pieces[id];
            let start = offset as usize / PAGE;
            // The rest of the range in this piece, at once.
            let n = (pages - k).min(piece.tiles as usize - start);
            for j in 0..n {
                piece.pending.insert((start + j) as u32, change(k + j));
            }
            self.dirty.insert(id);
            k += n;
        }
        outside
    }

    /// The heap tile that tile `tile` of piece `id` shows as flushed (`None`: the
    /// sink, or never mapped).
    pub(crate) fn shows(&self, id: usize, tile: u32) -> Option<HeapTile> {
        let piece = self.pieces.get(id)?;
        match piece.now.get(&tile).copied().or(piece.floor)? {
            Now::Heap(t) => Some(t),
            Now::Sink => None,
        }
    }

    /// Tiles with a change waiting.
    pub(crate) fn pending(&self) -> usize {
        self.dirty
            .iter()
            .map(|&id| self.pieces[id].pending.len())
            .sum()
    }

    /// The changes since the last flush, against what the tiles show: what
    /// the tiles show then is what was asked. `sink` is the zeroed tile
    /// unmapped tiles show.
    pub(crate) fn flush(&mut self, sink: HeapTile) -> Flush {
        let mut out = Flush::default();
        for id in std::mem::take(&mut self.dirty) {
            let never = self.never;
            let piece = &mut self.pieces[id];
            let pending = std::mem::take(&mut piece.pending);
            // Tiles to change, in order: (tile, what it shows now); and the
            // tiles to null first.
            let mut set: Vec<(TileRange, u32)> = Vec::new();
            let mut clear: Vec<u32> = Vec::new();
            let fill = piece.new && never == Never::Sink;
            let mut gap_from = 0u32;
            for (&tile, &change) in &pending {
                if fill && tile > gap_from {
                    set.push((range(gap_from, tile - gap_from, sink.tile, true), sink.heap));
                    out.sunk += (tile - gap_from) as usize;
                }
                gap_from = tile + 1;
                let shown = piece.now.get(&tile).copied().or(piece.floor);
                match change {
                    Change::Map(to) if shown == Some(Now::Heap(to)) => {}
                    Change::Map(to) => {
                        set.push((range(tile, 1, to.tile, false), to.heap));
                        if shown.is_some() {
                            clear.push(tile);
                            out.remapped += 1;
                        } else {
                            out.fresh += 1;
                        }
                        piece.now.insert(tile, Now::Heap(to));
                    }
                    // Never mapped (left so, or filled below), or the sink already.
                    Change::Unmap if !matches!(shown, Some(Now::Heap(_))) => {
                        if fill {
                            set.push((range(tile, 1, sink.tile, false), sink.heap));
                            out.sunk += 1;
                        }
                    }
                    Change::Unmap => {
                        clear.push(tile);
                        set.push((range(tile, 1, sink.tile, false), sink.heap));
                        out.sunk += 1;
                        if piece.floor == Some(Now::Sink) {
                            piece.now.remove(&tile);
                        } else {
                            piece.now.insert(tile, Now::Sink);
                        }
                    }
                }
            }
            if fill && piece.tiles > gap_from {
                set.push((
                    range(gap_from, piece.tiles - gap_from, sink.tile, true),
                    sink.heap,
                ));
                out.sunk += (piece.tiles - gap_from) as usize;
            }
            if fill {
                piece.floor = Some(Now::Sink);
                piece.now.retain(|_, now| *now != Now::Sink);
            }
            piece.new = false;
            // The nulls, consecutive tiles joined.
            let mut nulls: Vec<(u32, u32)> = Vec::new();
            for tile in clear {
                match nulls.last_mut() {
                    Some((start, count)) if *start + *count == tile => *count += 1,
                    _ => nulls.push((tile, 1)),
                }
            }
            if !nulls.is_empty() {
                out.clears.push((id, nulls));
            }
            // One update per heap, ranges joined.
            let mut by_heap: BTreeMap<u32, Vec<TileRange>> = BTreeMap::new();
            for (r, heap) in set {
                let ranges = by_heap.entry(heap).or_default();
                match ranges
                    .last_mut()
                    .and_then(|last| join(*last, r).map(|j| (last, j)))
                {
                    Some((last, joined)) => *last = joined,
                    None => ranges.push(r),
                }
            }
            out.updates
                .extend(by_heap.into_iter().map(|(heap, ranges)| Update {
                    piece: id,
                    heap,
                    ranges,
                }));
        }
        out
    }
}

/// A range, `one` only where it means something (2 tiles or more).
fn range(start: u32, count: u32, heap_start: u32, one: bool) -> TileRange {
    TileRange {
        start,
        count,
        heap_start,
        one: one && count > 1,
    }
}

/// `a` and `b` as one range, if `b` follows on from `a` in the piece and in
/// the heap (consecutive heap tiles), or both show one heap tile.
fn join(a: TileRange, b: TileRange) -> Option<TileRange> {
    if b.start != a.start + a.count {
        return None;
    }
    let count = a.count + b.count;
    let single = |r: TileRange| r.one || r.count == 1;
    let consecutive = |r: TileRange| !r.one || r.count == 1;
    if single(a) && single(b) && a.heap_start == b.heap_start {
        return Some(range(a.start, count, a.heap_start, true));
    }
    (consecutive(a) && consecutive(b) && b.heap_start == a.heap_start + a.count)
        .then(|| range(a.start, count, a.heap_start, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;
    /// A reservation's start (the title's are above 4 GiB, and one below).
    const A: usize = 0x10_0000_0000;
    const SINK: HeapTile = HeapTile {
        heap: 0,
        tile: 100_000,
    };

    fn heap(tile: u32) -> HeapTile {
        HeapTile { heap: 0, tile }
    }

    fn r(start: u32, count: u32, heap_start: u32, one: bool) -> TileRange {
        TileRange {
            start,
            count,
            heap_start,
            one,
        }
    }

    /// The tile mappings a flush asks for, per piece and heap.
    fn ranges(flush: &Flush) -> Vec<(usize, u32, Vec<TileRange>)> {
        flush
            .updates
            .iter()
            .map(|u| (u.piece, u.heap, u.ranges.clone()))
            .collect()
    }

    #[test]
    fn pieces_split_at_4_gib() {
        let mut plan = Plan::new(Never::Untouched);
        let nine = plan.reserve(A, 9 * GIB);
        assert_eq!(
            nine,
            Reserved {
                pieces: 0..3,
                dropped: vec![]
            }
        );
        // Pieces of 4 GiB less one tile: a table view of a buffer of exactly
        // 4 GiB reads zeros (probe 6).
        assert_eq!(PIECE, (4 << 30) - (64 << 10));
        assert_eq!(plan.piece(0), Some(A..A + PIECE));
        assert_eq!(plan.piece(1), Some(A + PIECE..A + 2 * PIECE));
        assert_eq!(plan.piece(2), Some(A + 2 * PIECE..A + 9 * GIB));
        assert_eq!(plan.pieces[0].tiles, 65_535);
        assert_eq!(plan.pieces[2].tiles, 16_386);
        // Exactly 4 GiB is two pieces, 4 GiB less a tile one; the 174 MiB
        // heap below 4 GiB one too. The game's 24 GiB reservations are 7.
        assert_eq!(plan.reserve(A + 16 * GIB, 4 * GIB).pieces, 3..5);
        assert_eq!(plan.reserve(A + 24 * GIB, PIECE).pieces, 5..6);
        assert_eq!(plan.reserve(0x1084_0000, 0xae0_0000).pieces, 6..7);
        assert_eq!(plan.reserve(A + 32 * GIB, 24 * GIB).pieces, 7..14);
        assert!((0..14).all(|id| plan.piece(id).is_some_and(|r| r.len() <= PIECE)));
        // Where a range lies: the piece, and its offset there (32 bits).
        assert_eq!(plan.locate(A, A + 1), Some((0, 0)));
        assert_eq!(
            plan.locate(A + PIECE - 8, A + PIECE),
            Some((0, (PIECE - 8) as u32))
        );
        assert_eq!(plan.locate(A + PIECE, A + PIECE + 8), Some((1, 0)));
        assert_eq!(
            plan.locate(A + 9 * GIB - 1, A + 9 * GIB),
            Some((2, (9 * GIB - 1 - 2 * PIECE) as u32))
        );
        assert_eq!(plan.locate(0x1084_0000 + 64, 0x1084_0100), Some((6, 64)));
        assert_eq!(plan.pieces_of(A + PIECE - PAGE, 2 * PAGE), vec![0, 1]);
        assert_eq!(plan.pieces_of(A + 9 * GIB, PAGE), Vec::<usize>::new());
        // Past the end, before the start, empty.
        assert_eq!(plan.locate(A + 9 * GIB, A + 9 * GIB + 1), None);
        assert_eq!(plan.locate(A - 1, A + 1), None);
        assert_eq!(plan.locate(A + 8, A + 8), None);
    }

    #[test]
    fn a_range_straddling_two_pieces_is_split() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, 5 * GIB);
        // A view's lookup across the pieces' line finds no one piece (a copy).
        assert_eq!(plan.locate(A + PIECE - 8, A + PIECE + 8), None);
        assert_eq!(plan.locate(A, A + 5 * GIB), None);
        // A view mapped across it: the last two tiles of piece 0 and the first
        // two of piece 1, heap tiles 10 to 13 in order.
        assert_eq!(plan.map(A + PIECE - 2 * PAGE, 4 * PAGE, heap(10)), 0);
        let flush = plan.flush(SINK);
        assert_eq!(
            ranges(&flush),
            vec![
                (0, 0, vec![r(65_533, 2, 10, false)]),
                (1, 0, vec![r(0, 2, 12, false)])
            ]
        );
        assert_eq!((flush.fresh, flush.remapped, flush.sunk), (4, 0, 0));
    }

    #[test]
    fn pending_mappings_coalesce() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, GIB);
        // Map, unmap and map again before a flush: the last wins.
        plan.map(A, 4 * PAGE, heap(0));
        plan.unmap(A + PAGE, 2 * PAGE);
        plan.map(A + 2 * PAGE, PAGE, heap(40));
        assert_eq!(plan.pending(), 4);
        let flush = plan.flush(SINK);
        // Tile 1 was never mapped as flushed: its unmap asks nothing. Heap
        // tiles 40 then 3 are not one run.
        assert_eq!(
            ranges(&flush),
            vec![(
                0,
                0,
                vec![r(0, 1, 0, false), r(2, 1, 40, false), r(3, 1, 3, false)]
            )]
        );
        assert_eq!((flush.fresh, flush.remapped, flush.sunk), (3, 0, 0));
        assert_eq!(plan.pending(), 0);
        assert_eq!(plan.flush(SINK), Flush::default(), "nothing left");
    }

    #[test]
    fn unmaps_go_to_the_sink() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, GIB);
        plan.map(A, 8 * PAGE, heap(0));
        plan.flush(SINK);
        plan.unmap(A + 2 * PAGE, 4 * PAGE);
        let flush = plan.flush(SINK);
        // Nulled first, then pointed at the sink.
        assert_eq!(flush.clears, vec![(0, vec![(2, 4)])]);
        assert_eq!(ranges(&flush), vec![(0, 0, vec![r(2, 4, SINK.tile, true)])]);
        assert_eq!((flush.fresh, flush.remapped, flush.sunk), (0, 0, 4));
        // Unmapped again, or never mapped: nothing (no null mapping ever).
        plan.unmap(A, PAGE * 64);
        let flush = plan.flush(SINK);
        assert_eq!(flush.clears, vec![(0, vec![(0, 2), (6, 2)])]);
        assert_eq!(
            ranges(&flush),
            vec![(
                0,
                0,
                vec![r(0, 2, SINK.tile, true), r(6, 2, SINK.tile, true)]
            )]
        );
        plan.unmap(A, PAGE * 64);
        assert_eq!(plan.flush(SINK), Flush::default());
        // A page mapped over the sink is a remap: nulled, then mapped.
        plan.map(A + 3 * PAGE, PAGE, heap(7));
        let flush = plan.flush(SINK);
        assert_eq!(flush.clears, vec![(0, vec![(3, 1)])]);
        assert_eq!(ranges(&flush), vec![(0, 0, vec![r(3, 1, 7, false)])]);
        assert_eq!((flush.fresh, flush.remapped), (0, 1));
    }

    #[test]
    fn never_mapped_tiles_can_go_to_the_sink_first() {
        let mut plan = Plan::new(Never::Sink);
        plan.reserve(A, 64 * PAGE);
        plan.map(A + 10 * PAGE, 2 * PAGE, heap(5));
        plan.unmap(A + 20 * PAGE, PAGE);
        let flush = plan.flush(SINK);
        // The gaps and the unmapped tile go to the sink, joined with it.
        assert_eq!(
            ranges(&flush),
            vec![(
                0,
                0,
                vec![
                    r(0, 10, SINK.tile, true),
                    r(10, 2, 5, false),
                    r(12, 52, SINK.tile, true)
                ]
            )]
        );
        assert_eq!((flush.fresh, flush.remapped, flush.sunk), (2, 0, 62));
        assert!(flush.clears.is_empty(), "nothing showed a mapping");
        // From then on the sink is what a tile shows: a map over it is a
        // remap, an unmap of it asks nothing.
        plan.map(A, PAGE, heap(9));
        plan.unmap(A + 30 * PAGE, PAGE);
        let flush = plan.flush(SINK);
        assert_eq!(flush.clears, vec![(0, vec![(0, 1)])]);
        assert_eq!(ranges(&flush), vec![(0, 0, vec![r(0, 1, 9, false)])]);
        assert_eq!((flush.fresh, flush.remapped, flush.sunk), (0, 1, 0));
        // A whole piece with no change at its first flush: one range.
        plan.reserve(A + GIB, PIECE);
        plan.unmap(A + GIB, PAGE);
        let flush = plan.flush(SINK);
        assert_eq!(
            ranges(&flush),
            vec![(1, 0, vec![r(0, 65_535, SINK.tile, true)])]
        );
    }

    #[test]
    fn ranges_join_into_calls() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, GIB);
        // Runs of consecutive heap tiles, out of order in the heap, one tile
        // apart in the piece: three ranges.
        plan.map(A, 3 * PAGE, heap(30));
        plan.map(A + 3 * PAGE, 2 * PAGE, heap(33));
        plan.map(A + 5 * PAGE, 2 * PAGE, heap(10));
        plan.map(A + 8 * PAGE, PAGE, heap(12));
        // One constant page at four addresses (`CopyPageMappingsBatchX`'s
        // kind 1, a page at a time): one range of one heap tile.
        for k in 0..4 {
            plan.map(A + (20 + k) * PAGE, PAGE, heap(50));
        }
        // A page of another section's heap in between.
        plan.map(A + 9 * PAGE, PAGE, HeapTile { heap: 1, tile: 13 });
        let flush = plan.flush(SINK);
        assert_eq!(
            ranges(&flush),
            vec![
                (
                    0,
                    0,
                    vec![
                        r(0, 5, 30, false),
                        r(5, 2, 10, false),
                        r(8, 1, 12, false),
                        r(20, 4, 50, true)
                    ]
                ),
                (0, 1, vec![r(9, 1, 13, false)]),
            ]
        );
        assert_eq!(flush.fresh, 13);
        // A run and the same tile repeated do not join: 7, 8, 8.
        let (a, b) = (r(0, 2, 7, false), r(2, 1, 8, false));
        assert_eq!(join(a, b), None);
        assert_eq!(
            join(r(0, 1, 7, false), r(1, 1, 8, false)),
            Some(r(0, 2, 7, false))
        );
        assert_eq!(
            join(r(0, 1, 8, false), r(1, 1, 8, false)),
            Some(r(0, 2, 8, true))
        );
        assert_eq!(join(r(0, 3, 8, true), r(3, 1, 9, false)), None);
        assert_eq!(
            join(r(0, 3, 8, true), r(4, 1, 8, false)),
            None,
            "not adjacent"
        );
    }

    #[test]
    fn one_page_at_two_addresses_is_mapped_at_both() {
        let mut plan = Plan::new(Never::Untouched);
        // The pool view and the tiled heap: two reservations.
        plan.reserve(A, GIB);
        plan.reserve(A + 2 * GIB, GIB);
        plan.map(A + PAGE, PAGE, heap(77));
        plan.map(A + 2 * GIB + 5 * PAGE, PAGE, heap(77));
        let flush = plan.flush(SINK);
        assert_eq!(
            ranges(&flush),
            vec![
                (0, 0, vec![r(1, 1, 77, false)]),
                (1, 0, vec![r(5, 1, 77, false)])
            ]
        );
        // Unmapped at one: the other still shows it.
        plan.unmap(A + PAGE, PAGE);
        let flush = plan.flush(SINK);
        assert_eq!(flush.clears, vec![(0, vec![(1, 1)])]);
        assert_eq!(
            ranges(&flush),
            vec![(0, 0, vec![r(1, 1, SINK.tile, false)])]
        );
        assert_eq!(plan.pieces[1].now.get(&5), Some(&Now::Heap(heap(77))));
    }

    #[test]
    fn a_remap_changes_only_what_changed() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, GIB);
        plan.map(A, 16 * PAGE, heap(100));
        plan.flush(SINK);
        // The same pages again: nothing.
        plan.map(A, 16 * PAGE, heap(100));
        assert_eq!(plan.flush(SINK), Flush::default());
        // `xmem`'s unmap of one page of a view: the view goes and its two
        // remnants are mapped again; only the middle page changes.
        plan.unmap(A, 16 * PAGE);
        plan.map(A, 7 * PAGE, heap(100));
        plan.map(A + 8 * PAGE, 8 * PAGE, heap(108));
        let flush = plan.flush(SINK);
        assert_eq!(flush.clears, vec![(0, vec![(7, 1)])]);
        assert_eq!(
            ranges(&flush),
            vec![(0, 0, vec![r(7, 1, SINK.tile, false)])]
        );
        // Another page over a mapped one: a remap, nulled first (unmap then
        // map: about 2 µs a tile against 10 to 13; probe 8).
        plan.map(A + 3 * PAGE, PAGE, heap(9));
        let flush = plan.flush(SINK);
        assert_eq!(flush.clears, vec![(0, vec![(3, 1)])]);
        assert_eq!(ranges(&flush), vec![(0, 0, vec![r(3, 1, 9, false)])]);
        assert_eq!((flush.fresh, flush.remapped, flush.sunk), (0, 1, 0));
    }

    #[test]
    fn memory_outside_the_reservations_is_not_recorded() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, 4 * PAGE);
        // Two pages before, all four inside, two after.
        assert_eq!(plan.map(A - 2 * PAGE, 8 * PAGE, heap(0)), 4);
        assert_eq!(plan.pending(), 4);
        let flush = plan.flush(SINK);
        assert_eq!(ranges(&flush), vec![(0, 0, vec![r(0, 4, 2, false)])]);
        assert_eq!(plan.unmap(A + 100 * PAGE, PAGE), 1);
        assert_eq!(plan.flush(SINK), Flush::default());
    }

    #[test]
    fn a_reservation_made_again_drops_the_old_pieces() {
        let mut plan = Plan::new(Never::Untouched);
        plan.reserve(A, 5 * GIB);
        plan.map(A, PAGE, heap(1));
        // The title released it through the host, and reserved part again.
        let again = plan.reserve(A + GIB, GIB);
        assert_eq!(
            again,
            Reserved {
                pieces: 2..3,
                dropped: vec![0, 1]
            }
        );
        assert_eq!(plan.piece(0), None);
        assert_eq!(plan.locate(A, A + 8), None);
        assert_eq!(plan.locate(A + GIB, A + GIB + 8), Some((2, 0)));
        assert_eq!(plan.pending(), 0, "the old pieces' changes go with them");
        assert_eq!(plan.flush(SINK), Flush::default());
    }

    /// Any sequence of maps and unmaps, flushed now and then, leaves the
    /// buffers showing what was asked, each flush changing only tiles that
    /// change: pseudo-random runs over the 4 GiB line of a reservation and a
    /// small one after it, some running off its end.
    #[test]
    fn flushes_leave_the_buffers_showing_what_was_asked() {
        for never in [Never::Untouched, Never::Sink] {
            let mut plan = Plan::new(never);
            plan.reserve(A, 5 * GIB);
            let small = A + 8 * GIB;
            plan.reserve(small, 16 * PAGE);
            let pages: Vec<usize> = (0..64)
                .map(|k| A + 4 * GIB - 32 * PAGE + k * PAGE)
                .chain((0..16).map(|k| small + k * PAGE))
                .collect();
            // What each page address is to show (`None`: nothing), whether
            // it ever showed a page, and what the buffers show.
            let mut asked: HashMap<usize, Option<HeapTile>> = HashMap::new();
            let mut ever: BTreeSet<usize> = BTreeSet::new();
            let mut shown: HashMap<(usize, u32), HeapTile> = HashMap::new();
            let mut x: u64 = 0x2545_f491_4f6c_dd1d;
            let mut next = |n: u64| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (x >> 33) % n
            };
            for step in 0..2_000 {
                let at = pages[next(pages.len() as u64) as usize];
                let n = 1 + next(6) as usize;
                if next(3) == 0 {
                    plan.unmap(at, n * PAGE);
                    for k in 0..n {
                        let page = at + k * PAGE;
                        if plan.locate(page, page + 1).is_some() {
                            asked.insert(page, None);
                        }
                    }
                } else {
                    let first = heap(next(40) as u32);
                    let first = HeapTile {
                        heap: next(2) as u32,
                        ..first
                    };
                    plan.map(at, n * PAGE, first);
                    for k in 0..n {
                        let page = at + k * PAGE;
                        if plan.locate(page, page + 1).is_some() {
                            let tile = HeapTile {
                                tile: first.tile + k as u32,
                                ..first
                            };
                            asked.insert(page, Some(tile));
                            ever.insert(page);
                        }
                    }
                }
                if next(5) != 0 && step != 1_999 {
                    continue;
                }
                let flush = plan.flush(SINK);
                // The nulls first: every nulled tile is mapped again below.
                let mut nulled: BTreeSet<(usize, u32)> = BTreeSet::new();
                for (piece, ranges) in &flush.clears {
                    for &(start, count) in ranges {
                        for t in start..start + count {
                            assert!(
                                shown.remove(&(*piece, t)).is_some(),
                                "{never:?}: a null of a tile showing nothing"
                            );
                            nulled.insert((*piece, t));
                        }
                    }
                }
                let mut tiles = 0;
                for update in &flush.updates {
                    let mut end = 0;
                    for r in &update.ranges {
                        assert!(
                            r.start >= end && r.count > 0,
                            "{never:?}: in order, apart: {update:?}"
                        );
                        assert!(!r.one || r.count > 1);
                        end = r.start + r.count;
                        for i in 0..r.count {
                            let to = HeapTile {
                                heap: update.heap,
                                tile: if r.one {
                                    r.heap_start
                                } else {
                                    r.heap_start + i
                                },
                            };
                            nulled.remove(&(update.piece, r.start + i));
                            let before = shown.insert((update.piece, r.start + i), to);
                            assert_ne!(
                                before,
                                Some(to),
                                "{never:?}: a tile mapped to what it shows"
                            );
                            tiles += 1;
                        }
                    }
                }
                assert_eq!(
                    tiles,
                    flush.fresh + flush.remapped + flush.sunk,
                    "{never:?}"
                );
                assert!(nulled.is_empty(), "{never:?}: tiles left null: {nulled:?}");
                for (&page, &want) in &asked {
                    let (piece, offset) = plan.locate(page, page + 1).expect("in a piece");
                    let now = shown.get(&(piece, offset / PAGE as u32)).copied();
                    match want {
                        Some(tile) => assert_eq!(now, Some(tile), "{never:?}: page {page:#x}"),
                        // Never asked to show a page: left alone.
                        None if never == Never::Untouched && !ever.contains(&page) => {
                            assert_eq!(now, None, "never mapped, left alone")
                        }
                        // Shown a page and unmapped (or mapped and unmapped
                        // between two flushes, left alone).
                        None if never == Never::Untouched => {
                            assert!(matches!(now, None | Some(SINK)), "page {page:#x}: {now:?}")
                        }
                        None => assert_eq!(now, Some(SINK), "{never:?}: page {page:#x} unmapped"),
                    }
                }
            }
        }
    }
}
