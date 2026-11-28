//! mac: stage 2's GPU side (`VICTORY_ALIAS_PHYSICAL=gpu`,
//! `notes/alias-physical-design.md` §2, increment I3b).
//!
//! Each placeholder reservation of the title's (`victory_hle`'s `xmem`) gets
//! one reserved buffer per piece of at most 4 GiB less one tile
//! (`crate::tile_plan`), and each heap-backed section (`crate::alias::
//! PhysicalHeap`) a number and a sink tile past its pages. As `xmem` maps and
//! unmaps the title's physical pages, after its CPU remap, the plan records
//! which heap tile each tile of a piece is to show. The submitting thread
//! drains it on the host queue just before the title's lists run
//! (`Host::run`): per piece one null `UpdateTileMappings` of the tiles that
//! change from a mapping, then one call per heap. The sink is zeroed before
//! each submission. A page from a section without a heap makes the pieces it
//! lands in fall back to copies.
//!
//! A start check before use: one tile of a reserved buffer over a small
//! heap, written on the CPU and copied out on the GPU, and the other way. If
//! it fails, the GPU side stays off (the heap backing stays).
//!
//! Views of the pieces are for `crate::shadow` (increment I4): [`view`].
//!
//! Canonical views (`=gpu`, I4b; `=gpu-tiles` leaves them out): each
//! heap-backed section also gets placed buffers over its heap, one per heap
//! piece (consecutive spans of at most 4 GiB less one tile), and a range that
//! is one unbroken run of one section's pages inside one heap piece is a view
//! of that buffer, so every title address of a page reaches one GPU range.
//! Views of pieces are left for scattered ranges. While views live, a page
//! is reached through one kind of range: each view holds its heap tiles
//! ([`Hold`]), and a range over tiles held by the other kind keeps a copy
//! ("mixed"). Section views do not follow remaps: a map or unmap of a page
//! under one drops it from `crate::shadow`'s registries. Piece views follow
//! (the flush remaps their tiles), but leave the registries when a remap
//! brings them a heap tile that section views hold.

use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{LazyLock, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use crate::tile_plan::{HeapTile, Never, PAGE, Plan};

/// A piece's reserved buffer and its GPU address.
struct Piece {
    buffer: ID3D12Resource,
    gpu: u64,
}

/// A section's heap: the heap, its pages and its sink tile's CPU address;
/// with canonical views (`=gpu`) its section buffers, placed buffers over its
/// pages, one per heap piece of `per` tiles from its start, with their GPU
/// addresses.
struct Heap {
    heap: ID3D12Heap,
    tiles: u32,
    sink: usize,
    sections: Vec<(ID3D12Resource, u64)>,
    per: u32,
}

/// What a view of title memory reaches its pages through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A placed buffer over a section's heap (canonical, `=gpu`).
    Section,
    /// A reservation's piece (its reserved buffer, tiled from the heaps).
    Piece,
}

/// A live view's claim on the heap tiles it reaches (canonical views,
/// `=gpu`): counted while its `crate::shadow::Shadow` lives, given back when
/// it goes. A section view's are heap tiles `first..first + n` of heap
/// `owner`; a piece view's the tiles `first..first + n` of piece `owner`,
/// counted against whatever heap tiles they are to show.
pub(crate) struct Hold {
    id: u64,
    kind: Kind,
    owner: usize,
    first: u32,
    n: u32,
    tracked: bool,
}

impl Hold {
    fn untracked(kind: Kind) -> Hold {
        Hold {
            id: 0,
            kind,
            owner: 0,
            first: 0,
            n: 0,
            tracked: false,
        }
    }

    pub(crate) fn kind(&self) -> Kind {
        self.kind
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        if self.tracked {
            release(self);
        }
    }
}

struct State {
    plan: Plan,
    /// By piece: its buffer, unless it could not be made or was dropped.
    pieces: Vec<Option<Piece>>,
    /// Pieces that fell back to copies.
    fallen: BTreeSet<usize>,
    /// By heap number.
    heaps: Vec<Option<Heap>>,
    /// Buffers of dropped pieces, kept until the next flush: the
    /// submission before it may still read them.
    graveyard: Vec<ID3D12Resource>,
    /// Canonical views (`=gpu`): live views per heap tile, of sections and
    /// of pieces.
    holds: HashMap<HeapTile, [u32; 2]>,
    /// Live piece views per piece tile, and the heap tile they are counted
    /// against (what the tile is to show; `None`: nothing).
    piece_held: HashMap<(usize, u32), (u32, Option<HeapTile>)>,
    /// Section views still in `crate::shadow`'s registries, by hold: their
    /// title pages; and those pages, with how many such views each.
    section_live: HashMap<u64, Range<usize>>,
    section_pages: HashMap<usize, u32>,
    next_hold: u64,
}

// SAFETY: D3D12 objects are free-threaded, and the state is only used
// under its lock.
unsafe impl Send for State {}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        plan: Plan::new(Never::Untouched),
        pieces: Vec::new(),
        fallen: BTreeSet::new(),
        heaps: Vec::new(),
        graveyard: Vec::new(),
        holds: HashMap::new(),
        piece_held: HashMap::new(),
        section_live: HashMap::new(),
        section_pages: HashMap::new(),
        next_hold: 1,
    })
});

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The counts of the physical pages line (`crate::physical_runs::GpuCounts`),
/// by index: up to [`SINCE_START`], since the last line; from it, since the
/// start.
static COUNTS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
// Since the last line: tiles mapped fresh, nulled and mapped again, pointed
// at the sink; `UpdateTileMappings` calls; their microseconds; views of
// sections made and their bytes; views of pieces made; ranges that kept a
// copy as mixed; views of sections and of pieces dropped by remaps.
const FRESH: usize = 0;
const US: usize = 4;
const SECTION_VIEWS: usize = 5;
const SECTION_BYTES: usize = 6;
const PIECE_VIEWS: usize = 7;
const MIXED: usize = 8;
const SECTION_DROPS: usize = 9;
const PIECE_DROPS: usize = 10;
const SINCE_START: usize = 11;
// Since the start: pieces made and fallen back; views of sections, of
// pieces, of pieces with a page at two or more addresses (as flushed);
// copies kept for ranges in a piece (across its end, a fallen piece,
// `gpu-single`), and mixed.
const MADE: usize = 11;
const FALLEN: usize = 12;
const SECTION_TOTAL: usize = 13;
const PIECE_TOTAL: usize = 14;
const SHARED: usize = 15;
const KEPT: usize = 16;
const MIXED_TOTAL: usize = 17;

fn count(i: usize, n: u64) {
    COUNTS[i].fetch_add(n, Relaxed);
}

/// Whether views of sections come first (`=gpu`).
fn canonical() -> bool {
    crate::mac_switches::alias_physical().canonical()
}

/// The tiles of a heap piece: [`crate::tile_plan::PIECE`]'s, or (tests) a
/// smaller span, to make heaps of a few pages with more than one piece.
fn heap_piece_tiles() -> u32 {
    #[cfg(test)]
    {
        let tiles = tests::HEAP_PIECE_TILES.load(Relaxed);
        if tiles != 0 {
            return tiles;
        }
    }
    (crate::tile_plan::PIECE / PAGE) as u32
}

/// Whether the GPU side is on: `VICTORY_ALIAS_PHYSICAL=gpu` (or
/// `gpu-single`), the heap backing on (`crate::alias::physical_on`), and the
/// start check passed. Decided once, at the title's first reservation or
/// physical allocation.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let mode = crate::mac_switches::alias_physical();
        if !mode.gpu() {
            return false;
        }
        if !crate::alias::physical_on() {
            tracing::warn!(
                "{}: the GPU side stays off: the heap backing is off (its own line says why)",
                mode.name()
            );
            return false;
        }
        match start_check() {
            Ok(()) => {
                crate::physical_runs::gpu_on();
                tracing::info!(
                    "{}: the GPU side is on: one reserved buffer per piece of at most 4 GiB less a tile of each \
                     placeholder reservation, tiled from the sections' heaps as the title maps its pages, flushed \
                     before each submission (nulls first, unmapped tiles on a zeroed sink){}; the start check passed",
                    mode.name(),
                    if mode.canonical() {
                        "; a range that is one run of a section's pages is a view of a placed buffer over the \
                         section's heap (canonical views), other ranges in one piece views of the piece"
                    } else {
                        "; ranges in one piece are views of the piece"
                    }
                );
                true
            }
            Err(why) => {
                tracing::warn!(
                    "{}: the GPU side stays off: the start check failed: {why} (the heap backing stays)",
                    mode.name()
                );
                false
            }
        }
    })
}

/// A heap-backed section's heap of `tiles` pages, its sink tile's CPU
/// address `sink` (zeroed here): its number.
pub(crate) fn heap_made(heap: &ID3D12Heap, len: usize, sink: usize) -> u32 {
    // SAFETY: the sink tile is the heap's last, mapped by its placed buffer.
    unsafe { std::ptr::write_bytes(sink as *mut u8, 0, PAGE) };
    let per = heap_piece_tiles();
    let sections = if canonical() {
        section_buffers(heap, len, per as usize * PAGE)
    } else {
        Vec::new()
    };
    let mut st = state();
    st.heaps.push(Some(Heap {
        heap: heap.clone(),
        tiles: (len / PAGE) as u32,
        sink,
        sections,
        per,
    }));
    (st.heaps.len() - 1) as u32
}

/// A heap's section buffers (canonical views): one placed UAV buffer per span
/// of `per` bytes of its first `len` (its pages; not its sink), not mapped by
/// the CPU. None if one cannot be made: its pages then get views of pieces.
fn section_buffers(heap: &ID3D12Heap, len: usize, per: usize) -> Vec<(ID3D12Resource, u64)> {
    let Some(host) = crate::host::host() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for at in (0..len).step_by(per) {
        let size = (len - at).min(per);
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: a placed buffer over the heap's bytes `[at, at + size)`,
        // inside it and 64 KiB aligned, with an output slot of ours.
        let made = unsafe {
            host.device.CreatePlacedResource(
                heap,
                at as u64,
                &buffer_desc(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made.and_then(|()| buffer.ok_or_else(windows::core::Error::empty)) {
            Ok(buffer) => {
                crate::host::name(&buffer, || format!("title section {len:#x} at {at:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                out.push((buffer, gpu));
            }
            Err(error) => {
                tracing::warn!(
                    len,
                    at,
                    %error,
                    "VICTORY_ALIAS_PHYSICAL=gpu: no placed buffer over this section's heap: its pages get views of \
                     pieces only"
                );
                return Vec::new();
            }
        }
    }
    out
}

/// The heap numbered `id` is released (its section's views are gone).
pub(crate) fn heap_dropped(id: u32) {
    let mut st = state();
    if let Some(heap) = st.heaps.get_mut(id as usize) {
        *heap = None;
    }
}

/// The sink unmapped tiles show: the largest heap's.
fn sink(st: &State) -> Option<(HeapTile, usize)> {
    st.heaps
        .iter()
        .enumerate()
        .filter_map(|(id, h)| h.as_ref().map(|h| (id, h)))
        .max_by_key(|(_, h)| h.tiles)
        .map(|(id, h)| {
            (
                HeapTile {
                    heap: id as u32,
                    tile: h.tiles,
                },
                h.sink,
            )
        })
}

/// A placeholder reservation of 64 KiB pages at `[at, at + len)`: its pieces'
/// reserved buffers (address space only until mapped).
pub fn reserved(at: usize, len: usize) {
    if !on() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    // Views made over the memory before (the title released it through the
    // host) are stale.
    crate::shadow::drop_views(at as u64, at.saturating_add(len) as u64, None);
    let mut st = state();
    let made = st.plan.reserve(at, len);
    for id in made.dropped {
        if let Some(piece) = st.pieces.get_mut(id).and_then(Option::take) {
            st.graveyard.push(piece.buffer);
        }
    }
    for id in made.pieces {
        let len = st.plan.piece(id).map_or(0, |r| r.len());
        let piece = match reserved_buffer(&host.device, len) {
            Ok(buffer) => {
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                count(MADE, 1);
                Some(Piece { buffer, gpu })
            }
            Err(error) => {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    len,
                    %error,
                    "VICTORY_ALIAS_PHYSICAL=gpu: no reserved buffer for this piece of a reservation: it keeps copies"
                );
                st.fallen.insert(id);
                count(FALLEN, 1);
                None
            }
        };
        if st.pieces.len() <= id {
            st.pieces.resize_with(id + 1, || None);
        }
        st.pieces[id] = piece;
    }
}

/// `xmem` mapped `[at, at + len)` (whole pages) to the pages from `first` on
/// of the heap numbered `heap`, or of a section without one (`None`).
pub fn mapped(at: usize, len: usize, heap: Option<u32>, first: u32) {
    if !on() {
        return;
    }
    let mut st = state();
    let mut fell = Vec::new();
    let mut drops = Vec::new();
    match heap.filter(|&h| st.heaps.get(h as usize).is_some_and(Option::is_some)) {
        Some(heap) => {
            if canonical() {
                drops = changing(&mut st, at, len, |k| {
                    Some(HeapTile {
                        heap,
                        tile: first + k as u32,
                    })
                });
            }
            st.plan.map(at, len, HeapTile { heap, tile: first });
        }
        None => {
            for id in st.plan.pieces_of(at, len) {
                if st.fallen.insert(id) {
                    fell.extend(st.plan.piece(id));
                    count(FALLEN, 1);
                    tracing::warn!(
                        at = format_args!("{at:#x}"),
                        len,
                        piece = id,
                        "VICTORY_ALIAS_PHYSICAL=gpu: a page of a section without a heap is mapped in a piece of a \
                         reservation: that piece keeps copies"
                    );
                }
            }
        }
    }
    drop(st);
    // Its views leave the registries; new copies take their ranges.
    for range in fell {
        crate::shadow::drop_views(range.start as u64, range.end as u64, None);
    }
    dropped(drops);
}

/// `xmem` unmapped `[at, at + len)` (whole pages).
pub fn unmapped(at: usize, len: usize) {
    if !on() {
        return;
    }
    let mut st = state();
    let drops = if canonical() {
        changing(&mut st, at, len, |_| None)
    } else {
        Vec::new()
    };
    st.plan.unmap(at, len);
    drop(st);
    dropped(drops);
}

/// Before the plan takes a change of title pages `[at, at + len)` (page `k`
/// to show `to(k)`), with canonical views: the live views it affects. A
/// section view over a page that changes is to leave the registries. Piece
/// views follow: their counts move to the tile's new heap tile, and they are
/// to leave the registries when section views hold that one. The title
/// ranges to drop, by kind.
fn changing(
    st: &mut State,
    at: usize,
    len: usize,
    to: impl Fn(usize) -> Option<HeapTile>,
) -> Vec<(Range<usize>, Kind)> {
    let mut out: Vec<(Range<usize>, Kind)> = Vec::new();
    if st.section_pages.is_empty() && st.piece_held.is_empty() {
        return out;
    }
    let push = |out: &mut Vec<(Range<usize>, Kind)>, page: usize, kind: Kind| {
        if let Some((range, _)) = out.iter_mut().rev().find(|(_, k)| *k == kind)
            && range.end == page
        {
            range.end += PAGE;
            return;
        }
        out.push((page..page + PAGE, kind));
    };
    let State {
        plan,
        holds,
        piece_held,
        section_pages,
        ..
    } = st;
    for k in 0..len / PAGE {
        let page = at + k * PAGE;
        let Some((id, tile)) = plan.tile_of(page) else {
            continue;
        };
        let new = to(k);
        if plan.current(id, tile) == new {
            continue;
        }
        if section_pages.get(&(page / PAGE)).is_some_and(|&n| n > 0) {
            push(&mut out, page, Kind::Section);
        }
        if let Some(entry) = piece_held.get_mut(&(id, tile)) {
            let (n, old) = *entry;
            entry.1 = new;
            if let Some(old) = old {
                take(holds, old, 1, n);
            }
            if let Some(new) = new {
                let held = holds.entry(new).or_default();
                held[1] += n;
                if held[0] > 0 {
                    push(&mut out, page, Kind::Piece);
                }
            }
        }
    }
    out
}

/// The views [`changing`] found leave the registries (no lock of ours held).
fn dropped(drops: Vec<(Range<usize>, Kind)>) {
    for (range, kind) in drops {
        let n = crate::shadow::drop_views(range.start as u64, range.end as u64, Some(kind));
        count(
            if kind == Kind::Section {
                SECTION_DROPS
            } else {
                PIECE_DROPS
            },
            n as u64,
        );
    }
}

/// `n` fewer live views of kind `kind` (0 sections, 1 pieces) reach heap
/// tile `t`.
fn take(holds: &mut HashMap<HeapTile, [u32; 2]>, t: HeapTile, kind: usize, n: u32) {
    if let Some(held) = holds.get_mut(&t) {
        held[kind] = held[kind].saturating_sub(n);
        if *held == [0, 0] {
            holds.remove(&t);
        }
    }
}

/// A section view's title pages are no longer in the registries.
fn leave(st: &mut State, id: u64) {
    if let Some(range) = st.section_live.remove(&id) {
        for page in range {
            if let Some(n) = st.section_pages.get_mut(&page) {
                *n -= 1;
                if *n == 0 {
                    st.section_pages.remove(&page);
                }
            }
        }
    }
}

/// `crate::shadow` dropped the view holding `hold` from its registries; it
/// may live on in lists.
pub(crate) fn left(hold: &Hold) {
    if hold.tracked && hold.kind == Kind::Section {
        leave(&mut state(), hold.id);
    }
}

/// A view is gone: its tiles are given back.
fn release(hold: &Hold) {
    let mut st = state();
    match hold.kind {
        Kind::Section => {
            for k in 0..hold.n {
                let t = HeapTile {
                    heap: hold.owner as u32,
                    tile: hold.first + k,
                };
                take(&mut st.holds, t, 0, 1);
            }
            leave(&mut st, hold.id);
        }
        Kind::Piece => {
            let State {
                holds, piece_held, ..
            } = &mut *st;
            for tile in hold.first..hold.first + hold.n {
                let key = (hold.owner, tile);
                let Some(entry) = piece_held.get_mut(&key) else {
                    continue;
                };
                if let Some(t) = entry.1 {
                    take(holds, t, 1, 1);
                }
                entry.0 -= 1;
                if entry.0 == 0 {
                    piece_held.remove(&key);
                }
            }
        }
    }
}

/// Before a submission's lists go to `queue`: the tile changes since the
/// last, and the sink zeroed. On the submitting thread, with the GPU idle
/// (`Host::run` has finished the submission before).
pub(crate) fn flush(queue: &ID3D12CommandQueue) {
    if !on() {
        return;
    }
    let mut st = state();
    st.graveyard.clear();
    let Some((sink, sink_cpu)) = sink(&st) else {
        // No heap yet: nothing can be mapped, so nothing is to change.
        return;
    };
    // SAFETY: the sink tile of a live heap; the GPU is idle.
    unsafe { std::ptr::write_bytes(sink_cpu as *mut u8, 0, PAGE) };
    if st.plan.pending() == 0 {
        return;
    }
    let started = Instant::now();
    let flush = st.plan.flush(sink);
    let mut calls = 0;
    for (piece, ranges) in &flush.clears {
        let Some(Some(p)) = st.pieces.get(*piece) else {
            continue;
        };
        let at: Vec<_> = ranges
            .iter()
            .map(|&(start, _)| D3D12_TILED_RESOURCE_COORDINATE {
                X: start,
                ..Default::default()
            })
            .collect();
        let sizes: Vec<_> = ranges
            .iter()
            .map(|&(_, count)| D3D12_TILE_REGION_SIZE {
                NumTiles: count,
                ..Default::default()
            })
            .collect();
        let flags = vec![D3D12_TILE_RANGE_FLAG_NULL; ranges.len()];
        let counts: Vec<u32> = ranges.iter().map(|&(_, count)| count).collect();
        // SAFETY: tiles of a live reserved buffer, null ranges of the same counts.
        unsafe {
            queue.UpdateTileMappings(
                &p.buffer,
                ranges.len() as u32,
                Some(at.as_ptr()),
                Some(sizes.as_ptr()),
                None::<&ID3D12Heap>,
                ranges.len() as u32,
                Some(flags.as_ptr()),
                None,
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        calls += 1;
    }
    for update in &flush.updates {
        let (Some(Some(p)), Some(Some(heap))) = (
            st.pieces.get(update.piece),
            st.heaps.get(update.heap as usize),
        ) else {
            continue;
        };
        let at: Vec<_> = update
            .ranges
            .iter()
            .map(|r| D3D12_TILED_RESOURCE_COORDINATE {
                X: r.start,
                ..Default::default()
            })
            .collect();
        let sizes: Vec<_> = update
            .ranges
            .iter()
            .map(|r| D3D12_TILE_REGION_SIZE {
                NumTiles: r.count,
                ..Default::default()
            })
            .collect();
        let flags: Vec<_> = update
            .ranges
            .iter()
            .map(|r| {
                if r.one {
                    D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE
                } else {
                    D3D12_TILE_RANGE_FLAG_NONE
                }
            })
            .collect();
        let starts: Vec<u32> = update.ranges.iter().map(|r| r.heap_start).collect();
        let counts: Vec<u32> = update.ranges.iter().map(|r| r.count).collect();
        // SAFETY: tiles of a live reserved buffer onto tiles of a live heap
        // (its pages and its sink), as many on both sides.
        unsafe {
            queue.UpdateTileMappings(
                &p.buffer,
                update.ranges.len() as u32,
                Some(at.as_ptr()),
                Some(sizes.as_ptr()),
                &heap.heap,
                update.ranges.len() as u32,
                Some(flags.as_ptr()),
                Some(starts.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        calls += 1;
    }
    for (i, n) in [flush.fresh, flush.remapped, flush.sunk, calls]
        .into_iter()
        .enumerate()
    {
        count(FRESH + i, n as u64);
    }
    count(US, started.elapsed().as_micros() as u64);
}

/// A view of title memory (I4, I4b): the buffer it is in (a piece's
/// reserved buffer, or a section buffer), the GPU address of the range's
/// first byte, its offset in the buffer, the title range it covers, and its
/// claim on the tiles.
pub(crate) struct View {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) gpu: u64,
    pub(crate) base: u64,
    pub(crate) range: Range<usize>,
    pub(crate) hold: Hold,
}

/// The view title memory `[from, to)` (a title buffer) gets, if any
/// (`crate::shadow`). With `=gpu` first a view of its section, when the range
/// is one unbroken run of one section's pages inside one heap piece. Else a
/// view of the piece it lies in wholly, if that has a buffer and has not
/// fallen back, and (`gpu-single`) has no page shown at two or more
/// addresses as last flushed. `None` (a copy) also when the range's tiles
/// are held by live views of the other kind (mixed, `=gpu`).
pub(crate) fn view(from: usize, to: usize) -> Option<View> {
    if !on() || to <= from {
        return None;
    }
    let mut st = state();
    // A range that starts in no piece is not ours to count.
    st.plan.locate(from, from + 1)?;
    if canonical() {
        match section_view(&mut st, from, to) {
            Some(Ok(view)) => return Some(view),
            Some(Err(Mixed)) => return None,
            None => {}
        }
    }
    piece_view(&mut st, from, to)
}

/// The view a range outside title buffers around `[address, end)` gets
/// (`crate::shadow`'s windows), if any: with `=gpu`, for a run, a view of its
/// section over the longest run holding it inside the `block`-aligned window
/// around it and inside its heap piece; else a view of its piece over its own
/// pages. With `=gpu-tiles` and `=gpu-single`, a view of its piece over the
/// 256 MiB block holding it, cut to the piece.
pub(crate) fn window(address: usize, end: usize, block: usize) -> Option<View> {
    if !on() || end <= address {
        return None;
    }
    let mut st = state();
    let (id, _) = st.plan.locate(address, address + 1)?;
    if st.fallen.contains(&id) || st.pieces.get(id).is_none_or(Option::is_none) {
        return None;
    }
    let piece = st.plan.piece(id)?;
    let (from, to) = if canonical() {
        let lo = address / block * block;
        let hi = end.checked_next_multiple_of(block)?;
        if let Some(run) = section_run(&st, address, end, lo, hi) {
            match section_view(&mut st, run.start, run.end) {
                Some(Ok(view)) => return Some(view),
                Some(Err(Mixed)) => return None,
                None => {}
            }
        }
        (
            (address / PAGE * PAGE).max(piece.start),
            end.checked_next_multiple_of(PAGE)?.min(piece.end),
        )
    } else {
        const BLOCK: usize = 256 << 20;
        (
            (address & !(BLOCK - 1)).max(piece.start),
            end.checked_next_multiple_of(BLOCK)?.min(piece.end),
        )
    };
    if end > to {
        return None;
    }
    piece_view(&mut st, from, to)
}

/// A range's tiles are held by live views of the other kind: it keeps a copy.
struct Mixed;

/// The longest run holding `[address, end)` within `[lo, hi)` and its heap
/// piece, if `[address, end)` is a run whose section has section buffers.
fn section_run(st: &State, address: usize, end: usize, lo: usize, hi: usize) -> Option<Range<usize>> {
    let (first, _) = st.plan.run(address, end)?;
    let heap = st.heaps.get(first.heap as usize)?.as_ref()?;
    if heap.sections.is_empty() {
        return None;
    }
    let j = first.tile / heap.per;
    let tiles = j * heap.per..((j + 1) * heap.per).min(heap.tiles);
    Some(st.plan.run_around(address, end, lo, hi, tiles)?.0)
}

/// `[from, to)` as a view of its section: `None` if it is not one run of a
/// section with section buffers inside one heap piece (or a page of it is in
/// a piece that fell back), and also when piece views hold its tiles only
/// through the range's own piece tiles (a title buffer inside a bigger one
/// that is a view of the piece: a view of the piece then reaches the same GPU
/// range those views do, no new one); `Mixed` if piece views hold one of its
/// tiles through other title addresses.
fn section_view(st: &mut State, from: usize, to: usize) -> Option<Result<View, Mixed>> {
    let (first, n) = st.plan.run(from, to)?;
    let heap = st.heaps.get(first.heap as usize)?.as_ref()?;
    let j = crate::tile_plan::heap_piece(first.tile, n, heap.per)?;
    if first.tile + n > heap.tiles {
        return None;
    }
    let (buffer, gpu) = heap.sections.get(j as usize)?.clone();
    let base = u64::from(first.tile - j * heap.per) * PAGE as u64 + (from % PAGE) as u64;
    let pages = from / PAGE * PAGE..to.next_multiple_of(PAGE);
    if st
        .plan
        .pieces_of(pages.start, pages.len())
        .iter()
        .any(|id| st.fallen.contains(id))
    {
        return None;
    }
    let tiles = (0..n).map(|k| HeapTile {
        heap: first.heap,
        tile: first.tile + k,
    });
    let mut held = false;
    for (k, t) in tiles.clone().enumerate() {
        let Some(pieces) = st.holds.get(&t).map(|h| h[1]).filter(|&n| n > 0) else {
            continue;
        };
        held = true;
        let own = st
            .plan
            .tile_of(pages.start + k * PAGE)
            .and_then(|key| st.piece_held.get(&key))
            .map_or(0, |entry| entry.0);
        if own != pieces {
            count(MIXED, 1);
            count(MIXED_TOTAL, 1);
            return Some(Err(Mixed));
        }
    }
    if held {
        return None;
    }
    for t in tiles {
        st.holds.entry(t).or_default()[0] += 1;
    }
    let id = st.next_hold;
    st.next_hold += 1;
    for page in pages.start / PAGE..pages.end / PAGE {
        *st.section_pages.entry(page).or_default() += 1;
    }
    st.section_live.insert(id, pages.start / PAGE..pages.end / PAGE);
    count(SECTION_VIEWS, 1);
    count(SECTION_TOTAL, 1);
    count(SECTION_BYTES, (to - from) as u64);
    Some(Ok(View {
        buffer,
        gpu: gpu + base,
        base,
        range: from..to,
        hold: Hold {
            id,
            kind: Kind::Section,
            owner: first.heap as usize,
            first: first.tile,
            n,
            tracked: true,
        },
    }))
}

/// `[from, to)` as a view of the piece it lies in wholly (I4's views), with
/// its tiles held when views of sections are on.
fn piece_view(st: &mut State, from: usize, to: usize) -> Option<View> {
    let kept = || {
        count(KEPT, 1);
        None
    };
    let Some((id, offset)) = st.plan.locate(from, to) else {
        return kept();
    };
    if st.fallen.contains(&id) {
        return kept();
    }
    let Some((buffer, gpu)) = st
        .pieces
        .get(id)
        .and_then(Option::as_ref)
        .map(|p| (p.buffer.clone(), p.gpu))
    else {
        return kept();
    };
    let (first, last) = (
        offset / PAGE as u32,
        (offset as usize + (to - from)).div_ceil(PAGE) as u32,
    );
    let shared = st.plan.shared(id, first, last - first);
    if shared
        && crate::mac_switches::alias_physical() == crate::mac_switches::AliasPhysical::GpuSingle
    {
        return kept();
    }
    let hold = if canonical() {
        let held = |t: u32| {
            st.plan
                .current(id, t)
                .is_some_and(|h| st.holds.get(&h).is_some_and(|h| h[0] > 0))
        };
        if (first..last).any(held) {
            count(MIXED, 1);
            count(MIXED_TOTAL, 1);
            return None;
        }
        for tile in first..last {
            let now = st.plan.current(id, tile);
            let entry = st.piece_held.entry((id, tile)).or_insert((0, now));
            entry.0 += 1;
            if let Some(t) = entry.1 {
                st.holds.entry(t).or_default()[1] += 1;
            }
        }
        let hold = st.next_hold;
        st.next_hold += 1;
        Hold {
            id: hold,
            kind: Kind::Piece,
            owner: id,
            first,
            n: last - first,
            tracked: true,
        }
    } else {
        Hold::untracked(Kind::Piece)
    };
    count(PIECE_VIEWS, 1);
    count(PIECE_TOTAL, 1);
    if shared {
        count(SHARED, 1);
    }
    Some(View {
        buffer,
        gpu: gpu + u64::from(offset),
        base: u64::from(offset),
        range: from..to,
        hold,
    })
}

/// The piece title address `at` is in, if it has a buffer and has not
/// fallen back: its title range (`crate::memory::pointer_pages`'s budget).
pub(crate) fn piece_at(at: usize) -> Option<std::ops::Range<usize>> {
    if !on() {
        return None;
    }
    let st = state();
    let (id, _) = st.plan.locate(at, at + 1)?;
    if st.fallen.contains(&id) || st.pieces.get(id).is_none_or(Option::is_none) {
        return None;
    }
    st.plan.piece(id)
}

/// For the physical pages line: the counts since its last, and since the
/// start.
pub(crate) fn take_counts() -> crate::physical_runs::GpuCounts {
    let mut n = [0u64; 18];
    for (i, c) in COUNTS.iter().enumerate() {
        n[i] = if i < SINCE_START {
            c.swap(0, Relaxed)
        } else {
            c.load(Relaxed)
        };
    }
    crate::physical_runs::GpuCounts {
        fresh: n[FRESH],
        remapped: n[FRESH + 1],
        sunk: n[FRESH + 2],
        calls: n[FRESH + 3],
        us: n[US],
        section_views: n[SECTION_VIEWS],
        section_bytes: n[SECTION_BYTES],
        piece_views: n[PIECE_VIEWS],
        mixed: n[MIXED],
        section_drops: n[SECTION_DROPS],
        piece_drops: n[PIECE_DROPS],
        pieces: n[MADE],
        fallen: n[FALLEN],
        section_total: n[SECTION_TOTAL],
        piece_total: n[PIECE_TOTAL],
        shared: n[SHARED],
        kept: n[KEPT],
        mixed_total: n[MIXED_TOTAL],
    }
}

/// Live views holding heap tile `(heap, tile)`: of sections, of pieces.
#[cfg(test)]
fn holds_at(heap: u32, tile: u32) -> [u32; 2] {
    state()
        .holds
        .get(&HeapTile { heap, tile })
        .copied()
        .unwrap_or_default()
}

/// The heap tile `at`'s page shows through its piece, as last flushed:
/// (heap number, tile); `None` for the sink or a tile never mapped. Tests.
pub fn tile_at(at: usize) -> Option<(u32, u32)> {
    let st = state();
    let (id, offset) = st.plan.locate(at, at + 1)?;
    st.plan
        .shows(id, offset / PAGE as u32)
        .map(|t| (t.heap, t.tile))
}

/// Flushes on a queue of the host device's own (tests of `victory_hle`'s
/// hooks, which have no submission).
pub fn flush_now() -> bool {
    let Some(host) = crate::host::host() else {
        return false;
    };
    let Ok(rig) = Rig::new(&host.device) else {
        return false;
    };
    flush(&rig.queue);
    rig.wait().is_ok()
}

fn buffer_desc(size: usize, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A reserved UAV buffer of `len` bytes, in the common state.
fn reserved_buffer(device: &ID3D12Device, len: usize) -> windows::core::Result<ID3D12Resource> {
    let mut out = None;
    // SAFETY: plain creation with an output slot of ours.
    unsafe {
        device.CreateReservedResource(
            &buffer_desc(len, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut out,
        )?;
    }
    out.ok_or_else(windows::core::Error::empty)
}

/// A queue, list and fence of our own, for the start check and tests.
struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: std::cell::Cell<u64>,
    event: windows::Win32::Foundation::HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            Ok(Rig {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                done: std::cell::Cell::new(0),
                event: CreateEventW(None, false, false, None)?,
            })
        }
    }

    /// Waits for everything sent to the queue (10 s at most).
    fn wait(&self) -> windows::core::Result<()> {
        self.done.set(self.done.get() + 1);
        let done = self.done.get();
        // SAFETY: our fence, queue and event.
        unsafe {
            self.queue.Signal(&self.fence, done)?;
            if self.fence.GetCompletedValue() < done {
                self.fence.SetEventOnCompletion(done, self.event)?;
                if WaitForSingleObject(self.event, 10_000).0 != 0 {
                    return Err(windows::core::Error::empty());
                }
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    /// Records with `record` and runs it (after `before`, which may queue
    /// tile updates), then waits.
    fn run(
        &self,
        before: impl FnOnce(&ID3D12CommandQueue),
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
        }
        record(&self.list);
        before(&self.queue);
        // SAFETY: as above.
        unsafe {
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
        }
        self.wait()
    }

    fn committed(
        &self,
        kind: D3D12_HEAP_TYPE,
        size: usize,
        state: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let mut out = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE),
                state,
                None,
                &mut out,
            )?;
        }
        out.ok_or_else(windows::core::Error::empty)
    }
}

/// The CPU mapping of a committed upload or readback buffer.
fn mapped_bytes(buffer: &ID3D12Resource) -> windows::core::Result<*mut u8> {
    let mut data = std::ptr::null_mut();
    // SAFETY: a CPU-visible buffer of ours.
    unsafe { buffer.Map(0, None, Some(&mut data))? };
    Ok(data.cast())
}

/// One tile of a reserved buffer over a heap: the CPU's words read by a GPU
/// copy, a GPU copy's words read by the CPU, and a tile never mapped reads
/// zero (as the design takes it, `tile_plan::Never::Untouched`).
fn start_check() -> Result<(), String> {
    const WORDS: usize = 256;
    let host = crate::host::host().ok_or("no host device")?;
    let device = &host.device;
    let heap = crate::alias::PhysicalHeap::new(device, 2 * PAGE)
        .map_err(|e| format!("CreateHeap: {e}"))?;
    let buffer =
        reserved_buffer(device, 2 * PAGE).map_err(|e| format!("CreateReservedResource: {e}"))?;
    let rig = Rig::new(device).map_err(|e| format!("a queue: {e}"))?;
    let back = rig
        .committed(
            D3D12_HEAP_TYPE_READBACK,
            2 * WORDS * 4,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
        .map_err(|e| format!("a readback buffer: {e}"))?;
    let up = rig
        .committed(
            D3D12_HEAP_TYPE_UPLOAD,
            WORDS * 4,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .map_err(|e| format!("an upload buffer: {e}"))?;
    let word = |i: usize, seed: u32| (i as u32).wrapping_mul(0x9e37_79b1) ^ seed;
    let cpu = heap.cpu_address() as *mut u32;
    // SAFETY: the heap's first tile, mapped by its placed buffer; the upload
    // buffer's mapping of `WORDS` words.
    unsafe {
        for i in 0..WORDS {
            cpu.add(i).write_volatile(word(i, 0x5a5a_0001));
        }
        let up_words = mapped_bytes(&up)
            .map_err(|e| format!("Map: {e}"))?
            .cast::<u32>();
        for i in 0..WORDS {
            up_words.add(i).write(word(i, 0xc3c3_0002));
        }
    }
    // Buffer tile 1 onto heap tile 0; tile 0 is left as made.
    let map = |queue: &ID3D12CommandQueue| {
        let at = D3D12_TILED_RESOURCE_COORDINATE {
            X: 1,
            ..Default::default()
        };
        let size = D3D12_TILE_REGION_SIZE {
            NumTiles: 1,
            ..Default::default()
        };
        // SAFETY: one tile of a live reserved buffer onto one of a live heap.
        unsafe {
            queue.UpdateTileMappings(
                &buffer,
                1,
                Some(&at),
                Some(&size),
                &heap._heap,
                1,
                Some(&D3D12_TILE_RANGE_FLAG_NONE),
                Some(&0),
                Some(&1),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    };
    let bytes = (WORDS * 4) as u64;
    rig.run(map, |list| {
        // SAFETY: live buffers, ranges inside them.
        unsafe {
            list.CopyBufferRegion(&back, 0, &buffer, PAGE as u64, bytes);
            list.CopyBufferRegion(&back, bytes, &buffer, 0, bytes);
            list.CopyBufferRegion(&buffer, (PAGE + 1024) as u64, &up, 0, bytes);
        }
    })
    .map_err(|e| format!("the copies: {e}"))?;
    // SAFETY: the readback buffer's mapping of `2 * WORDS` words; the heap's
    // first tile.
    let (read, never, written) = unsafe {
        let back = std::slice::from_raw_parts(
            mapped_bytes(&back)
                .map_err(|e| format!("Map: {e}"))?
                .cast::<u32>(),
            2 * WORDS,
        );
        let written = (0..WORDS).all(|i| cpu.add(256 + i).read_volatile() == word(i, 0xc3c3_0002));
        (
            (0..WORDS).all(|i| back[i] == word(i, 0x5a5a_0001)),
            back[WORDS..].iter().all(|&w| w == 0),
            written,
        )
    };
    match (read, written) {
        (true, true) => {
            if !never {
                tracing::warn!(
                    "VICTORY_ALIAS_PHYSICAL=gpu: a tile never mapped does not read zero here (the design takes it \
                     to, as probe 7 found)"
                );
            }
            Ok(())
        }
        (false, _) => Err("a GPU copy out of the tile did not read the CPU's words".into()),
        (true, false) => Err("the CPU did not read a GPU copy's words in the heap".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::s;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    use crate::compute::Param;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    /// The tiles of a heap piece for heaps made while it is set (0: the
    /// default), to make heaps of a few pages with more than one piece.
    pub(super) static HEAP_PIECE_TILES: AtomicU32 = AtomicU32::new(0);

    /// Canonical views are on (`VICTORY_ALIAS_PHYSICAL=gpu` under Wine).
    fn canonical_on() -> bool {
        if on() && canonical() {
            return true;
        }
        eprintln!("skipped: VICTORY_ALIAS_PHYSICAL=gpu (canonical views) is not on here");
        false
    }

    const SHADER: &str = r"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Params : register(b0) { uint words; uint add; uint pad0; uint pad1; };
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= words) return;
    dst.Store(i * 4, src.Load(i * 4) + add);
}
";
    const WORDS: usize = PAGE / 4;

    fn word(i: usize, seed: u32) -> u32 {
        (i as u32).wrapping_mul(0x9e37_79b1) ^ seed
    }

    /// Title memory of `pages` pages, its own reservation in the plan.
    struct Title {
        at: usize,
        pages: usize,
    }

    impl Title {
        fn new(pages: usize) -> Title {
            // SAFETY: a fresh allocation for the test.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    pages * PAGE,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(at, 0);
            reserved(at, pages * PAGE);
            Title { at, pages }
        }

        fn page(&self, k: usize) -> usize {
            assert!(k < self.pages);
            self.at + k * PAGE
        }

        /// Page `k` shows heap page `p`, on the CPU (as `xmem` remaps its
        /// views) and in the plan.
        fn show(&self, heap: &crate::alias::PhysicalHeap, k: usize, p: usize) {
            // SAFETY: our own title memory.
            assert!(unsafe { heap.alias_view(self.page(k), p * PAGE, PAGE) });
            mapped(self.page(k), PAGE, heap.id(), p as u32);
        }

        /// The CPU's words at page `k`.
        fn words(&self, k: usize) -> &[u32] {
            // SAFETY: a mapped page of ours.
            unsafe { std::slice::from_raw_parts(self.page(k) as *const u32, WORDS) }
        }

        fn fill(&self, k: usize, seed: u32) {
            // SAFETY: a mapped page of ours.
            let page = unsafe { std::slice::from_raw_parts_mut(self.page(k) as *mut u32, WORDS) };
            for (i, w) in page.iter_mut().enumerate() {
                *w = word(i, seed);
            }
        }
    }

    impl Drop for Title {
        fn drop(&mut self) {
            // SAFETY: our own allocation.
            unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// The GPU address of title page `at` through its piece (with `=gpu` too,
    /// where a lookup would give a single page a view of its section).
    fn gpu_at(at: usize) -> u64 {
        let st = state();
        let (id, offset) = st.plan.locate(at, at + PAGE).expect("in a piece");
        st.pieces[id].as_ref().expect("a piece's buffer").gpu + u64::from(offset)
    }

    /// Scattered title pages through the pieces: a dispatch reads a page
    /// and writes another, the CPU reads the written page at both of its
    /// addresses (one RAM); a remap is seen at the next submission; an
    /// unmapped page reads zero; a page written through one piece reads new
    /// through another in the next submission after a polled fence.
    /// `VICTORY_ALIAS_PHYSICAL=gpu` (with `VICTORY_ALIAS_BUFFERS=1` and
    /// `VICTORY_COPY_ORDER=split`) under Wine on Rosetta; else skipped.
    #[test]
    fn title_pages_reach_a_dispatch_through_the_pieces_and_back() {
        if !on() {
            eprintln!(
                "skipped: VICTORY_ALIAS_PHYSICAL=gpu is not on here (set it, with its prerequisites, under Wine)"
            );
            return;
        }
        let host = crate::host::host().expect("a host device");
        let heap = crate::alias::physical_heap(8 * PAGE).expect("a heap");
        assert!(heap.id().is_some(), "the heap is one of the pieces'");
        let (a, b) = (Title::new(8), Title::new(4));
        // Scattered: a's pages 0, 1 and 2 show heap pages 3, 5 and 1; heap
        // page 1 also at a's page 6 and at b's page 0 (another piece).
        a.show(&heap, 0, 3);
        a.show(&heap, 1, 5);
        a.show(&heap, 2, 1);
        a.show(&heap, 6, 1);
        b.show(&heap, 0, 1);
        b.show(&heap, 2, 0);
        a.fill(0, 0x1111);
        a.fill(1, 0x2222);
        let pipeline = crate::compute::pipeline(
            SHADER,
            s!("alias_physical test"),
            0,
            &[
                Param::Srv {
                    register: 0,
                    space: 0,
                },
                Param::Uav { register: 0 },
                Param::Constants {
                    register: 0,
                    count: 4,
                },
            ],
        )
        .expect("a pipeline");
        let rig = Rig::new(&host.device).expect("a queue");
        // A dispatch adding `add` to each word of page `src` into page `dst`,
        // tile changes flushed first; waited for on the fence's event, or
        // (`polled`) by polling the fence alone, as `Host::run` may.
        let dispatch = |src: usize, dst: usize, add: u32, polled: bool| {
            let constants = [WORDS as u32, add, 0, 0];
            let record = |list: &ID3D12GraphicsCommandList| {
                // SAFETY: live pipeline, views of live pieces, constants on this stack.
                unsafe {
                    list.SetComputeRootSignature(&pipeline.0);
                    list.SetPipelineState(&pipeline.1);
                    list.SetComputeRootShaderResourceView(0, gpu_at(src));
                    list.SetComputeRootUnorderedAccessView(1, gpu_at(dst));
                    list.SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
                    list.Dispatch((WORDS / 64) as u32, 1, 1);
                }
            };
            if !polled {
                return rig.run(flush, record).expect("the dispatch ran");
            }
            // SAFETY: the rig's own list, queue and fence, one run at a time.
            unsafe {
                rig.allocator.Reset().unwrap();
                rig.list.Reset(&rig.allocator, None).unwrap();
                record(&rig.list);
                rig.list.Close().unwrap();
                flush(&rig.queue);
                rig.queue
                    .ExecuteCommandLists(&[Some(rig.list.cast().unwrap())]);
                rig.done.set(rig.done.get() + 1);
                rig.queue.Signal(&rig.fence, rig.done.get()).unwrap();
                let started = Instant::now();
                while rig.fence.GetCompletedValue() < rig.done.get() {
                    assert!(started.elapsed().as_secs() < 10, "the dispatch ran");
                    std::hint::spin_loop();
                }
            }
        };
        // Page 0 (heap 3) + 1 into page 2 (heap 1): seen at both of heap
        // page 1's addresses in a, and at b's.
        dispatch(a.page(0), a.page(2), 1, false);
        let want: Vec<u32> = (0..WORDS).map(|i| word(i, 0x1111) + 1).collect();
        assert_eq!(
            a.words(2),
            &want[..],
            "the dispatch's words at the page it wrote"
        );
        assert_eq!(a.words(6), &want[..], "and at the page's second address");
        assert_eq!(
            b.words(0),
            &want[..],
            "and at its third, in another reservation"
        );
        assert_eq!(tile_at(a.page(2)), heap.id().map(|h| (h, 1)));
        // A remap: page 0 now shows heap page 7; the next submission reads it.
        a.show(&heap, 0, 7);
        a.fill(0, 0x3333);
        dispatch(a.page(0), a.page(2), 2, false);
        let want: Vec<u32> = (0..WORDS).map(|i| word(i, 0x3333) + 2).collect();
        assert_eq!(
            a.words(2),
            &want[..],
            "a remap is seen at the next submission"
        );
        // Unmapped: page 0 reads zero (the sink).
        unmapped(a.page(0), PAGE);
        dispatch(a.page(0), a.page(2), 3, false);
        assert!(
            a.words(2).iter().all(|&w| w == 3),
            "an unmapped page reads zero"
        );
        assert_eq!(tile_at(a.page(0)), None);
        // Heap page 1 read through b (range B, which caches it), written
        // through a (range A), read through B again, each submission waited
        // for by polling its fence alone: fresh (probe 8: a CPU wait drops
        // B's lines; is a polled fence one?).
        b.show(&heap, 3, 6);
        dispatch(b.page(0), b.page(3), 0, true);
        dispatch(a.page(1), a.page(2), 4, true);
        dispatch(b.page(0), b.page(2), 5, true);
        let want: Vec<u32> = (0..WORDS).map(|i| word(i, 0x2222) + 4 + 5).collect();
        let stale = b.words(2).iter().zip(&want).filter(|(g, w)| g != w).count();
        assert_eq!(
            stale, 0,
            "words of b's page 2 not from a's write read through b, after polled fences"
        );
        // SAFETY: querying device health.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Reads `bytes` of `buffer` from `offset` with a GPU copy.
    fn read_back(rig: &Rig, buffer: &ID3D12Resource, offset: u64, bytes: usize) -> Vec<u32> {
        let back = rig
            .committed(
                D3D12_HEAP_TYPE_READBACK,
                bytes,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .expect("a readback buffer");
        // SAFETY: live buffers, ranges inside them.
        rig.run(flush, |list| unsafe {
            list.CopyBufferRegion(&back, 0, buffer, offset, bytes as u64)
        })
        .expect("the copy ran");
        // SAFETY: the readback buffer's mapping of `bytes`.
        unsafe {
            std::slice::from_raw_parts(mapped_bytes(&back).unwrap().cast::<u32>(), bytes / 4)
                .to_vec()
        }
    }

    /// A title buffer (`crate::shadow::register`) in a piece of a
    /// reservation, and a range outside buffers there, are views of the
    /// piece, not copies: aliased, clean, at their offset in the piece's
    /// buffer, and the GPU reads the title's bytes through them, by a copy
    /// and by a descriptor-table view that starts at the view's base (as
    /// root views and title memory's page-table pairs now do).
    #[test]
    fn a_range_in_a_reservation_is_a_view() {
        if !on() {
            eprintln!("skipped: VICTORY_ALIAS_PHYSICAL=gpu is not on here");
            return;
        }
        let host = crate::host::host().expect("a host device");
        let heap = crate::alias::physical_heap(4 * PAGE).expect("a heap");
        let t = Title::new(4);
        t.show(&heap, 1, 2);
        t.show(&heap, 2, 0);
        t.fill(1, 0x4444);
        t.fill(2, 0x5555);
        let (start, len) = (t.page(1) as u64, 2 * PAGE as u64);
        crate::shadow::register(0xa1_0001, start, len);
        let s = crate::shadow::find(start + 64, 128).expect("a view");
        assert!(s.is_view() && s.is_aliased(), "a view, aliased");
        assert!(s.clean(), "nothing to refresh");
        assert_eq!(s.update(start as usize, len as usize), 0, "nothing copied");
        assert_eq!(s.range(), (start, len));
        assert_eq!(s.base(), PAGE as u64, "page 1 of the piece");
        assert_eq!(
            s.offset(start + 64),
            PAGE as u64 + 64,
            "from the buffer's start"
        );
        assert_eq!(s.gpu(start + 64), gpu_at(t.page(1)) + 64);
        let rig = Rig::new(&host.device).expect("a queue");
        let words = read_back(&rig, &s.buffer, s.offset(start), 2 * PAGE);
        assert!(
            words[..WORDS]
                .iter()
                .enumerate()
                .all(|(i, &w)| w == word(i, 0x4444)),
            "page 1 through the view"
        );
        assert!(
            words[WORDS..]
                .iter()
                .enumerate()
                .all(|(i, &w)| w == word(i, 0x5555)),
            "page 2 through the view"
        );
        // A table view from the view's base, read at `address - start`.
        t.show(&heap, 3, 3);
        // SAFETY: plain creation of a heap of ours.
        let views: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        }
        .expect("a view heap");
        let out = view(t.page(3), t.page(3) + PAGE).expect("page 3 in the piece");
        // SAFETY: slots of our heap, views of live buffers.
        unsafe {
            let cpu = views.GetCPUDescriptorHandleForHeapStart();
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: s.base() / 4,
                        NumElements: (len / 4) as u32,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            host.device
                .CreateShaderResourceView(&s.buffer, Some(&srv), cpu);
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: out.base / 4,
                        NumElements: WORDS as u32,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            host.device.CreateUnorderedAccessView(
                &out.buffer,
                None,
                Some(&uav),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + step,
                },
            );
        }
        // The shader reads word `i` of the view's second page: `address - start` is a page.
        let constants = [WORDS as u32, 7, 0, 0];
        let shifted = "ByteAddressBuffer src : register(t0);\nRWByteAddressBuffer dst : register(u0);\n\
            cbuffer Params : register(b0) { uint words; uint add; uint pad0; uint pad1; };\n\
            [numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
            uint i = id.x; if (i >= words) return; dst.Store(i * 4, src.Load(65536 + i * 4) + add); }\n";
        let table = crate::compute::pipeline(
            shifted,
            s!("alias_physical test (table, page 2)"),
            0,
            &[
                Param::Table(&[crate::compute::srv(0, 1), crate::compute::uav(0, 1)]),
                Param::Constants {
                    register: 0,
                    count: 4,
                },
            ],
        )
        .expect("a pipeline");
        rig.run(flush, |list| {
            // SAFETY: live pipeline and heap, constants on this stack.
            unsafe {
                list.SetDescriptorHeaps(&[Some(views.clone())]);
                list.SetComputeRootSignature(&table.0);
                list.SetPipelineState(&table.1);
                list.SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
                list.SetComputeRoot32BitConstants(1, 4, constants.as_ptr().cast(), 0);
                list.Dispatch((WORDS / 64) as u32, 1, 1);
            }
        })
        .expect("the dispatch ran");
        assert!(
            t.words(3)
                .iter()
                .enumerate()
                .all(|(i, &w)| w == word(i, 0x5555) + 7),
            "a table view from the view's base reads its second page at offset 64 KiB"
        );
        // A range outside buffers in the piece: a view of the piece (a
        // window cut to it).
        let window = crate::shadow::find(t.page(0) as u64 + 8, 16).expect("a window");
        assert!(window.is_view());
        let (from, n) = window.range();
        assert!(from <= t.page(0) as u64 && t.page(0) as u64 + 24 <= from + n);
        assert!(
            from >= t.at as u64 && from + n <= (t.at + 4 * PAGE) as u64,
            "cut to the piece"
        );
        crate::shadow::unregister(0xa1_0001, start);
    }

    /// A range across the line between two pieces keeps a copy; one inside
    /// the second piece is a view.
    #[test]
    fn a_straddling_range_keeps_a_copy() {
        if !on() {
            eprintln!("skipped: VICTORY_ALIAS_PHYSICAL=gpu is not on here");
            return;
        }
        use windows_sys::Win32::System::Memory::PAGE_NOACCESS;
        let len = crate::tile_plan::PIECE + 4 * PAGE;
        // SAFETY: address space for the test, two pages at the line committed.
        let at =
            unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(at, 0);
        reserved(at, len);
        let line = at + crate::tile_plan::PIECE;
        // SAFETY: inside the reservation above.
        assert!(
            !unsafe {
                VirtualAlloc(
                    (line - PAGE) as *const _,
                    2 * PAGE,
                    MEM_COMMIT,
                    PAGE_READWRITE,
                )
            }
            .is_null()
        );
        // Inside first: the copy window the range across would make holds it.
        let inside = crate::shadow::find(line as u64 + 64, 16).expect("a view");
        assert!(inside.is_view(), "inside the second piece: a view");
        assert!(inside.range().0 >= line as u64);
        let across = crate::shadow::find(line as u64 - 64, 128).expect("a copy");
        assert!(!across.is_view(), "across the pieces' line: a copy");
        // SAFETY: our reservation.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// A page of a section without a heap mapped into a piece makes it fall
    /// back: its views leave the registries, and the same title buffer gets
    /// a copy.
    #[test]
    fn a_fallen_back_reservation_gives_copies() {
        if !on() {
            eprintln!("skipped: VICTORY_ALIAS_PHYSICAL=gpu is not on here");
            return;
        }
        let heap = crate::alias::physical_heap(2 * PAGE).expect("a heap");
        let t = Title::new(2);
        t.show(&heap, 0, 1);
        t.fill(0, 0x6666);
        let start = t.page(0) as u64;
        crate::shadow::register(0xa1_0002, start, PAGE as u64);
        let before = crate::shadow::find(start, 64).expect("a view");
        assert!(before.is_view());
        mapped(t.page(1), PAGE, None, 0);
        assert!(
            view(t.page(0), t.page(0) + PAGE).is_none(),
            "the piece fell back"
        );
        let after = crate::shadow::find(start, 64).expect("a copy");
        assert!(!after.is_view() && !after.is_aliased(), "a copy now");
        crate::shadow::unregister(0xa1_0002, start);
    }

    /// I4b: a title buffer over one unbroken run of a section's pages is a
    /// view of the section's buffer, at the run's place in its heap piece; a
    /// page of it shown at another title address gets the same GPU range
    /// there; the GPU reads the title's bytes through it; its heap tiles are
    /// held while it lives.
    #[test]
    fn a_run_is_a_view_of_its_section() {
        if !canonical_on() {
            return;
        }
        let host = crate::host::host().expect("a host device");
        let heap = crate::alias::physical_heap(6 * PAGE).expect("a heap");
        let id = heap.id().expect("a heap of the pieces'");
        let (t, u) = (Title::new(4), Title::new(2));
        // t's pages 0 to 2 show heap pages 2, 3 and 4; u's page 1 shows heap
        // page 3 too.
        t.show(&heap, 0, 2);
        t.show(&heap, 1, 3);
        t.show(&heap, 2, 4);
        u.show(&heap, 1, 3);
        for k in 0..3 {
            t.fill(k, 0x7000 + k as u32);
        }
        let (start, len) = (t.page(0) as u64 + 0x100, 3 * PAGE as u64 - 0x200);
        crate::shadow::register(0xa1_0010, start, len);
        let s = crate::shadow::find(start + 64, 128).expect("a view");
        assert_eq!(s.view_kind(), Some(Kind::Section), "a view of the section");
        assert!(s.is_aliased() && s.clean(), "aliased, nothing to refresh");
        assert_eq!(s.range(), (start, len));
        assert_eq!(
            s.base(),
            2 * PAGE as u64 + 0x100,
            "heap page 2's place in its heap piece"
        );
        assert_eq!(holds_at(id, 3), [1, 0]);
        let rig = Rig::new(&host.device).expect("a queue");
        let words = read_back(&rig, &s.buffer, s.offset(t.page(1) as u64), PAGE);
        assert!(
            words.iter().enumerate().all(|(i, &w)| w == word(i, 0x7001)),
            "heap page 3 through the section's buffer"
        );
        // The page's other title address: the same GPU range.
        crate::shadow::register(0xa1_0011, u.page(1) as u64, PAGE as u64);
        let v = crate::shadow::find(u.page(1) as u64, 4).expect("a view");
        assert_eq!(v.view_kind(), Some(Kind::Section));
        assert_eq!(
            v.gpu(u.page(1) as u64),
            s.gpu(t.page(1) as u64),
            "one GPU range for the page at both of its addresses"
        );
        assert_eq!(holds_at(id, 3), [2, 0]);
        crate::shadow::unregister(0xa1_0010, start);
        crate::shadow::unregister(0xa1_0011, u.page(1) as u64);
        drop((s, v));
        assert_eq!(holds_at(id, 3), [0, 0], "given back with the views");
    }

    /// I4b: a run across the edge between two heap pieces is a view of its
    /// reservation's piece; a run inside the second heap piece is a view of
    /// that heap piece's buffer, from the heap piece's start.
    #[test]
    fn a_run_across_a_heap_pieces_edge_is_a_view_of_a_piece() {
        if !canonical_on() {
            return;
        }
        let host = crate::host::host().expect("a host device");
        HEAP_PIECE_TILES.store(2, Relaxed);
        let heap = crate::alias::physical_heap(4 * PAGE);
        HEAP_PIECE_TILES.store(0, Relaxed);
        let heap = heap.expect("a heap");
        let t = Title::new(4);
        // Heap pages 1 and 2 (across the line at 2), and 3 (the second piece).
        t.show(&heap, 0, 1);
        t.show(&heap, 1, 2);
        t.show(&heap, 3, 3);
        t.fill(1, 0x7101);
        t.fill(3, 0x7103);
        crate::shadow::register(0xa1_0020, t.page(0) as u64, 2 * PAGE as u64);
        crate::shadow::register(0xa1_0021, t.page(3) as u64, PAGE as u64);
        let across = crate::shadow::find(t.page(0) as u64, 16).expect("a view");
        assert_eq!(across.view_kind(), Some(Kind::Piece), "across the heap pieces' edge");
        let second = crate::shadow::find(t.page(3) as u64, 16).expect("a view");
        assert_eq!(second.view_kind(), Some(Kind::Section), "inside the second heap piece");
        assert_eq!(second.base(), PAGE as u64, "heap page 3 is page 1 of heap piece 1");
        let rig = Rig::new(&host.device).expect("a queue");
        let words = read_back(&rig, &second.buffer, second.offset(t.page(3) as u64), PAGE);
        assert!(
            words.iter().enumerate().all(|(i, &w)| w == word(i, 0x7103)),
            "heap page 3 through heap piece 1's buffer"
        );
        let words = read_back(&rig, &across.buffer, across.offset(t.page(1) as u64), PAGE);
        assert!(
            words.iter().enumerate().all(|(i, &w)| w == word(i, 0x7101)),
            "heap page 2 through the reservation's piece"
        );
        crate::shadow::unregister(0xa1_0020, t.page(0) as u64);
        crate::shadow::unregister(0xa1_0021, t.page(3) as u64);
    }

    /// I4b: while views of one kind hold a heap tile, a range over it that
    /// would be a view of the other kind keeps a copy (mixed), both ways.
    #[test]
    fn a_mixed_range_is_a_copy() {
        if !canonical_on() {
            return;
        }
        let heap = crate::alias::physical_heap(4 * PAGE).expect("a heap");
        let id = heap.id().expect("a heap of the pieces'");
        let (t, u) = (Title::new(3), Title::new(2));
        // t's pages 0 and 1 show heap pages 3 and 1 (scattered); u's page 0
        // shows heap page 1 (a run of one).
        t.show(&heap, 0, 3);
        t.show(&heap, 1, 1);
        u.show(&heap, 0, 1);
        let (scattered, run) = (t.page(0) as u64, u.page(0) as u64);
        crate::shadow::register(0xa1_0030, scattered, 2 * PAGE as u64);
        let piece = crate::shadow::find(scattered, 16).expect("a view");
        assert_eq!(piece.view_kind(), Some(Kind::Piece));
        assert_eq!(holds_at(id, 1), [0, 1]);
        crate::shadow::register(0xa1_0031, run, PAGE as u64);
        let copy = crate::shadow::find(run, 16).expect("a copy");
        assert!(!copy.is_view() && !copy.is_aliased(), "piece views hold its tile: a copy");
        // A title buffer inside the scattered one, a run of one page: its
        // tile is held only through its own piece tile, so a view of the
        // piece, the same GPU range as the outer view's.
        let inner_at = t.page(1) as u64;
        crate::shadow::register(0xa1_0034, inner_at, PAGE as u64);
        let inner = crate::shadow::find(inner_at, 16).expect("a view");
        assert_eq!(inner.view_kind(), Some(Kind::Piece), "inside a piece view: a piece view");
        assert_eq!(inner.gpu(inner_at), piece.gpu(inner_at), "the same GPU range");
        assert_eq!(holds_at(id, 1), [0, 2]);
        crate::shadow::unregister(0xa1_0034, inner_at);
        crate::shadow::unregister(0xa1_0030, scattered);
        crate::shadow::unregister(0xa1_0031, run);
        drop((piece, copy, inner));
        assert_eq!(holds_at(id, 1), [0, 0]);
        // The other way: a view of the section first.
        crate::shadow::register(0xa1_0032, run, PAGE as u64);
        let section = crate::shadow::find(run, 16).expect("a view");
        assert_eq!(section.view_kind(), Some(Kind::Section));
        crate::shadow::register(0xa1_0033, scattered, 2 * PAGE as u64);
        let copy = crate::shadow::find(scattered, 16).expect("a copy");
        assert!(!copy.is_view(), "a section view holds a tile of it: a copy");
        assert_eq!(holds_at(id, 1), [1, 0]);
        crate::shadow::unregister(0xa1_0032, run);
        crate::shadow::unregister(0xa1_0033, scattered);
    }

    /// I4b: a remap of a page under a view of its section drops the view
    /// from the registry (it does not follow remaps); while it lives on, a
    /// new lookup there keeps a copy (mixed). A view of a piece follows
    /// remaps and stays.
    #[test]
    fn a_remap_drops_a_section_view() {
        if !canonical_on() {
            return;
        }
        let heap = crate::alias::physical_heap(4 * PAGE).expect("a heap");
        let t = Title::new(2);
        t.show(&heap, 0, 0);
        t.show(&heap, 1, 1);
        let start = t.page(0) as u64;
        crate::shadow::register(0xa1_0040, start, 2 * PAGE as u64);
        let section = crate::shadow::find(start + 8, 8).expect("a view");
        assert_eq!(section.view_kind(), Some(Kind::Section));
        assert_eq!(section.base(), 0);
        // Page 1 shows heap page 2 now: no run, and the old view is out.
        t.show(&heap, 1, 2);
        let next = crate::shadow::find(start + 8, 8).expect("a copy");
        assert!(!Arc::ptr_eq(&section, &next), "the section view left the registry");
        assert!(!next.is_view(), "the old view still holds heap page 0: a copy (mixed)");
        crate::shadow::unregister(0xa1_0040, start);
        drop((section, next));
        // A view of the piece, then a remap under it: it follows, and stays.
        crate::shadow::register(0xa1_0041, start, 2 * PAGE as u64);
        let piece = crate::shadow::find(start + 8, 8).expect("a view");
        assert_eq!(piece.view_kind(), Some(Kind::Piece));
        t.show(&heap, 1, 3);
        let again = crate::shadow::find(start + 8, 8).expect("a view");
        assert!(Arc::ptr_eq(&piece, &again), "the piece view stays");
        let id = heap.id().expect("a heap of the pieces'");
        assert_eq!(holds_at(id, 3), [0, 1], "its hold moved to the new heap tile");
        assert_eq!(holds_at(id, 2), [0, 0]);
        crate::shadow::unregister(0xa1_0041, start);
    }

    /// I4b, the coordinator's GPU test: one page at two title addresses (two
    /// reservations), each a title buffer, both bound in one submission: read
    /// through the second, a shader writes through the first, a UAV barrier,
    /// a read through the second in the next dispatch. With `=gpu` both are
    /// the section's one GPU range, and the read is fresh. With `=gpu-tiles`
    /// they are two pieces, two ranges, and the read is stale (probes 5 and
    /// 8: the GPU caches lines per range; no barrier helps).
    #[test]
    fn one_page_at_two_addresses_is_fresh_through_one_range() {
        let mode = crate::mac_switches::alias_physical();
        if !on() || mode == crate::mac_switches::AliasPhysical::GpuSingle {
            eprintln!("skipped: needs VICTORY_ALIAS_PHYSICAL=gpu or =gpu-tiles under Wine");
            return;
        }
        let host = crate::host::host().expect("a host device");
        let heap = crate::alias::physical_heap(4 * PAGE).expect("a heap");
        let (a, b, c) = (Title::new(2), Title::new(2), Title::new(2));
        // Heap page 1 at a's page 0 and b's page 1; the source at a's page 1
        // (heap page 2); the outputs at c's pages (heap pages 3 and 0).
        a.show(&heap, 0, 1);
        b.show(&heap, 1, 1);
        a.show(&heap, 1, 2);
        c.show(&heap, 0, 3);
        c.show(&heap, 1, 0);
        a.fill(0, 0x7400);
        a.fill(1, 0x7401);
        let (x, y, src, out, first) = (a.page(0), b.page(1), a.page(1), c.page(0), c.page(1));
        for (k, at) in [x, y, src, out, first].into_iter().enumerate() {
            crate::shadow::register(0xa1_0050 + k, at as u64, PAGE as u64);
        }
        let views: Vec<_> = [x, y, src, out, first]
            .into_iter()
            .map(|at| crate::shadow::find(at as u64, 16).expect("a view"))
            .collect();
        assert!(views.iter().all(|v| v.is_view()), "all views");
        let gpu = |k: usize, at: usize| views[k].gpu(at as u64);
        if mode.canonical() {
            assert_eq!(gpu(0, x), gpu(1, y), "=gpu: one GPU range for the page");
        } else {
            assert_ne!(gpu(0, x), gpu(1, y), "=gpu-tiles: a range per address");
        }
        let pipeline = crate::compute::pipeline(
            SHADER,
            s!("alias_physical test (two addresses)"),
            0,
            &[
                Param::Srv {
                    register: 0,
                    space: 0,
                },
                Param::Uav { register: 0 },
                Param::Constants {
                    register: 0,
                    count: 4,
                },
            ],
        )
        .expect("a pipeline");
        let rig = Rig::new(&host.device).expect("a queue");
        rig.run(flush, |list| {
            let pass = |from: u64, to: u64, add: u32| {
                let constants = [WORDS as u32, add, 0, 0];
                // SAFETY: live pipeline and views, constants on this stack.
                unsafe {
                    list.SetComputeRootShaderResourceView(0, from);
                    list.SetComputeRootUnorderedAccessView(1, to);
                    list.SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
                    list.Dispatch((WORDS / 64) as u32, 1, 1);
                    list.ResourceBarrier(&[crate::host::uav_barrier()]);
                }
            };
            // SAFETY: a live pipeline.
            unsafe {
                list.SetComputeRootSignature(&pipeline.0);
                list.SetPipelineState(&pipeline.1);
            }
            // Read through y (its lines cached), write through x, read through y.
            pass(gpu(1, y), gpu(4, first), 0);
            pass(gpu(2, src), gpu(0, x), 1);
            pass(gpu(1, y), gpu(3, out), 0);
        })
        .expect("the submission ran");
        let want: Vec<u32> = (0..WORDS).map(|i| word(i, 0x7401) + 1).collect();
        assert_eq!(a.words(0), &want[..], "the write through x reached the page");
        let old = c.words(1).iter().enumerate().all(|(i, &w)| w == word(i, 0x7400));
        assert!(old, "the first read through y saw the page before the write");
        let stale = c.words(0).iter().zip(&want).filter(|(g, w)| g != w).count();
        eprintln!(
            "{}: {stale} of {WORDS} words read through the second address stale after the write and a UAV barrier",
            mode.name()
        );
        if mode.canonical() {
            assert_eq!(stale, 0, "=gpu: fresh through one range");
        } else {
            assert!(
                stale > 0,
                "=gpu-tiles: probes 5 and 8 found two ranges stale under a UAV barrier; fresh here is a finding"
            );
        }
        for (k, at) in [x, y, src, out, first].into_iter().enumerate() {
            crate::shadow::unregister(0xa1_0050 + k, at as u64);
        }
        // SAFETY: querying device health.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
