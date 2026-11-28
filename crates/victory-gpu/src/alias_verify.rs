//! mac: stage 2's verify mode (`VICTORY_ALIAS_PHYSICAL=verify`, and
//! `=verify-tiles` for I4's order; `notes/alias-physical-debug.md`).
//!
//! Everything is drawn from host copies, as with the switch off (the heaps
//! back the title's pages as with `=cpu`, and the GPU side's pieces and tile
//! flushes run as with `=gpu`). Where `=gpu` would have bound a view of a
//! piece or of a section in a copy's place, the view is made beside the copy
//! (`crate::shadow`: [`Attached`], and this module's own registry for
//! windows), holding its tiles as a bound view would, and every binding of
//! the copy is checked on the CPU: the binding's own arithmetic, applied to
//! the view (root views' first byte and offset, table views' first element,
//! vertex, index and indirect buffers' GPU addresses, page-table pairs),
//! gives buffer bytes; through the tile plan those are heap tiles
//! (`crate::verify_plan`), which must be the tiles the title shows at the
//! same addresses, with the same bytes. A sample of each binding is read
//! (its start, its end and a few pages between, 64 bytes each); each
//! distinct binding once a presented frame, up to a budget a frame.
//!
//! Also counted: bindings `=gpu` would leave on copies, and table views that
//! would be exact copies because the view's base puts them off their
//! elements (two GPU ranges over one page then); uses of a copy whose view a
//! remap made leave the registries, read through that old view (what a heap
//! slot or a list made before the remap would read with `=gpu`); per
//! submission, heap tiles reached through two GPU ranges with a write
//! through one of them (stale within the submission, `notes/tile-probe.md`
//! probes 5 and 8) and pages of section views the title remapped before the
//! submission ran; title copies into view memory; textures uploaded from
//! view memory, and from pages the GPU wrote through would-be views.
//!
//! Lines: `alias verify: ...` every 5 s beside the physical pages line, and
//! the first 20 differences and 10 hazards in full.
//!
//! The one-range rule (`crate::one_range`) sees `=gpu`'s copies: a title
//! buffer's copy `=gpu` would make too holds its pages, and so do the window
//! copies and views' own copies `=gpu` would make, kept here (window copies
//! in [`WINDOWS`] as `=gpu`'s window registry holds them, views' own copies
//! while asked for).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use crate::shadow::Shadow;
use crate::tile_plan::{HeapTile, PAGE, Plan};
use crate::verify_plan::{self, Found, Outcome, Reach};

/// What a host copy holds in verify mode: the view `=gpu` would bind in its
/// place.
#[derive(Default)]
pub(crate) struct Attached(Mutex<Held>);

#[derive(Default)]
struct Held {
    what: What,
    /// The view `=gpu` would bind in the copy's place now (`None`: a copy
    /// with `=gpu` too).
    view: Option<Arc<Shadow>>,
    /// A title buffer's view left with a remap: the next check asks
    /// `crate::alias_physical::view` again, as `=gpu`'s next lookup would.
    ask: bool,
    /// The last view that left: with `=gpu`, heap slots and lists made
    /// before the remap still read through it.
    stale: Option<Arc<Shadow>>,
    /// The page-table pair of `view` was checked (once a view: its
    /// arithmetic does not change; the bytes are other kinds' to check).
    paired: bool,
}

/// What a copy is, for verify mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum What {
    /// Neither below (aliased regions, exact copies): `=gpu` binds the same.
    #[default]
    Other,
    /// A title buffer's copy, whose view was asked for when it was made.
    Buffer,
    /// A window outside title buffers: the view is asked per range.
    Window,
}

impl Attached {
    fn held(&self) -> std::sync::MutexGuard<'_, Held> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A title buffer's copy, and the view `=gpu` would have made of it
    /// (`None`: `=gpu` makes a copy too).
    pub(crate) fn buffer(&self, view: Option<Arc<Shadow>>) {
        let mut h = self.held();
        h.what = What::Buffer;
        h.view = view;
        h.ask = false;
        h.paired = false;
    }

    /// A window's copy.
    pub(crate) fn window(&self) {
        self.held().what = What::Window;
    }

    /// The title remapped `[from, to)`: the view, if it is over it (and of
    /// `kind`), leaves as views leave `=gpu`'s registries, and stays as the
    /// stale one. It is returned, to be told (`crate::alias_physical::left`).
    pub(crate) fn drop_over(
        &self,
        from: u64,
        to: u64,
        kind: Option<crate::alias_physical::Kind>,
    ) -> Option<Arc<Shadow>> {
        let mut h = self.held();
        let over = h.view.as_ref().is_some_and(|v| {
            let (start, len) = v.range();
            start < to && from < start + len && (kind.is_none() || v.view_kind() == kind)
        });
        if !over {
            return None;
        }
        let view = h.view.take()?;
        h.ask = true;
        h.paired = false;
        let old = h.stale.replace(view.clone());
        drop(h);
        // Its tiles go back outside the lock (`crate::alias_physical`'s).
        drop(old);
        Some(view)
    }
}

/// A binding kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Bind {
    Root,
    RootUav,
    Table,
    TableUav,
    Slot,
    SlotUav,
    Constant,
    Vertex,
    Index,
    Arguments,
    Count,
    PageTable,
}

const BINDS: usize = 12;

impl Bind {
    fn name(self) -> &'static str {
        [
            "root views",
            "root UAVs",
            "table views",
            "table UAVs",
            "heap slots",
            "heap slot UAVs",
            "constants",
            "vertex buffers",
            "index buffers",
            "indirect arguments",
            "indirect counts",
            "page-table pairs",
        ][self as usize]
    }

    fn uav(self) -> bool {
        matches!(self, Bind::RootUav | Bind::TableUav | Bind::SlotUav)
    }

    fn all() -> [Bind; BINDS] {
        use Bind::*;
        [
            Root, RootUav, Table, TableUav, Slot, SlotUav, Constant, Vertex, Index, Arguments,
            Count, PageTable,
        ]
    }
}

/// Per binding kind, since the last line.
#[derive(Clone, Copy)]
enum N {
    /// Bindings seen.
    Asked,
    /// Checked (the frame's first of its kind, within the budget).
    Checked,
    /// `=gpu` binds a copy too.
    Copy,
    /// Table views `=gpu` would give an exact copy: off their elements
    /// from the view's base.
    Exact,
    /// Through a view: equal, different (by tile or byte), by tile.
    Equal,
    Different,
    Tiles,
    /// Bytes compared.
    Bytes,
    /// Checked through a view a remap made leave, and different.
    Stale,
    StaleDiffer,
    /// Not checked: the frame's budget was spent.
    Skipped,
}

const NS: usize = 11;

static COUNTS: [[AtomicU64; NS]; BINDS] = [const { [const { AtomicU64::new(0) }; NS] }; BINDS];

fn count(bind: Bind, n: N, by: u64) {
    COUNTS[bind as usize][n as usize].fetch_add(by, Relaxed);
}

/// Since the last line: differences through views of sections and of
/// pieces; two-range hazards (heap tiles) and submissions with one; pages
/// of section views stale at submission; title copies into view memory and
/// their bytes; texture refreshes from view memory, their bytes, and those
/// from pages the GPU wrote through would-be views.
static OTHER: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];
const DIFF_SECTION: usize = 0;
const DIFF_PIECE: usize = 1;
const HAZARDS: usize = 2;
const HAZARD_SUBMISSIONS: usize = 3;
const STALE_AT_SUBMIT: usize = 4;
const COPIES_IN: usize = 5;
const COPIED_IN: usize = 6;
const TEXTURES: usize = 7;
const TEXTURE_BYTES: usize = 8;
const TEXTURES_WRITTEN: usize = 9;
const TEXTURES_MISTOLD: usize = 10;
const WITHHELD: usize = 11;
const WITHHELD_BYTES: usize = 12;
const LANDED: usize = 13;
const LANDED_BYTES: usize = 14;

/// Withheld writes in aliased memory since the last line, by the aliased
/// region (its start and length): bytes.
static LANDED_IN: Mutex<BTreeMap<(u64, u64), u64>> = Mutex::new(BTreeMap::new());

/// A submission's shaders wrote `[address, address + len)` of aliased
/// memory `shadow` (one of `VICTORY_ALIAS_BUFFERS`' regions), and the write
/// is withheld (`crate::memory::withholding`): it is in title memory
/// already, and stays. `programs`: the title-memory programs of the lists
/// that declared it. Counted by region; the first 20 logged in full.
pub(crate) fn landed(shadow: &Shadow, address: usize, len: usize, programs: Option<&Vec<String>>) {
    let (start, n) = shadow.range();
    let (from, to) = ((address as u64).max(start), ((address + len) as u64).min(start + n));
    if from >= to {
        return;
    }
    OTHER[LANDED].fetch_add(1, Relaxed);
    OTHER[LANDED_BYTES].fetch_add(to - from, Relaxed);
    *LANDED_IN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry((start, n))
        .or_default() += to - from;
    static FULL: AtomicU64 = AtomicU64::new(0);
    if FULL.fetch_add(1, Relaxed) < 20 {
        tracing::warn!(
            "alias verify: a withheld write is in title memory anyway: {from:#x}+{:#x} of the aliased region \
             {start:#x}+{n:#x} (frame {}), declared by lists using the title-memory programs {:?}",
            to - from,
            crate::queue::presented(),
            programs.map_or(&[][..], |p| &p[..]),
        );
    }
}

/// A submission's shaders wrote `[address, address + len)` of `copy`, and
/// the write is withheld from title memory (`crate::memory::withholding`: a
/// list using title memory read memory with no copy, so its results and
/// those computed from them are garbage). Copies keep it out; through a
/// view `=gpu` binds it is in title memory already, and stays. Counted
/// where `=gpu` would have bound a view.
pub(crate) fn withheld(copy: &Shadow, address: usize, len: usize) {
    if !crate::alias_physical::verifying() || len == 0 {
        return;
    }
    let what = copy.verify.held().what;
    let view = match what {
        What::Buffer => copy.verify.held().view.clone(),
        What::Window => window_view(address as u64, (address + len) as u64),
        What::Other => None,
    };
    let Some(view) = view else {
        return;
    };
    let (start, n) = view.range();
    let (from, to) = ((address as u64).max(start), ((address + len) as u64).min(start + n));
    if from < to {
        OTHER[WITHHELD].fetch_add(1, Relaxed);
        OTHER[WITHHELD_BYTES].fetch_add(to - from, Relaxed);
    }
}

/// Heap tiles a UAV binding would write through a view that reaches other
/// tiles than the title shows at its addresses: `=gpu` tells the title's
/// pages there (`crate::watch::notify_written`), not these tiles' trackers.
static MISTOLD: LazyLock<Mutex<HashSet<HeapTile>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
static ANY_MISTOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A tracked texture over title memory `[start, start + size)` is asked
/// for its changes: with `=gpu`, would a GPU write into its pages have
/// reached its tracker? Counted where a UAV binding wrote one of its heap
/// tiles through a view telling other pages. (Nothing to do, at one load,
/// while no such binding was seen.)
pub(crate) fn texture_asked(start: usize, size: usize) {
    if !ANY_MISTOLD.load(Relaxed) || size == 0 || !crate::alias_physical::verifying() {
        return;
    }
    // Once a frame per texture.
    {
        static ASKED: LazyLock<Mutex<(u64, HashSet<usize>)>> =
            LazyLock::new(|| Mutex::new((u64::MAX, HashSet::new())));
        let frame = crate::queue::presented();
        let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
        if asked.0 != frame {
            asked.0 = frame;
            asked.1.clear();
        }
        if !asked.1.insert(start) {
            return;
        }
    }
    let mistold = MISTOLD.lock().unwrap_or_else(|e| e.into_inner());
    let hit = crate::alias_physical::with_plan(|plan, _| {
        (start / PAGE..(start + size).div_ceil(PAGE))
            .take(4096)
            .any(|p| plan.current_at(p * PAGE).is_some_and(|t| mistold.contains(&t)))
    });
    if hit {
        OTHER[TEXTURES_MISTOLD].fetch_add(1, Relaxed);
    }
}

/// Differences and hazards logged in full so far.
static LOGGED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bindings checked in this presented frame (copy, address, length,
/// kind), and the frame.
static SEEN: LazyLock<Mutex<Seen>> = LazyLock::new(|| Mutex::new((u64::MAX, HashSet::new())));

/// The frame, and the bindings checked in it.
type Seen = (u64, HashSet<(usize, u64, u64, u8)>);

/// Checks a frame at most (`VICTORY_ALIAS_VERIFY_CHECKS`, default 20,000).
fn budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("VICTORY_ALIAS_VERIFY_CHECKS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(20_000)
    })
}

/// Whether this binding is the frame's first of its kind and within the
/// budget (else counted as skipped).
fn first_this_frame(bind: Bind, copy: &Shadow, address: u64, len: u64) -> bool {
    let frame = crate::queue::presented();
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.0 != frame {
        seen.0 = frame;
        seen.1.clear();
    }
    let key = (std::ptr::from_ref(copy) as usize, address, len, bind as u8);
    if seen.1.contains(&key) {
        return false;
    }
    if seen.1.len() >= budget() {
        count(bind, N::Skipped, 1);
        return false;
    }
    seen.1.insert(key);
    true
}

/// What `=gpu`'s windows registry would hold: a view, or (one-range rule) a
/// window copy, which holds its pages as `=gpu`'s would.
enum Win {
    View(Arc<Shadow>),
    Copy(#[allow(dead_code)] Option<crate::alias_physical::CopyHold>),
}

/// The views of windows (ranges outside title buffers) `=gpu` would have
/// made, and its window copies, by (start, length), as `crate::shadow`'s
/// windows registry holds them with `=gpu`.
static WINDOWS: Mutex<BTreeMap<(u64, u64), Win>> = Mutex::new(BTreeMap::new());

/// The view `=gpu` would bind for `[address, address + len)` of `copy`, if
/// any.
fn view_for(copy: &Shadow, address: u64, len: u64) -> Option<Arc<Shadow>> {
    let what = copy.verify.held().what;
    match what {
        What::Other => None,
        What::Buffer => {
            {
                let h = copy.verify.held();
                if !h.ask {
                    return h.view.clone();
                }
            }
            // As `=gpu`'s next lookup after the view left: asked again.
            let (start, n) = copy.range();
            let view = crate::alias_physical::view(start as usize, (start + n) as usize)
                .map(Shadow::new_view_checked);
            let none = view.is_none();
            let found = {
                let mut h = copy.verify.held();
                if h.ask {
                    h.ask = false;
                    h.paired = false;
                    h.view = view;
                }
                h.view.clone()
            };
            // `=gpu` makes a copy now: it holds its pages (one-range rule).
            if none && found.is_none() && !copy.holds_copy() {
                copy.hold_copy(crate::alias_physical::copy_made(start as usize, (start + n) as usize));
            }
            found
        }
        What::Window => window_view(address, address.saturating_add(len.max(1))),
    }
}

/// The window view `=gpu` would hold `[address, end)` in, made if need be
/// (`None`: `=gpu` reads it from a window copy, made here as `=gpu` would).
fn window_view(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let mut map = WINDOWS.lock().unwrap_or_else(|e| e.into_inner());
    // `crate::shadow::loose`'s search: the one starting closest below.
    let found = map
        .range(..=(address, u64::MAX))
        .rev()
        .take(64)
        .find(|&(&(s, n), _)| s <= address && end <= s + n)
        .map(|(_, w)| match w {
            Win::View(v) => Some(v.clone()),
            Win::Copy(_) => None,
        });
    if let Some(found) = found {
        return found;
    }
    let block = crate::shadow::loose_window_bytes();
    if let Some(view) = crate::alias_physical::window(address as usize, end as usize, block as usize) {
        let (from, n) = (view.range.start as u64, view.range.len() as u64);
        let view = Shadow::new_view_checked(view);
        map.insert((from, n), Win::View(view.clone()));
        return Some(view);
    }
    // `=gpu` makes the window's copy: with the one-range rule it holds its
    // pages, and the views over them leave (outside this lock).
    let start = address & !(block - 1);
    let len = end.checked_next_multiple_of(block)? - start;
    let made = crate::alias_physical::copy_made(start as usize, (start + len) as usize);
    let (hold, leave) = made.map_or((None, None), |(h, l)| (Some(h), Some(l)));
    map.insert((start, len), Win::Copy(hold));
    drop(map);
    if let Some(leave) = leave {
        leave.now();
    }
    None
}

/// The window views over `[from, to)` (of `kind`) leave, as with `=gpu`.
pub(crate) fn drop_windows(
    from: u64,
    to: u64,
    kind: Option<crate::alias_physical::Kind>,
) -> Vec<Arc<Shadow>> {
    let mut map = WINDOWS.lock().unwrap_or_else(|e| e.into_inner());
    let keys: Vec<(u64, u64)> = map
        .iter()
        .filter(|&(&(s, n), w)| {
            s < to && from < s + n && matches!(w, Win::View(v) if kind.is_none() || v.view_kind() == kind)
        })
        .map(|(&k, _)| k)
        .collect();
    keys.into_iter()
        .filter_map(|k| match map.remove(&k) {
            Some(Win::View(v)) => Some(v),
            _ => None,
        })
        .collect()
}

/// Tests: the windows (views and copies) over `[from, to)` go.
#[cfg(test)]
pub(crate) fn forget_windows(from: u64, to: u64) {
    let gone: Vec<Win> = {
        let mut map = WINDOWS.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map.keys().filter(|&&(s, n)| s < to && from < s + n).copied().collect();
        keys.into_iter().filter_map(|k| map.remove(&k)).collect()
    };
    drop(gone);
}

/// The copies of views' own bytes `=gpu` would make (`crate::shadow::for_view`
/// off the view's elements), by (address, length, element), each holding its
/// pages for the one-range rule, with the frame it was last asked for in:
/// `=gpu` keeps one while heap slots and lists hold it, here while asked for
/// within [`OWN_FRAMES`] frames.
#[allow(clippy::type_complexity)]
static OWN: LazyLock<Mutex<HashMap<(u64, u64, u64), (Option<crate::alias_physical::CopyHold>, u64)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const OWN_FRAMES: u64 = 60;

/// `=gpu` would give the view of `[address, address + len)` of `element`-byte
/// elements a copy of its own: hold its pages as that copy would.
fn own_copy(address: u64, len: u64, element: u64) {
    let frame = crate::queue::presented();
    let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = own.get_mut(&(address, len, element)) {
        entry.1 = frame;
        return;
    }
    let mut gone = Vec::new();
    if own.len() >= 4096 {
        let old: Vec<(u64, u64, u64)> = own
            .iter()
            .filter(|(_, (_, at))| frame.saturating_sub(*at) >= OWN_FRAMES)
            .map(|(&k, _)| k)
            .collect();
        gone.extend(old.into_iter().filter_map(|k| own.remove(&k)));
    }
    let made = crate::alias_physical::copy_made(address as usize, address.saturating_add(len) as usize);
    let (hold, leave) = made.map_or((None, None), |(h, l)| (Some(h), Some(l)));
    own.insert((address, len, element), (hold, frame));
    drop(own);
    // Their holds go outside the lock (`crate::alias_physical`'s).
    drop(gone);
    if let Some(leave) = leave {
        leave.now();
    }
}

/// A binding of `[address, address + len)` of `copy`, of kind `bind` (root
/// views, vertex, index and indirect buffers, page-table pairs): checked
/// through the view `=gpu` would bind there, if any.
pub(crate) fn check(bind: Bind, copy: &Arc<Shadow>, address: u64, len: u64) {
    if !crate::alias_physical::verifying() || len == 0 || copy.is_view() {
        return;
    }
    // Page tables name every copy at each rewrite; with `=gpu` a window's
    // pairs are its views', made per range: only title buffers' pairs are
    // checked, once a view.
    if bind == Bind::PageTable {
        let mut h = copy.verify.held();
        if h.what != What::Buffer || (h.paired && !h.ask) {
            return;
        }
        h.paired = true;
    }
    count(bind, N::Asked, 1);
    if bind != Bind::PageTable && !first_this_frame(bind, copy, address, len) {
        return;
    }
    count(bind, N::Checked, 1);
    let Some(view) = view_for(copy, address, len) else {
        count(bind, N::Copy, 1);
        note_copy(bind, copy, address, len, None);
        return;
    };
    let Some(place) = place(bind, &view, address) else {
        count(bind, N::Different, 1);
        return;
    };
    compare(bind, &view, address, len, place, false);
}

/// A table view (through the title's tables or the heap, or a constant
/// buffer) of `[address, address + len)` whose elements are `element`
/// bytes: `=gpu` makes its first element `offset / element` in the view
/// `crate::shadow::for_view` finds, or an exact copy when the offset is not
/// a whole number of elements.
pub(crate) fn table(bind: Bind, address: u64, len: u64, element: u64) {
    if !crate::alias_physical::verifying() || len == 0 {
        return;
    }
    let Some(copy) = crate::shadow::find(address, len) else {
        return;
    };
    if copy.is_view() {
        return;
    }
    count(bind, N::Asked, 1);
    if !first_this_frame(bind, &copy, address, len) {
        return;
    }
    count(bind, N::Checked, 1);
    // A heap slot or a list made before a remap made the view leave reads
    // through the old one with `=gpu`.
    let stale = copy.verify.held().stale.clone();
    if let Some(stale) = stale {
        let offset = stale.offset(address);
        let (start, n) = stale.range();
        if address >= start && address + len <= start + n && offset.is_multiple_of(element.max(1)) {
            compare(bind, &stale, address, len, offset, true);
        }
    }
    let Some(view) = view_for(&copy, address, len) else {
        count(bind, N::Copy, 1);
        note_copy(bind, &copy, address, len, None);
        return;
    };
    let offset = view.offset(address);
    if element > 1 && !offset.is_multiple_of(element) {
        // `=gpu` would give the view an exact copy of its own, beside the
        // view over the same pages: two GPU ranges.
        count(bind, N::Exact, 1);
        // With the one-range rule `=gpu`'s copy holds its pages, as here.
        own_copy(address, len, element);
        note_copy(bind, &copy, address, len, Some(address));
        return;
    }
    let element = element.max(1);
    let place = offset / element * element;
    compare(bind, &view, address, len, place, false);
}

/// The buffer byte a binding of kind `bind` puts title address `address` at
/// in view `view`, by the binding's own arithmetic (`None`: two of its
/// formulas disagree).
fn place(bind: Bind, view: &Shadow, address: u64) -> Option<u64> {
    // SAFETY: a live buffer.
    let va = unsafe { view.buffer.GetGPUVirtualAddress() };
    match bind {
        // `crate::draw`'s root views: a raw view from `first`, `offset` in.
        Bind::Root | Bind::RootUav => {
            let (first, offset) = view.root_place(address);
            Some(first + offset)
        }
        Bind::Table | Bind::TableUav | Bind::Slot | Bind::SlotUav | Bind::Constant => {
            Some(view.offset(address))
        }
        // A GPU address (`IASetVertexBuffers`, `IASetIndexBuffer`,
        // `ExecuteIndirect`'s count).
        Bind::Vertex | Bind::Index | Bind::Count => view.gpu(address).checked_sub(va),
        // `ExecuteIndirect`'s arguments: the GPU address, and the clamp
        // batch's buffer start (`gpu(at) - offset(at)`, `crate::draw`).
        Bind::Arguments => {
            let byte = view.gpu(address).checked_sub(va)?;
            (view.gpu(address) - view.offset(address) == va).then_some(byte)
        }
        // `crate::memory`'s pairs: a raw view from the copy's base, and the
        // copy's title start in the table's entry.
        Bind::PageTable => Some(view.base() + (address - view.range().0)),
    }
}

/// Compares `[address, address + len)` (cut to the view) as a binding of
/// kind `bind` reads it through `view`, title address `address` at buffer
/// byte `place`; counts, logs the first differences, and notes the pages
/// for the submission's checks.
fn compare(bind: Bind, view: &Shadow, address: u64, len: u64, place: u64, stale: bool) {
    let Some(reach) = view.reach() else {
        return;
    };
    let (vstart, vlen) = view.range();
    let to = address
        .saturating_add(len)
        .min(vstart + vlen)
        .max(address + 1);
    let byte_of = |a: usize| place + (a as u64 - address);
    let samples = verify_plan::samples(address as usize, to as usize, byte_of, 64, 6);
    // One check in 64 reads the title's own addresses (through the kernel,
    // which refuses an unmapped page): that the title shows the heap's RAM
    // there (`=cpu`'s remap). The others read the tile the title shows
    // through the heap, which is the same RAM when that holds.
    static REAL: AtomicU64 = AtomicU64::new(0);
    let real = REAL.fetch_add(1, Relaxed).is_multiple_of(64);
    let uav = bind.uav();
    let mut notes = Vec::with_capacity(samples.len());
    let mut mistold = Vec::new();
    let outcome: Outcome = crate::alias_physical::with_plan(|plan, heap| {
        let mut title = |at: usize, out: &mut [u8]| {
            if real {
                crate::host::read_into(at as *const u8, out)
            } else {
                match plan.current_at(at / PAGE * PAGE) {
                    Some(t) => heap(t, at % PAGE, out),
                    None => false,
                }
            }
        };
        let o = verify_plan::compare(plan, reach, &samples, &mut title, &mut |t, o, b| {
            heap(t, o, b)
        });
        for s in &samples {
            let title_tile = plan.current_at(s.at / PAGE * PAGE);
            if let Ok(Some(t)) = reach.tile(plan, s.byte) {
                // A write through this view lands in tile `t`, not in the
                // one the title shows at its address, whose trackers its
                // copy-back tells: both tiles' textures go wrong.
                if uav && !stale && title_tile != Some(t) {
                    mistold.push(t);
                    mistold.extend(title_tile);
                }
                notes.push(Note {
                    page: s.at / PAGE,
                    tile: t,
                    range: reach.range_id(s.byte),
                    uav,
                    view: true,
                    section: matches!(reach, Reach::Section { .. }),
                });
            }
        }
        o
    });
    // (Outside the plan's lock: `texture_asked` takes them the other way.)
    if !mistold.is_empty() {
        MISTOLD.lock().unwrap_or_else(|e| e.into_inner()).extend(mistold);
        ANY_MISTOLD.store(true, Relaxed);
    }
    if !stale {
        note(notes);
    }
    count(bind, N::Bytes, outcome.bytes);
    if stale {
        count(bind, N::Stale, 1);
        if outcome.differs() {
            count(bind, N::StaleDiffer, 1);
        }
    } else if outcome.differs() {
        count(bind, N::Different, 1);
        if outcome.tiles_differ > 0 {
            count(bind, N::Tiles, 1);
        }
        let i = if matches!(reach, Reach::Section { .. }) {
            DIFF_SECTION
        } else {
            DIFF_PIECE
        };
        OTHER[i].fetch_add(1, Relaxed);
    } else {
        count(bind, N::Equal, 1);
    }
    if outcome.differs()
        && let Some(f) = outcome.first
    {
        *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some((bind, stale, f));
        if LOGGED[0].fetch_add(1, Relaxed) < 20 {
            log_difference(bind, view, reach, address, len, place, stale, &outcome, &f);
        }
    }
}

/// The last difference found: the binding kind, whether through a view a
/// remap made leave, and the difference (tests).
static LAST: Mutex<Option<(Bind, bool, Found)>> = Mutex::new(None);

#[allow(clippy::too_many_arguments)]
fn log_difference(
    bind: Bind,
    view: &Shadow,
    reach: Reach,
    address: u64,
    len: u64,
    place: u64,
    stale: bool,
    o: &Outcome,
    f: &Found,
) {
    let (vstart, vlen) = view.range();
    let tile = |t: Option<HeapTile>| {
        t.map_or("nothing".to_owned(), |t| {
            format!("heap {} tile {}", t.heap, t.tile)
        })
    };
    let reached = match f.view_tile {
        Ok(t) => tile(t),
        Err(_) => "past the view's buffer".to_owned(),
    };
    tracing::warn!(
        "alias verify: a difference in {} at {address:#x}+{len:#x}{}: through a view of {} over {vstart:#x}+{vlen:#x} \
         (base {:#x}), the binding puts the address at buffer byte {place:#x}; first difference at {:#x} (buffer \
         byte {:#x}): the title shows {} there, the view reaches {}; expected {:08x?}, found {:08x?}; {} of {} \
         stretches differ by tile, {} by bytes ({} bytes compared)",
        bind.name(),
        if stale {
            " (through a view a remap made leave: heap slots and lists made before it read this)"
        } else {
            ""
        },
        match reach {
            Reach::Piece { piece } => format!("piece {piece}"),
            Reach::Section { heap, tile0, tiles } =>
                format!("heap {heap}'s section buffer at tile {tile0} ({tiles} tiles)"),
        },
        view.base(),
        f.at,
        f.byte,
        tile(f.title_tile),
        reached,
        f.expected,
        f.found,
        o.tiles_differ,
        o.stretches,
        o.bytes_differ,
        o.bytes,
    );
}

/// A page a submission's binding reaches: the title page, the heap tile,
/// the GPU range (buffer and tile in it), whether written, whether through
/// a view (else a copy `=gpu` keeps), and whether through a section view
/// (fixed tiles).
#[derive(Clone, Copy)]
struct Note {
    page: usize,
    tile: HeapTile,
    range: (u64, u32),
    uav: bool,
    view: bool,
    section: bool,
}

/// The current submission's notes (at most [`NOTES`]).
static PENDING: Mutex<Vec<Note>> = Mutex::new(Vec::new());
const NOTES: usize = 200_000;

fn note(notes: Vec<Note>) {
    let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let room = NOTES.saturating_sub(p.len());
    p.extend(notes.into_iter().take(room));
}

/// A binding `=gpu` leaves on a copy (`exact`: an exact copy of its own at
/// that address): its pages in reservations, as a GPU range of their own.
fn note_copy(bind: Bind, copy: &Shadow, address: u64, len: u64, exact: Option<u64>) {
    let id = 1 << 62 | exact.unwrap_or(std::ptr::from_ref(copy) as u64) & ((1 << 62) - 1);
    let samples = verify_plan::samples(
        address as usize,
        address.saturating_add(len) as usize,
        |a| a as u64,
        4,
        6,
    );
    let notes: Vec<Note> = crate::alias_physical::with_plan(|plan, _| {
        samples
            .iter()
            .filter_map(|s| {
                plan.current_at(s.at / PAGE * PAGE).map(|t| Note {
                    page: s.at / PAGE,
                    tile: t,
                    range: (id, (s.at / PAGE) as u32),
                    uav: bind.uav(),
                    view: false,
                    section: false,
                })
            })
            .collect()
    });
    note(notes);
}

/// A submission goes to the GPU (`crate::alias_physical::flush`, under its
/// lock): its notes against `plan` as it is now. Heap tiles reached through
/// two GPU ranges with a write through one, and section views' pages the
/// title remapped since they were bound.
pub(crate) fn submitted(plan: &Plan) {
    let notes = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    if notes.is_empty() {
        return;
    }
    // Per heap tile: the first range, whether written, whether a view
    // reached it, and whether it was counted.
    let mut by_tile: HashMap<HeapTile, ((u64, u32), bool, bool, bool)> = HashMap::new();
    let mut hazards = 0u64;
    let mut stale = 0u64;
    for n in &notes {
        match by_tile.entry(n.tile) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert((n.range, n.uav, n.view, false));
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let (range, uav, view, flagged) = e.get_mut();
                *uav |= n.uav;
                *view |= n.view;
                // Two copies over one page are as before stage 2: only
                // pairs with a view count.
                if *range != n.range && *uav && *view && !*flagged {
                    *flagged = true;
                    hazards += 1;
                    if LOGGED[1].fetch_add(1, Relaxed) < 10 {
                        tracing::warn!(
                            "alias verify: heap {} tile {} is reached through two GPU ranges in one submission, one \
                             of them written (title page {:#x}; ranges {:x?} and {:x?}): the second reads stale \
                             bytes within the submission (probes 5 and 8)",
                            n.tile.heap,
                            n.tile.tile,
                            n.page * PAGE,
                            range,
                            n.range
                        );
                    }
                }
            }
        }
        if n.section && plan.current_at(n.page * PAGE) != Some(n.tile) {
            stale += 1;
        }
    }
    if hazards > 0 {
        OTHER[HAZARDS].fetch_add(hazards, Relaxed);
        OTHER[HAZARD_SUBMISSIONS].fetch_add(1, Relaxed);
    }
    OTHER[STALE_AT_SUBMIT].fetch_add(stale, Relaxed);
}

/// A title copy (run on the CPU at submission) into `[dst, dst + len)`.
pub(crate) fn title_copy(dst: usize, len: usize) {
    if !crate::alias_physical::verifying() || len == 0 {
        return;
    }
    let inside = crate::alias_physical::with_plan(|plan, _| plan.locate(dst, dst + 1).is_some());
    if inside {
        OTHER[COPIES_IN].fetch_add(1, Relaxed);
        OTHER[COPIED_IN].fetch_add(len as u64, Relaxed);
    }
}

/// Title pages (64 KiB numbers) the GPU wrote through copies with a view
/// beside them, since the last line.
static WRITTEN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Copy-back put what the GPU wrote in `copy` into title memory
/// `[at, at + len)`.
pub(crate) fn gpu_wrote(copy: &Shadow, at: usize, len: usize) {
    if !crate::alias_physical::verifying() || len == 0 {
        return;
    }
    let held = copy.verify.held();
    if held.view.is_none() && held.what != What::Window {
        return;
    }
    drop(held);
    let mut w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    let set = w.get_or_insert_with(HashSet::new);
    if set.len() < 1 << 20 {
        set.extend(at / PAGE..(at + len).div_ceil(PAGE));
    }
}

/// A texture over title memory `[start, start + size)` is refreshed for
/// its changed runs `changed`.
pub(crate) fn texture(start: usize, size: usize, changed: &[(usize, usize)]) {
    if !crate::alias_physical::verifying() || size == 0 {
        return;
    }
    let inside =
        crate::alias_physical::with_plan(|plan, _| plan.locate(start, start + 1).is_some());
    if !inside {
        return;
    }
    OTHER[TEXTURES].fetch_add(1, Relaxed);
    OTHER[TEXTURE_BYTES].fetch_add(changed.iter().map(|&(f, e)| (e - f) as u64).sum(), Relaxed);
    let w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = w.as_ref()
        && changed
            .iter()
            .any(|&(f, e)| (f / PAGE..e.div_ceil(PAGE)).any(|p| set.contains(&p)))
    {
        OTHER[TEXTURES_WRITTEN].fetch_add(1, Relaxed);
    }
}

/// [`LANDED_IN`] since the last line, per frame, the largest first, taken.
fn landed_by_region(frames: f64) -> String {
    let regions = std::mem::take(&mut *LANDED_IN.lock().unwrap_or_else(|e| e.into_inner()));
    let mut regions: Vec<((u64, u64), u64)> = regions.into_iter().collect();
    regions.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
    if regions.is_empty() {
        return "none".to_owned();
    }
    regions
        .iter()
        .take(9)
        .map(|((s, n), b)| format!("{s:#x}+{n:#x} {:.1} MB", *b as f64 / frames.max(1.0) / 1e6))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The verify mode's line, per frame over `frames` (`None` when off).
pub(crate) fn line(frames: f64) -> Option<String> {
    if !crate::alias_physical::verifying() {
        return None;
    }
    let f = |n: u64| n as f64 / frames.max(1.0);
    let mut parts = Vec::new();
    let (mut checked, mut skipped) = (0u64, 0u64);
    for bind in Bind::all() {
        let c = COUNTS[bind as usize].each_ref().map(|n| n.swap(0, Relaxed));
        let n = |k: N| c[k as usize];
        checked += n(N::Checked);
        skipped += n(N::Skipped);
        if n(N::Asked) == 0 {
            continue;
        }
        parts.push(format!(
            "{}: {:.0} checked of {:.0} bindings ({:.0} copies with =gpu too, {:.0} exact copies for the view's \
             base), {:.0} equal, {:.0} different ({:.0} by tile), {:.0} KB compared, {:.0} of {:.0} through views \
             left by remaps different",
            bind.name(),
            f(n(N::Checked)),
            f(n(N::Asked)),
            f(n(N::Copy)),
            f(n(N::Exact)),
            f(n(N::Equal)),
            f(n(N::Different)),
            f(n(N::Tiles)),
            f(n(N::Bytes)) / 1e3,
            f(n(N::StaleDiffer)),
            f(n(N::Stale)),
        ));
    }
    let o = OTHER.each_ref().map(|n| n.swap(0, Relaxed));
    let written = WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .map_or(0, |s| s.len());
    let share = if checked + skipped == 0 {
        100.0
    } else {
        100.0 * checked as f64 / (checked + skipped) as f64
    };
    Some(format!(
        "alias verify, per frame: {}; differences through views of sections {:.1}, of pieces {:.1}; {:.1}% of the \
         frame's distinct bindings checked ({:.0} over the budget); heap tiles reached through two GPU ranges with \
         a write {:.1} (in {:.1} submissions); section views' pages remapped before their submission ran {:.1}; \
         title copies into view memory {:.1} ({:.1} KB); textures refreshed from view memory {:.1} ({:.1} MB), {:.1} \
         of them over pages the GPU wrote through would-be views; {written} such pages since the last line; \
         textures asked for changes over heap tiles written through views that tell other pages {:.1}; \
         withheld writes (copy mode keeps them out of title memory) that views would put there {:.1} ({:.1} MB); \
         withheld writes in aliased regions, in title memory anyway, {:.1} ({:.1} MB; by region: {})",
        if parts.is_empty() {
            "no bindings".to_owned()
        } else {
            parts.join("; ")
        },
        f(o[DIFF_SECTION]),
        f(o[DIFF_PIECE]),
        share,
        f(skipped),
        f(o[HAZARDS]),
        f(o[HAZARD_SUBMISSIONS]),
        f(o[STALE_AT_SUBMIT]),
        f(o[COPIES_IN]),
        f(o[COPIED_IN]) / 1e3,
        f(o[TEXTURES]),
        f(o[TEXTURE_BYTES]) / 1e6,
        f(o[TEXTURES_WRITTEN]),
        f(o[TEXTURES_MISTOLD]),
        f(o[WITHHELD]),
        f(o[WITHHELD_BYTES]) / 1e6,
        f(o[LANDED]),
        f(o[LANDED_BYTES]) / 1e6,
        landed_by_region(frames),
    ))
}

/// Tests: the differences counted for `bind` since the last line or test
/// take, and the first difference logged's count; resets them.
#[cfg(test)]
pub(crate) fn take_for_test(bind: Bind) -> [u64; 4] {
    let c = COUNTS[bind as usize].each_ref().map(|n| n.swap(0, Relaxed));
    [
        c[N::Checked as usize],
        c[N::Equal as usize],
        c[N::Different as usize],
        c[N::Copy as usize],
    ]
}

/// Tests: forget the frame's checked bindings, so a binding is checked
/// again.
#[cfg(test)]
pub(crate) fn forget_seen() {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).1.clear();
}

/// Tests: the last difference found, taken.
#[cfg(test)]
pub(crate) fn take_last() -> Option<(Bind, bool, Found)> {
    LAST.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Tests: stale checks since the last take, and those that differed.
#[cfg(test)]
pub(crate) fn take_stale(bind: Bind) -> [u64; 2] {
    let c = &COUNTS[bind as usize];
    [
        c[N::Stale as usize].swap(0, Relaxed),
        c[N::StaleDiffer as usize].swap(0, Relaxed),
    ]
}

/// Tests: the view kept beside `copy`, if any.
#[cfg(test)]
pub(crate) fn view_of(copy: &Shadow) -> Option<Arc<Shadow>> {
    copy.verify.held().view.clone()
}
