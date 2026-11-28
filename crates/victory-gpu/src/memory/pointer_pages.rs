//! Title memory shaders reach only through pointers gets copies
//! (`VICTORY_POINTER_PAGES=1`, off by default; `notes/rt-builder.md`,
//! section 7).
//!
//! The page table names the host copies that draws and dispatches made
//! through views (`crate::shadow`). Memory reached only by an address a
//! shader computes, such as the nodes of the ray-tracing structures, which
//! the builder and the tracers follow by pointer, has no copy: it reads 0
//! and takes no writes. Every ray then finds nothing. And as each such
//! read is a miss, every frame withholds what shaders wrote
//! (`super::withholding`). In the runs of 27 November with the builder on,
//! 5 to 11 pages and one or two whole 256 MiB regions a frame missed.
//!
//! Here each missed page that is committed title memory gets a copy
//! ([`crate::shadow::find`]: the title buffer holding it, else a loose
//! window), so the next submission's page table names it. A region with
//! no block is known by its number only (the shader marks the region
//! where it has no page entries): the first committed memory in it gets a
//! copy, so the region has a block, and later misses in it name their
//! pages. A page that is not committed (a stale or garbage pointer) stays
//! missing, as before.
//!
//! The idea is fxceless/victory-fork 578d459c and 9a9091d1 ("copies shaders
//! reached through pointers", from fault notes) and DexNom/VIctory
//! bc3cd4b0; their title memory has another layout, so the code is this
//! line's. As avatarmanz/VIctory-7900XTX b0ba5faa found, a bound on such
//! pages (2,048 there) is met within a minute of ray tracing: there is
//! none here.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// `VICTORY_POINTER_PAGES=1` (or `2`, [`null_page_free`]): missed pages
/// get copies. Unset or anything else: off.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(level(), 1..=3);
        if on {
            tracing::info!(
                "VICTORY_POINTER_PAGES=1: title memory shaders reach by address with no copy gets one from the next submission on"
            );
        }
        on
    })
}

/// `VICTORY_POINTER_PAGES`'s value: 0 (off), 1 or 2.
fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_POINTER_PAGES").as_deref().map(str::trim) {
        Ok("1") => 1,
        Ok("2") => 2,
        Ok("3") => 3,
        _ => 0,
    })
}

/// `VICTORY_POINTER_PAGES=2`: also, a read of the null page (an address
/// below 64 KiB, which no copy can hold) is no miss: the list's writes are
/// not withheld for it.
///
/// Every frame some list reads the null page (2 to 4 pages a frame in the
/// runs of 27 November, builder on or off), so every frame withholds what
/// shaders wrote (`super::withholding`): nothing a shader writes goes back
/// into title memory, and the ranges are taken back from title memory at
/// the frame's end, or right after a submission with such a list. So no
/// structure the builder makes outlasts its frame, or its submission. The
/// null page holds nothing on the console either (guess: such reads are the
/// lanes of empty slots, which read zeros there too). Its writes are still
/// dropped.
///
/// `VICTORY_NULL_PAGE=1` (unset or `0`: off) turns on this alone, without
/// the copies of `=1`. Upstream VIctory-macOS PR #634 (7a489245) makes the
/// same exemption, on by default there (`VICTORY_NULL_PAGE=0` turns it off).
/// The frame line then counts the reads let pass ([`note_null_read`]) and the
/// MB of views that went back into title memory only because of it
/// ([`note_spared`]).
pub(crate) fn null_page_free() -> bool {
    #[cfg(test)]
    if NULL_FREE_HERE.with(std::cell::Cell::get) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let alone = crate::switch::on("VICTORY_NULL_PAGE");
        let on = level() >= 2 || alone;
        if on {
            let by = if alone { "VICTORY_NULL_PAGE=1" } else { "VICTORY_POINTER_PAGES=2" };
            tracing::info!("{by}: reads of the null page are no miss (the list's writes are not withheld for them)");
        }
        on
    })
}

/// Null-page reads [`null_page_free`] let pass since the last
/// [`take_null`] (the lists' own count, after their run).
static NULL_READS: AtomicU64 = AtomicU64::new(0);

/// Whether a list's miss check let a null-page read pass since the last
/// [`take_spared`] (checks run one submission at a time).
static SPARED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bytes of views that went back into title memory since the last
/// [`take_null`] that would have been kept without [`null_page_free`].
static SPARED_BYTES: AtomicU64 = AtomicU64::new(0);

/// A list read the null page and [`null_page_free`] let it pass.
pub(crate) fn note_null_read() {
    NULL_READS.fetch_add(1, Ordering::Relaxed);
}

/// A miss check let a null-page read pass (`super::has_missed`).
pub(crate) fn note_spared_check() {
    #[cfg(test)]
    SPARED_HERE.with(|noted| noted.set(true));
    SPARED.store(true, Ordering::Relaxed);
}

/// Whether a miss check let a null-page read pass since the last call.
pub(crate) fn take_spared() -> bool {
    SPARED.swap(false, Ordering::Relaxed)
}

/// `bytes` of views went back into title memory that the frame would have
/// kept without [`null_page_free`] (`crate::host`).
pub(crate) fn note_spared(bytes: u64) {
    SPARED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Null-page reads let pass, and bytes of views no longer kept for them,
/// since the last call.
pub(crate) fn take_null() -> (u64, u64) {
    (NULL_READS.swap(0, Ordering::Relaxed), SPARED_BYTES.swap(0, Ordering::Relaxed))
}

/// `VICTORY_POINTER_PAGES=3` (includes 2): also, a read of memory no one
/// allocated (`MEM_FREE` in the host's memory map: a page, or a whole
/// 256 MiB region with no block) is no miss.
///
/// Run rtk (27 November 19:44:24) read `0x7150000000`, `0x7160000000`
/// (whole regions) and `0x70df200000` every frame, all free: addresses the
/// title never allocated (stray pointers, or ones derived from data the
/// host does not have). Each such read made every frame withhold what
/// shaders wrote (112 MB a frame), so the BVH builder's structures still
/// never outlasted their submission or frame. Such memory has nothing to
/// copy, and the console would read nothing there either (guess). Its
/// writes are still dropped.
pub(crate) fn unallocated_free() -> bool {
    #[cfg(test)]
    if UNALLOCATED_FREE.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = level() == 3;
        if on {
            tracing::info!("VICTORY_POINTER_PAGES=3: reads of memory no one allocated are no miss (the list's writes are not withheld for them)");
        }
        on
    })
}

/// Tests: [`unallocated_free`] whatever the variable says.
#[cfg(test)]
pub(crate) static UNALLOCATED_FREE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Misses [`never_allocated`] let pass since the last [`take_unallocated`].
static UNALLOCATED: AtomicU64 = AtomicU64::new(0);

/// Frames a verdict of [`never_allocated`] is kept before the memory map is
/// asked again (the title may allocate there later).
const VERDICT_FRAMES: u64 = 60;

/// Whether `missed` lies in memory no one allocated: a page whose memory
/// map entry is free, or a region free from its start to its end. Asked of
/// the memory map once per address every [`VERDICT_FRAMES`] frames, and
/// counted when it is (the list's own count, after its run).
pub(crate) fn never_allocated(missed: Missed) -> bool {
    let free = is_never_allocated(missed);
    if free {
        UNALLOCATED.fetch_add(1, Ordering::Relaxed);
    }
    free
}

/// [`never_allocated`], not counted (whether a submission missed).
pub(crate) fn is_never_allocated(missed: Missed) -> bool {
    static VERDICTS: std::sync::Mutex<std::collections::BTreeMap<u64, (u64, bool)>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let (at, span) = match missed {
        Missed::Page(a) => (a & !(PAGE - 1), PAGE),
        Missed::Region(a) => (a & !(REGION - 1), REGION),
    };
    if at < LOWEST {
        return false;
    }
    let now = crate::frame::current();
    let free = {
        let mut verdicts = VERDICTS.lock().unwrap_or_else(|e| e.into_inner());
        match verdicts.get(&at) {
            Some(&(when, free)) if now.saturating_sub(when) < VERDICT_FRAMES => free,
            _ => {
                let free = free_span(at, span);
                verdicts.insert(at, (now, free));
                free
            }
        }
    };
    free
}

/// Whether `[at, at + span)` is free memory in the host's memory map, all
/// of it.
fn free_span(at: u64, span: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return false;
    }
    info.State == MEM_FREE && info.BaseAddress as u64 + info.RegionSize as u64 >= at + span
}

/// Misses let pass as never allocated since the last call.
pub(crate) fn take_unallocated() -> u64 {
    UNALLOCATED.swap(0, Ordering::Relaxed)
}

/// Whether page-table entry `entry` (of `blocks`, the level-1 index of each
/// block) is the null page: region 0's first page.
pub(crate) fn is_null_page(blocks: &[usize], entry: usize, block_len: usize) -> bool {
    let block = entry / block_len;
    block != 0 && blocks.get(block) == Some(&0) && entry % block_len == 0
}

#[cfg(test)]
thread_local! {
    /// [`null_page_free`] holds on this thread ([`NullFreeHere`]).
    static NULL_FREE_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Whether a miss check of this thread let a null-page read pass since
    /// its last [`take_spared_here`].
    static SPARED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: while it lives, [`null_page_free`] holds on this thread, whatever
/// the variables say (other threads' tests keep the default).
#[cfg(test)]
pub(crate) struct NullFreeHere;

#[cfg(test)]
impl NullFreeHere {
    pub(crate) fn new() -> Self {
        NULL_FREE_HERE.with(|on| on.set(true));
        NullFreeHere
    }
}

#[cfg(test)]
impl Drop for NullFreeHere {
    fn drop(&mut self) {
        NULL_FREE_HERE.with(|on| on.set(false));
    }
}

/// Tests: whether a miss check of this thread let a null-page read pass
/// since its last call. [`take_spared`] tells of every thread's checks, and
/// other tests' submissions call it at any time (`super::note_submission`).
#[cfg(test)]
pub(crate) fn take_spared_here() -> bool {
    SPARED_HERE.with(std::cell::Cell::take)
}

/// Tests: [`on`] whatever the variable says.
#[cfg(test)]
pub(crate) static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bytes of a page of the page table.
const PAGE: u64 = 1 << 16;
/// Bytes of a region of the page table's level 1.
const REGION: u64 = 1 << 28;
/// Below this nothing is mapped (the null page).
const LOWEST: u64 = 0x10000;

/// Copies made for missed pages and regions, and misses left (not
/// committed), since the last [`take`].
static MADE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The copies [`learn`] made (start, length), and their bytes in all.
static HELD: std::sync::Mutex<(std::collections::BTreeSet<(u64, u64)>, u64)> =
    std::sync::Mutex::new((std::collections::BTreeSet::new(), 0));

/// Misses not given a copy because the copies made so far fill the budget
/// ([`budget`]), since the last [`take_over`].
static OVER: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_POINTER_PAGES_MB`: how many MB of copies pointer misses may
/// make in all (default 4,096); past it a miss gets no copy and reads 0, as
/// without the switch.
///
/// After fxceless/victory-fork 3e573d25, whose pages from faults are a
/// budget of 512 MB held at a time, recycled oldest first. That recycling
/// is not taken: there a page made from a fault is that line's alone,
/// while here a miss makes or reuses an ordinary copy (its title buffer
/// or a 4 MiB loose window) that views take too. Giving one back while a
/// view holds it would leave two copies of the same memory, each with its
/// own writes. So this is a bound for the run, high enough that the cell
/// and the boat stay well under it (about 300 MB in run pp2, guess from
/// its copy sizes), and the log says when it is met.
fn budget() -> u64 {
    static MB: OnceLock<u64> = OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_POINTER_PAGES_MB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4096)
    }) << 20
}

/// Whether a copy of `len` bytes at `start` that pointer misses made fits
/// the budget (`budget` bytes, `held` so far); counts it if it does. A copy
/// counted before is free.
fn admit(held: &mut (std::collections::BTreeSet<(u64, u64)>, u64), start: u64, len: u64, budget: u64) -> bool {
    if held.0.contains(&(start, len)) {
        return true;
    }
    if held.1 + len > budget {
        return false;
    }
    held.0.insert((start, len));
    held.1 += len;
    true
}

/// What a missed address is to [`learn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Missed {
    /// A page with no copy: its address.
    Page(u64),
    /// A 256 MiB region with no block: its first address.
    Region(u64),
}

/// The address to copy for a missed page or region, if any: the page
/// itself, or the first committed byte of the region (found by
/// `first_committed` over `[start, end)`); none below the null page's end.
pub(crate) fn target(missed: Missed, first_committed: impl Fn(u64, u64) -> Option<u64>) -> Option<u64> {
    match missed {
        Missed::Page(at) if at >= LOWEST => Some(at & !(PAGE - 1)),
        Missed::Page(_) => None,
        Missed::Region(at) => {
            let start = (at & !(REGION - 1)).max(LOWEST);
            first_committed(start, (at & !(REGION - 1)) + REGION)
        }
    }
}

/// After a list ran: copies for the pages and regions it missed (each once
/// per list; the copies' own lookups keep repeats cheap).
pub(crate) fn learn(missed: &[Missed]) {
    for &m in missed {
        let page = matches!(m, Missed::Page(_));
        let made = target(m, first_committed).filter(|&at| crate::texture::committed(at as *const u8, 1));
        // A copy that already holds it costs nothing; a new one counts
        // against the budget. The window or buffer a new copy would take is
        // known only once made: a copy past the budget is made, and stays
        // (views may take it), but nothing more is made after it.
        // mac: a view of a piece of a reservation (`VICTORY_ALIAS_PHYSICAL=gpu`)
        // copies nothing, and is no part of the budget.
        let view = made.is_some_and(|at| crate::alias_physical::piece_at(at as usize).is_some());
        if made.is_some() && !view {
            let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
            if held.1 >= budget() {
                drop(held);
                OVER.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        }
        let copy = made.and_then(|at| crate::shadow::find(at, 1));
        if let Some(c) = copy.as_ref().filter(|c| !c.is_view()) {
            let (start, len) = c.range();
            let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
            if !admit(&mut held, start, len, budget()) {
                // Over the budget with this one: it counts, and is the last.
                held.0.insert((start, len));
                held.1 += len;
                static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        held_mb = held.1 >> 20,
                        budget_mb = budget() >> 20,
                        "VICTORY_POINTER_PAGES: the copies pointer misses made fill the budget (VICTORY_POINTER_PAGES_MB); later misses get none"
                    );
                }
            }
        }
        match copy {
            Some(_) => {
                MADE[usize::from(!page)].fetch_add(1, Ordering::Relaxed);
            }
            None => {
                MADE[2].fetch_add(1, Ordering::Relaxed);
                describe_once(m);
            }
        }
    }
}

/// What the host's memory map holds at a miss left without a copy, logged
/// once per 256 MiB region (up to 32): its state, type and allocation at
/// the address, and the first allocation in the region if any. Runs of 27
/// November missed the regions at 0x7150000000 and 0x7160000000 every
/// frame with nothing committed in them.
fn describe_once(missed: Missed) {
    static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
    let at = match missed {
        Missed::Page(a) | Missed::Region(a) => a,
    };
    let region = at & !(REGION - 1);
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.contains(&region) || seen.len() >= 32 {
            return;
        }
        seen.push(region);
    }
    let here = query(at.max(LOWEST));
    let first = first_allocated(region.max(LOWEST), region + REGION).map(query);
    tracing::info!(
        address = format_args!("{at:#x}"),
        page = matches!(missed, Missed::Page(_)),
        ?here,
        ?first,
        "title memory reached by a pointer has no committed memory to copy (VICTORY_POINTER_PAGES)"
    );
}

/// The host's memory map at `at`: (state, type, protection, allocation
/// base, region base, region size), as hex text.
fn query(at: u64) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return "none".into();
    }
    format!(
        "state {:#x} type {:#x} protect {:#x} allocation {:#x} region {:#x}+{:#x}",
        info.State, info.Type, info.Protect, info.AllocationBase as u64, info.BaseAddress as u64, info.RegionSize
    )
}

/// The first address in `[start, end)` that is reserved or committed.
fn first_allocated(start: u64, end: u64) -> Option<u64> {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = start;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) } == 0 {
            return None;
        }
        if info.State != MEM_FREE {
            return Some(p.max(info.BaseAddress as u64));
        }
        let next = info.BaseAddress as u64 + info.RegionSize as u64;
        if next <= p {
            return None;
        }
        p = next;
    }
    None
}

/// Copies asked for since the last call: (for pages, for regions, left
/// missing).
pub(crate) fn take() -> [u64; 3] {
    MADE.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}

/// Misses refused for the budget since the last call, and the MB the
/// copies pointer misses made hold in all.
pub(crate) fn take_over() -> (u64, u64) {
    let held = HELD.lock().unwrap_or_else(|e| e.into_inner()).1;
    (OVER.swap(0, Ordering::Relaxed), held >> 20)
}

/// The first committed, readable address in `[start, end)`, by the
/// host's memory map.
fn first_committed(start: u64, end: u64) -> Option<u64> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = start;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return None;
        }
        let readable = info.Protect != 0 && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0;
        if info.State == MEM_COMMIT && readable {
            return Some(p.max(info.BaseAddress as u64));
        }
        let next = info.BaseAddress as u64 + info.RegionSize as u64;
        if next <= p {
            return None;
        }
        p = next;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missed page is copied at its page; one in the null page is not; a
    /// region is copied at its first committed byte, looked for from the
    /// region's start (past the null page in region 0) to its end.
    #[test]
    fn missed_pages_and_regions_name_what_to_copy() {
        let none = |_: u64, _: u64| None;
        assert_eq!(target(Missed::Page(0x9b2c4_1234), none), Some(0x9b2c4_0000));
        assert_eq!(target(Missed::Page(0x8000), none), None, "the null page");
        assert_eq!(target(Missed::Page(0), none), None);
        let asked = std::cell::Cell::new((0, 0));
        let at = |s: u64, e: u64| {
            asked.set((s, e));
            Some(s + 0x40_0000)
        };
        assert_eq!(target(Missed::Region(0x71_5000_0000), at), Some(0x71_5040_0000));
        assert_eq!(asked.get(), (0x71_5000_0000, 0x71_6000_0000));
        assert_eq!(target(Missed::Region(0), at), Some(0x10000 + 0x40_0000));
        assert_eq!(asked.get(), (0x10000, 0x1000_0000));
        assert_eq!(target(Missed::Region(0x71_5000_0000), none), None, "nothing committed");
    }

    /// The null page is region 0's first page, in a block of its own; not
    /// block 0 (regions with no block), nor another page or region.
    #[test]
    fn the_null_page_is_region_0s_first_page() {
        let blocks = [usize::MAX, 0, 7];
        let n = 4096;
        assert!(is_null_page(&blocks, n, n));
        assert!(!is_null_page(&blocks, n + 1, n), "the second page");
        assert!(!is_null_page(&blocks, 2 * n, n), "region 7");
        assert!(!is_null_page(&blocks, 0, n), "block 0");
        assert!(!is_null_page(&blocks, 3 * n, n), "past the blocks");
    }

    /// The budget admits new copies until they would pass it; one counted
    /// before is free; the count is the copies' bytes.
    #[test]
    fn the_budget_admits_copies_until_it_is_full() {
        let mut held = Default::default();
        assert!(admit(&mut held, 0x1000_0000, 4 << 20, 10 << 20));
        assert!(admit(&mut held, 0x2000_0000, 4 << 20, 10 << 20));
        assert!(admit(&mut held, 0x1000_0000, 4 << 20, 10 << 20), "counted before");
        assert!(!admit(&mut held, 0x3000_0000, 4 << 20, 10 << 20), "past the budget");
        assert_eq!(held.1, 8 << 20);
        assert!(admit(&mut held, 0x3000_0000, 2 << 20, 10 << 20), "a smaller one fits");
    }

    /// Memory no one allocated is told from memory that is: a free
    /// region far from any allocation, against a live allocation and the
    /// null page.
    #[test]
    fn memory_no_one_allocated_is_told_apart() {
        let held = vec![1u8; 1 << 16];
        let at = held.as_ptr() as u64;
        assert!(!free_span(at & !(PAGE - 1), PAGE), "a live allocation");
        assert!(!never_allocated(Missed::Page(0)), "the null page");
        // An address just below 2^47, which nothing maps.
        let far = (1u64 << 47) - 2 * REGION;
        assert!(free_span(far, REGION));
        assert!(never_allocated(Missed::Region(far)));
        assert!(never_allocated(Missed::Page(far + 3 * PAGE)));
    }

    /// The host's memory map: a committed allocation is found from below
    /// it, and nothing past it.
    #[test]
    fn the_first_committed_byte_is_found_by_the_memory_map() {
        let held = vec![1u8; 3 << 16];
        let at = held.as_ptr() as u64;
        assert_eq!(first_committed(at, at + 1), Some(at));
        let page = at.next_multiple_of(1 << 16);
        assert_eq!(first_committed(page, page + (1 << 16)), Some(page));
    }
}
