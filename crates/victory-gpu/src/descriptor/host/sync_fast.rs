//! mac: multi-page parallel heap sync with staged CPU descriptors
//! (`VICTORY_HEAP_SYNC_FAST=1`, `crate::mac_switches::heap_sync_fast`).
//!
//! Kept in its own file so upstream changes to `descriptor/host.rs` and
//! `descriptor/host/quick.rs` apply cleanly.
//!
//! What changes when `VICTORY_HEAP_SYNC_FAST=1` is on:
//! 1. Dirty 64 KiB heap pages of a submission are scanned together instead of
//!    one page at a time, skipping empty slots (`tag != VIEW_TAG` with
//!    `made == 0`) and already-verified slots (`block_forgot == views_forgot()`
//!    with `made == key`) without taking `VIEWS.read()` or allocating per page.
//! 2. Buffer SRVs and buffer UAVs are resolved into a non-shader-visible CPU
//!    staging `ID3D12DescriptorHeap` (cached across passes via `Weak<Shadow>`
//!    and `shadow::lookup_epoch()`) and copied into the shader-visible heap via
//!    `CopyDescriptorsSimple` (~0.01 us/view on D3DMetal vs ~0.12-0.41 us for
//!    `CreateShaderResourceView`), with slot 0 holding `null_view` for cleared
//!    records. Workers read `StagedState` directly without per-pass cloning.
//!    A cached view found at another `LOOKUP_EPOCH` (any buffer made or
//!    released since) is looked up again and kept if the same copy comes
//!    back. Found views are checked under one shared hold of `VIEWS` before
//!    anything of them is used, and only views still current there have
//!    their resources pinned (`try_add_ref`), as `view_at` pins; the others
//!    are misses. A miss drops the view's old entry before its slot is
//!    claimed again.
//! 3. `descriptor_as_data` address words (`[u32; 4]`) are precomputed once per
//!    cached record instead of per slot.
//! 4. Ring-heap overwrites whose `kept` slot already holds the exact same
//!    `(ID3D12Resource, None)` skip COM `AddRef`/`Release` and global
//!    `RETIRED` lock traffic; actual retirements flush in batches via
//!    `crate::host::retire_many`.
//! 5. The census goes in place (`bindless::Inner::enter_batch`), chosen by
//!    the pass for every batch its workers make, whatever the switch reads
//!    meanwhile (`crate::ab`).
//!
//! The frame line's "heap sync fast" line ([`log`]) counts, with the switch
//! or its A/B: heap sync time on either path, slots looked at, synced and
//! skipped, and staged views found, looked up again, missed and dropped.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::{
    HeldSource, HostHeap, PUBLISHING, SYNC_BLOCK_SLOTS, address_words, buffer_element, element_size,
    in_buffer, null_view, srv_desc, typed_buffer, uav_desc, under_wine, view_kind, view_stride,
    widened_format, widened_w,
};
use crate::bindless::Seen;
use crate::census_maps::Hashing;
use crate::descriptor::pool::Pool;
use crate::descriptor::{
    VIEW_TAG, View, ViewPin, inline, view_at, views_current_into, views_forgot, views_read,
};
use crate::host::{Host, host, retire, retire_many};
use crate::object::This;
use crate::shadow::{Shadow, Widened};

/// Total slots in the CPU-only staging descriptor heap per `HostHeap`
/// (`32,768 * 32 B = 1 MiB`). Slot 0 is reserved for `null_view`; slots
/// `1..STAGED_CAPACITY` hold cached buffer SRV and UAV descriptors.
const STAGED_CAPACITY: u32 = 32_768;

/// Minimum pages before parallelizing the slot scan across `Pool` workers
/// (16 pages = 32,768 slots = 1 MiB; smaller scans finish in ~15 us on the
/// calling thread without a Wine Condvar round-trip).
const PARALLEL_SCAN_PAGES: usize = 16;

/// Minimum cold misses in one pass before staging descriptors in parallel.
const PARALLEL_STAGE_MIN: usize = 64;

/// Maximum retired `ID3D12Resource` handles buffered per worker before flushing.
const RETIRE_BATCH: usize = 256;

/// Since the last frame line ([`log`]): fast passes and their nanoseconds;
/// slots they looked at, passed by as verified (`block_forgot`, `made`)
/// and as empty, and made (stale); staged views found, missed, and found
/// but dropped under `VIEWS` (released since their lookup), and of those
/// found the ones looked up again at a new `LOOKUP_EPOCH`; waits for a
/// heap's `staged_fast` and their nanoseconds; heap syncs' nanoseconds
/// before submissions and ahead of them (both paths, [`Timed`]). Log only.
static COUNTS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
const PASSES: usize = 0;
const PASS_NANOS: usize = 1;
const LOOKED: usize = 2;
const VERIFIED: usize = 3;
const EMPTY: usize = 4;
const SYNCED: usize = 5;
const HITS: usize = 6;
const MISSES: usize = 7;
const DROPPED: usize = 8;
const WAITS: usize = 9;
const WAIT_NANOS: usize = 10;
const SYNC_NANOS: usize = 11;
const AHEAD_NANOS: usize = 12;
const REVALIDATED: usize = 13;

#[inline]
fn count(at: usize, n: u64) {
    if n != 0 {
        COUNTS[at].fetch_add(n, Relaxed);
    }
}

/// A heap's staging state; a wait for it (a pass on the same heap by the
/// other syncing thread) is counted.
fn lock_staged(staged: &Mutex<Option<StagedState>>) -> MutexGuard<'_, Option<StagedState>> {
    match staged.try_lock() {
        Ok(state) => state,
        Err(TryLockError::Poisoned(e)) => e.into_inner(),
        Err(TryLockError::WouldBlock) => {
            let began = std::time::Instant::now();
            let state = staged.lock().unwrap_or_else(|e| e.into_inner());
            count(WAITS, 1);
            count(WAIT_NANOS, began.elapsed().as_nanos() as u64);
            state
        }
    }
}

/// mac: the time of one heap sync (`crate::descriptor::sync`), whichever
/// path it takes, so that both sides of `VICTORY_AB=VICTORY_HEAP_SYNC_FAST`
/// show it ([`log`]); `ahead`: a round while the GPU runs.
pub(crate) struct Timed {
    began: std::time::Instant,
    ahead: bool,
}

impl Timed {
    pub(crate) fn start(ahead: bool) -> Self {
        Timed { began: std::time::Instant::now(), ahead }
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        let at = if self.ahead { AHEAD_NANOS } else { SYNC_NANOS };
        count(at, self.began.elapsed().as_nanos() as u64);
    }
}

/// With the frame line (`frames` since the last one), while
/// `VICTORY_HEAP_SYNC_FAST=1` is on or `VICTORY_AB` flips it: the heap
/// syncs' time on either path, and the fast passes' slots, staged views and
/// waits. Log only.
pub(crate) fn log(frames: f64) {
    let n: [f64; 14] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
    let on = crate::mac_switches::heap_sync_fast();
    if !on && !crate::ab::names("VICTORY_HEAP_SYNC_FAST") {
        return;
    }
    tracing::info!(
        "heap sync fast (VICTORY_HEAP_SYNC_FAST={}): heap syncs {:.2} ms/frame before submissions and {:.2} ms \
         ahead; fast passes {:.1}/frame in {:.2} ms: {:.0} slots/frame looked at, {:.0} synced, {:.0} skipped as \
         verified, {:.0} skipped as empty; staged views {:.0} found ({:.0} of them looked up again), {:.0} missed, \
         {:.0} found but released since; waits for a heap's staged views {:.1} ({:.2} ms)",
        u8::from(on),
        n[SYNC_NANOS] / 1e6,
        n[AHEAD_NANOS] / 1e6,
        n[PASSES],
        n[PASS_NANOS] / 1e6,
        n[LOOKED],
        n[SYNCED],
        n[VERIFIED],
        n[EMPTY],
        n[HITS],
        n[REVALIDATED],
        n[MISSES],
        n[DROPPED],
        n[WAITS],
        n[WAIT_NANOS] / 1e6,
    );
}

type CandidateScratch = (Vec<(u32, [u32; 3])>, Vec<bool>);

thread_local! {
    static SCAN_SCRATCH: Cell<CandidateScratch> = const { Cell::new((Vec::new(), Vec::new())) };
}

fn with_scan_scratch<R>(f: impl FnOnce(&mut Vec<(u32, [u32; 3])>, &mut Vec<bool>) -> R) -> R {
    let (mut candidates, mut current) = SCAN_SCRATCH.take();
    candidates.clear();
    current.clear();
    let out = f(&mut candidates, &mut current);
    candidates.clear();
    current.clear();
    SCAN_SCRATCH.set((candidates, current));
    out
}

/// Releases pinned `Resource` refcounts outside all heap locks at pass end.
struct RawPins(Vec<usize>);

impl Drop for RawPins {
    fn drop(&mut self) {
        for r in self.0.drain(..) {
            crate::object::release_ref(r as This);
        }
    }
}

#[derive(Clone)]
enum DescSpec {
    Srv(D3D12_SHADER_RESOURCE_VIEW_DESC),
    Uav(D3D12_UNORDERED_ACCESS_VIEW_DESC),
}

/// Persistent cached metadata for a staged buffer SRV/UAV (`own == false`,
/// `widened == None`). Holds only a `Weak<Shadow>`, never extending the
/// lifetime of any `Shadow` or `ID3D12Resource`.
struct CachedView {
    record: [u32; 3],
    slot: u32,
    raw_buffer: usize,
    shadow: Weak<Shadow>,
    /// `crate::shadow::lookup_epoch` before the lookup it was resolved (or
    /// last looked up again) from; a hit at another epoch looks it up again.
    lookup_epoch: u64,
    /// The `crate::shadow::find` it was resolved from (address, bytes), and
    /// whether that lookup alone decides it ([`ResolvedFresh::revalidate`]).
    lookup: (u64, u64),
    revalidate: bool,
    descriptor_as_data: bool,
    stride: u32,
    mark: Option<(u32, u32)>,
    addr_words: [u32; 4],
    seen_addr: u64,
    seen_len: u64,
    seen_uav: bool,
}

/// Non-persistent pass entry for `own == true`, `widened.is_some()`, or staging
/// capacity overflow (rare; empty on 99.9% of passes).
struct OverflowEntry {
    record: [u32; 3],
    staged_cpu: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    fallback_spec: Option<DescSpec>,
    buffer: ID3D12Resource,
    raw_buffer: usize,
    shadow: Arc<Shadow>,
    own: bool,
    widened: Option<Arc<Widened>>,
    stride: u32,
    mark: Option<(u32, u32)>,
    addr_words: [u32; 4],
    seen_addr: u64,
    seen_len: u64,
    seen_uav: bool,
}

impl OverflowEntry {
    #[inline]
    fn held(&self) -> Option<HeldSource> {
        match &self.widened {
            Some(w) => Some(HeldSource::Widened(w.clone(), self.shadow.clone())),
            None => self.own.then(|| HeldSource::Shadow(self.shadow.clone())),
        }
    }
}

pub(super) struct StagedState {
    #[allow(dead_code)] // keeps the D3D12 CPU descriptor heap alive
    heap: ID3D12DescriptorHeap,
    cpu_start: usize,
    step: usize,
    pass_id: u64,
    clock: u32,
    slot_owner: Box<[u64]>,
    slot_pass: Box<[u64]>,
    /// Index into `pass_shadows` for each staging slot active in `pass_id`.
    slot_shadow_idx: Box<[u32]>,
    /// Upgraded `Arc<Shadow>`s kept alive for the duration of the current pass.
    pass_shadows: Vec<Arc<Shadow>>,
    by_key: HashMap<u64, CachedView, Hashing>,
}

impl StagedState {
    fn new(host: &Host) -> Option<Self> {
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain CPU-only descriptor heap creation.
        let (heap, cpu_start, step) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: STAGED_CAPACITY,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .ok()?;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = host.device.GetDescriptorHandleIncrementSize(kind) as usize;
            (heap, start, step)
        };
        let state = StagedState {
            heap,
            cpu_start,
            step,
            pass_id: 0,
            clock: 1,
            slot_owner: vec![0u64; STAGED_CAPACITY as usize].into_boxed_slice(),
            slot_pass: vec![0u64; STAGED_CAPACITY as usize].into_boxed_slice(),
            slot_shadow_idx: vec![0u32; STAGED_CAPACITY as usize].into_boxed_slice(),
            pass_shadows: Vec::with_capacity(256),
            by_key: HashMap::with_capacity_and_hasher(4096, Hashing::new(true)),
        };
        // Slot 0 permanently holds `null_view` for cleared records.
        null_view(state.cpu(0));
        Some(state)
    }

    #[inline]
    fn cpu(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu_start + slot as usize * self.step,
        }
    }

    /// Claims a staging slot in `1..STAGED_CAPACITY` not yet used in `pass_id`,
    /// evicting the oldest unreferenced slot via clock hand. The caller has
    /// dropped `key`'s own entry ([`StagedState::drop_key`]).
    fn alloc_slot(&mut self, key: u64) -> Option<u32> {
        let cap = STAGED_CAPACITY - 1;
        for _ in 0..cap {
            let s = self.clock;
            self.clock = if s + 1 >= STAGED_CAPACITY { 1 } else { s + 1 };
            if self.slot_pass[s as usize] == self.pass_id {
                continue;
            }
            // The entry staged here goes, only if it is the one here.
            let prev = self.slot_owner[s as usize];
            if prev != 0 && self.by_key.get(&prev).is_some_and(|c| c.slot == s) {
                self.by_key.remove(&prev);
            }
            self.slot_owner[s as usize] = key;
            self.slot_pass[s as usize] = self.pass_id;
            return Some(s);
        }
        None
    }

    /// A miss for `key`: its entry goes, and its staging slot is free again
    /// unless this pass took it already.
    fn drop_key(&mut self, key: u64) {
        if let Some(old) = self.by_key.remove(&key) {
            let s = old.slot as usize;
            if self.slot_owner[s] == key && self.slot_pass[s] != self.pass_id {
                self.slot_owner[s] = 0;
            }
        }
    }

    /// A hit found no longer current under `VIEWS`: its entry and its slot
    /// go, so that the apply step finds neither.
    fn evict(&mut self, key: u64) {
        if let Some(old) = self.by_key.remove(&key) {
            self.slot_owner[old.slot as usize] = 0;
            self.slot_pass[old.slot as usize] = 0;
        }
    }
}

struct ResolvedFresh {
    /// The `crate::shadow::find` (address, bytes) it was resolved from.
    lookup: (u64, u64),
    /// Whether the same copy from that lookup gives the same descriptor,
    /// mark and census range: every view staged here but a shader view in a
    /// format the host widens (whose widened copy is made or not apart from
    /// the lookup, `Shadow::widened`).
    revalidate: bool,
    spec: DescSpec,
    buffer: ID3D12Resource,
    raw_buffer: usize,
    shadow: Arc<Shadow>,
    own: bool,
    widened: Option<Arc<Widened>>,
    stride: u32,
    mark: Option<(u32, u32)>,
    addr_words: [u32; 4],
    seen_addr: u64,
    seen_len: u64,
    seen_uav: bool,
}

/// Resolves buffer SRVs (`Constant`, `Shader` buffer, `Placed(false, ..)`
/// buffer) and standard buffer UAVs (`Unordered` buffer, `Placed(true, ..)`
/// buffer). Returns `None` for textures, samplers, 3-channel typed UAVs, or
/// views the host refuses so `HostHeap::mirror` handles them identically.
fn resolve_fast(view: &View, dad: bool) -> Option<ResolvedFresh> {
    match view {
        &View::Constant(address, size) => {
            let len = u64::from(size);
            if size == 0 || !size.is_multiple_of(4) {
                return None;
            }
            let shadow = crate::shadow::find(address, len)?;
            if shadow.left(address) < len || !shadow.offset(address).is_multiple_of(4) {
                return None;
            }
            let first = shadow.offset(address) / 4;
            let count = size / 4;
            if !in_buffer(&shadow.buffer, first, count, 4) {
                return None;
            }
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first,
                        NumElements: count,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            let buffer = shadow.buffer.clone();
            let raw_buffer = Interface::as_raw(&buffer) as usize;
            let seen_len = len.min(shadow.left(address));
            Some(ResolvedFresh {
                lookup: (address, len),
                revalidate: true,
                spec: DescSpec::Srv(desc),
                buffer,
                raw_buffer,
                shadow,
                own: false,
                widened: None,
                stride: 4,
                mark: None,
                addr_words: address_words(Some(view), dad),
                seen_addr: address,
                seen_len,
                seen_uav: false,
            })
        }
        View::Shader(resource, Some(words)) => {
            let resource = *resource as This;
            let (data, _) = crate::resource::buffer_bytes(resource)?;
            if crate::resource::host_copy(resource).is_some() {
                return None;
            }
            resolve_buffer_srv(view, data as u64, srv_desc(words), dad)
        }
        View::Placed(false, resource, words) if resource[0] == 1 => {
            let address = u64::from(words[10]) | u64::from(words[11]) << 32;
            resolve_buffer_srv(view, address, srv_desc(words[..10].try_into().ok()?), dad)
        }
        View::Unordered(resource, _, Some(words)) => {
            let resource = *resource as This;
            if crate::resource::host_copy(resource).is_some() {
                return None;
            }
            let (data, _) = crate::resource::buffer_bytes(resource)?;
            resolve_buffer_uav(view, data as u64, uav_desc(words), dad)
        }
        View::Placed(true, resource, words) if resource[0] == 1 => {
            let address = u64::from(words[10]) | u64::from(words[11]) << 32;
            resolve_buffer_uav(view, address, uav_desc(words[..10].try_into().ok()?), dad)
        }
        _ => None,
    }
}

fn resolve_buffer_srv(
    view: &View,
    address: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    dad: bool,
) -> Option<ResolvedFresh> {
    let element = element_size(&desc);
    if element == 0 || desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: verified buffer SRV union variant.
    let (first_elem, count) = unsafe { (desc.Anonymous.Buffer.FirstElement, desc.Anonymous.Buffer.NumElements) };
    let start = address + first_elem * element;
    let len = u64::from(count) * element;
    if len == 0 {
        return None;
    }
    let (shadow, own) = crate::shadow::for_view(start, len, element)?;
    let first = shadow.offset(start) / element;
    if !in_buffer(&shadow.buffer, first, count, element) {
        return None;
    }
    let widen = (under_wine() && matches!(desc.Format.0, 6..=8))
        .then(|| widened_w(desc.Format.0))
        .flatten();
    let revalidate = widen.is_none();
    let widened = match widen {
        Some(w) => shadow.widened(start, u64::from(count) * 12, w),
        None => None,
    };
    let buffer = match &widened {
        Some(w) => {
            desc.Format = widened_format(desc.Format.0)?;
            desc.Anonymous.Buffer.FirstElement = 0;
            w.buffer.clone()
        }
        None => {
            desc.Anonymous.Buffer.FirstElement = first;
            shadow.buffer.clone()
        }
    };
    let raw_buffer = Interface::as_raw(&buffer) as usize;
    let seen_len = len.min(shadow.left(start));
    Some(ResolvedFresh {
        lookup: (start, len),
        revalidate,
        spec: DescSpec::Srv(desc),
        buffer,
        raw_buffer,
        shadow,
        own,
        widened,
        stride: view_stride(view) | view_kind(view),
        mark: None,
        addr_words: address_words(Some(view), dad),
        seen_addr: start,
        seen_len,
        seen_uav: false,
    })
}

fn resolve_buffer_uav(
    view: &View,
    address: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    dad: bool,
) -> Option<ResolvedFresh> {
    if desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: verified buffer UAV union variant.
    let buf = unsafe { desc.Anonymous.Buffer };
    let raw = buf.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buf.StructureByteStride, raw, desc.Format.0)?;
    if element == 0 {
        return None;
    }
    if typed_buffer(buf.StructureByteStride, raw) && matches!(desc.Format.0, 6..=8) {
        return None;
    }
    let start = address + buf.FirstElement * element;
    let len = u64::from(buf.NumElements) * element;
    if len == 0 {
        return None;
    }
    let (shadow, own) = crate::shadow::for_view(start, len, element)?;
    // SAFETY: live D3D12 buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        return None;
    }
    let first = shadow.offset(start) / element;
    if !in_buffer(&shadow.buffer, first, buf.NumElements, element) {
        return None;
    }
    desc.Anonymous.Buffer.FirstElement = first;
    desc.Anonymous.Buffer.CounterOffsetInBytes = 0;
    let buffer = shadow.buffer.clone();
    let raw_buffer = Interface::as_raw(&buffer) as usize;
    let mark = shadow.mark_entry(start);
    let seen_len = len.min(shadow.left(start));
    Some(ResolvedFresh {
        lookup: (start, len),
        revalidate: true,
        spec: DescSpec::Uav(desc),
        buffer,
        raw_buffer,
        shadow,
        own,
        widened: None,
        stride: view_stride(view) | view_kind(view),
        mark,
        addr_words: address_words(Some(view), dad),
        seen_addr: start,
        seen_len,
        seen_uav: true,
    })
}

#[inline]
fn write_descriptor(
    host: &Host,
    buffer: &ID3D12Resource,
    spec: &DescSpec,
    dst: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    match spec {
        DescSpec::Srv(desc) => {
            // SAFETY: live buffer and valid CPU descriptor handle.
            unsafe { host.device.CreateShaderResourceView(buffer, Some(desc), dst) };
        }
        DescSpec::Uav(desc) => {
            // SAFETY: live UAV-capable buffer and valid CPU descriptor handle.
            unsafe {
                host.device.CreateUnorderedAccessView(
                    buffer,
                    None::<&ID3D12Resource>,
                    Some(desc),
                    dst,
                )
            };
        }
    }
}

impl HostHeap {
    /// Scans `pages` (`(first_slot, last_slot)` ranges) and updates all stale
    /// slots in one combined pass (`VICTORY_HEAP_SYNC_FAST=1`).
    pub(crate) fn sync_pages_fast(&self, pages: &[(u32, u32)], start: u64, step: u64) {
        self.sync_pages_fast_with(pages, start, step, crate::descriptor::pool::get());
    }

    pub(crate) fn sync_pages_fast_with(
        &self,
        pages: &[(u32, u32)],
        start: u64,
        step: u64,
        pool: Option<&Pool>,
    ) {
        if pages.is_empty() {
            return;
        }
        let began = std::time::Instant::now();
        let forgot = views_forgot();
        let split_pool = pool.filter(|_| self.made.split());
        let stale = match split_pool {
            Some(pool) if pages.len() >= PARALLEL_SCAN_PAGES => {
                let parts = pages.len().min(pool.threads() * 2).max(2);
                let each = pages.len().div_ceil(parts);
                let out: Vec<Mutex<Vec<u32>>> = (0..parts).map(|_| Mutex::new(Vec::new())).collect();
                pool.run(parts, &|k| {
                    let lo = (k * each).min(pages.len());
                    let hi = ((k + 1) * each).min(pages.len());
                    if lo < hi {
                        let v = self.scan_pages_slice(&pages[lo..hi], start, step, forgot);
                        *out[k].lock().unwrap_or_else(|e| e.into_inner()) = v;
                    }
                });
                let total: usize = out
                    .iter()
                    .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()).len())
                    .sum();
                let mut merged = Vec::with_capacity(total);
                for m in out {
                    merged.extend(m.into_inner().unwrap_or_else(|e| e.into_inner()));
                }
                merged
            }
            _ => self.scan_pages_slice(pages, start, step, forgot),
        };

        count(SYNCED, stale.len() as u64);
        if !stale.is_empty() {
            let can_stage = !self.sampler
                && super::quick::on()
                && !crate::alias_physical::verifying()
                && host().is_some();
            if can_stage {
                self.make_stale_staged(&stale, start, step, split_pool);
            } else {
                self.make_stale(stale, start, step);
            }
        }

        // As of the pass's start: a `forget` during it leaves the blocks to
        // be looked at again.
        for &(first, last) in pages {
            self.verified(first, last.min(self.count), forgot);
        }
        count(PASSES, 1);
        count(PASS_NANOS, began.elapsed().as_nanos() as u64);
    }

    /// The mapped view addresses' host address, if they are made. Read under
    /// the slot's lock, as `note_address` reads it: `view_addresses` fills
    /// them under every slot's lock, so a slot made after that writes its
    /// own entry and one made before is in the fill.
    #[inline]
    fn address_data(&self) -> Option<usize> {
        self.addresses.get().and_then(Option::as_ref).map(|&(_, data, _)| data)
    }

    fn scan_pages_slice(
        &self,
        pages: &[(u32, u32)],
        start: u64,
        step: u64,
        forgot: u64,
    ) -> Vec<u32> {
        let mut stale = Vec::new();
        let (mut looked, mut verified_skips, mut empty_skips) = (0u64, 0u64, 0u64);
        with_scan_scratch(|candidates, current| {
            for &(first, last) in pages {
                looked += u64::from(last.saturating_sub(first));
                let mut cur = first;
                while cur < last {
                    let block_end = if cur < self.count {
                        let b = cur / SYNC_BLOCK_SLOTS;
                        ((b + 1) * SYNC_BLOCK_SLOTS).min(last).min(self.count)
                    } else {
                        last
                    };
                    let verified = cur < self.count
                        && self.block_forgot[(cur / SYNC_BLOCK_SLOTS) as usize]
                            .load(Ordering::Acquire)
                            == forgot;
                    for i in cur..block_end {
                        let addr = start + u64::from(i) * step;
                        // SAFETY: caller verified the heap page is mapped.
                        let record = unsafe { (addr as *const [u32; 3]).read_unaligned() };
                        let [tag, view, generation] = record;
                        if tag != VIEW_TAG {
                            match self.made.peek(i) {
                                Some(0) => empty_skips += 1,
                                _ => stale.push(i),
                            }
                            continue;
                        }
                        let key = u64::from(generation) << 32 | u64::from(view);
                        if verified && key != 0 && self.made.peek(i) == Some(key) {
                            verified_skips += 1;
                            continue;
                        }
                        candidates.push((i, record));
                    }
                    cur = block_end;
                }
            }
            if !candidates.is_empty() {
                let records: Vec<[u32; 3]> = candidates.iter().map(|&(_, r)| r).collect();
                views_current_into(&records, current);
                let _held = (!self.made.split()).then(|| self.made.lock(0));
                for (&(i, [_, view, generation]), &is_current) in candidates.iter().zip(current.iter()) {
                    if crate::descriptor::slots::stale(is_current, view, generation, self.made.peek(i)) {
                        stale.push(i);
                    }
                }
                stale.sort_unstable();
            }
        });
        count(LOOKED, looked);
        count(VERIFIED, verified_skips);
        count(EMPTY, empty_skips);
        stale
    }

    fn make_stale_staged(&self, stale: &[u32], start: u64, step: u64, split_pool: Option<&Pool>) {
        let Some(host) = host() else {
            return;
        };
        if crate::descriptor::ahead::on() {
            crate::descriptor::ahead::changed(stale.len());
        }
        let at = |i: u32| start + u64::from(i) * step;

        let mut guard = lock_staged(&self.staged_fast);
        if guard.is_none() {
            *guard = StagedState::new(host);
        }
        let Some(state) = guard.as_mut() else {
            drop(guard);
            self.make_stale(stale.to_vec(), start, step);
            return;
        };
        state.pass_id = state.pass_id.wrapping_add(1).max(1);
        let pass_id = state.pass_id;
        state.pass_shadows.clear();
        let null_cpu = state.cpu(0);
        let dad = crate::shader_cache::descriptor_as_data();

        // Only populated when a record misses `StagedState` as `own`/`widened`,
        // overflows `STAGED_CAPACITY`, or cannot be resolved (textures/refused).
        let mut overflow_map: HashMap<u64, Option<usize>, Hashing> =
            HashMap::with_hasher(Hashing::new(true));
        let mut overflow_entries: Vec<OverflowEntry> = Vec::new();
        let mut cold_writes: Vec<(D3D12_CPU_DESCRIPTOR_HANDLE, ID3D12Resource, DescSpec)> = Vec::new();
        // Cached views found for this pass: key, record and the index of
        // their copy in `pass_shadows`. Nothing of the view is used before
        // the check under `VIEWS` below.
        let mut hits: Vec<(u64, [u32; 3], u32)> = Vec::new();
        let mut pins: Vec<ViewPin> = Vec::new();
        let (mut misses, mut revalidated) = (0u64, 0u64);

        for &index in stale {
            #[cfg(test)]
            tests::hook(tests::Stage::Lookup(index));
            if index >= self.count {
                continue;
            }
            let handle = at(index);
            // SAFETY: caller verified the heap page is mapped.
            let record = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            let [tag, view_idx, generation] = record;
            if tag != VIEW_TAG {
                continue;
            }
            let key = u64::from(generation) << 32 | u64::from(view_idx);

            if let Some(cached) = state.by_key.get_mut(&key) {
                let s = cached.slot as usize;
                if state.slot_pass[s] == pass_id {
                    // Taken in this pass already (a hit, or made just now).
                    continue;
                }
                // The shadow tables as of this lookup, not as of the pass's
                // start. They change with every buffer the title makes or
                // releases, anywhere: an entry from another epoch is looked
                // up again, and stays only if the lookup brings the same
                // copy (its descriptor, mark and range follow from that copy
                // alone); else it is a miss.
                let epoch = crate::shadow::lookup_epoch();
                if cached.descriptor_as_data == dad
                    && let Some(shadow) = cached.shadow.upgrade()
                    && Interface::as_raw(&shadow.buffer) as usize == cached.raw_buffer
                    && (cached.lookup_epoch == epoch
                        || cached.revalidate
                            && crate::shadow::find(cached.lookup.0, cached.lookup.1)
                                .is_some_and(|found| Arc::ptr_eq(&found, &shadow)))
                {
                    if cached.lookup_epoch != epoch {
                        cached.lookup_epoch = epoch;
                        revalidated += 1;
                    }
                    let s_idx = state.pass_shadows.len() as u32;
                    state.pass_shadows.push(shadow);
                    state.slot_shadow_idx[s] = s_idx;
                    state.slot_pass[s] = pass_id;
                    hits.push((key, record, s_idx));
                    continue;
                }
            }

            if overflow_map.contains_key(&key) {
                continue;
            }

            // A miss. Its old entry goes first, whatever the record resolves
            // to now: an entry left in `by_key` whose slot this pass takes
            // again would send the apply step to it (`apply_chunk_fast`
            // looks there first) with a `slot_shadow_idx` from another pass.
            misses += 1;
            state.drop_key(key);
            // Read before the lookups: a buffer made or released during them
            // makes the new entry miss at its next lookup.
            let lookup_epoch = crate::shadow::lookup_epoch();
            let Some(view_pin) = view_at(view_idx, generation) else {
                overflow_map.insert(key, None);
                continue;
            };
            let Some(fresh) = resolve_fast(&view_pin, dad) else {
                overflow_map.insert(key, None);
                continue;
            };
            pins.push(view_pin);

            let staged_slot = state.alloc_slot(key);
            let staged_cpu = staged_slot.map(|s| state.cpu(s));
            if let Some(cpu_dst) = staged_cpu {
                cold_writes.push((cpu_dst, fresh.buffer.clone(), fresh.spec.clone()));
                if !fresh.own && fresh.widened.is_none() && let Some(slot) = staged_slot {
                    let s_idx = state.pass_shadows.len() as u32;
                    state.pass_shadows.push(fresh.shadow);
                    state.slot_shadow_idx[slot as usize] = s_idx;
                    state.slot_pass[slot as usize] = pass_id;
                    state.by_key.insert(
                        key,
                        CachedView {
                            record,
                            slot,
                            raw_buffer: fresh.raw_buffer,
                            shadow: Arc::downgrade(&state.pass_shadows[s_idx as usize]),
                            lookup_epoch,
                            lookup: fresh.lookup,
                            revalidate: fresh.revalidate,
                            descriptor_as_data: dad,
                            stride: fresh.stride,
                            mark: fresh.mark,
                            addr_words: fresh.addr_words,
                            seen_addr: fresh.seen_addr,
                            seen_len: fresh.seen_len,
                            seen_uav: fresh.seen_uav,
                        },
                    );
                    continue;
                }
            }

            let entry_idx = overflow_entries.len();
            overflow_entries.push(OverflowEntry {
                record,
                staged_cpu,
                fallback_spec: staged_cpu.is_none().then_some(fresh.spec),
                buffer: fresh.buffer,
                raw_buffer: fresh.raw_buffer,
                shadow: fresh.shadow,
                own: fresh.own,
                widened: fresh.widened,
                stride: fresh.stride,
                mark: fresh.mark,
                addr_words: fresh.addr_words,
                seen_addr: fresh.seen_addr,
                seen_len: fresh.seen_len,
                seen_uav: fresh.seen_uav,
            });
            overflow_map.insert(key, Some(entry_idx));
        }

        #[cfg(test)]
        tests::hook(tests::Stage::Pin);

        // Under one shared hold of `VIEWS`, as `view_at` pins: a hit is used
        // only if its view is still current, and only then are the
        // resources it names pinned, each once for the pass. `forget` takes
        // `VIEWS` alone before a resource's memory goes, so the resources of
        // a current entry are still there to count on; a hit whose view the
        // title released since its lookup (or whose resource is already on
        // its way out) is a miss, made by `mirror` as the default path makes it.
        let mut raw_pins = RawPins(Vec::new());
        let mut dropped = 0u64;
        if !hits.is_empty() {
            let views = views_read();
            let mut pinned: HashMap<usize, bool, Hashing> = HashMap::with_hasher(Hashing::new(true));
            for &(key, [_, view_idx, generation], s_idx) in &hits {
                let current = if inline::kind(generation) != 0 {
                    // Named whole in its record: no resource object to pin.
                    inline::current(view_idx, generation)
                } else {
                    match views.list.get(view_idx as usize) {
                        Some((view, current)) if *current == generation => {
                            view.resources().into_iter().filter(|&r| r != 0).all(|r| {
                                *pinned.entry(r).or_insert_with(|| {
                                    let held = crate::object::try_add_ref(r as This);
                                    if held {
                                        raw_pins.0.push(r);
                                    }
                                    held
                                })
                            })
                        }
                        _ => false,
                    }
                };
                if current {
                    state.pass_shadows[s_idx as usize].touch();
                } else {
                    dropped += 1;
                    state.evict(key);
                    overflow_map.insert(key, None);
                }
            }
            drop(views);
        }
        count(HITS, hits.len() as u64 - dropped);
        count(MISSES, misses);
        count(DROPPED, dropped);
        count(REVALIDATED, revalidated);
        #[cfg(test)]
        tests::LAST_PASS.set([hits.len() as u64 - dropped, misses, dropped, revalidated]);

        // Materialize cold staging descriptors (in parallel when many).
        if !cold_writes.is_empty() {
            match split_pool {
                Some(pool) if cold_writes.len() >= PARALLEL_STAGE_MIN => {
                    let parts = cold_writes.len().div_ceil(32).min(pool.threads() * 4).max(2);
                    let each = cold_writes.len().div_ceil(parts);
                    pool.run(parts, &|k| {
                        let lo = (k * each).min(cold_writes.len());
                        let hi = ((k + 1) * each).min(cold_writes.len());
                        for (dst, buf, spec) in &cold_writes[lo..hi] {
                            write_descriptor(host, buf, spec, *dst);
                        }
                    });
                }
                _ => {
                    for (dst, buf, spec) in &cold_writes {
                        write_descriptor(host, buf, spec, *dst);
                    }
                }
            }
        }

        let worker_on = crate::heap_worker::on();
        let state_ref: &StagedState = &*state;

        match split_pool {
            Some(pool) => {
                let wide = stale.len() >= crate::descriptor::pool::WORTH;
                crate::descriptor::pool::note(stale.len(), wide);
                if wide {
                    // One contiguous chunk per worker thread so each worker
                    // flushes one `Batch` to `Viewed` instead of 4x chunks.
                    let parts = pool.threads().min(stale.len().div_ceil(16)).max(1);
                    let each = stale.len().div_ceil(parts);
                    let (round, holds) = (
                        crate::descriptor::ahead::running(),
                        crate::host::holds_submissions(),
                    );
                    pool.run(parts, &|k| {
                        let lo = (k * each).min(stale.len());
                        let hi = ((k + 1) * each).min(stale.len());
                        if lo >= hi {
                            return;
                        }
                        let _syncing = crate::descriptor::slots::Syncing::enter();
                        crate::descriptor::ahead::as_round(round, || {
                            crate::host::as_holder(holds, || {
                                self.apply_chunk_fast(
                                    host,
                                    &stale[lo..hi],
                                    &at,
                                    state_ref,
                                    pass_id,
                                    &overflow_map,
                                    &overflow_entries,
                                    null_cpu,
                                    worker_on,
                                );
                            })
                        });
                    });
                } else {
                    self.apply_chunk_fast(
                        host,
                        stale,
                        &at,
                        state_ref,
                        pass_id,
                        &overflow_map,
                        &overflow_entries,
                        null_cpu,
                        worker_on,
                    );
                }
            }
            None => {
                self.apply_chunk_fast(
                    host,
                    stale,
                    &at,
                    state_ref,
                    pass_id,
                    &overflow_map,
                    &overflow_entries,
                    null_cpu,
                    worker_on,
                );
            }
        }

        // Release `staged_fast` lock before dropping `pins` / `raw_pins` so
        // any resource destructor runs with zero heap locks held.
        state.pass_shadows.clear();
        drop(guard);
        drop(overflow_entries);
        drop(pins);
        drop(raw_pins);
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_chunk_fast(
        &self,
        host: &Host,
        indices: &[u32],
        at: &impl Fn(u32) -> u64,
        state: &StagedState,
        pass_id: u64,
        overflow_map: &HashMap<u64, Option<usize>, Hashing>,
        overflow_entries: &[OverflowEntry],
        null_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        worker_on: bool,
    ) {
        // In place, whichever thread this is and whatever the switch reads
        // now: one pass enters its census one way (`crate::ab`).
        let mut census = self.viewed.batch_in_place();
        let pass = self.viewed.pass_in_place();
        let mut retired_local: Vec<ID3D12Resource> = Vec::new();
        let has_overflow = !overflow_entries.is_empty();

        for &index in indices {
            let handle = at(index);
            if index >= self.count {
                self.mirror(index, handle, false);
                pass.enter_if_full();
                continue;
            }
            // SAFETY: caller verified the heap page is mapped.
            let record = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            let [tag, view_idx, generation] = record;
            let done = if tag != VIEW_TAG {
                self.clear_slot_fast(
                    host,
                    index,
                    handle,
                    null_cpu,
                    &mut census,
                    &mut retired_local,
                )
            } else {
                let key = u64::from(generation) << 32 | u64::from(view_idx);
                if let Some(cached) = state.by_key.get(&key)
                    && state.slot_pass[cached.slot as usize] == pass_id
                {
                    let shadow = &state.pass_shadows[state.slot_shadow_idx[cached.slot as usize] as usize];
                    self.show_cached_slot(
                        host,
                        index,
                        handle,
                        cached,
                        shadow,
                        state.cpu(cached.slot),
                        worker_on,
                        &mut census,
                        &mut retired_local,
                    )
                } else if has_overflow
                    && let Some(r) = overflow_map.get(&key).copied().flatten()
                {
                    self.show_overflow_slot(
                        host,
                        index,
                        handle,
                        &overflow_entries[r],
                        worker_on,
                        &mut census,
                        &mut retired_local,
                    )
                } else {
                    false
                }
            };
            if !done {
                self.mirror(index, handle, false);
            }
            census.enter_if_full();
            pass.enter_if_full();
            if retired_local.len() >= RETIRE_BATCH {
                retire_many(&mut retired_local);
            }
        }
        retire_many(&mut retired_local);
        drop(census);
        drop(pass);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn show_cached_slot(
        &self,
        host: &Host,
        index: u32,
        handle: u64,
        cached: &CachedView,
        shadow: &Arc<Shadow>,
        src_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        worker_on: bool,
        census: &mut crate::bindless::Batch<'_>,
        retired_local: &mut Vec<ID3D12Resource>,
    ) -> bool {
        let key = u64::from(cached.record[2]) << 32 | u64::from(cached.record[1]);
        let mut made = self.made.lock(index);
        // SAFETY: caller verified the heap page is mapped.
        if unsafe { (handle as *const [u32; 3]).read_unaligned() } != cached.record {
            return false;
        }
        if made.get(index) == key {
            return true;
        }
        self.ready[index as usize].store(0, Ordering::Relaxed);
        self.write_stride(index, cached.stride);
        if let Some(data) = self.address_data()
            && index < self.address_slots()
        {
            // SAFETY: mapped `addresses` upload buffer holds 16 bytes per slot below `address_slots()`.
            unsafe { ((data + 16 * index as usize) as *mut [u32; 4]).write_unaligned(cached.addr_words) };
        }
        made.set(index, key);
        crate::descriptor::slots::made();
        // SAFETY: `src_cpu` is in our CPU staging heap and `self.cpu(index)` is in our shader-visible heap.
        unsafe {
            host.device.CopyDescriptorsSimple(
                1,
                self.cpu(index),
                src_cpu,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            )
        };
        self.set_mark(index, cached.mark);
        census.set_seen(
            index,
            Some(Seen::Buffer(
                shadow.clone(),
                cached.seen_addr,
                cached.seen_len,
                cached.seen_uav,
            )),
        );
        self.keep_fast(
            index,
            Some((&shadow.buffer, cached.raw_buffer, None)),
            retired_local,
        );
        self.ready[index as usize].store(key, Ordering::Release);
        drop(made);
        if worker_on && !PUBLISHING.get() {
            self.uploads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(index);
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn show_overflow_slot(
        &self,
        host: &Host,
        index: u32,
        handle: u64,
        entry: &OverflowEntry,
        worker_on: bool,
        census: &mut crate::bindless::Batch<'_>,
        retired_local: &mut Vec<ID3D12Resource>,
    ) -> bool {
        let key = u64::from(entry.record[2]) << 32 | u64::from(entry.record[1]);
        let mut made = self.made.lock(index);
        // SAFETY: caller verified the heap page is mapped.
        if unsafe { (handle as *const [u32; 3]).read_unaligned() } != entry.record {
            return false;
        }
        if made.get(index) == key {
            return true;
        }
        self.ready[index as usize].store(0, Ordering::Relaxed);
        self.write_stride(index, entry.stride);
        if let Some(data) = self.address_data()
            && index < self.address_slots()
        {
            // SAFETY: mapped `addresses` upload buffer holds 16 bytes per slot below `address_slots()`.
            unsafe { ((data + 16 * index as usize) as *mut [u32; 4]).write_unaligned(entry.addr_words) };
        }
        made.set(index, key);
        crate::descriptor::slots::made();
        match entry.staged_cpu {
            Some(src_cpu) => {
                // SAFETY: `src_cpu` is in our CPU staging heap and `self.cpu(index)` is in our shader-visible heap.
                unsafe {
                    host.device.CopyDescriptorsSimple(
                        1,
                        self.cpu(index),
                        src_cpu,
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    )
                };
            }
            None => {
                if let Some(spec) = &entry.fallback_spec {
                    write_descriptor(host, &entry.buffer, spec, self.cpu(index));
                }
            }
        }
        self.set_mark(index, entry.mark);
        census.set_seen(
            index,
            Some(Seen::Buffer(
                entry.shadow.clone(),
                entry.seen_addr,
                entry.seen_len,
                entry.seen_uav,
            )),
        );
        self.keep_fast(
            index,
            Some((&entry.buffer, entry.raw_buffer, entry.held())),
            retired_local,
        );
        self.ready[index as usize].store(key, Ordering::Release);
        drop(made);
        if worker_on && !PUBLISHING.get() {
            self.uploads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(index);
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn clear_slot_fast(
        &self,
        host: &Host,
        index: u32,
        handle: u64,
        null_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        census: &mut crate::bindless::Batch<'_>,
        retired_local: &mut Vec<ID3D12Resource>,
    ) -> bool {
        let mut made = self.made.lock(index);
        // SAFETY: caller verified the heap page is mapped.
        let [tag, _, _] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag == VIEW_TAG {
            return false;
        }
        if made.get(index) == 0 {
            return true;
        }
        self.ready[index as usize].store(0, Ordering::Relaxed);
        made.set(index, 0);
        crate::descriptor::slots::made();
        // SAFETY: `null_cpu` is slot 0 of our CPU staging heap, initialized with `null_view`.
        unsafe {
            host.device.CopyDescriptorsSimple(
                1,
                self.cpu(index),
                null_cpu,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            )
        };
        self.keep_fast(index, None, retired_local);
        self.write_stride(index, 0);
        self.set_mark(index, None);
        if let Some(data) = self.address_data()
            && index < self.address_slots()
        {
            // SAFETY: mapped `addresses` upload buffer holds 16 bytes per slot below `address_slots()`.
            unsafe { ((data + 16 * index as usize) as *mut [u32; 4]).write_unaligned([0; 4]) };
        }
        if index < self.count {
            census.set_seen(index, None);
        }
        true
    }

    fn keep_fast(
        &self,
        index: u32,
        new: Option<(&ID3D12Resource, usize, Option<HeldSource>)>,
        retired_local: &mut Vec<ID3D12Resource>,
    ) {
        let split = self.kept.split();
        let old = self.kept.with(index as usize, |kept, at| {
            let slot = kept.slot(at, new.is_some())?;
            if let (Some((old_buf, None)), Some((_, new_raw, None))) = (slot.as_ref(), &new)
                && Interface::as_raw(old_buf) as usize == *new_raw
            {
                // Ring-heap overwrite onto the exact same host buffer with no
                // widened/own source: the slot already holds this resource.
                return None;
            }
            let old_widened = slot.as_ref().and_then(|(_, source)| match source {
                Some(HeldSource::Widened(view, _)) => Some(view.clone()),
                _ => None,
            });
            let resource = new.map(|(buf, _, src)| (buf.clone(), src));
            let new_widened = resource.as_ref().and_then(|(_, source)| match source {
                Some(HeldSource::Widened(view, _)) => Some(view.clone()),
                _ => None,
            });
            let old = std::mem::replace(slot, resource);
            self.note_widened(old_widened.as_ref(), false);
            self.note_widened(new_widened.as_ref(), true);
            #[cfg(feature = "diagnostics")]
            if let Some((old, _)) = &old
                && !self.sampler
                && slot.as_ref().map(|s| Interface::as_raw(&s.0)) != Some(Interface::as_raw(old))
                // SAFETY: a live resource.
                && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            {
                let left = crate::resource::life::holders(old);
                crate::resource::life::note(
                    old.as_raw() as usize,
                    crate::resource::life::SLOT,
                    u64::from(index),
                    left,
                );
            }
            if split {
                return old.map(|(b, _)| b);
            }
            if let Some((old, _)) = old {
                retire(old);
            }
            None
        });
        if let Some(old_buf) = old {
            retired_local.push(old_buf);
        }
    }
}


#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    use super::*;
    use crate::descriptor::host::split_tests::{Title, random, shown};

    /// Where a pass is (`HostHeap::make_stale_staged`): at the lookup of
    /// stale slot `index`, or between the lookups and the pins under `VIEWS`.
    pub(super) enum Stage {
        Lookup(u32),
        Pin,
    }

    type Hook = Box<dyn FnMut(Stage)>;

    thread_local! {
        /// What a test runs at each [`Stage`] of the passes on its thread.
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
        /// The last pass on this thread: staged views found, missed, found
        /// but released before their pin, and found after a new lookup.
        pub(super) static LAST_PASS: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
    }

    pub(super) fn hook(stage: Stage) {
        HOOK.with(|h| {
            if let Some(f) = h.borrow_mut().as_mut() {
                f(stage);
            }
        });
    }

    fn set_hook(f: Option<Hook>) {
        HOOK.with(|h| *h.borrow_mut() = f);
    }

    /// A title resource's release, through its own table.
    fn release_of(resource: This) -> extern "system" fn(This) -> u32 {
        crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10)
    }

    /// The shader view record of `count` raw words of `resource` from word
    /// `first`, as the title writes it.
    fn srv_record(resource: This, first: u32, count: u32) -> [u32; 8] {
        let mut record = [0u32; 8];
        crate::descriptor::create_srv(
            std::ptr::null(),
            resource,
            &[62, 1, 0x1688, 0, first, 0, count, 0, 0, 0],
            record.as_mut_ptr() as u64,
        );
        record
    }

    /// What slot `index` keeps: the host resource, and as what (0 alone, 1
    /// with its copy, 2 widened).
    fn kept_of(heap: &HostHeap, index: u32) -> Option<(usize, u8)> {
        heap.kept.get(index as usize, |held| {
            held.map(|(buffer, source)| {
                let kind = match source {
                    None => 0,
                    Some(HeldSource::Shadow(_)) => 1,
                    Some(HeldSource::Widened(..)) => 2,
                };
                (Interface::as_raw(buffer) as usize, kind)
            })
        })
    }

    /// Slot `index`'s mark words, if the heap has marks.
    fn mark_of(heap: &HostHeap, index: u32) -> Option<(u32, u32)> {
        heap.marks.as_ref().map(|(_, data)| {
            // SAFETY: 2 dwords per slot in the mapped `marks` buffer.
            unsafe {
                let p = (*data as *const u32).add(2 * index as usize);
                (p.read_volatile(), p.add(1).read_volatile())
            }
        })
    }

    /// Every slot of `fast` shows what `def` shows: `shown` (made, kept,
    /// ready, stride, census range), its view address, its marks, and the
    /// host resource it keeps and as what; and the two censuses count the
    /// same ([`crate::bindless::Census`]: entries, range and texture counts,
    /// ends, hulls, unordered ranges). (A descriptor's bytes are not memory
    /// the CPU can read on D3DMetal: its CPU handles are not addresses. What
    /// the GPU reads through the slots is compared by
    /// `the_gpu_reads_the_same_through_fast_and_default_slots`.)
    fn same_heaps(def: &HostHeap, fast: &HostHeap, slots: u32, what: &str) {
        for i in 0..slots {
            assert_eq!(shown(fast, i), shown(def, i), "{what}: slot {i}");
            assert_eq!(fast.address_at(i), def.address_at(i), "{what}: slot {i} view address");
            assert_eq!(mark_of(fast, i), mark_of(def, i), "{what}: slot {i} mark");
            assert_eq!(kept_of(fast, i), kept_of(def, i), "{what}: slot {i} kept");
        }
        assert_eq!(fast.viewed.census(), def.viewed.census(), "{what}: census");
    }

    /// The default path's sync of `pages`, as `sync_slots` makes them
    /// (several threads for a wide page).
    fn sync_default(heap: &HostHeap, pool: &Pool, pages: &[(u32, u32)], start: u64, step: u64) {
        let at = |i: u32| start + u64::from(i) * step;
        for &(first, last) in pages {
            let stale = heap.stale_slots(first, last, start, step);
            if stale.len() >= crate::descriptor::pool::WORTH {
                heap.sync_wide(pool, &stale, &at);
            } else {
                heap.make_stale(stale, start, step);
            }
        }
    }

    /// Two heaps of `slots` as a world's are (split locks, census batches
    /// and passes, view addresses made), or none without a host GPU.
    fn two_heaps(slots: u32) -> Option<(HostHeap, HostHeap)> {
        let (Some(mut def), Some(mut fast)) = (HostHeap::new_as(false, slots, true), HostHeap::new_as(false, slots, true))
        else {
            return None;
        };
        def.viewed = Arc::new(crate::bindless::Viewed::with_batch(true, true));
        fast.viewed = Arc::new(crate::bindless::Viewed::with_batch(true, true));
        assert!(def.view_addresses().is_some() && fast.view_addresses().is_some());
        Some((def, fast))
    }

    /// Differential test on real D3D12 + Wine heaps: `VICTORY_HEAP_SYNC_FAST=1`
    /// leaves every slot as the default path does ([`same_heaps`]) across
    /// buffer SRVs, CBVs, buffer UAVs, placed buffer SRVs (including
    /// off-element own copies), refused views, cleared slots, resource
    /// release (`forget`) and ring-heap overwrites. Pool workers enter the
    /// census in place: no switch is read for it.
    #[test]
    fn heap_sync_fast_matches_default_across_views_overwrites_and_forget() {
        let Some(host_dev) = host() else {
            return;
        };
        crate::descriptor::slots::spin_in_tests();
        let pool = Pool::new(4);

        const PAGES: u32 = 4;
        const SLOTS_PER_PAGE: u32 = 2048;
        const SLOTS: u32 = PAGES * SLOTS_PER_PAGE;

        let res_a = crate::resource::test_buffer(8192);
        let res_b = crate::resource::test_buffer(4096);
        let release = release_of(res_a);
        let mem_a = crate::resource::buffer_bytes(res_a).unwrap().0 as u64;

        let mut records = vec![[0u32; 8]; SLOTS as usize].into_boxed_slice();
        let start = records.as_mut_ptr() as u64;
        let at = |i: u32| start + 32 * u64::from(i);
        let null = std::ptr::null();

        let Some((def_heap, fast_heap)) = two_heaps(SLOTS) else {
            release(res_a);
            release(res_b);
            return;
        };

        let page_ranges: Vec<(u32, u32)> = (0..PAGES)
            .map(|p| (p * SLOTS_PER_PAGE, (p + 1) * SLOTS_PER_PAGE))
            .collect();

        for round in 0..4u32 {
            match round {
                0 | 1 => {
                    for i in 0..SLOTS {
                        if i.is_multiple_of(5) {
                            continue;
                        }
                        let off = ((i + round * 13) % 32) * 4;
                        match (i + round) % 6 {
                            0 => crate::descriptor::create_srv(
                                null,
                                res_a,
                                &[62, 1, 0x1688, 0, off, 0, 64, 0, 0, 0],
                                at(i),
                            ),
                            1 => crate::descriptor::create_cbv(
                                null,
                                &[
                                    (mem_a + 64 * u64::from(off % 16)) as u32,
                                    ((mem_a + 64 * u64::from(off % 16)) >> 32) as u32,
                                    128,
                                    0,
                                ],
                                at(i),
                            ),
                            2 => crate::descriptor::create_uav(
                                null,
                                res_a,
                                null,
                                &[39, 1, 0, 0, off, 0, 16, 0, 1, 0],
                                at(i),
                            ),
                            3 => {
                                let mut placed = [0u32; 16];
                                placed[..10].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 8, 12, 0, 0]);
                                let p_addr = mem_a + 4;
                                (placed[10], placed[11]) = (p_addr as u32, (p_addr >> 32) as u32);
                                let mut buf = [0u32; 14];
                                (buf[0], buf[4]) = (1, 512);
                                crate::descriptor::create_placed_srv(null, &buf, &placed, at(i));
                            }
                            4 => crate::descriptor::create_srv(
                                null,
                                res_b,
                                &[62, 1, 0x1688, 0, 0, 0, 32, 0, 0, 0],
                                at(i),
                            ),
                            _ => crate::descriptor::create_srv(
                                null,
                                res_a,
                                &[62, 1, 0x1688, 0, 0, 0, 0, 0, 0, 0],
                                at(i),
                            ),
                        }
                    }
                }
                2 => {
                    release(res_b);
                }
                _ => {
                    for i in (0..SLOTS).step_by(2) {
                        records[i as usize] = [0u32; 8];
                    }
                }
            }

            sync_default(&def_heap, &pool, &page_ranges, start, 32);
            fast_heap.sync_pages_fast_with(&page_ranges, start, 32, Some(&pool));
            same_heaps(&def_heap, &fast_heap, SLOTS, &format!("round {round}"));
        }

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host_dev.log_messages(), None);
        drop((def_heap, fast_heap));
        crate::host::release_retired();
        release(res_a);
    }

    /// A pass whose cached views go stale while it runs, as the title's
    /// buffers come and go in a world frame. Every view is cached. Halfway
    /// through the pass's lookups the title buffer under the constant
    /// buffer views `a` is released and another made over the same memory
    /// (`LOOKUP_EPOCH` moves, and their lookup now brings another copy), so
    /// `a`'s views miss and are resolved again, while the shader views `c2`
    /// looked up after it are looked up again and found (same copy).
    /// Between the lookups and the pins the title releases the buffer under
    /// the views `b` (`forget`: `VIEWS_FORGOT` moves, and its object is
    /// freed), so `b`'s views, found before, are dropped unpinned and made as
    /// the default path makes them; `c1`'s stay found. Each round every slot
    /// shows what the default path shows ([`same_heaps`]). Other tests making
    /// buffers meanwhile change nothing here: a hit at a new epoch is looked
    /// up again.
    #[test]
    fn heap_sync_fast_drops_hits_released_before_their_pin_and_misses_after_a_lookup_change() {
        let Some(host_dev) = host() else {
            return;
        };
        crate::descriptor::slots::spin_in_tests();
        let pool = Pool::new(4);
        const SLOTS: u32 = 4096;
        const VIEWS: u64 = 32;
        let pages = [(0, SLOTS / 2), (SLOTS / 2, SLOTS)];

        let (res_b, res_c) = (crate::resource::test_buffer(4096), crate::resource::test_buffer(4096));
        let release = release_of(res_b);
        // Title memory for `a`'s constant buffer views, a title buffer of
        // its own (registered as one is when made).
        let memory = vec![0u8; 0x1000];
        let fake = memory.as_ptr() as u64;
        crate::shadow::register(fake as usize, fake, 0x1000);
        let views = |f: &dyn Fn(u32) -> [u32; 8]| (0..VIEWS as u32).map(f).collect::<Vec<_>>();
        let a = views(&|k| {
            let mut record = [0u32; 8];
            let address = fake + 64 * u64::from(k);
            crate::descriptor::create_cbv(
                std::ptr::null(),
                &[address as u32, (address >> 32) as u32, 64, 0],
                record.as_mut_ptr() as u64,
            );
            record
        });
        let b = views(&|k| srv_record(res_b, 4 * k, 16));
        let c1 = views(&|k| srv_record(res_c, 4 * k, 16));
        let c2 = views(&|k| srv_record(res_c, 2048 + 4 * k, 32));

        let mut records = vec![[0u32; 8]; SLOTS as usize].into_boxed_slice();
        let start = records.as_mut_ptr() as u64;
        let Some((def_heap, fast_heap)) = two_heaps(SLOTS) else {
            release(res_b);
            release(res_c);
            crate::shadow::unregister(fake as usize, fake);
            return;
        };
        // Below half: `b` and `c1` in turn; above: `a` and `c2`. Each shift
        // moves every record, so every slot is stale and each view is looked
        // up once.
        let mut fill = |shift: u32| {
            for i in 0..SLOTS {
                let k = ((i / 2 + shift) % VIEWS as u32) as usize;
                records[i as usize] = match (i < SLOTS / 2, i % 2) {
                    (true, 0) => b[k],
                    (true, _) => c1[k],
                    (false, 0) => a[k],
                    (false, _) => c2[k],
                };
            }
        };

        fill(0);
        sync_default(&def_heap, &pool, &pages, start, 32);
        fast_heap.sync_pages_fast_with(&pages, start, 32, Some(&pool));
        same_heaps(&def_heap, &fast_heap, SLOTS, "views cached");
        let [found, missed, dropped, _] = LAST_PASS.get();
        assert_eq!([found, missed, dropped], [0, 4 * VIEWS, 0], "every view missed and was staged once");

        fill(1);
        let (epoch, forgot) = (crate::shadow::lookup_epoch(), views_forgot());
        set_hook(Some(Box::new(move |stage| match stage {
            Stage::Lookup(index) if index == SLOTS / 2 => {
                // `a`'s buffer goes, and another is made over its memory.
                crate::shadow::unregister(fake as usize, fake);
                crate::shadow::register(fake as usize + 1, fake, 0x1000);
            }
            Stage::Lookup(_) => {}
            Stage::Pin => {
                release(res_b);
            }
        })));
        fast_heap.sync_pages_fast_with(&pages, start, 32, Some(&pool));
        set_hook(None);
        assert!(crate::shadow::lookup_epoch() > epoch && views_forgot() > forgot);
        let [found, missed, dropped, again] = LAST_PASS.get();
        assert_eq!(
            [found, missed, dropped],
            [2 * VIEWS, VIEWS, VIEWS],
            "c1's and c2's views found, a's missed (another copy), b's found and dropped"
        );
        assert!(again >= VIEWS, "c2's views were looked up again: {again}");
        sync_default(&def_heap, &pool, &pages, start, 32);
        same_heaps(&def_heap, &fast_heap, SLOTS, "b released before the pin");
        assert!((0..SLOTS / 2).step_by(2).all(|i| !fast_heap.kept(i)), "b's slots are null");

        // The dropped entries are gone: a's (new copy), c1's and c2's are
        // found; b's records name no view now, and their slots, null
        // already, are not stale.
        fill(2);
        fast_heap.sync_pages_fast_with(&pages, start, 32, Some(&pool));
        sync_default(&def_heap, &pool, &pages, start, 32);
        same_heaps(&def_heap, &fast_heap, SLOTS, "after the drop");
        let [found, missed, dropped, _] = LAST_PASS.get();
        assert_eq!([found, missed, dropped], [3 * VIEWS, 0, 0], "a's, c1's and c2's found");

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host_dev.log_messages(), None);
        drop((def_heap, fast_heap));
        crate::host::release_retired();
        release(res_c);
        crate::shadow::unregister(fake as usize + 1, fake);
    }

    /// Copies the first 16 words of the raw view at a table's first slot
    /// into `dst` at `slot * 64` bytes. (Wine's compiler takes no
    /// `GetDimensions` of a raw buffer.)
    const READ_SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Slot : register(b0) { uint slot; };
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(slot * 64 + id.x * 4, src.Load(id.x * 4));
}
"#;

    /// A queue, a list and a fence of our own, to run a list and wait.
    struct Runner {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        done: Cell<u64>,
    }

    impl Runner {
        fn new(device: &ID3D12Device) -> Runner {
            // SAFETY: plain creation of objects of our own.
            unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                    .unwrap();
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                list.Close().unwrap();
                let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                Runner { queue, allocator, list, fence, done: Cell::new(0) }
            }
        }

        /// Records with `record`, runs it and waits for it (10 s at most).
        fn run(&self, record: impl FnOnce(&ID3D12GraphicsCommandList)) {
            // SAFETY: our own list, queue and fence, one run at a time.
            unsafe {
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
                record(&self.list);
                self.list.Close().unwrap();
                self.queue.ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
                self.done.set(self.done.get() + 1);
                self.queue.Signal(&self.fence, self.done.get()).unwrap();
                let began = Instant::now();
                while self.fence.GetCompletedValue() < self.done.get() {
                    assert!(began.elapsed().as_secs() < 10, "the list ran");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
    }

    fn committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, size: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    state,
                    None,
                    &mut out,
                )
                .unwrap();
        }
        out.unwrap()
    }

    /// The GPU reads the same words through every slot of the fast heap as
    /// through the default heap's, and the title's bytes: raw shader views
    /// and constant buffer views of a title buffer at many offsets, made
    /// fresh (each view staged by `Create*View` into the CPU heap, then
    /// `CopyDescriptorsSimple`), found in the staging cache (copied from
    /// descriptors staged passes ago), and found after a new
    /// `LOOKUP_EPOCH` (looked up again). The default heap's slots are made
    /// by `Create*View` in place. So a descriptor copied from the CPU heap
    /// keeps its buffer resident for the GPU on D3DMetal as one made in
    /// place does.
    #[test]
    fn the_gpu_reads_the_same_through_fast_and_default_slots() {
        let Some(host_dev) = host() else {
            return;
        };
        crate::descriptor::slots::spin_in_tests();
        let pool = Pool::new(4);
        const SLOTS: u32 = 256;
        const VIEWS: u32 = 48;
        let resource = crate::resource::test_buffer(4096);
        let release = release_of(resource);
        let memory = crate::resource::buffer_bytes(resource).unwrap().0 as u64;
        // SAFETY: the buffer's own 4096 bytes of title memory.
        unsafe {
            let words = std::slice::from_raw_parts_mut(memory as *mut u32, 1024);
            for (i, w) in words.iter_mut().enumerate() {
                *w = 0x5000_0000 + i as u32;
            }
        }
        // Raw shader views of 16 words from word `4k + 1`, and constant
        // buffer views of 64 bytes from byte `64k`: (record, first word).
        let views: Vec<([u32; 8], u32)> = (0..VIEWS)
            .map(|k| {
                if k % 2 == 0 {
                    let mut record = [0u32; 8];
                    let first = 4 * k + 1;
                    crate::descriptor::create_srv(
                        std::ptr::null(),
                        resource,
                        &[39, 1, 0x1688, 0, first, 0, 16, 0, 1, 0],
                        record.as_mut_ptr() as u64,
                    );
                    (record, first)
                } else {
                    let mut record = [0u32; 8];
                    let address = memory + 64 * u64::from(k);
                    crate::descriptor::create_cbv(
                        std::ptr::null(),
                        &[address as u32, (address >> 32) as u32, 64, 0],
                        record.as_mut_ptr() as u64,
                    );
                    (record, 16 * k)
                }
            })
            .collect();
        let mut records = vec![[0u32; 8]; SLOTS as usize].into_boxed_slice();
        let start = records.as_mut_ptr() as u64;
        let pages = [(0, SLOTS)];
        let Some((def_heap, fast_heap)) = two_heaps(SLOTS) else {
            release(resource);
            return;
        };
        let shadow = crate::shadow::find(memory, 4096).expect("the buffer's host copy");
        shadow.update(memory as usize, 4096);

        let (root, pso) = crate::compute::pipeline(
            READ_SOURCE,
            windows::core::s!("read"),
            0,
            &[
                crate::compute::Param::Table(&[crate::compute::srv(0, 1)]),
                crate::compute::Param::Constants { register: 0, count: 1 },
                crate::compute::Param::Uav { register: 0 },
            ],
        )
        .expect("the read shader");
        let device = &host_dev.device;
        let runner = Runner::new(device);
        let bytes = 2 * u64::from(SLOTS) * 64;
        let out = committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let back = committed(device, D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        for (round, what) in ["made fresh", "found in the cache", "looked up again"].into_iter().enumerate() {
            let shift = round as u32 * 5;
            let which = |i: u32| ((i * 7 + shift) % VIEWS) as usize;
            for i in 0..SLOTS {
                records[i as usize] = views[which(i)].0;
            }
            if round == 2 {
                // A buffer made and released: the shadow tables changed.
                let other = vec![0u8; 0x1000];
                let at = other.as_ptr() as u64;
                crate::shadow::register(at as usize, at, 0x1000);
                crate::shadow::unregister(at as usize, at);
            }
            sync_default(&def_heap, &pool, &pages, start, 32);
            fast_heap.sync_pages_fast_with(&pages, start, 32, Some(&pool));
            same_heaps(&def_heap, &fast_heap, SLOTS, what);
            let [found, missed, _, again] = LAST_PASS.get();
            match round {
                0 => assert_eq!((found, missed), (0, u64::from(VIEWS)), "{what}"),
                _ => assert_eq!((found, missed), (u64::from(VIEWS), 0), "{what}"),
            }
            if round == 2 {
                assert_eq!(again, u64::from(VIEWS), "{what}");
            }

            // Each side in a list of its own, the fast one first: nothing the
            // default heap's list makes resident is there for the fast one's.
            for (side, heap) in [(1u32, &fast_heap), (0, &def_heap)] {
                let barrier = |before, after| D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(out.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: before,
                            StateAfter: after,
                        }),
                    },
                };
                let half = bytes / 2;
                runner.run(|list| {
                    // SAFETY: live objects of ours; tables inside the heap.
                    unsafe {
                        list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                        list.SetComputeRootSignature(&root);
                        list.SetPipelineState(&pso);
                        list.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
                        for i in 0..SLOTS {
                            list.SetComputeRootDescriptorTable(0, heap.gpu(i));
                            list.SetComputeRoot32BitConstant(1, side * SLOTS + i, 0);
                            list.Dispatch(1, 1, 1);
                        }
                        // (The barriers' references to `out` are left to the
                        // test's end, as a test may.)
                        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                        list.CopyBufferRegion(&back, u64::from(side) * half, &out, u64::from(side) * half, half);
                        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
                    }
                });
            }
            let mut data = std::ptr::null_mut();
            // SAFETY: our readback buffer, read after its copy ran.
            let words: Vec<u32> = unsafe {
                back.Map(0, None, Some(&mut data)).unwrap();
                let words = std::slice::from_raw_parts(data.cast::<u32>(), (bytes / 4) as usize).to_vec();
                back.Unmap(0, None);
                words
            };
            for i in 0..SLOTS {
                let at = |side: u32| {
                    let first = ((side * SLOTS + i) * 16) as usize;
                    first..first + 16
                };
                let first = views[which(i)].1;
                let want: Vec<u32> = (0..16).map(|w| 0x5000_0000 + first + w).collect();
                assert_eq!(words[at(0)], want[..], "{what}: slot {i} through the default heap");
                assert_eq!(words[at(1)], want[..], "{what}: slot {i} through the fast heap");
            }
        }

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host_dev.log_messages(), None);
        drop((def_heap, fast_heap));
        crate::host::release_retired();
        release(resource);
    }

    /// Passes while a title thread makes buffers, writes views of them over
    /// the heap, and releases them (`LOOKUP_EPOCH` and `VIEWS_FORGOT` move
    /// throughout every pass, and cached views are released between their
    /// lookups and their pins), and another rewrites slots with views of a
    /// buffer that stays. Once the title rests, a last pass leaves every
    /// slot as a heap made by the default path from the same records.
    #[test]
    fn heap_sync_fast_matches_default_while_the_title_makes_and_releases_buffers() {
        let Some(host_dev) = host() else {
            return;
        };
        crate::descriptor::slots::spin_in_tests();
        let pool = Pool::new(4);
        const SLOTS: u32 = 8192;
        let title = Title::new(SLOTS, 64);
        let pages: Vec<(u32, u32)> = (0..SLOTS / 2048).map(|p| (p * 2048, (p + 1) * 2048)).collect();
        let Some((def_heap, fast_heap)) = two_heaps(SLOTS) else {
            return;
        };
        // Every slot made once on both, so that a slot the title clears
        // later was made and cleared on both.
        for i in 0..SLOTS {
            title.write(i, Some(&title.views[i as usize % title.views.len()]));
        }
        sync_default(&def_heap, &pool, &pages, title.start(), 32);
        fast_heap.sync_pages_fast_with(&pages, title.start(), 32, Some(&pool));
        let stop = AtomicBool::new(false);
        let mut passes = 0u32;
        let [mut found, mut missed, mut dropped, mut again] = [0u64; 4];
        std::thread::scope(|scope| {
            // Buffers that come and go, their views over the heap.
            scope.spawn(|| {
                let mut next = random(5);
                while !stop.load(Ordering::Relaxed) {
                    let resource = crate::resource::test_buffer(4096);
                    let release = release_of(resource);
                    let views: Vec<_> = (0..8).map(|k| srv_record(resource, 8 * k, 16)).collect();
                    for _ in 0..64 {
                        let slot = (next() % u64::from(SLOTS)) as u32;
                        title.write(slot, Some(&views[(next() % 8) as usize]));
                    }
                    std::thread::yield_now();
                    release(resource);
                }
            });
            // Slots rewritten with views that stay, or cleared.
            scope.spawn(|| {
                let mut next = random(9);
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..256 {
                        let slot = (next() % u64::from(SLOTS)) as u32;
                        let view = (!next().is_multiple_of(11)).then(|| &title.views[(next() % 64) as usize]);
                        title.write(slot, view);
                    }
                    std::thread::yield_now();
                }
            });
            let began = Instant::now();
            while began.elapsed().as_millis() < 1500 || passes < 20 {
                fast_heap.sync_pages_fast_with(&pages, title.start(), 32, Some(&pool));
                let [f, m, d, g] = LAST_PASS.get();
                (found, missed, dropped, again) = (found + f, missed + m, dropped + d, again + g);
                passes += 1;
                crate::host::release_retired();
            }
            stop.store(true, Ordering::Relaxed);
        });
        fast_heap.sync_pages_fast_with(&pages, title.start(), 32, Some(&pool));
        sync_default(&def_heap, &pool, &pages, title.start(), 32);
        same_heaps(&def_heap, &fast_heap, SLOTS, "after the title rests");
        eprintln!(
            "{passes} passes: staged views found {found} ({again} looked up again), missed {missed}, found but released since {dropped}"
        );
        assert!(found > 0 && missed > 0, "both hits and misses ran");

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host_dev.log_messages(), None);
        drop((def_heap, fast_heap));
        crate::host::release_retired();
        release_of(title.resource)(title.resource);
    }

    /// One frame of `SUBMISSIONS` heap syncs per side, alternating, with
    /// `play-main-mac.sh`'s settings; `churn`: a buffer made, viewed and
    /// released before each submission, so `LOOKUP_EPOCH` and
    /// `VIEWS_FORGOT` move between passes as buffers come and go in a world
    /// frame (staged views are then looked up again; the fast side's found,
    /// missed and dropped counts are printed).
    #[test]
    #[ignore = "benchmarks world-scale multi-submission descriptor heap sync under Wine + D3DMetal"]
    fn world_heap_sync_ab_bench() {
        const PAGES: u32 = 176; // 176 dirty 64 KiB pages = 360,448 slots/heap
        const SLOTS_PER_PAGE: u32 = 2048;
        const SLOTS: u32 = PAGES * SLOTS_PER_PAGE;
        const DISTINCT_VIEWS: u32 = 2048;
        const SUBMISSIONS: usize = 20;

        let Some(host_dev) = host() else {
            return;
        };
        let _quiet = host_dev.quiet();
        crate::descriptor::slots::spin_in_tests();
        let pool = Pool::new(4);
        let title = Title::new(SLOTS, DISTINCT_VIEWS);

        let Some((def_heap, fast_heap)) = two_heaps(SLOTS) else {
            return;
        };

        for (scenario, pages_per_sub, writes_per_page, churn) in [
            (
                "world-sparse (737K slots scanned across 20 subs, ~72 changed/page = 25,920 slots/frame)",
                18usize,
                72u32,
                false,
            ),
            (
                "world-heavy (737K slots scanned across 20 subs, ~112 changed/page = 40,320 slots/frame)",
                18usize,
                112u32,
                false,
            ),
            (
                "world-sparse, a buffer made and released before each submission",
                18usize,
                72u32,
                true,
            ),
        ] {
            let mut def_us = Vec::new();
            let mut fast_us = Vec::new();
            let mut next = random(42);
            let mut fast_counts = [0u64; 4];

            for i in 0..SLOTS {
                if !i.is_multiple_of(3) {
                    title.write(i, Some(&title.views[(i as usize * 7) % title.views.len()]));
                }
            }
            let all_pages: Vec<(u32, u32)> = (0..PAGES)
                .map(|p| (p * SLOTS_PER_PAGE, (p + 1) * SLOTS_PER_PAGE))
                .collect();
            sync_default(&def_heap, &pool, &all_pages, title.start(), 32);
            fast_heap.sync_pages_fast_with(&all_pages, title.start(), 32, Some(&pool));
            crate::host::release_retired();

            // 10 alternating A/B frames (skipping first 2 warm-up pairs).
            for round in 0..10u32 {
                for fast_on in [false, true] {
                    let heap = if fast_on { &fast_heap } else { &def_heap };
                    let mut frame_us = 0u64;
                    for sub in 0..SUBMISSIONS {
                        let base_page = ((sub * pages_per_sub) % PAGES as usize) as u32;
                        let mut dirty_pages = Vec::with_capacity(pages_per_sub);
                        for p_off in 0..pages_per_sub as u32 {
                            let p = (base_page + p_off) % PAGES;
                            let first = p * SLOTS_PER_PAGE;
                            let stride = (SLOTS_PER_PAGE / writes_per_page).max(1);
                            for k in 0..writes_per_page {
                                let slot = first
                                    + (k * stride + round + u32::from(fast_on)) % SLOTS_PER_PAGE;
                                let r = next();
                                let rec = (!r.is_multiple_of(19))
                                    .then(|| &title.views[(r as usize) % title.views.len()]);
                                title.write(slot, rec);
                            }
                            dirty_pages.push((first, first + SLOTS_PER_PAGE));
                        }
                        if churn {
                            let resource = crate::resource::test_buffer(4096);
                            let _ = srv_record(resource, 0, 16);
                            release_of(resource)(resource);
                        }

                        if fast_on {
                            let t0 = Instant::now();
                            heap.sync_pages_fast_with(&dirty_pages, title.start(), 32, Some(&pool));
                            frame_us += t0.elapsed().as_micros() as u64;
                            if round >= 2 {
                                let last = LAST_PASS.get();
                                (0..4).for_each(|i| fast_counts[i] += last[i]);
                            }
                        } else {
                            let at = |i: u32| title.at(i);
                            let t0 = Instant::now();
                            for &(f, l) in &dirty_pages {
                                let stale = heap.stale_slots(f, l, title.start(), 32);
                                if crate::descriptor::ahead::on() {
                                    crate::descriptor::ahead::changed(stale.len());
                                }
                                let wide = stale.len() >= crate::descriptor::pool::WORTH;
                                crate::descriptor::pool::note(stale.len(), wide);
                                if wide {
                                    heap.sync_wide(&pool, &stale, &at);
                                } else {
                                    heap.make_stale(stale, title.start(), 32);
                                }
                            }
                            frame_us += t0.elapsed().as_micros() as u64;
                        }
                        crate::host::release_retired();
                    }
                    if round >= 2 {
                        if fast_on {
                            fast_us.push(frame_us);
                        } else {
                            def_us.push(frame_us);
                        }
                    }
                }
            }

            def_us.sort_unstable();
            fast_us.sort_unstable();
            let d_med = def_us[def_us.len() / 2];
            let f_med = fast_us[fast_us.len() / 2];
            let frames = fast_us.len() as u64;
            eprintln!(
                "[{scenario}]\n  default: median {:>6} us ({:.2} ms/frame)\n  fast:    median {:>6} us ({:.2} ms/frame)  ({:.1}x faster, {:+.1}%)\n  fast, staged views per frame: {} found ({} looked up again), {} missed, {} found but released since",
                d_med,
                d_med as f64 / 1e3,
                f_med,
                f_med as f64 / 1e3,
                d_med as f64 / f_med.max(1) as f64,
                (f_med as f64 - d_med as f64) * 100.0 / d_med.max(1) as f64,
                fast_counts[0] / frames,
                fast_counts[3] / frames,
                fast_counts[1] / frames,
                fast_counts[2] / frames,
            );
        }

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        drop((def_heap, fast_heap));
        crate::host::release_retired();
    }
}
