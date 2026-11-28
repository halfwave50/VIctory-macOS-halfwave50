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
//! 3. `descriptor_as_data` address words (`[u32; 4]`) are precomputed once per
//!    cached record instead of per slot.
//! 4. Ring-heap overwrites whose `kept` slot already holds the exact same
//!    `(ID3D12Resource, None)` skip COM `AddRef`/`Release` and global
//!    `RETIRED` lock traffic; actual retirements flush in batches via
//!    `crate::host::retire_many`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::Ordering;

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
    lookup_epoch: u64,
    verified_forgot: u64,
    descriptor_as_data: bool,
    stride: u32,
    mark: Option<(u32, u32)>,
    addr_words: [u32; 4],
    seen_addr: u64,
    seen_len: u64,
    seen_uav: bool,
    resources: [usize; 2],
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
    /// evicting the oldest unreferenced slot via clock hand.
    fn alloc_slot(&mut self, key: u64) -> Option<u32> {
        let cap = STAGED_CAPACITY - 1;
        for _ in 0..cap {
            let s = self.clock;
            self.clock = if s + 1 >= STAGED_CAPACITY { 1 } else { s + 1 };
            if self.slot_pass[s as usize] == self.pass_id {
                continue;
            }
            let prev = self.slot_owner[s as usize];
            if prev != 0 && prev != key {
                self.by_key.remove(&prev);
            }
            self.slot_owner[s as usize] = key;
            self.slot_pass[s as usize] = self.pass_id;
            return Some(s);
        }
        None
    }
}

struct ResolvedFresh {
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
    let widened = match (under_wine() && matches!(desc.Format.0, 6..=8))
        .then(|| widened_w(desc.Format.0))
        .flatten()
    {
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

        if !stale.is_empty() {
            let can_stage = !self.sampler
                && super::quick::on()
                && !crate::alias_physical::verifying()
                && host().is_some();
            if can_stage {
                self.make_stale_staged(&stale, start, step, forgot, split_pool);
            } else {
                self.make_stale(stale, start, step);
            }
        }

        for &(first, last) in pages {
            self.verified(first, last.min(self.count), forgot);
        }
    }

    fn scan_pages_slice(
        &self,
        pages: &[(u32, u32)],
        start: u64,
        step: u64,
        forgot: u64,
    ) -> Vec<u32> {
        let mut stale = Vec::new();
        with_scan_scratch(|candidates, current| {
            for &(first, last) in pages {
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
                                Some(0) => {}
                                _ => stale.push(i),
                            }
                            continue;
                        }
                        let key = u64::from(generation) << 32 | u64::from(view);
                        if verified && key != 0 && self.made.peek(i) == Some(key) {
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
        stale
    }

    fn make_stale_staged(
        &self,
        stale: &[u32],
        start: u64,
        step: u64,
        forgot: u64,
        split_pool: Option<&Pool>,
    ) {
        let Some(host) = host() else {
            return;
        };
        if crate::descriptor::ahead::on() {
            crate::descriptor::ahead::changed(stale.len());
        }
        let at = |i: u32| start + u64::from(i) * step;

        let mut guard = self.staged_fast.lock().unwrap_or_else(|e| e.into_inner());
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
        let lookup_epoch = crate::shadow::lookup_epoch();
        let dad = crate::shader_cache::descriptor_as_data();

        // Only populated when a record misses `StagedState` as `own`/`widened`,
        // overflows `STAGED_CAPACITY`, or cannot be resolved (textures/refused).
        let mut overflow_map: HashMap<u64, Option<usize>, Hashing> =
            HashMap::with_hasher(Hashing::new(true));
        let mut overflow_entries: Vec<OverflowEntry> = Vec::new();
        let mut cold_writes: Vec<(D3D12_CPU_DESCRIPTOR_HANDLE, ID3D12Resource, DescSpec)> = Vec::new();
        let mut need_forgot_verify: Vec<(u64, [u32; 3])> = Vec::new();
        let mut distinct_resources: Vec<usize> = Vec::new();
        let mut pins: Vec<ViewPin> = Vec::new();

        for &index in stale {
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

            // Fast O(1) check on persistent staged cache: if this slot was
            // already activated in `pass_id`, skip immediately with zero allocs
            // or refcount traffic!
            if let Some(cached) = state.by_key.get_mut(&key) {
                let s = cached.slot as usize;
                if state.slot_pass[s] == pass_id {
                    continue;
                }
                if cached.lookup_epoch == lookup_epoch
                    && cached.descriptor_as_data == dad
                    && let Some(shadow) = cached.shadow.upgrade()
                    && Interface::as_raw(&shadow.buffer) as usize == cached.raw_buffer
                {
                    shadow.touch();
                    let s_idx = state.pass_shadows.len() as u32;
                    state.pass_shadows.push(shadow);
                    state.slot_shadow_idx[s] = s_idx;
                    state.slot_pass[s] = pass_id;
                    if cached.verified_forgot != forgot && cached.resources != [0, 0] {
                        need_forgot_verify.push((key, record));
                    }
                    for r in cached.resources.into_iter().filter(|&r| r != 0) {
                        if !distinct_resources.contains(&r) {
                            distinct_resources.push(r);
                        }
                    }
                    continue;
                }
            }

            if overflow_map.contains_key(&key) {
                continue;
            }

            // Cache miss or invalidated epoch: resolve fresh.
            let Some(view_pin) = view_at(view_idx, generation) else {
                overflow_map.insert(key, None);
                continue;
            };
            let Some(fresh) = resolve_fast(&view_pin, dad) else {
                overflow_map.insert(key, None);
                continue;
            };
            let resources = match &*view_pin {
                View::Shader(r, _) => [*r, 0],
                View::Unordered(r, c, _) => [*r, *c],
                _ => [0, 0],
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
                            verified_forgot: forgot,
                            descriptor_as_data: dad,
                            stride: fresh.stride,
                            mark: fresh.mark,
                            addr_words: fresh.addr_words,
                            seen_addr: fresh.seen_addr,
                            seen_len: fresh.seen_len,
                            seen_uav: fresh.seen_uav,
                            resources,
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

        // Pin each distinct `Resource` once for the pass under a single shared
        // hold of `VIEWS`, and evict any cached view invalidated by `forget()`.
        let mut raw_pins = RawPins(Vec::with_capacity(distinct_resources.len()));
        if !distinct_resources.is_empty() || !need_forgot_verify.is_empty() {
            let views = views_read();
            let mut dead_resources: Vec<usize> = Vec::new();
            for r in distinct_resources {
                if crate::object::try_add_ref(r as This) {
                    raw_pins.0.push(r);
                } else {
                    dead_resources.push(r);
                }
            }
            for (key, [_, view_idx, generation]) in need_forgot_verify {
                let alive = if inline::kind(generation) != 0 {
                    inline::current(view_idx, generation)
                } else {
                    views
                        .list
                        .get(view_idx as usize)
                        .is_some_and(|(_, cur)| *cur == generation)
                };
                if alive {
                    if let Some(c) = state.by_key.get_mut(&key) {
                        c.verified_forgot = forgot;
                    }
                } else if let Some(evicted) = state.by_key.remove(&key) {
                    state.slot_owner[evicted.slot as usize] = 0;
                    state.slot_pass[evicted.slot as usize] = 0;
                    overflow_map.insert(key, None);
                }
            }
            if !dead_resources.is_empty() {
                state.by_key.retain(|&key, c| {
                    let hit_dead = c.resources.iter().any(|r| *r != 0 && dead_resources.contains(r));
                    if hit_dead {
                        state.slot_owner[c.slot as usize] = 0;
                        state.slot_pass[c.slot as usize] = 0;
                        overflow_map.insert(key, None);
                    }
                    !hit_dead
                });
            }
        }

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

        let addresses_data = self
            .addresses
            .get()
            .and_then(Option::as_ref)
            .map(|&(_, data, _)| data);
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
                                    addresses_data,
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
                        addresses_data,
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
                    addresses_data,
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
        addresses_data: Option<usize>,
        worker_on: bool,
    ) {
        let mut census = self.viewed.batch();
        let pass = self.viewed.pass();
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
                    addresses_data,
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
                        addresses_data,
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
                        addresses_data,
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
        addresses_data: Option<usize>,
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
        if let Some(data) = addresses_data
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
        addresses_data: Option<usize>,
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
        if let Some(data) = addresses_data
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
        addresses_data: Option<usize>,
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
        if let Some(data) = addresses_data
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
    use std::time::Instant;

    use super::*;
    use crate::descriptor::host::split_tests::{Title, random, shown};
    use crate::mac_switches::tests::HeapSyncFastHere;

    /// Differential test on real D3D12 + Wine heaps: verifies that
    /// `VICTORY_HEAP_SYNC_FAST=1` leaves every slot's `made`, `kept`, `ready`,
    /// `strides`, `marks`, `view_addresses`, and `bindless::Viewed` state
    /// identical to the default path across buffer SRVs, CBVs, buffer UAVs,
    /// placed buffer SRVs (including off-element own copies), refused views,
    /// cleared slots, resource release (`forget`), and ring-heap overwrites.
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
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(res_a, 0x10);
        let mem_a = crate::resource::buffer_bytes(res_a).unwrap().0 as u64;

        let mut records = vec![[0u32; 8]; SLOTS as usize].into_boxed_slice();
        let start = records.as_mut_ptr() as u64;
        let at = |i: u32| start + 32 * u64::from(i);
        let null = std::ptr::null();

        let (Some(mut def_heap), Some(mut fast_heap)) = (
            HostHeap::new_as(false, SLOTS, true),
            HostHeap::new_as(false, SLOTS, true),
        ) else {
            release(res_a);
            release(res_b);
            return;
        };
        def_heap.viewed = Arc::new(crate::bindless::Viewed::with_batch(true, true));
        fast_heap.viewed = Arc::new(crate::bindless::Viewed::with_batch(true, true));
        assert!(def_heap.view_addresses().is_some());
        assert!(fast_heap.view_addresses().is_some());

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

            for &(first, last) in &page_ranges {
                let stale = def_heap.stale_slots(first, last, start, 32);
                if stale.len() >= crate::descriptor::pool::WORTH {
                    def_heap.sync_wide(&pool, &stale, &at);
                } else {
                    def_heap.make_stale(stale, start, 32);
                }
            }
            {
                let _pin = HeapSyncFastHere::new();
                fast_heap.sync_pages_fast_with(&page_ranges, start, 32, Some(&pool));
            }

            for i in 0..SLOTS {
                assert_eq!(
                    shown(&fast_heap, i),
                    shown(&def_heap, i),
                    "round {round} slot {i} mismatch"
                );
                assert_eq!(
                    fast_heap.address_at(i),
                    def_heap.address_at(i),
                    "round {round} slot {i} view_addresses mismatch"
                );
                let mark_read = |h: &HostHeap| {
                    h.marks.as_ref().map(|(_, data)| {
                        // SAFETY: 2 dwords per slot in mapped `marks` buffer.
                        unsafe {
                            let p = (*data as *const u32).add(2 * i as usize);
                            (p.read_volatile(), p.add(1).read_volatile())
                        }
                    })
                };
                assert_eq!(
                    mark_read(&fast_heap),
                    mark_read(&def_heap),
                    "round {round} slot {i} mark mismatch"
                );
            }
        }

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host_dev.log_messages(), None);
        drop((def_heap, fast_heap));
        crate::host::release_retired();
        release(res_a);
    }

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

        let (Some(mut def_heap), Some(mut fast_heap)) = (
            HostHeap::new_as(false, SLOTS, true),
            HostHeap::new_as(false, SLOTS, true),
        ) else {
            return;
        };
        def_heap.viewed = Arc::new(crate::bindless::Viewed::with_batch(true, true));
        fast_heap.viewed = Arc::new(crate::bindless::Viewed::with_batch(true, true));
        let _ = def_heap.view_addresses();
        let _ = fast_heap.view_addresses();

        for (scenario, pages_per_sub, writes_per_page) in [
            (
                "world-sparse (737K slots scanned across 20 subs, ~72 changed/page = 25,920 slots/frame)",
                18usize,
                72u32,
            ),
            (
                "world-heavy (737K slots scanned across 20 subs, ~112 changed/page = 40,320 slots/frame)",
                18usize,
                112u32,
            ),
        ] {
            let mut def_us = Vec::new();
            let mut fast_us = Vec::new();
            let mut next = random(42);

            for i in 0..SLOTS {
                if !i.is_multiple_of(3) {
                    title.write(i, Some(&title.views[(i as usize * 7) % title.views.len()]));
                }
            }
            let all_pages: Vec<(u32, u32)> = (0..PAGES)
                .map(|p| (p * SLOTS_PER_PAGE, (p + 1) * SLOTS_PER_PAGE))
                .collect();
            for &(f, l) in &all_pages {
                let s = def_heap.stale_slots(f, l, title.start(), 32);
                def_heap.sync_wide(&pool, &s, &|i| title.at(i));
            }
            {
                let _pin = HeapSyncFastHere::new();
                fast_heap.sync_pages_fast_with(&all_pages, title.start(), 32, Some(&pool));
            }
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

                        if fast_on {
                            let _pin = HeapSyncFastHere::new();
                            let t0 = Instant::now();
                            heap.sync_pages_fast_with(&dirty_pages, title.start(), 32, Some(&pool));
                            frame_us += t0.elapsed().as_micros() as u64;
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
            eprintln!(
                "[{scenario}]\n  default: median {:>6} us ({:.2} ms/frame)\n  fast:    median {:>6} us ({:.2} ms/frame)  ({:.1}x faster, {:+.1}%)",
                d_med,
                d_med as f64 / 1e3,
                f_med,
                f_med as f64 / 1e3,
                d_med as f64 / f_med.max(1) as f64,
                (f_med as f64 - d_med as f64) * 100.0 / d_med.max(1) as f64,
            );
        }

        // SAFETY: plain device health check.
        assert!(unsafe { host_dev.device.GetDeviceRemovedReason() }.is_ok());
        drop((def_heap, fast_heap));
        crate::host::release_retired();
    }
}
