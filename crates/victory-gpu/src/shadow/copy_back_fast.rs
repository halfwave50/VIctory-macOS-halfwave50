//! mac: fast marked and snapshot copy-back (`VICTORY_COPY_BACK_SPANS=1`).
//!
//! Kept in its own file so upstream changes to `shadow.rs` apply cleanly.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

#[cfg(test)]
use super::MARKED_SPLIT;
use super::{
    SPLIT_MIN_PAGES, SPLIT_PARTS, SNAPSHOTS, Shadow, Snapshot, copy_back_split, give_back,
    record_marked_blocks, split_said,
};
use crate::copy_back_spans::{COALESCE_GAP, block_count, diff_spans, merge_spans, page_runs};
use crate::watch::{MarkedPage, PAGE};

type ScanOut = (crate::watch::Runs, Vec<usize>, u64);

/// Chunk size when splitting large contiguous writeback spans across workers
/// under `VICTORY_COPY_BACK_SPLIT=1` (256 KiB = 4 × 64 KiB pages).
const WRITE_CHUNK_BYTES: usize = 4 * PAGE;

const BLOCK: usize = 4096;

thread_local! {
    /// Warm 64 KiB page buffer per thread, avoiding repeated heap allocations
    /// and zero-fill page faults while staying hot in L1/L2 cache.
    static SCAN_BUF: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}

fn with_scan_buf<R>(need: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let mut buf = SCAN_BUF.take();
    if buf.len() < need {
        buf.resize(need.max(PAGE), 0);
    }
    let out = f(&mut buf[..need]);
    SCAN_BUF.set(buf);
    out
}

impl Shadow {
    /// `Shadow::copy_back_or_not` with word-scanned three-way merge and
    /// coalesced span writes, reading, writing, and re-guarding one 64 KiB
    /// piece at a time so concurrent title writes outside the GPU spans are
    /// never cleared from the changed set without reaching the shadow.
    pub(super) fn copy_back_or_not_spans(&self, address: usize, len: usize, back: bool) -> usize {
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Snapshot(segments, compared, spans) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let before = if !spans.is_empty() {
            spans.covers(from, to).then_some(None)
        } else {
            segments
                .iter()
                .find(|(at, bytes)| *at <= from && to <= at + bytes.len())
                .map(|(taken, before)| Some(&before[from - taken..to - taken]))
        };
        if let Some(before) = before
            && back
            && from < to
        {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(started, crate::copystat::C::CompareUs);
            const PIECE: usize = 64 << 10;
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            let mut scratch = Vec::new();
            let mut piece_writes: Vec<(usize, usize)> = Vec::new();
            with_scan_buf(PIECE, |title_buf| {
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the mapping; only the submitting thread writes it.
                    let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let title = &mut title_buf[..n];
                    if n == PIECE && !compared.insert(start + at) {
                        at += n;
                        continue;
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                    SNAPSHOTS[4].fetch_add(n as u64, Ordering::Relaxed);
                    let old = match before {
                        Some(before) => {
                            let old = &before[at - from..at - from + n];
                            (*host != *old).then_some(old)
                        }
                        None => spans.changed_piece(at, host, &mut scratch),
                    };
                    if let Some(old) = old
                        && crate::host::read_into((start + at) as *const u8, title)
                    {
                        piece_writes.clear();
                        if merge_spans(title, host, old, start + at, COALESCE_GAP, &mut piece_writes)
                        {
                            let _writing = crate::watch::writing(start + at, n);
                            for &(w_from, w_to) in &piece_writes {
                                let w_len = w_to - w_from;
                                // SAFETY: `w_from..w_to` lies inside `[start + at, start + at + n)`
                                // and `merge_spans` stored the merged bytes in `self.data`.
                                let src = unsafe { self.data.add(w_from - start) };
                                crate::host::write_mapped(w_from as *mut u8, src, w_len);
                                crate::alias_verify::gpu_wrote(self, w_from, w_len);
                                written += w_len;
                            }
                        }
                        let page = (start + at) / PAGE;
                        if let Some(t) = &self.tracked
                            && (first..last).contains(&page)
                        {
                            t.taken(&[page]);
                        }
                    }
                    if self.has_widened.load(Ordering::Acquire) {
                        self.refresh_widened(start + at, start + at + n);
                    }
                    at += n;
                }
            });
            SNAPSHOTS[5].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if !back && from < to && self.has_widened.load(Ordering::Acquire) {
            self.refresh_widened(start + from, start + to);
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            spans.clear();
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// `Shadow::copy_back_marked` with batched tracker pre-pass, warm L2-sized
    /// scan buffers, jump-ahead span diffing, coalesced changed-span writes,
    /// and a single batched guard open/re-guard round trip. Worker splitting
    /// runs only when `VICTORY_COPY_BACK_SPLIT` asks for it and no page of the
    /// copy is shown at two addresses (matching `Shadow::copy_back_marked`).
    pub(super) fn copy_back_marked_spans(&self, pages: &[usize]) -> usize {
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let has_widened = self.has_widened.load(Ordering::Acquire);
        let long = copy_back_split() && pages.len() >= SPLIT_MIN_PAGES && !has_widened;
        let split = long && !crate::watch::aliased(start, end - start);
        if long {
            split_said(split, pages.len());
        }
        #[cfg(test)]
        MARKED_SPLIT.with(|taken| taken.set(Some(split)));
        let planned = self.plan_marked(t, pages, start, end);
        if !split {
            let (writes, mut taken, blocks_compared) =
                self.scan_marked(t, &planned, start, has_widened);
            t.unarm_marked(&planned, &mut taken);
            record_marked_blocks(block_count(&writes, BLOCK), blocks_compared);
            return self.write_marked_spans(t, &writes, &taken, start, false);
        }
        let parts = planned.len().div_ceil(SPLIT_MIN_PAGES / 2).clamp(1, SPLIT_PARTS);
        let (writes, mut taken, blocks_compared) = if parts >= 2 {
            let each = planned.len().div_ceil(parts);
            let done: Vec<Mutex<ScanOut>> = (0..parts).map(|_| Mutex::default()).collect();
            crate::texture::in_parallel(parts, |k| {
                let lo = (k * each).min(planned.len());
                let hi = ((k + 1) * each).min(planned.len());
                *done[k].lock().unwrap_or_else(|e| e.into_inner()) =
                    self.scan_marked(t, &planned[lo..hi], start, false);
            });
            let (mut writes, mut taken, mut blocks_compared) = (Vec::new(), Vec::new(), 0u64);
            for part in done {
                let (w, p, bc) = part.into_inner().unwrap_or_else(|e| e.into_inner());
                blocks_compared += bc;
                for (w_from, w_to) in w {
                    if let Some(&mut (_, ref mut prev_to)) = writes.last_mut()
                        && *prev_to == w_from
                    {
                        *prev_to = w_to;
                        continue;
                    }
                    writes.push((w_from, w_to));
                }
                taken.extend(p);
            }
            (writes, taken, blocks_compared)
        } else {
            self.scan_marked(t, &planned, start, false)
        };
        t.unarm_marked(&planned, &mut taken);
        record_marked_blocks(block_count(&writes, BLOCK), blocks_compared);
        self.write_marked_spans(t, &writes, &taken, start, true)
    }

    fn plan_marked(
        &self,
        t: &crate::watch::Tracked,
        pages: &[usize],
        start: usize,
        end: usize,
    ) -> Vec<MarkedPage> {
        let candidates: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&page| {
                let (from, to) = (page.max(start), (page + PAGE).min(end));
                from < to && !self.kept_early(page)
            })
            .collect();
        t.prepare_marked(&candidates, start, end)
    }

    fn scan_marked(
        &self,
        t: &crate::watch::Tracked,
        planned: &[MarkedPage],
        start: usize,
        has_widened: bool,
    ) -> ScanOut {
        let mut writes = Vec::new();
        let mut taken = Vec::new();
        let mut blocks_compared = 0u64;
        with_scan_buf(PAGE, |buf| {
            for item in planned {
                let n = item.to - item.from;
                if crate::host::read_into(item.from as *const u8, &mut buf[..n]) {
                    blocks_compared += (item.to.div_ceil(BLOCK) - item.from / BLOCK) as u64;
                    let late_saved = if item.saved.is_none() && !t.clean() {
                        t.saved(item.page / PAGE)
                    } else {
                        None
                    };
                    self.scan_one_marked(
                        item,
                        late_saved.as_deref(),
                        &mut buf[..n],
                        start,
                        has_widened,
                        &mut writes,
                        &mut taken,
                    );
                }
            }
        });
        (writes, taken, blocks_compared)
    }

    #[inline]
    fn scan_one_marked(
        &self,
        item: &MarkedPage,
        late_saved: Option<&[u8]>,
        title: &mut [u8],
        start: usize,
        has_widened: bool,
        writes: &mut Vec<(usize, usize)>,
        taken: &mut Vec<usize>,
    ) {
        let n = item.to - item.from;
        // SAFETY: `item.from..item.to` is inside this shadow and disjoint across workers.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(item.from - start), n) };
        match item.saved.as_deref().or(late_saved) {
            None => {
                diff_spans(host, title, item.from, COALESCE_GAP, writes);
            }
            Some(before) => {
                let before = &before[item.from - item.page..item.to - item.page];
                merge_spans(title, host, before, item.from, COALESCE_GAP, writes);
            }
        }
        if has_widened {
            self.refresh_widened(item.from, item.to);
        }
        if item.from == item.page && item.to == item.page + PAGE {
            taken.push(item.page / PAGE);
        }
    }

    fn write_marked_spans(
        &self,
        t: &crate::watch::Tracked,
        writes: &[(usize, usize)],
        taken: &[usize],
        start: usize,
        parallel: bool,
    ) -> usize {
        if writes.is_empty() && taken.is_empty() {
            return 0;
        }
        let dirty_pages = page_runs(writes, PAGE);
        let writing = crate::watch::writing_runs(&dirty_pages);
        let write_one = |&(from, to): &(usize, usize)| {
            let len = to - from;
            // SAFETY: `from..to` is inside `self.data`, which holds the merged bytes.
            let src = unsafe { self.data.add(from - start) };
            crate::host::write_mapped(from as *mut u8, src, len);
            len
        };
        let total_bytes: usize = writes.iter().map(|&(a, b)| b - a).sum();
        let written = if parallel && total_bytes >= WRITE_CHUNK_BYTES {
            let mut chunks: Vec<(usize, usize)> = Vec::with_capacity(writes.len().max(SPLIT_PARTS));
            for &(from, to) in writes {
                if to - from <= WRITE_CHUNK_BYTES {
                    chunks.push((from, to));
                } else {
                    let mut at = from;
                    while at < to {
                        let next = ((at / WRITE_CHUNK_BYTES + 1) * WRITE_CHUNK_BYTES).min(to);
                        chunks.push((at, next));
                        at = next;
                    }
                }
            }
            if chunks.len() >= 2 {
                let parts = chunks.len().min(SPLIT_PARTS);
                let each = chunks.len().div_ceil(parts);
                let total = std::sync::atomic::AtomicUsize::new(0);
                crate::texture::in_parallel(parts, |k| {
                    let lo = (k * each).min(chunks.len());
                    let hi = ((k + 1) * each).min(chunks.len());
                    let n: usize = chunks[lo..hi].iter().map(write_one).sum();
                    total.fetch_add(n, Ordering::Relaxed);
                });
                total.into_inner()
            } else {
                chunks.iter().map(write_one).sum()
            }
        } else {
            writes.iter().map(write_one).sum()
        };
        for &(p0, p1) in &dirty_pages {
            crate::alias_verify::gpu_wrote(self, p0 * PAGE, (p1 - p0) * PAGE);
        }
        writing.finish_and_take(t, taken);
        written
    }
}
