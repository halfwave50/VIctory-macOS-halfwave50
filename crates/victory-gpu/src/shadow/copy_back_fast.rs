//! mac: fast marked and snapshot copy-back (`VICTORY_COPY_BACK_SPANS=1`).
//!
//! Kept in its own file so upstream changes to `shadow.rs` apply cleanly.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use super::{
    SPLIT_MIN_PAGES, SPLIT_PARTS, SNAPSHOTS, Shadow, Snapshot, copy_back_split, give_back,
    parallel_copies_enabled,
};
#[cfg(test)]
use super::{MARKED_SPLIT, SPLIT_FORCED};
use crate::copy_back_spans::{COALESCE_GAP, diff_spans, merge_spans, page_runs};
use crate::watch::{MarkedPage, PAGE};

type ScanOut = (crate::watch::Runs, Vec<usize>);

/// Chunk size when splitting large contiguous writeback spans across workers
/// (256 KiB = 4 × 64 KiB pages).
const WRITE_CHUNK_BYTES: usize = 4 * PAGE;

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
    /// `Shadow::copy_back_or_not` with word-scanned three-way merge, coalesced
    /// span writes, and a single guard open/close+re-guard pair.
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
            let mut scratch = Vec::new();
            let mut synced = Vec::new();
            let mut writes: Vec<(usize, usize)> = Vec::new();
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
                        merge_spans(title, host, old, start + at, COALESCE_GAP, &mut writes);
                        synced.push((start + at) / PAGE);
                    }
                    if self.has_widened.load(Ordering::Acquire) {
                        self.refresh_widened(start + at, start + at + n);
                    }
                    at += n;
                }
            });
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
            let dirty_pages = page_runs(&writes, PAGE);
            let writing = crate::watch::writing_runs(&dirty_pages, self.tracked.as_ref());
            for &(w_from, w_to) in &writes {
                let w_len = w_to - w_from;
                // SAFETY: `w_from..w_to` lies inside `[start + from, start + to)` and
                // `merge_spans` stored the merged bytes in `self.data`.
                let src = unsafe { self.data.add(w_from - start) };
                crate::host::write_mapped(w_from as *mut u8, src, w_len);
                written += w_len;
            }
            for &(p0, p1) in &dirty_pages {
                crate::alias_verify::gpu_wrote(self, p0 * PAGE, (p1 - p0) * PAGE);
            }
            match &self.tracked {
                Some(t) => writing.finish_and_take(t, &whole),
                None => drop(writing),
            }
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
    /// scan buffers, per-page alias partitioning for parallel workers,
    /// jump-ahead span diffing, chunked parallel writes, and vetted re-guarding.
    pub(super) fn copy_back_marked_spans(&self, pages: &[usize]) -> usize {
        let _source = self.widened_source.lock().unwrap_or_else(|e| e.into_inner());
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let has_widened = self.has_widened.load(Ordering::Acquire);
        let want_parallel = {
            #[cfg(test)]
            {
                SPLIT_FORCED
                    .with(std::cell::Cell::get)
                    .unwrap_or_else(|| parallel_copies_enabled() || copy_back_split())
            }
            #[cfg(not(test))]
            {
                parallel_copies_enabled() || copy_back_split()
            }
        };
        let (unaliased, aliased) = if want_parallel && pages.len() >= SPLIT_MIN_PAGES && !has_widened {
            crate::watch::partition_unaliased(pages)
        } else {
            (Vec::new(), Vec::new())
        };
        let split = want_parallel && unaliased.len() >= SPLIT_MIN_PAGES && !has_widened;
        #[cfg(test)]
        MARKED_SPLIT.with(|taken| taken.set(Some(split)));
        if !split {
            let planned = self.plan_marked(t, pages, start, end);
            let (writes, taken) = self.scan_marked(&planned, start, has_widened);
            return self.write_marked_spans(t, &writes, &taken, start, false);
        }
        let planned = self.plan_marked(t, &unaliased, start, end);
        let parts = planned.len().div_ceil(SPLIT_MIN_PAGES / 2).clamp(1, SPLIT_PARTS);
        let (writes, taken) = if parts >= 2 {
            let each = planned.len().div_ceil(parts);
            let done: Vec<Mutex<ScanOut>> = (0..parts).map(|_| Mutex::default()).collect();
            crate::texture::in_parallel(parts, |k| {
                let lo = (k * each).min(planned.len());
                let hi = ((k + 1) * each).min(planned.len());
                *done[k].lock().unwrap_or_else(|e| e.into_inner()) =
                    self.scan_marked(&planned[lo..hi], start, false);
            });
            let (mut writes, mut taken) = (Vec::new(), Vec::new());
            for part in done {
                let (w, p) = part.into_inner().unwrap_or_else(|e| e.into_inner());
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
            (writes, taken)
        } else {
            self.scan_marked(&planned, start, false)
        };
        let mut total = self.write_marked_spans(t, &writes, &taken, start, true);
        if !aliased.is_empty() {
            let aliased_planned = self.plan_marked(t, &aliased, start, end);
            let (a_writes, a_taken) = self.scan_marked(&aliased_planned, start, has_widened);
            total += self.write_marked_spans(t, &a_writes, &a_taken, start, false);
        }
        total
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
        planned: &[MarkedPage],
        start: usize,
        has_widened: bool,
    ) -> ScanOut {
        let mut writes = Vec::new();
        let mut taken = Vec::new();
        with_scan_buf(PAGE, |buf| {
            for item in planned {
                let n = item.to - item.from;
                if crate::host::read_into(item.from as *const u8, &mut buf[..n]) {
                    self.scan_one_marked(item, &mut buf[..n], start, has_widened, &mut writes, &mut taken);
                }
            }
        });
        (writes, taken)
    }

    #[inline]
    fn scan_one_marked(
        &self,
        item: &MarkedPage,
        title: &mut [u8],
        start: usize,
        has_widened: bool,
        writes: &mut Vec<(usize, usize)>,
        taken: &mut Vec<usize>,
    ) {
        let n = item.to - item.from;
        // SAFETY: `item.from..item.to` is inside this shadow and disjoint across workers.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(item.from - start), n) };
        let differs = match &item.saved {
            None => diff_spans(host, title, item.from, COALESCE_GAP, writes),
            Some(before) => {
                let before = &before[item.from - item.page..item.to - item.page];
                merge_spans(title, host, before, item.from, COALESCE_GAP, writes)
            }
        };
        if has_widened {
            self.refresh_widened(item.from, item.to);
        }
        if item.from == item.page && item.to == item.page + PAGE && (differs || item.was_changed) {
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
        let writing = crate::watch::writing_runs(&dirty_pages, Some(t));
        let write_one = |&(from, to): &(usize, usize)| {
            let len = to - from;
            // SAFETY: `from..to` is inside `self.data`, which holds the merged bytes.
            let src = unsafe { self.data.add(from - start) };
            crate::host::write_mapped(from as *mut u8, src, len);
            len
        };
        let total_bytes: usize = writes.iter().map(|&(a, b)| b - a).sum();
        let written = if parallel && total_bytes >= WRITE_CHUNK_BYTES {
            // Chunk multi-megabyte merged spans into 256 KiB slices so multiple
            // cores saturate memory bandwidth even when dense pages merge into
            // a single contiguous run.
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
