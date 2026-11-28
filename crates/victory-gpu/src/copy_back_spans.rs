//! mac: word-scanned diff, three-way merge and coalesced writeback spans for
//! [`crate::shadow`] (`VICTORY_COPY_BACK_SPANS=1`).
//!
//! Marked and snapshot copy-backs previously compared and merged byte by byte
//! (writing every byte of `host`, a GPU-mapped buffer, even where nothing
//! changed) and wrote whole 64 KiB pages back into title memory whenever a
//! single byte differed. Here clean 4 KiB blocks are skipped by slice
//! comparison, differing words are scanned 8 bytes at a time without touching
//! unchanged GPU memory, and adjacent GPU changes within [`COALESCE_GAP`]
//! bytes are joined into one writeback span without bridging across page
//! boundaries unless the spans meet byte-for-byte.
//!
//! Plain std, so its tests run on every host.

#![cfg_attr(not(windows), allow(dead_code))]

/// Maximum unchanged gap (in bytes) bridged within one already-mapped piece so
/// strided shader writes do not issue a kernel write per element. Never
/// bridges across piece boundaries unless two spans touch (`end == start`).
pub(crate) const COALESCE_GAP: usize = 1024;

const BLOCK: usize = 4096;

/// Appends `[from, to)` to `out`, joining with the previous span when both lie
/// inside the current piece (`last_end >= piece_base`) and are at most
/// `coalesce` bytes apart, or when they meet (`last_end == from`) across
/// adjacent pieces.
#[inline]
pub(crate) fn push_span(
    out: &mut Vec<(usize, usize)>,
    piece_base: usize,
    from: usize,
    to: usize,
    coalesce: usize,
) {
    if from >= to {
        return;
    }
    if let Some(&mut (_, ref mut end)) = out.last_mut() {
        let gap_limit = if *end >= piece_base { coalesce } else { 0 };
        if from <= end.saturating_add(gap_limit) {
            *end = (*end).max(to);
            return;
        }
    }
    out.push((from, to));
}

#[inline]
fn diff_word_bounds(diff: u64) -> (usize, usize) {
    debug_assert_ne!(diff, 0);
    let first = (diff.trailing_zeros() / 8) as usize;
    let last = 8 - (diff.leading_zeros() / 8) as usize;
    (first, last)
}

#[inline]
fn first_diff_in(h: &[u8], t: &[u8]) -> Option<usize> {
    let n = h.len();
    let words = n & !7;
    let mut off = 0;
    while off < words {
        let hw = u64::from_le_bytes(h[off..off + 8].try_into().unwrap());
        let tw = u64::from_le_bytes(t[off..off + 8].try_into().unwrap());
        let d = hw ^ tw;
        if d != 0 {
            return Some(off + (d.trailing_zeros() / 8) as usize);
        }
        off += 8;
    }
    while off < n {
        if h[off] != t[off] {
            return Some(off);
        }
        off += 1;
    }
    None
}

#[inline]
fn last_diff_end_in(h: &[u8], t: &[u8]) -> Option<usize> {
    let mut i = h.len();
    let tail = i & 7;
    for _ in 0..tail {
        i -= 1;
        if h[i] != t[i] {
            return Some(i + 1);
        }
    }
    while i >= 8 {
        i -= 8;
        let hw = u64::from_le_bytes(h[i..i + 8].try_into().unwrap());
        let tw = u64::from_le_bytes(t[i..i + 8].try_into().unwrap());
        let d = hw ^ tw;
        if d != 0 {
            return Some(i + 8 - (d.leading_zeros() / 8) as usize);
        }
    }
    None
}

/// Compares `host` against `title` (of equal length, starting at `base_addr`
/// in title memory), appending coalesced `[from, to)` byte spans where `host`
/// differs from `title` to `out`. Returns whether any byte differed.
///
/// Clean 4 KiB blocks are skipped by slice equality, and open spans extend by
/// probing backwards from `span_end + coalesce + 1`, so dense 64 KiB pages
/// touch one word per `coalesce` window instead of scanning every word.
pub(crate) fn diff_spans(
    host: &[u8],
    title: &[u8],
    base_addr: usize,
    coalesce: usize,
    out: &mut Vec<(usize, usize)>,
) -> bool {
    debug_assert_eq!(host.len(), title.len());
    let len = host.len();
    let mut any = false;
    let mut pos = 0;
    while pos < len {
        let blk_end = ((pos / BLOCK + 1) * BLOCK).min(len);
        if host[pos..blk_end] == title[pos..blk_end] {
            pos = blk_end;
            continue;
        }
        let Some(rel_start) = first_diff_in(&host[pos..blk_end], &title[pos..blk_end]) else {
            pos = blk_end;
            continue;
        };
        any = true;
        let span_start = pos + rel_start;
        let mut span_end = span_start + 1;
        while span_end < len {
            let limit = (span_end + coalesce + 1).min(len);
            match last_diff_end_in(&host[span_end..limit], &title[span_end..limit]) {
                Some(rel_end) => span_end += rel_end,
                None => {
                    pos = limit;
                    break;
                }
            }
            if span_end == len {
                pos = len;
                break;
            }
        }
        if span_end == len {
            pos = len;
        }
        push_span(out, base_addr, base_addr + span_start, base_addr + span_end, coalesce);
    }
    any
}

/// Three-way bytewise writeback merge of `title`, `host` and `old` (starting
/// at `base_addr` in title memory):
/// - Where `host[i] == old[i]`, the title's byte `title[i]` is kept and copied
///   into `host[i]`.
/// - Where `host[i] != old[i]`, the GPU's byte `host[i]` wins and is written
///   into `title[i]`.
///
/// Unchanged 8-byte words (`host == title`) are skipped without reading `old`
/// or writing either slice, so clean cache lines in GPU-mapped `host` memory
/// stay untouched. Appends coalesced `[from, to)` spans where the merged
/// result differs from `title` to `out`, and returns whether any byte of
/// `title` changed.
pub(crate) fn merge_spans(
    title: &mut [u8],
    host: &mut [u8],
    old: &[u8],
    base_addr: usize,
    coalesce: usize,
    out: &mut Vec<(usize, usize)>,
) -> bool {
    debug_assert_eq!(title.len(), host.len());
    debug_assert_eq!(title.len(), old.len());
    let len = title.len();
    let words = len & !7;
    let mut changed = false;
    let mut off = 0;
    while off < words {
        let t = u64::from_le_bytes(title[off..off + 8].try_into().unwrap());
        let h = u64::from_le_bytes(host[off..off + 8].try_into().unwrap());
        if h == t {
            off += 8;
            continue;
        }
        let o = u64::from_le_bytes(old[off..off + 8].try_into().unwrap());
        if h == o {
            // Only the CPU changed this word: host takes title's bytes.
            host[off..off + 8].copy_from_slice(&t.to_le_bytes());
        } else if t == o {
            // Only the GPU changed this word: title takes host's bytes, host
            // already holds them.
            title[off..off + 8].copy_from_slice(&h.to_le_bytes());
            let (b0, b1) = diff_word_bounds(h ^ t);
            let abs = base_addr + off;
            push_span(out, base_addr, abs + b0, abs + b1, coalesce);
            changed = true;
        } else {
            // Both CPU and GPU modified bytes within the same 8-byte word.
            for b in 0..8 {
                let i = off + b;
                let (tb, hb, ob) = (title[i], host[i], old[i]);
                if hb == ob {
                    if hb != tb {
                        host[i] = tb;
                    }
                } else if tb != hb {
                    title[i] = hb;
                    let abs = base_addr + i;
                    push_span(out, base_addr, abs, abs + 1, coalesce);
                    changed = true;
                }
            }
        }
        off += 8;
    }
    while off < len {
        let (tb, hb, ob) = (title[off], host[off], old[off]);
        if hb == ob {
            if hb != tb {
                host[off] = tb;
            }
        } else if tb != hb {
            title[off] = hb;
            let abs = base_addr + off;
            push_span(out, base_addr, abs, abs + 1, coalesce);
            changed = true;
        }
        off += 1;
    }
    changed
}

/// Converts sorted, non-overlapping byte spans `[from, to)` into sorted,
/// merged page-number intervals `[first_page, last_page)` for `page_size`.
pub(crate) fn page_runs(spans: &[(usize, usize)], page_size: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &(from, to) in spans {
        if from >= to {
            continue;
        }
        let (p0, p1) = (from / page_size, to.div_ceil(page_size));
        match runs.last_mut() {
            Some((_, last)) if p0 <= *last => *last = (*last).max(p1),
            _ => runs.push((p0, p1)),
        }
    }
    runs
}

/// Counts distinct `block_size`-aligned blocks overlapped by sorted,
/// non-overlapping byte spans `[from, to)` without allocating.
pub(crate) fn block_count(spans: &[(usize, usize)], block_size: usize) -> u64 {
    let mut total = 0u64;
    let mut covered_end = 0usize;
    for &(from, to) in spans {
        if from >= to {
            continue;
        }
        let b0 = (from / block_size).max(covered_end);
        let b1 = to.div_ceil(block_size);
        if b1 > b0 {
            total += (b1 - b0) as u64;
            covered_end = b1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle_merge(title: &[u8], host: &[u8], old: &[u8]) -> (Vec<u8>, bool) {
        let expected: Vec<u8> = host
            .iter()
            .zip(old)
            .zip(title)
            .map(|((&gpu, &prev), &cpu)| if gpu == prev { cpu } else { gpu })
            .collect();
        let changed = expected != title;
        (expected, changed)
    }

    #[test]
    fn diff_spans_finds_exact_bytes_and_coalesces_within_piece_only() {
        let page = 64 << 10;
        let title = vec![0x11u8; 2 * page];
        let mut host = title.clone();
        // Page 0: two changes 200 bytes apart (coalesced at COALESCE_GAP=1024),
        // and one 4000 bytes further on (separate span), ending at page end.
        host[10] = 0xaa;
        host[210] = 0xbb;
        host[4500] = 0xcc;
        host[page - 1] = 0xdd;
        // Page 1: starts at byte 0 (touches page 0's end -> merges across boundary),
        // and another change at +5000.
        host[page] = 0xee;
        host[page + 5000] = 0xff;

        let mut spans = Vec::new();
        assert!(diff_spans(&host[..page], &title[..page], 0x10_0000, COALESCE_GAP, &mut spans));
        assert!(diff_spans(
            &host[page..],
            &title[page..],
            0x10_0000 + page,
            COALESCE_GAP,
            &mut spans
        ));
        assert_eq!(
            spans,
            vec![
                (0x10_0000 + 10, 0x10_0000 + 211),
                (0x10_0000 + 4500, 0x10_0000 + 4501),
                (0x10_0000 + page - 1, 0x10_0000 + page + 1),
                (0x10_0000 + page + 5000, 0x10_0000 + page + 5001),
            ]
        );

        // Non-touching changes across a piece boundary never bridge the boundary
        // even when their distance is below `coalesce`.
        let mut gap_spans = Vec::new();
        let mut h2 = title.clone();
        h2[page - 4] = 0x77;
        h2[page + 4] = 0x88;
        assert!(diff_spans(&h2[..page], &title[..page], 0, COALESCE_GAP, &mut gap_spans));
        assert!(diff_spans(&h2[page..], &title[page..], page, COALESCE_GAP, &mut gap_spans));
        assert_eq!(gap_spans, vec![(page - 4, page - 3), (page + 4, page + 5)]);
    }

    #[test]
    fn merge_spans_matches_bytewise_oracle_and_reconstructs_title_via_spans() {
        for len in [0usize, 1, 7, 8, 9, 63, 64, 65, 193, 4103, 65_536] {
            let old: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(17) ^ 0x5a) as u8).collect();
            for (cpu_stride, gpu_stride) in [(3, 5), (1, 67), (64, 128), (len + 1, 9), (7, len + 1)] {
                let mut title = old.clone();
                let mut host = old.clone();
                for i in (0..len).step_by(cpu_stride) {
                    title[i] = title[i].wrapping_add(0x21);
                }
                for i in (0..len).step_by(gpu_stride) {
                    host[i] = if i % 2 == 0 { title[i] } else { old[i] ^ 0xa5 };
                }
                let orig_title = title.clone();
                let (expected, expected_changed) = oracle_merge(&title, &host, &old);

                let base = 0x40_0000;
                let mut spans = Vec::new();
                let changed = merge_spans(&mut title, &mut host, &old, base, COALESCE_GAP, &mut spans);
                assert_eq!(changed, expected_changed, "len={len} strides=({cpu_stride},{gpu_stride})");
                assert_eq!(title, expected, "title mismatch at len={len}");
                assert_eq!(host, expected, "host mismatch at len={len}");

                // Applying only the emitted spans from `host` onto `orig_title`
                // must also produce `expected` byte-for-byte.
                let mut applied = orig_title;
                for &(from, to) in &spans {
                    applied[from - base..to - base].copy_from_slice(&host[from - base..to - base]);
                }
                assert_eq!(applied, expected, "span replay mismatch at len={len}");
            }
        }
    }

    #[test]
    fn page_runs_merges_overlapping_and_adjacent_pages() {
        let p = 64 << 10;
        let spans = [
            (10, 20),
            (500, 700),
            (p - 4, p + 4),
            (2 * p + 100, 2 * p + 200),
            (4 * p, 5 * p),
        ];
        assert_eq!(page_runs(&spans, p), vec![(0, 3), (4, 5)]);
    }

    #[test]
    fn diff_spans_matches_bytewise_oracle_across_coalesce_sizes_and_patterns() {
        let oracle = |host: &[u8], title: &[u8], base: usize, coalesce: usize| {
            let mut out = Vec::new();
            for i in 0..host.len() {
                if host[i] != title[i] {
                    push_span(&mut out, base, base + i, base + i + 1, coalesce);
                }
            }
            out
        };
        for len in [0usize, 1, 7, 8, 9, 64, 1023, 1024, 1025, 4096, 8195, 65_536] {
            let title = vec![0x42u8; len];
            for coalesce in [0usize, 1, 8, 64, 1024] {
                // Clean, full dense, strided, and pseudo-random bursts.
                for mode in 0..4 {
                    let mut host = title.clone();
                    match mode {
                        0 => {}
                        1 => host.fill(0x99),
                        2 => {
                            for i in (0..len).step_by(317) {
                                host[i] = 0x77;
                            }
                        }
                        _ => {
                            let mut x = 0x1234_5678_9abc_def0u64 ^ len as u64;
                            for _ in 0..24 {
                                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                                if len > 0 {
                                    let idx = (x as usize) % len;
                                    let run = ((x >> 32) as usize % 48).min(len - idx);
                                    host[idx..idx + run].fill(0xcc);
                                }
                            }
                        }
                    }
                    let expected = oracle(&host, &title, 0x20_0000, coalesce);
                    let mut actual = Vec::new();
                    let differs = diff_spans(&host, &title, 0x20_0000, coalesce, &mut actual);
                    assert_eq!(differs, !expected.is_empty());
                    assert_eq!(actual, expected, "len={len} coalesce={coalesce} mode={mode}");
                    if coalesce < BLOCK {
                        let base = 0x20_0000;
                        let end = base + len;
                        let mut oracle_blocks = 0u64;
                        for blk in base / BLOCK..end.div_ceil(BLOCK) {
                            let a = (blk * BLOCK).max(base) - base;
                            let b = ((blk + 1) * BLOCK).min(end) - base;
                            if a < b && host[a..b] != title[a..b] {
                                oracle_blocks += 1;
                            }
                        }
                        assert_eq!(
                            block_count(&actual, BLOCK),
                            oracle_blocks,
                            "4 KiB block count mismatch at len={len} coalesce={coalesce} mode={mode}"
                        );
                    }
                }
            }
        }
    }
}
