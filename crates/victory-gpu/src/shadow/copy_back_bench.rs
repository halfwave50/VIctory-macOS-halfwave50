//! Integration tests and A/B benchmark for `VICTORY_COPY_BACK_SPANS`
//! (`crate::shadow::copy_back_fast`).
//!
//! Run the benchmark alone, in release, under Wine + D3DMetal:
//! `tools/macos/test-wine.sh --release -p victory-gpu --lib world_marked_and_snapshot_copy_back -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use super::*;
use crate::mac_switches::tests::CopyBackSpansHere;
use crate::watch::{PAGE, tests as watch_tests};

/// Differential check on real D3D12 + Wine memory: runs the default path and
/// `VICTORY_COPY_BACK_SPANS=1` on identical setups (partial end pages, clean
/// marked pages, single-byte GPU writes, armed pages with concurrent title
/// writes, strided GPU writes, and an aliased page elsewhere in a multi-page
/// buffer) and verifies every guest and shadow byte matches.
#[test]
fn copy_back_spans_matches_default_across_marked_and_snapshot_cases() {
    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();

    for spans_on in [false, true] {
        let _pin = spans_on.then(CopyBackSpansHere::new);
        let n_pages = 2 * SPLIT_MIN_PAGES + 6;
        let base = watch_tests::pool(n_pages);
        let start = base + 48;
        let end = base + n_pages * PAGE - 80;
        let len = end - start;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, start as u64, len as u64);

        // Link the last page as aliased to itself-neighbour in watch state to
        // exercise per-page alias partitioning on a large buffer.
        let alias_pool = watch_tests::pool(1);
        crate::watch::title_memory_aliased(alias_pool, &[base + (n_pages - 2) * PAGE]);

        {
            let _w = crate::watch::writing(base, n_pages * PAGE);
            // SAFETY: owned pool allocation.
            unsafe { (base as *mut u8).write_bytes(0x23, n_pages * PAGE) };
        }
        copy.update(start, len);
        copy.writing(start, len, true);

        let write_cpu = |off: usize, val: u8| {
            // SAFETY: owned pool allocation.
            unsafe { ((base + off) as *mut u8).write_volatile(val) };
        };

        // 1) Partial first & last pages: GPU writes near edges.
        copy.write_host(start as u64, &[0xa1, 0xa2, 0xa3]);
        copy.write_host((end - 2) as u64, &[0xe1, 0xe2]);

        // 2) Armed page 1: both CPU and GPU write nearby bytes & same-word bytes.
        write_cpu(PAGE + 11, 0x51);
        write_cpu(PAGE + 20, 0x52);
        copy.write_host((base + PAGE + 10) as u64, &[0xb1]);
        copy.write_host((base + PAGE + 12) as u64, &[0xb2]);
        copy.write_host((base + PAGE + 24) as u64, &[0xb3]);

        // 3) Armed page 2: CPU writes only, GPU leaves untouched.
        write_cpu(2 * PAGE + 64, 0x77);

        // 4) Interior pages: mix of clean, sparse single-dword, strided, and dense.
        for p in 3..n_pages - 1 {
            match p % 4 {
                0 => {} // clean marked page
                1 => copy.write_host((base + p * PAGE + 128) as u64, &[0xc4; 16]),
                2 => {
                    for s in (0..PAGE).step_by(256) {
                        copy.write_host((base + p * PAGE + s) as u64, &[0xd5; 4]);
                    }
                }
                _ => {
                    let block = vec![0xf6u8; PAGE];
                    copy.write_host((base + p * PAGE) as u64, &block);
                }
            }
        }

        let pages: Vec<usize> = (0..n_pages).map(|i| base + i * PAGE).collect();
        let written = copy.copy_back_marked(&pages);
        if spans_on {
            assert!(written < (n_pages / 2) * PAGE, "spans should write far less than full pages: {written}");
        } else {
            assert!(written > (n_pages / 2) * PAGE);
        }
        // Second call must find everything synced and write 0 bytes.
        assert_eq!(copy.copy_back_marked(&pages), 0, "spans_on={spans_on}");

        // SAFETY: owned pool allocation.
        let guest = unsafe { std::slice::from_raw_parts(base as *const u8, n_pages * PAGE) };
        assert_eq!(guest[47], 0x23, "before start untouched");
        assert_eq!(&guest[48..51], &[0xa1, 0xa2, 0xa3]);
        assert_eq!(&guest[n_pages * PAGE - 82..n_pages * PAGE - 80], &[0xe1, 0xe2]);
        assert_eq!(guest[n_pages * PAGE - 80], 0x23, "past end untouched");
        assert_eq!(
            [guest[PAGE + 10], guest[PAGE + 11], guest[PAGE + 12], guest[PAGE + 20], guest[PAGE + 24]],
            [0xb1, 0x51, 0xb2, 0x52, 0xb3],
            "same-word & nearby three-way merge exact (spans_on={spans_on})"
        );
        assert_eq!(guest[2 * PAGE + 64], 0x77, "CPU-only armed page preserved");
        assert_eq!(
            copy.read_host((base + 2 * PAGE + 64) as u64, 1)[0] & 0xff,
            0x77,
            "host shadow took CPU byte on armed page"
        );
        for p in 3..n_pages - 1 {
            match p % 4 {
                0 => assert_eq!(guest[p * PAGE + 128], 0x23),
                1 => {
                    assert_eq!(&guest[p * PAGE + 128..p * PAGE + 144], &[0xc4; 16]);
                    assert_eq!(guest[p * PAGE + 144], 0x23);
                }
                2 => {
                    assert_eq!(&guest[p * PAGE..p * PAGE + 4], &[0xd5; 4]);
                    assert_eq!(guest[p * PAGE + 4], 0x23);
                    assert_eq!(&guest[p * PAGE + 256..p * PAGE + 260], &[0xd5; 4]);
                }
                _ => assert!(guest[p * PAGE..(p + 1) * PAGE].iter().all(|&b| b == 0xf6)),
            }
        }

        // Also check snapshot (`copy_back`) path with CPU + GPU writes.
        copy.copy_back(start, len);
        copy.writing(base + PAGE, 2 * PAGE, false);
        write_cpu(PAGE + 500, 0x31);
        copy.write_host((base + PAGE + 504) as u64, &[0x9a; 12]);
        let snap_written = copy.copy_back(base + PAGE, 2 * PAGE);
        if spans_on {
            assert_eq!(snap_written, 12);
        } else {
            assert_eq!(snap_written, PAGE);
        }
        assert_eq!(guest[PAGE + 500], 0x31);
        assert_eq!(&guest[PAGE + 504..PAGE + 516], &[0x9a; 12]);

        crate::watch::title_memory_remapped(alias_pool, PAGE);
        crate::watch::title_memory_remapped(base, n_pages * PAGE);
        drop(copy);
        watch_tests::free(alias_pool);
        watch_tests::free(base);
    }
}

#[test]
#[ignore = "benchmarks 2,033 marked pages on a 174 MiB D3D12 shadow buffer"]
fn world_marked_and_snapshot_copy_back() {
    const TOTAL_PAGES: usize = 2_784; // ~174 MiB heap
    const MARKED_PAGES: usize = 2_033;
    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();

    let base = watch_tests::pool(TOTAL_PAGES);
    let alias_pool = watch_tests::pool(1);
    let len = TOTAL_PAGES * PAGE;
    let (buffer, data) = host.shared_buffer(len as u64).unwrap();
    // SAFETY: live D3D12 buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);
    // Real world heap has aliased pages elsewhere in the 174 MiB range.
    crate::watch::title_memory_aliased(alias_pool, &[base + (TOTAL_PAGES - 1) * PAGE]);

    {
        let _w = crate::watch::writing(base, len);
        // SAFETY: owned pool allocation.
        unsafe { (base as *mut u8).write_bytes(0x11, len) };
    }
    copy.update(base, len);

    // 2,033 marked pages across several clusters in the 174 MiB heap.
    let pages: Vec<usize> = (0..MARKED_PAGES)
        .map(|i| base + (i + (i / 64) * 16) * PAGE)
        .collect();

    for (scenario, dense_mod) in [
        ("sparse-records (27% clean, 73% 64B counter/indirect records)", usize::MAX),
        ("world-mixed (27% clean, 65% sparse/strided, 8% full-page)", 12usize),
        ("dense-73pct (1,525 full 64 KiB pages = 99.9 MB)", 4usize),
    ] {
        let mut default_us = Vec::new();
        let mut spans_us = Vec::new();
        let mut default_mb = 0.0;
        let mut spans_mb = 0.0;
        let mut fill = vec![0u8; PAGE];

        // 10 alternating A/B rounds (skipping first 2 warm-up pairs).
        for round in 0..10u8 {
            for spans_on in [false, true] {
                let seed = round.wrapping_mul(13).wrapping_add(u8::from(spans_on) * 7).max(1);
                copy.writing(base, (TOTAL_PAGES - 4) * PAGE, true);
                for (k, &page) in pages.iter().enumerate() {
                    if k % 4 == 0 {
                        // ~25-27% clean marked pages (GPU wrote identical value)
                        continue;
                    }
                    let val = seed ^ (k as u8);
                    if dense_mod == 4 || k % dense_mod == 1 {
                        // Full 64 KiB page overwrite
                        fill.fill(val);
                        copy.write_host(page as u64, &fill);
                    } else if k % 5 == 0 {
                        // Strided indirect/instance records (16 B every 512 B)
                        let patch = [val; 16];
                        for off in (0..PAGE).step_by(512) {
                            copy.write_host((page + off) as u64, &patch);
                        }
                    } else {
                        // Sparse shader counter / record write (64 B)
                        let off = (k * 192) % (PAGE - 64);
                        copy.write_host((page + off) as u64, &[val; 64]);
                    }
                }

                let _pin = spans_on.then(CopyBackSpansHere::new);
                let t0 = Instant::now();
                let bytes = copy.copy_back_marked(&pages);
                let us = t0.elapsed().as_micros() as u64;
                copy.copy_back(base, (TOTAL_PAGES - 4) * PAGE);
                if round >= 2 {
                    if spans_on {
                        spans_us.push(us);
                        spans_mb = bytes as f64 / 1e6;
                    } else {
                        default_us.push(us);
                        default_mb = bytes as f64 / 1e6;
                    }
                }
            }
        }
        default_us.sort_unstable();
        spans_us.sort_unstable();
        let d_med = default_us[default_us.len() / 2];
        let s_med = spans_us[spans_us.len() / 2];
        eprintln!(
            "[{scenario}]\n  default: median {:>6} us ({:.2} ms), {:.2} MB written\n  spans:   median {:>6} us ({:.2} ms), {:.2} MB written  ({:.1}x faster, {:+.1}%)",
            d_med,
            d_med as f64 / 1e3,
            default_mb,
            s_med,
            s_med as f64 / 1e3,
            spans_mb,
            d_med as f64 / s_med.max(1) as f64,
            (s_med as f64 - d_med as f64) * 100.0 / d_med.max(1) as f64,
        );
    }

    crate::watch::title_memory_remapped(alias_pool, PAGE);
    crate::watch::title_memory_remapped(base, len);
    drop(copy);
    watch_tests::free(alias_pool);
    watch_tests::free(base);
}
