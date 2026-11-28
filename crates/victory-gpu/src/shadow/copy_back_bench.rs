//! Integration tests and A/B benchmark for `VICTORY_COPY_BACK_SPANS`
//! (`crate::shadow::copy_back_fast`).
//!
//! Run the benchmark alone, in release, under Wine + D3DMetal:
//! `tools/macos/test-wine.sh --release -p victory-gpu --lib world_marked_and_snapshot_copy_back -- --ignored --nocapture --test-threads=1`

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::*;
use crate::mac_switches::tests::CopyBackSpansHere;
use crate::watch::{PAGE, tests as watch_tests};

fn page_protections(base: usize, n_pages: usize) -> Vec<u32> {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    (0..n_pages)
        .map(|k| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe {
                VirtualQuery(
                    (base + k * PAGE) as *const _,
                    &mut info,
                    size_of_val(&info),
                )
            };
            info.Protect
        })
        .collect()
}

fn page_changed_flags(copy: &Shadow, base: usize, n_pages: usize) -> Vec<bool> {
    let t = copy.tracked.as_ref().expect("tracked shadow");
    (0..n_pages).map(|k| t.changed(base / PAGE + k)).collect()
}

fn shadow_bytes(copy: &Shadow) -> Vec<u8> {
    // SAFETY: `copy.data` spans `copy.len` bytes and no submission is running.
    unsafe { std::slice::from_raw_parts(copy.data as *const u8, copy.len as usize).to_vec() }
}

/// Allocates a placeholder reservation of `slots.len()` 64 KiB pages and maps
/// each page `i` as its own `MapViewOfFile3` view onto physical page
/// `slots[i]` of a shared pagefile section (`section_pages` pages).
fn mapped_placeholder_views(section_pages: usize, slots: &[usize]) -> usize {
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
        MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc2,
        VirtualFree, MEM_RELEASE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let total_len = slots.len() * PAGE;
    // SAFETY: a pagefile section and placeholder reservation of our own, split
    // into one view per entry of `slots`.
    unsafe {
        let section = CreateFileMappingW(
            -1isize as _,
            std::ptr::null(),
            PAGE_READWRITE,
            0,
            (section_pages * PAGE) as u32,
            std::ptr::null(),
        );
        assert!(!section.is_null());
        let base = VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            total_len,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        ) as usize;
        assert_ne!(base, 0);
        for (i, &phys) in slots.iter().enumerate() {
            assert!(phys < section_pages);
            let at = base + i * PAGE;
            if i + 1 < slots.len() {
                VirtualFree(at as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at as *const _,
                (phys * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, at);
        }
        base
    }
}

/// The line's text is as before without the switch (the default path counts
/// no blocks), and the switch's 4 KiB block counts come at its end.
#[test]
fn marked_back_line_keeps_its_text_and_ends_with_the_spans_blocks() {
    let line_off = format_marked_back_line([2033.0, 4.0, 12_300_000.0, 14_200.0, 0.0, 0.0], false);
    assert_eq!(
        line_off,
        "; marked pages copied back: 2033 compared, 12.3 MB written back from them, 4 more in aliased memory told only, in 14.2 ms"
    );
    let line_on = format_marked_back_line([2033.0, 0.0, 12_300_000.0, 14_200.0, 2980.0, 32528.0], true);
    assert_eq!(
        line_on,
        "; marked pages copied back: 2033 compared, 12.3 MB written back from them, 0 more in aliased memory told only, in 14.2 ms, 4 KiB blocks 2980 written back of 32528 compared (VICTORY_COPY_BACK_SPANS)"
    );
}

/// Pages `[0, n_pages)` from `base` whose `armed` state is set (tests).
fn page_armed_flags(copy: &Shadow, base: usize, n_pages: usize) -> Vec<bool> {
    let t = copy.tracked.as_ref().expect("a tracked copy");
    (0..n_pages).map(|p| t.armed_now(base / PAGE + p)).collect()
}

/// The 4 KiB blocks of `[start, end)` whose bytes differ between `a` and `b`
/// (both from `base`).
fn blocks_differing(a: &[u8], b: &[u8], base: usize, start: usize, end: usize) -> u64 {
    const BLOCK: usize = 4096;
    (start / BLOCK..end.div_ceil(BLOCK))
        .filter(|&blk| {
            let (from, to) = ((blk * BLOCK).max(start) - base, ((blk + 1) * BLOCK).min(end) - base);
            a[from..to] != b[from..to]
        })
        .count() as u64
}

/// Differential check on real D3D12 + Wine memory: runs the default path and
/// `VICTORY_COPY_BACK_SPANS=1` on identical setups (partial end pages, clean
/// marked pages, single-byte GPU writes, armed pages with same-word and nearby
/// concurrent CPU + GPU writes, CPU-only armed pages, strided GPU writes, and
/// full-page GPU writes) and compares EVERY byte of every guest page, EVERY
/// byte of the `Shadow` host buffer, per-page `VirtualQuery` protection state,
/// per-page `Tracked::changed` and armed flags (a page the title wrote before
/// it was armed is kept and unarmed on both sides), `Shadow::clean()`, and the
/// switch's 4 KiB block counts against the blocks whose bytes the copy-back
/// changed (the default path counts none) after both marked and snapshot
/// copy-backs.
#[test]
fn copy_back_spans_matches_default_across_marked_and_snapshot_cases() {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};

    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();

    struct Outcome {
        marked_guest: Vec<u8>,
        marked_shadow: Vec<u8>,
        marked_prot: Vec<u32>,
        marked_changed: Vec<bool>,
        marked_armed: Vec<bool>,
        marked_clean: bool,
        marked_blocks: (u64, u64),
        marked_blocks_differing: u64,
        marked_blocks_of_compared_pages: u64,
        marked_split: Option<bool>,
        snap_guest: Vec<u8>,
        snap_shadow: Vec<u8>,
        snap_prot: Vec<u32>,
        snap_changed: Vec<bool>,
        snap_clean: bool,
        marked_written: usize,
        snap_written: usize,
    }

    let n_pages = 2 * SPLIT_MIN_PAGES + 6;
    let run_side = |spans_on: bool| -> Outcome {
        let _pin = spans_on.then(CopyBackSpansHere::new);
        let base = watch_tests::pool(n_pages);
        let start = base + 48;
        let end = base + n_pages * PAGE - 80;
        let len = end - start;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, start as u64, len as u64);

        {
            let _w = crate::watch::writing(base, n_pages * PAGE);
            // SAFETY: owned pool allocation; deterministic per-byte pattern.
            let slice = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, n_pages * PAGE) };
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i.wrapping_mul(31) ^ (i >> 9) ^ 0x23) as u8;
            }
        }
        let write_cpu = |off: usize, val: u8| {
            // SAFETY: owned pool allocation.
            unsafe { ((base + off) as *mut u8).write_volatile(val) };
        };
        copy.update(start, len);
        // 0) Page 5, written before it is armed: changed with no bytes saved,
        //    so kept (and unarmed) by both paths.
        write_cpu(5 * PAGE + 3000, 0x66);
        copy.writing(start, len, true);

        // 1) Partial first & last pages: GPU writes near edges.
        copy.write_host(start as u64, &[0xa1, 0xa2, 0xa3]);
        copy.write_host((end - 2) as u64, &[0xe1, 0xe2]);

        // 2) Armed page 1: CPU and GPU write both same-word and nearby bytes.
        write_cpu(PAGE + 11, 0x51);
        write_cpu(PAGE + 20, 0x52);
        write_cpu(PAGE + 6000, 0x53);
        copy.write_host((base + PAGE + 10) as u64, &[0xb1]);
        copy.write_host((base + PAGE + 12) as u64, &[0xb2]);
        copy.write_host((base + PAGE + 24) as u64, &[0xb3]);

        // 3) Armed page 2: CPU writes only, GPU leaves untouched.
        write_cpu(2 * PAGE + 64, 0x77);
        write_cpu(2 * PAGE + 9000, 0x78);

        // 4) Unarmed page 3 (disarmed via unarm): CPU write keeps title bytes.
        copy.tracked.as_ref().unwrap().unarm(base / PAGE + 3);
        write_cpu(3 * PAGE + 32, 0x44);
        copy.write_host((base + 3 * PAGE + 40) as u64, &[0xee; 8]);

        // 5) Interior pages: clean, sparse single-record, strided, and full-page.
        for p in 4..n_pages - 1 {
            match p % 4 {
                0 => {}
                1 => copy.write_host((base + p * PAGE + 128) as u64, &[0xc4; 16]),
                2 => {
                    for s in (0..PAGE).step_by(256) {
                        copy.write_host((base + p * PAGE + s) as u64, &[0xd5; 4]);
                    }
                }
                _ => {
                    let block: Vec<u8> = (0..PAGE).map(|i| (0xf6u8).wrapping_add(i as u8)).collect();
                    copy.write_host((base + p * PAGE) as u64, &block);
                }
            }
        }

        let _ = take_last_marked_blocks();
        MARKED_SPLIT.with(|s| s.set(None));
        let pages: Vec<usize> = (0..n_pages).map(|i| base + i * PAGE).collect();
        // SAFETY: owned pool allocation.
        let guest_before = unsafe { std::slice::from_raw_parts(base as *const u8, n_pages * PAGE).to_vec() };
        let marked_written = copy.copy_back_marked(&pages);
        let marked_blocks = take_last_marked_blocks();
        let marked_split = MARKED_SPLIT.with(std::cell::Cell::get);
        let marked_armed = page_armed_flags(&copy, base, n_pages);

        // Second call finds all whole pages synced; partial end pages match too.
        assert_eq!(copy.copy_back_marked(&pages), 0, "spans_on={spans_on}");
        let _ = take_last_marked_blocks();

        // SAFETY: owned pool allocation.
        let marked_guest =
            unsafe { std::slice::from_raw_parts(base as *const u8, n_pages * PAGE).to_vec() };
        let marked_shadow = shadow_bytes(&copy);
        let marked_prot = page_protections(base, n_pages);
        let marked_changed = page_changed_flags(&copy, base, n_pages);
        let marked_clean = copy.clean();
        let marked_blocks_differing = blocks_differing(&guest_before, &marked_guest, base, start, end);
        // Every page but the kept pages 3 and 5 is compared.
        let marked_blocks_of_compared_pages: u64 = (0..n_pages)
            .filter(|&p| p != 3 && p != 5)
            .map(|p| {
                let (from, to) = ((base + p * PAGE).max(start), (base + (p + 1) * PAGE).min(end));
                (to.div_ceil(4096) - from / 4096) as u64
            })
            .sum();

        // Clear the armed snapshot from above, sync whole range, then exercise
        // snapshot `copy_back` across multiple pages (CPU-only, GPU-only,
        // same-word CPU+GPU, and clean pages).
        copy.copy_back(start, len);
        copy.update(start, len);
        copy.writing(base + PAGE, 4 * PAGE, false);
        write_cpu(PAGE + 500, 0x31);
        copy.write_host((base + PAGE + 502) as u64, &[0x9a; 12]);
        write_cpu(2 * PAGE + 1024, 0x32);
        copy.write_host((base + 3 * PAGE + 2048) as u64, &[0x9b; 24]);
        let snap_written = copy.copy_back(base + PAGE, 4 * PAGE);

        // SAFETY: owned pool allocation.
        let snap_guest =
            unsafe { std::slice::from_raw_parts(base as *const u8, n_pages * PAGE).to_vec() };
        let snap_shadow = shadow_bytes(&copy);
        let snap_prot = page_protections(base, n_pages);
        let snap_changed = page_changed_flags(&copy, base, n_pages);
        let snap_clean = copy.clean();

        crate::watch::title_memory_remapped(base, n_pages * PAGE);
        drop(copy);
        watch_tests::free(base);

        Outcome {
            marked_guest,
            marked_shadow,
            marked_prot,
            marked_changed,
            marked_armed,
            marked_clean,
            marked_blocks,
            marked_blocks_differing,
            marked_blocks_of_compared_pages,
            marked_split,
            snap_guest,
            snap_shadow,
            snap_prot,
            snap_changed,
            snap_clean,
            marked_written,
            snap_written,
        }
    };

    let def = run_side(false);
    let spn = run_side(true);

    // Worker splitting must stay off when `VICTORY_COPY_BACK_SPLIT` is unset.
    if std::env::var_os("VICTORY_COPY_BACK_SPLIT").is_none() {
        assert_eq!(def.marked_split, Some(false));
        assert_eq!(spn.marked_split, Some(false));
    }

    assert!(spn.marked_written < def.marked_written);
    assert_eq!(spn.snap_written, 12 + 24);
    assert_eq!(def.snap_written, 2 * PAGE);

    // The default path counts no blocks (a page compared whole, as before);
    // the switch's count is the blocks whose bytes the copy-back changed, of
    // the blocks of every page it compared (all but the kept pages 3 and 5).
    assert_eq!(def.marked_blocks, (0, 0), "the default path counts no blocks");
    assert_eq!(spn.marked_blocks_differing, def.marked_blocks_differing);
    assert_eq!(spn.marked_blocks.0, spn.marked_blocks_differing, "4 KiB blocks written back");
    assert_eq!(spn.marked_blocks.1, spn.marked_blocks_of_compared_pages, "4 KiB blocks compared");
    assert!(spn.marked_blocks.0 > 0 && spn.marked_blocks.0 < spn.marked_blocks.1);
    assert_eq!(spn.marked_armed, def.marked_armed, "armed pages after copy_back_marked");
    assert!(!spn.marked_armed[5], "the kept page is unarmed");
    assert_eq!(spn.marked_guest, def.marked_guest, "every guest byte after copy_back_marked");
    assert_eq!(spn.marked_shadow, def.marked_shadow, "every shadow byte after copy_back_marked");
    assert_eq!(spn.marked_prot, def.marked_prot, "every page protection after copy_back_marked");
    assert_eq!(spn.marked_changed, def.marked_changed, "changed set after copy_back_marked");
    assert_eq!(spn.marked_clean, def.marked_clean);

    // Partial end pages stay open/changed (not taken whole); interior synced pages are guarded.
    assert_eq!(spn.marked_prot[0], PAGE_READWRITE, "partial first page not guarded by taken");
    assert_eq!(spn.marked_prot[1], PAGE_READONLY, "whole synced page guarded");
    assert_eq!(spn.marked_prot[2], PAGE_READONLY, "CPU-only armed page synced and guarded");
    assert_eq!(spn.marked_prot[3], PAGE_READWRITE, "unarmed kept page stays open and changed");
    assert!(spn.marked_changed[3] && !spn.marked_changed[1] && !spn.marked_changed[2]);

    assert_eq!(spn.snap_guest, def.snap_guest, "every guest byte after snapshot copy_back");
    assert_eq!(spn.snap_shadow, def.snap_shadow, "every shadow byte after snapshot copy_back");
    assert_eq!(spn.snap_prot, def.snap_prot, "every page protection after snapshot copy_back");
    assert_eq!(spn.snap_changed, def.snap_changed, "changed set after snapshot copy_back");
    assert_eq!(spn.snap_clean, def.snap_clean);
    for p in [1, 3, 4] {
        assert_eq!(spn.snap_prot[p], PAGE_READONLY, "snapshot page {p} guarded");
        assert!(!spn.snap_changed[p], "snapshot page {p} cleared from changed");
    }
    // Page 2 had a CPU-only write (`*host == *old`), so snapshot leaves it open
    // and marked changed on both sides for the next `Shadow::update`.
    assert_eq!(spn.snap_prot[2], PAGE_READWRITE);
    assert!(spn.snap_changed[2]);
}

/// Tests a physical page shown at two virtual addresses BOTH inside the same
/// `Shadow` copy (`[base + 0 * PAGE]` and `[base + 2 * PAGE]` backed by the
/// same section offset) AND across two separate `Shadow` copies (`twin_views`),
/// written through both addresses by CPU and GPU. Verifies that copying back
/// through the first address saves pre-write bytes on the armed alias inside
/// the same copy before writing, and compares whole pages byte-for-byte,
/// protections, and `changed` sets against the default path.
#[test]
fn alias_inside_same_copy_and_across_copies_matches_whole_pages() {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;

    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();

    // Part 1: Self-alias inside one 4-page `Shadow` copy (pages 0 and 2 share physical page 0).
    let run_self_alias = |spans_on: bool| {
        let _pin = spans_on.then(CopyBackSpansHere::new);
        let base = mapped_placeholder_views(3, &[0, 1, 0, 2]);
        crate::watch::title_memory_aliased(base + 2 * PAGE, &[base]);
        let len = 4 * PAGE;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);

        {
            let _w = crate::watch::writing(base, len);
            // SAFETY: mapped views owned by this test.
            let slice = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, len) };
            for (i, b) in slice.iter_mut().enumerate() {
                *b = ((i % PAGE).wrapping_mul(19) ^ 0x42) as u8;
            }
        }
        copy.update(base, len);
        assert!(copy.clean());

        // Arm the whole 4-page copy. GPU writes page 0 at +16 (`0xaa`) and
        // page 2 (the alias of page 0!) at +64 (`0xbb`).
        copy.writing(base, len, true);
        copy.write_host((base + 16) as u64, &[0xaa; 8]);
        copy.write_host((base + 2 * PAGE + 64) as u64, &[0xbb; 8]);

        // Step 1: Copy back page 0 first. `writing_runs` must save page 2's
        // pre-write bytes (`old[16..24]`) BEFORE writing `0xaa` to physical
        // page 0, and leave page 2 marked changed in `copy`.
        copy.copy_back_marked(&[base]);
        assert!(
            copy.tracked.as_ref().unwrap().changed((base + 2 * PAGE) / PAGE),
            "alias page 2 inside same copy must be marked changed when page 0 writes (spans_on={spans_on})"
        );

        // Step 2: CPU writes through BOTH virtual addresses of physical page 0
        // (through page 2 at +128 and through page 0 at +256).
        // SAFETY: mapped views of physical page 0.
        unsafe {
            ((base + 2 * PAGE + 128) as *mut [u8; 4]).write_volatile([0x51, 0x52, 0x53, 0x54]);
            ((base + 256) as *mut [u8; 4]).write_volatile([0x61, 0x62, 0x63, 0x64]);
        }

        // Step 3: Copy back page 2. Because page 2 saved pre-write bytes from
        // before step 1, `host[2*PAGE + 16..24]` equals `old[16..24]`, so step
        // 1's `[0xaa; 8]` AND step 2's CPU writes at +128 and +256 ALL survive
        // alongside page 2's `[0xbb; 8]` at +64!
        copy.copy_back_marked(&[base + 2 * PAGE]);
        copy.copy_back(base, len);

        let prot_before_update = page_protections(base, 4);
        let changed_before_update = page_changed_flags(&copy, base, 4);

        // Refresh `copy` so both virtual pages in `Shadow` converge to physical memory.
        copy.update(base, len);
        let prot_after_update = page_protections(base, 4);
        let changed_after_update = page_changed_flags(&copy, base, 4);
        let clean_after_update = copy.clean();
        // SAFETY: mapped views owned by this test.
        let guest = unsafe { std::slice::from_raw_parts(base as *const u8, len).to_vec() };
        let shadow = shadow_bytes(&copy);

        crate::watch::title_memory_remapped(base, len);
        drop(copy);
        (
            guest,
            shadow,
            prot_before_update,
            changed_before_update,
            prot_after_update,
            changed_after_update,
            clean_after_update,
        )
    };

    let def = run_self_alias(false);
    let spn = run_self_alias(true);
    assert_eq!(spn.0, def.0, "whole-buffer guest bytes on self-alias");
    assert_eq!(spn.1, def.1, "whole-buffer shadow bytes on self-alias");
    assert_eq!(&spn.0[16..24], &[0xaa; 8], "page 0 GPU write preserved");
    assert_eq!(&spn.0[64..72], &[0xbb; 8], "page 2 GPU write preserved");
    assert_eq!(&spn.0[128..132], &[0x51, 0x52, 0x53, 0x54], "CPU write via alias B preserved");
    assert_eq!(&spn.0[256..260], &[0x61, 0x62, 0x63, 0x64], "CPU write via alias A preserved");
    assert_eq!(&spn.0[..PAGE], &spn.0[2 * PAGE..3 * PAGE], "both views show identical whole page");
    assert_eq!(&spn.1[..PAGE], &spn.0[..PAGE], "shadow page 0 equals guest whole page");
    assert_eq!(&spn.1[2 * PAGE..3 * PAGE], &spn.0[..PAGE], "shadow page 2 equals guest whole page");
    assert_eq!(spn.2, def.2, "protections before update");
    assert_eq!(spn.3, def.3, "changed flags before update");
    assert_eq!(spn.4, vec![PAGE_READONLY; 4]);
    assert_eq!(spn.5, vec![false; 4]);
    assert!(spn.6 && def.6);

    // Part 2: Twin views in two separate `Shadow` copies written through both addresses.
    let run_twin = |spans_on: bool| {
        let _pin = spans_on.then(CopyBackSpansHere::new);
        let (view_a, view_b) = watch_tests::twin_views(2);
        let len = 2 * PAGE;
        let (buf_a, data_a) = host.shared_buffer(len as u64).unwrap();
        let (buf_b, data_b) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffers.
        let (gpu_a, gpu_b) = unsafe { (buf_a.GetGPUVirtualAddress(), buf_b.GetGPUVirtualAddress()) };
        let copy_a = Shadow::new(buf_a, data_a, gpu_a, view_a as u64, len as u64);
        let copy_b = Shadow::new(buf_b, data_b, gpu_b, view_b as u64, len as u64);

        {
            let _w = crate::watch::writing(view_a, len);
            // SAFETY: twin views of 2 pages.
            let s = unsafe { std::slice::from_raw_parts_mut(view_a as *mut u8, len) };
            for (i, b) in s.iter_mut().enumerate() {
                *b = (i.wrapping_mul(7) ^ 0x5c) as u8;
            }
        }
        copy_a.update(view_a, len);
        copy_b.update(view_b, len);
        copy_a.writing(view_a, len, true);
        copy_b.writing(view_b, len, true);

        // Write CPU bytes via `view_b` on page 0 and via `view_a` on page 1,
        // and GPU bytes on `copy_a` page 0 and `copy_b` page 1.
        // SAFETY: twin views.
        unsafe {
            ((view_b + 40) as *mut u32).write_volatile(0x1122_3344);
            ((view_a + PAGE + 80) as *mut u32).write_volatile(0x5566_7788);
        }
        copy_a.write_host((view_a + 300) as u64, &[0xd1; 20]);
        copy_b.write_host((view_b + PAGE + 600) as u64, &[0xd2; 20]);

        copy_a.copy_back_marked(&[view_a]);
        copy_b.copy_back_marked(&[view_b + PAGE]);
        copy_a.copy_back(view_a, len);
        copy_b.copy_back(view_b, len);
        copy_a.update(view_a, len);
        copy_b.update(view_b, len);

        // SAFETY: twin views.
        let guest_a = unsafe { std::slice::from_raw_parts(view_a as *const u8, len).to_vec() };
        let shadow_a = shadow_bytes(&copy_a);
        let shadow_b = shadow_bytes(&copy_b);
        let prot_a = page_protections(view_a, 2);
        let prot_b = page_protections(view_b, 2);
        let clean = copy_a.clean() && copy_b.clean();

        crate::watch::title_memory_remapped(view_a, len);
        crate::watch::title_memory_remapped(view_b, len);
        drop(copy_a);
        drop(copy_b);
        (guest_a, shadow_a, shadow_b, prot_a, prot_b, clean)
    };

    let t_def = run_twin(false);
    let t_spn = run_twin(true);
    assert_eq!(t_spn.0, t_def.0, "twin views whole-page guest bytes");
    assert_eq!(t_spn.1, t_def.1, "twin views copy_a whole-page bytes");
    assert_eq!(t_spn.2, t_def.2, "twin views copy_b whole-page bytes");
    assert_eq!(t_spn.1, t_spn.0, "copy_a matches guest after update");
    assert_eq!(t_spn.2, t_spn.0, "copy_b matches guest after update");
    assert_eq!(t_spn.3, vec![PAGE_READONLY; 2]);
    assert_eq!(t_spn.4, vec![PAGE_READONLY; 2]);
    assert!(t_spn.5 && t_def.5);
}

/// Tests memory mapped with one `MapViewOfFile3` view per 64 KiB page: verifies
/// that `VICTORY_COPY_BACK_SPANS=1` respects per-view region ends in `protect`
/// (one `VirtualProtect` call per dirty view, zero failing cross-view calls)
/// and matches the default path across every byte, protection, and `changed`
/// flag.
#[test]
fn one_view_per_page_guards_one_view_per_call_and_matches_whole_pages() {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;

    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();

    let n = 6;
    let slots: Vec<usize> = (0..n).collect();
    let run_views = |spans_on: bool| {
        let _pin = spans_on.then(CopyBackSpansHere::new);
        let base = mapped_placeholder_views(n, &slots);
        let len = n * PAGE;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);

        {
            let _w = crate::watch::writing(base, len);
            // SAFETY: 6 one-page views.
            let s = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, len) };
            for (i, b) in s.iter_mut().enumerate() {
                *b = (i.wrapping_mul(13) ^ 0x37) as u8;
            }
        }
        copy.update(base, len);
        assert_eq!(page_protections(base, n), vec![PAGE_READONLY; n]);

        // Dirty 4 consecutive one-page views (pages 1..5) on GPU + CPU on page 2,
        // leaving pages 0 and 5 untouched (already guarded).
        copy.writing(base, len, true);
        // SAFETY: one-page view 2.
        unsafe { ((base + 2 * PAGE + 24) as *mut u32).write_volatile(0xdead_beef) };
        for p in 1..5 {
            copy.write_host((base + p * PAGE + 128) as u64, &[0xa0 + p as u8; 32]);
        }

        let pages: Vec<usize> = (0..n).map(|k| base + k * PAGE).collect();
        let before_calls = watch_tests::PROTECTS.with(std::cell::Cell::get);
        copy.copy_back_marked(&pages);
        let protect_calls = watch_tests::PROTECTS.with(std::cell::Cell::get) - before_calls;
        copy.copy_back(base, len);

        // Pages 1..5 (4 one-page views) needed re-guarding; pages 0 and 5 stayed
        // guarded, so `protect` must make exactly 4 `VirtualProtect` calls (never
        // a 5th failing multi-view call across view boundaries).
        assert_eq!(
            protect_calls, 4,
            "expected 1 VirtualProtect per dirty 1-page view without cross-view failure (spans_on={spans_on})"
        );

        let prot = page_protections(base, n);
        let changed = page_changed_flags(&copy, base, n);
        let clean = copy.clean();
        // SAFETY: 6 one-page views.
        let guest = unsafe { std::slice::from_raw_parts(base as *const u8, len).to_vec() };
        let shadow = shadow_bytes(&copy);

        crate::watch::title_memory_remapped(base, len);
        drop(copy);
        (guest, shadow, prot, changed, clean)
    };

    let def = run_views(false);
    let spn = run_views(true);
    assert_eq!(spn.0, def.0, "whole-page guest bytes on 1-view-per-page memory");
    assert_eq!(spn.1, def.1, "whole-page shadow bytes on 1-view-per-page memory");
    assert_eq!(spn.2, vec![PAGE_READONLY; n]);
    assert_eq!(spn.3, vec![false; n]);
    assert!(spn.4 && def.4);
}

/// Tests a concurrent title thread writing non-GPU bytes during both snapshot
/// `copy_back` and `copy_back_marked` under `VICTORY_COPY_BACK_SPANS=1`:
/// 1) Deterministically writes to an already-completed piece AND a not-yet-read
///    piece mid-snapshot, proving neither title write is lost from the shadow.
/// 2) Hammers non-GPU atomic counters from a background title thread across
///    repeated marked and snapshot copy-backs: no GPU write is lost, and the
///    copy converges whole after `Shadow::update`. The title's last sweep
///    marks every page changed again, so a title write lost on the way (one
///    landing on an open page between a copy-back's read and its re-guard) is
///    not seen here: `probe_title_writes_lost_between_a_copy_backs_read_and_its_reguard`
///    counts those on both paths.
#[test]
fn concurrent_title_thread_writes_during_copy_back_are_never_lost() {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;

    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();
    let _pin = CopyBackSpansHere::new();

    // Part 1: Deterministic mid-snapshot write to an already-synced piece (page 0)
    // and an upcoming piece (page 96) as soon as piece 2's GPU span lands.
    {
        let n_pages = 128;
        let base = watch_tests::pool(n_pages);
        let len = n_pages * PAGE;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);

        {
            let _w = crate::watch::writing(base, len);
            // SAFETY: owned pool allocation.
            unsafe { (base as *mut u8).write_bytes(0x11, len) };
        }
        copy.update(base, len);
        copy.writing(base, len, false);
        for p in 0..n_pages {
            copy.write_host((base + p * PAGE) as u64, &[0xaa; 32]);
        }

        let started = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                // Wait until piece 2 has written `0xaa` (so piece 0 is already
                // re-guarded and piece 96 has not been reached yet), or until
                // copy_back finishes on a very fast run.
                let probe = (base + 2 * PAGE) as *const u8;
                // SAFETY: inside `base .. base + len`.
                while unsafe { probe.read_volatile() } != 0xaa {
                    std::hint::spin_loop();
                }
                // SAFETY: non-GPU offsets on page 0 and page 96.
                unsafe {
                    ((base + 4096) as *mut [u8; 16]).write_volatile([0x7e; 16]);
                    ((base + 96 * PAGE + 4096) as *mut [u8; 16]).write_volatile([0x7f; 16]);
                }
            });
            started.store(true, Ordering::Release);
            let written = copy.copy_back(base, len);
            assert_eq!(written, n_pages * 32);
            writer.join().unwrap();
        });

        // Refresh `copy` and verify every single byte of all 128 pages matches!
        copy.update(base, len);
        // SAFETY: owned pool allocation.
        let guest = unsafe { std::slice::from_raw_parts(base as *const u8, len) };
        let shadow = shadow_bytes(&copy);
        assert_eq!(&guest[4096..4112], &[0x7e; 16], "guest kept piece-0 title write");
        assert_eq!(&guest[96 * PAGE + 4096..96 * PAGE + 4112], &[0x7f; 16]);
        assert_eq!(
            shadow.as_slice(),
            guest,
            "shadow must converge to every byte of all 128 pages (no missed title write)"
        );
        assert_eq!(page_protections(base, n_pages), vec![PAGE_READONLY; n_pages]);
        assert!(copy.clean());

        crate::watch::title_memory_remapped(base, len);
        drop(copy);
        watch_tests::free(base);
    }

    // Part 2: Free-running concurrent title writer across marked & snapshot rounds.
    {
        let n_pages = 32;
        let base = watch_tests::pool(n_pages);
        let len = n_pages * PAGE;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);

        {
            let _w = crate::watch::writing(base, len);
            // SAFETY: owned pool allocation zeroed before atomic accesses.
            unsafe { (base as *mut u8).write_bytes(0, len) };
        }
        copy.update(base, len);
        let pages: Vec<usize> = (0..n_pages).map(|k| base + k * PAGE).collect();

        let stop = AtomicBool::new(false);
        let final_seq = AtomicU64::new(0);
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                let mut seq = 0u64;
                while !stop.load(Ordering::Acquire) {
                    seq += 1;
                    for p in 0..n_pages {
                        // SAFETY: 8-byte aligned offset 8192 outside GPU spans `[0..64)`.
                        let slot = unsafe { &*((base + p * PAGE + 8192) as *const AtomicU64) };
                        slot.store(seq, Ordering::Release);
                    }
                }
                // One final sweep once copy-backs stop, faulting on guarded pages.
                seq += 1;
                for p in 0..n_pages {
                    // SAFETY: as above.
                    let slot = unsafe { &*((base + p * PAGE + 8192) as *const AtomicU64) };
                    slot.store(seq, Ordering::Release);
                }
                final_seq.store(seq, Ordering::Release);
            });

            for round in 1..=12u8 {
                // Match real submission order: arm first (`writing`), then
                // refresh (`update`) so any title fault after guarding saves
                // pre-fault bytes instead of triggering unarmed `MarkedKept`.
                copy.writing(base, len, true);
                copy.update(base, len);
                for p in 0..n_pages {
                    copy.write_host((base + p * PAGE) as u64, &[0x80 | round; 64]);
                }
                copy.copy_back_marked(&pages);
                copy.copy_back(base, len);

                copy.update(base, len);
                copy.writing(base, len, false);
                for p in 0..n_pages {
                    copy.write_host((base + p * PAGE + 128) as u64, &[0x40 | round; 64]);
                }
                copy.copy_back(base, len);
            }
            stop.store(true, Ordering::Release);
            writer.join().unwrap();
        });

        let want_seq = final_seq.load(Ordering::Acquire);
        assert!(want_seq >= 2);
        copy.update(base, len);
        // SAFETY: owned pool allocation, writer thread joined.
        let guest = unsafe { std::slice::from_raw_parts(base as *mut u8, len) };
        let shadow = shadow_bytes(&copy);
        assert_eq!(shadow.as_slice(), guest, "whole-buffer convergence after concurrent writes");
        for p in 0..n_pages {
            assert_eq!(&guest[p * PAGE..p * PAGE + 64], &[0x80 | 12; 64]);
            assert_eq!(&guest[p * PAGE + 128..p * PAGE + 192], &[0x40 | 12; 64]);
            let got = u64::from_le_bytes(guest[p * PAGE + 8192..p * PAGE + 8200].try_into().unwrap());
            assert_eq!(got, want_seq, "page {p} kept title thread's latest atomic counter");
        }
        assert_eq!(page_protections(base, n_pages), vec![PAGE_READONLY; n_pages]);
        assert!(copy.clean());

        crate::watch::title_memory_remapped(base, len);
        drop(copy);
        watch_tests::free(base);
    }
}

/// A probe, not a pass/fail test: the title thread of
/// `concurrent_title_thread_writes_during_copy_back_are_never_lost` part 2,
/// on the default path and with `VICTORY_COPY_BACK_SPANS=1`, checked after
/// each copy-back while it still writes: a page must be changed (its next
/// refresh brings the title's word) or its copy must hold the title's word.
/// Prints the pages found neither, per path and per call. Both paths show
/// some. As we read the code: a page the title has written is open, later
/// writes to it raise no fault, and the copy-back re-guards it (clearing
/// `changed`) only after it read the page, so a write between the two is
/// in title memory but not in the copy, until the title writes the page
/// again.
#[test]
#[ignore = "probe: counts title writes lost between a copy-back's read and its re-guard on both paths"]
fn probe_title_writes_lost_between_a_copy_backs_read_and_its_reguard() {
    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();
    for spans in [false, true] {
        let _pin = spans.then(CopyBackSpansHere::new);
        let n_pages = 32;
        let base = watch_tests::pool(n_pages);
        let len = n_pages * PAGE;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);
        {
            let _w = crate::watch::writing(base, len);
            // SAFETY: owned pool allocation zeroed before atomic accesses.
            unsafe { (base as *mut u8).write_bytes(0, len) };
        }
        copy.update(base, len);
        let pages: Vec<usize> = (0..n_pages).map(|k| base + k * PAGE).collect();
        let tracked = copy.tracked.as_ref().expect("a tracked copy");
        // Pages neither changed nor holding the title's word in the copy;
        // changed is read before and after the words, as a write to a
        // guarded page faults (and marks it changed) before it lands.
        let lost = || {
            (0..n_pages)
                .filter(|&p| {
                    let page = base / PAGE + p;
                    let before = tracked.changed(page);
                    // SAFETY: the writer's word, read atomically.
                    let title = unsafe { &*((base + p * PAGE + 8192) as *const AtomicU64) }.load(Ordering::Acquire);
                    // SAFETY: inside the copy's mapping, read while no copy-back runs.
                    let copied = unsafe { (copy.data.add(p * PAGE + 8192) as *const u64).read_volatile() };
                    !before && !tracked.changed(page) && copied != title
                })
                .count()
        };
        let stop = AtomicBool::new(false);
        let (mut after_marked, mut after_its_end, mut after_snapshot) = (0usize, 0usize, 0usize);
        const ROUNDS: u8 = 48;
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut seq = 0u64;
                while !stop.load(Ordering::Acquire) {
                    seq += 1;
                    for p in 0..n_pages {
                        // SAFETY: 8-byte aligned offset 8192 outside GPU spans.
                        let slot = unsafe { &*((base + p * PAGE + 8192) as *const AtomicU64) };
                        slot.store(seq, Ordering::Release);
                    }
                }
            });
            for round in 1..=ROUNDS {
                copy.writing(base, len, true);
                copy.update(base, len);
                for p in 0..n_pages {
                    copy.write_host((base + p * PAGE) as u64, &[0x80 | (round & 0x3f); 64]);
                }
                copy.copy_back_marked(&pages);
                after_marked += lost();
                // Ends the marked submission: no snapshot was kept for it.
                copy.copy_back(base, len);
                after_its_end += lost();
                copy.update(base, len);
                copy.writing(base, len, false);
                for p in 0..n_pages {
                    copy.write_host((base + p * PAGE + 128) as u64, &[0x40 | (round & 0x3f); 64]);
                }
                copy.copy_back(base, len);
                after_snapshot += lost();
            }
            stop.store(true, Ordering::Release);
        });
        eprintln!(
            "{}: pages found lost, of {} looked at each: after `copy_back_marked` {after_marked}, after the `copy_back` that ends its submission {after_its_end}, after a snapshot `copy_back` {after_snapshot}",
            if spans { "VICTORY_COPY_BACK_SPANS=1" } else { "default" },
            n_pages * usize::from(ROUNDS),
        );
        crate::watch::title_memory_remapped(base, len);
        drop(copy);
        watch_tests::free(base);
    }
}

#[test]
#[ignore = "benchmarks 2,033 marked pages on a 174 MiB D3D12 shadow buffer (serial vs serial)"]
fn world_marked_and_snapshot_copy_back() {
    const TOTAL_PAGES: usize = 2_784; // ~174 MiB heap
    const MARKED_PAGES: usize = 2_033;
    let Some(host) = host() else {
        return;
    };
    let _quiet = host.quiet();

    let base = watch_tests::pool(TOTAL_PAGES);
    let len = TOTAL_PAGES * PAGE;
    let (buffer, data) = host.shared_buffer(len as u64).unwrap();
    // SAFETY: live D3D12 buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let copy = Shadow::new(buffer, data, gpu, base as u64, len as u64);

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

    // Keep worker splitting off on BOTH sides so the A/B measures spans alone.
    SPLIT_FORCED.with(|f| f.set(Some(false)));

    for (scenario, dense_mod) in [
        ("sparse-records (27% clean, 73% 64B counter/indirect records)", usize::MAX),
        ("world-mixed (27% clean, 65% sparse/strided, 8% full-page)", 12usize),
        ("dense-73pct (1,525 full 64 KiB pages = 99.9 MB)", 4usize),
    ] {
        let mut default_us = Vec::new();
        let mut spans_us = Vec::new();
        let mut default_mb = 0.0;
        let mut spans_mb = 0.0;
        let mut blocks_pair = (0u64, 0u64);
        let mut fill = vec![0u8; PAGE];

        // 10 alternating A/B rounds (skipping first 2 warm-up pairs).
        for round in 0..10u8 {
            for spans_on in [false, true] {
                let seed = round.wrapping_mul(13).wrapping_add(u8::from(spans_on) * 7).max(1);
                copy.writing(base, (TOTAL_PAGES - 4) * PAGE, true);
                for (k, &page) in pages.iter().enumerate() {
                    if k % 4 == 0 {
                        continue;
                    }
                    let val = seed ^ (k as u8);
                    if dense_mod == 4 || k % dense_mod == 1 {
                        fill.fill(val);
                        copy.write_host(page as u64, &fill);
                    } else if dense_mod != usize::MAX && k % 5 == 0 {
                        let patch = [val; 16];
                        for off in (0..PAGE).step_by(512) {
                            copy.write_host((page + off) as u64, &patch);
                        }
                    } else {
                        let off = (k * 192) % (PAGE - 64);
                        copy.write_host((page + off) as u64, &[val; 64]);
                    }
                }

                let _pin = spans_on.then(CopyBackSpansHere::new);
                let _ = take_last_marked_blocks();
                MARKED_SPLIT.with(|s| s.set(None));
                let t0 = Instant::now();
                let bytes = copy.copy_back_marked(&pages);
                let us = t0.elapsed().as_micros() as u64;
                let blocks = take_last_marked_blocks();
                assert_eq!(MARKED_SPLIT.with(std::cell::Cell::get), Some(false), "serial on both sides");
                copy.copy_back(base, (TOTAL_PAGES - 4) * PAGE);
                if round >= 2 {
                    blocks_pair = blocks;
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
            "[{scenario}] ({} 4 KiB blocks written back of {} compared on the spans side, serial)\n  default: median {:>6} us ({:.2} ms), {:.2} MB written\n  spans:   median {:>6} us ({:.2} ms), {:.2} MB written  ({:.2}x faster, {:+.1}%)",
            blocks_pair.0,
            blocks_pair.1,
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

    SPLIT_FORCED.with(|f| f.set(None));
    crate::watch::title_memory_remapped(base, len);
    drop(copy);
    watch_tests::free(base);
}
