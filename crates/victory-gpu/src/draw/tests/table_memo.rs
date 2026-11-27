//! mac: `VICTORY_TABLE_MEMO` (from VIctory-macOS #462 31485db9), its two
//! host tests adapted: copies made are held on the test's thread, as other
//! tests make copies meanwhile.

use super::*;
use crate::draw::table_memo::{self, CHANGED, EPOCH, FIRST, HITS, LOOKUPS, ModeHere, REWRITTEN, VERIFY_EPOCH, VERIFY_STEPS};

/// Holds `copies_made` on this thread at `g` while it lives.
struct Generation;

impl Generation {
    fn hold(g: u64) -> Self {
        table_memo::GENERATION_HERE.with(|c| c.set(Some(g)));
        Generation
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        table_memo::GENERATION_HERE.with(|c| c.set(None));
    }
}

fn counted() -> [u64; 9] {
    table_memo::counted_here()
}

/// A constant view and a texture's shader view, each prepared twice: in
/// verify mode the second runs the whole path and changes nothing; with
/// the switch on, it is skipped and adds nothing.
#[test]
fn remembered_entries_are_skipped_and_verify_without_mismatches() {
    let Some(host) = crate::host::host() else { return };
    let Ok(list) = host.new_list() else { return };
    let _generation = Generation::hold(1);
    let p = crate::watch::tests::pool(1);
    let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
    let tex_mem = crate::watch::tests::pool(1) as u64;
    let tex = texture_at_address(tex_mem, &texture_words(32, 32, 28, 0x100, 0));
    let (heap_raw, title_heap, _host) = view_heap(8);
    let (slot0, slot1) = (title_heap.start, title_heap.start + 32);
    let cbv: [u32; 4] = [p as u32, (p as u64 >> 32) as u32, 256, 0];
    crate::descriptor::create_cbv(std::ptr::null(), &cbv, slot0);
    crate::descriptor::create_srv(std::ptr::null(), tex, std::ptr::null(), slot1);
    let mut state = State::default();
    state.set_heaps(&[heap_raw]);
    state.set_table(false, 0, title_heap.start);
    let u_cbv = used(ResourceType::Cbv);
    let u_srv = UsedResource { table_offset_dwords: 8, ..used(ResourceType::Srv) };
    {
        let _mode = ModeHere::set(2);
        let before = counted();
        for u in [&u_cbv, &u_srv] {
            state.table_entry(&list, u, false);
            let (epoch, steps) = (list.epoch(), list.deferred_steps());
            state.table_entry(&list, u, false);
            assert_eq!((list.epoch(), list.deferred_steps()), (epoch, steps), "{:?}", u.kind);
        }
        let c = counted();
        assert_eq!(c[LOOKUPS] - before[LOOKUPS], 4);
        assert_eq!(c[HITS] - before[HITS], 2, "the second of each is remembered");
        assert_eq!(c[VERIFY_EPOCH] - before[VERIFY_EPOCH], 0);
        assert_eq!(c[VERIFY_STEPS] - before[VERIFY_STEPS], 0);
    }
    {
        let _mode = ModeHere::set(1);
        let before = counted();
        let (epoch, steps) = (list.epoch(), list.deferred_steps());
        state.table_entry(&list, &u_srv, false);
        state.table_entry(&list, &u_cbv, false);
        assert_eq!((list.epoch(), list.deferred_steps()), (epoch, steps));
        let c = counted();
        assert_eq!((c[LOOKUPS] - before[LOOKUPS], c[HITS] - before[HITS]), (2, 2));
    }
    list.close();
    crate::shadow::unregister(buf as usize, p as u64);
    let release = crate::object::slot::<extern "system" fn(This) -> u32>;
    assert_eq!(release(tex, 0x10)(tex), 0);
    assert_eq!(release(buf, 0x10)(buf), 0);
    assert_eq!(release(heap_raw, 0x10)(heap_raw), 0);
}

/// A remembered constant view is prepared again when its slot is
/// rewritten (also back to the first record), when the list's refresh set
/// is cleared (a title copy), and when a buffer's host copy is made or
/// given up; not when only a texture state changed. A cut and a reset
/// forget it.
#[test]
fn a_remembered_entry_misses_when_what_it_depends_on_changes() {
    let Some(host) = crate::host::host() else { return };
    let Ok(list) = host.new_list() else { return };
    let _generation = Generation::hold(1);
    let p = crate::watch::tests::pool(1);
    let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
    let (heap_raw, title_heap, _host) = view_heap(8);
    let slot0 = title_heap.start;
    let cbv_a: [u32; 4] = [p as u32, (p as u64 >> 32) as u32, 256, 0];
    let cbv_b: [u32; 4] = [p as u32, (p as u64 >> 32) as u32, 512, 0];
    crate::descriptor::create_cbv(std::ptr::null(), &cbv_a, slot0);
    let mut state = State::default();
    state.set_heaps(&[heap_raw]);
    state.set_table(false, 0, title_heap.start);
    let u = used(ResourceType::Cbv);
    let _mode = ModeHere::set(1);
    let base = counted();
    let delta = |i: usize| counted()[i] - base[i];

    state.table_entry(&list, &u, false);
    state.table_entry(&list, &u, false);
    assert_eq!((delta(LOOKUPS), delta(HITS), delta(FIRST)), (2, 1, 1), "first, then remembered");

    crate::descriptor::create_cbv(std::ptr::null(), &cbv_b, slot0);
    state.table_entry(&list, &u, false);
    assert_eq!((delta(HITS), delta(REWRITTEN)), (1, 1), "a rewritten slot");
    state.table_entry(&list, &u, false);
    assert_eq!(delta(HITS), 2);
    crate::descriptor::create_cbv(std::ptr::null(), &cbv_a, slot0);
    state.table_entry(&list, &u, false);
    assert_eq!((delta(HITS), delta(REWRITTEN)), (2, 2), "the first record again is a new record");

    let src = [0x5au8; 16];
    list.defer_copy(p as *mut u8, src.as_ptr(), src.len());
    state.table_entry(&list, &u, false);
    assert_eq!((delta(HITS), delta(EPOCH)), (2, 1), "a title copy clears the refresh set");

    list.changed_something();
    state.table_entry(&list, &u, false);
    assert_eq!(delta(HITS), 3, "a texture state change does not reach a constant view");

    drop(_generation);
    let _generation = Generation::hold(2);
    state.table_entry(&list, &u, false);
    assert_eq!((delta(HITS), delta(CHANGED)), (3, 1), "a host copy made or given up");

    state.cut(&list);
    assert!(state.memo.is_empty(), "a cut forgets");
    state.table_entry(&list, &u, false);
    let (epoch, deferred) = (list.epoch(), list.deferred_epoch());
    state.reset(std::ptr::null());
    assert!(state.memo.is_empty(), "a reset forgets");
    list.close();
    list.reset();
    assert_ne!((list.epoch(), list.deferred_epoch()), (epoch, deferred), "a list reset moves both epochs");

    list.close();
    crate::shadow::unregister(buf as usize, p as u64);
    let release = crate::object::slot::<extern "system" fn(This) -> u32>;
    assert_eq!(release(buf, 0x10)(buf), 0);
    assert_eq!(release(heap_raw, 0x10)(heap_raw), 0);
}
