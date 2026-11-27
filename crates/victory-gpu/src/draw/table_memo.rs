//! mac: from VIctory-macOS #462 31485db9 (gdemontalivet), "opt-in per-list
//! descriptor table entry memoization" (`VICTORY_TABLE_MEMO`, off by
//! default as upstream).
//!
//! A list's draws bind the same table entries again and again. Each one
//! mirrors the title's slot into the host heap (a look in the view table,
//! under its lock), finds the buffer's host copy, asks for its refresh and
//! for the slot's (both kept out by the list's sets once asked), and moves
//! the texture to the state it needs. With the switch, an entry is
//! remembered per list (title slot, kind) once prepared, and a later draw
//! skips it while all of these still stand:
//!
//! - the slot's record bytes are the same (a rewrite of the slot misses);
//! - for buffers, constants and samplers: the list's sets of refreshes and
//!   slots asked are the same (`HostList::deferred_epoch`: a title copy or
//!   a reset clears them), and no host copy of a buffer was made or given
//!   up since (`crate::shadow::generation`; changed from upstream, so that a
//!   copy made for the range later is asked for);
//! - for textures: no texture of the list changed state and no depth twin
//!   was copied or written since (`HostList::epoch`), and the title
//!   texture's host copy is the one it was;
//! - the host slot still holds the view made from this record
//!   (`HostHeap::texture_slot_made_from`; tables made afresh have none).
//!
//! Never remembered: unordered views of buffers (each draw declares its
//! writes), three-channel buffer views (their widened reads), textures whose
//! memory another title texture writes (their copy each time).
//!
//! `VICTORY_TABLE_MEMO=2` (or `verify`) runs the whole preparation on hits
//! too, and counts hits after which it changed the list's epochs or added
//! steps (must be 0). Left out from upstream: its `TESTED_DEPTH` bookkeeping,
//! which nothing on this line reads.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::ResourceType;

use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;

/// `VICTORY_TABLE_MEMO`: 0 off (unset), 1 skip remembered entries, 2 verify.
pub(super) fn mode() -> u8 {
    #[cfg(test)]
    if let Some(v) = OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = mode_rule(std::env::var("VICTORY_TABLE_MEMO").ok().as_deref());
        if mode != 0 {
            tracing::info!(
                mode,
                "VICTORY_TABLE_MEMO: table entries a list prepared already are not prepared again while nothing they \
                 depend on changed (VIctory-macOS #462; 2 prepares them anyway and checks)"
            );
        }
        mode
    })
}

fn mode_rule(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("2" | "verify") => 2,
        _ => 0,
    }
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// While it lives, [`mode`] is the one given on this thread (tests).
#[cfg(test)]
pub(super) struct ModeHere(Option<u8>);

#[cfg(test)]
impl ModeHere {
    pub(super) fn set(mode: u8) -> Self {
        Self(OVERRIDE.with(|c| c.replace(Some(mode))))
    }
}

#[cfg(test)]
impl Drop for ModeHere {
    fn drop(&mut self) {
        OVERRIDE.with(|c| c.set(self.0));
    }
}

/// FNV-1a over the key's words (upstream's `MemoHasher`).
#[derive(Default)]
pub(super) struct Fnv(u64);

impl std::hash::Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, i: u64) {
        let h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        self.0 = (h ^ i).wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// What a remembered entry depended on beyond the list's epochs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seen {
    /// A buffer view, constants or a sampler: the host copies made so far.
    Buffers(u64),
    /// A title texture and its host copy (a placed texture's, `placed`).
    Texture { resource: usize, copy: usize, placed: bool },
}

#[derive(Clone, Copy)]
struct Remembered {
    bytes: [u64; 4],
    epoch: u64,
    seen: Seen,
}

/// A list's remembered entries: (title slot, kind).
#[derive(Default)]
pub(super) struct Memo(std::collections::HashMap<(u64, ResourceType), Remembered, std::hash::BuildHasherDefault<Fnv>>);

impl Memo {
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// What [`Memo::begin`] decided.
pub(super) enum Entry {
    Off,
    /// Remembered and standing: nothing to prepare.
    Skip,
    /// Prepare it, then [`Memo::end`].
    Do { key: (u64, ResourceType), bytes: [u64; 4], verify: Option<(u64, u64, usize)> },
}

/// Host copies of buffers made and given up so far
/// (`crate::shadow::generation`; tests may hold it on their thread).
fn copies_made() -> u64 {
    #[cfg(test)]
    if let Some(g) = GENERATION_HERE.with(std::cell::Cell::get) {
        return g;
    }
    crate::shadow::generation()
}

#[cfg(test)]
thread_local! {
    /// [`copies_made`] on this thread, while set (other tests make copies).
    pub(super) static GENERATION_HERE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

fn copy_of(r: usize) -> usize {
    crate::resource::made_copy(r as This).map_or(0, |c| std::sync::Arc::as_ptr(&c) as usize)
}

/// What an entry for `view` read as `kind` depends on, or `None` if it is
/// prepared every time.
fn seen_for(view: &View, kind: ResourceType) -> Option<Seen> {
    let uav = kind == ResourceType::Uav;
    if kind == ResourceType::Srv && crate::descriptor::host::three_channel_srv(view) {
        return None;
    }
    let buffers = Seen::Buffers(copies_made());
    let texture = |r: usize, placed: bool| {
        if crate::resource::buffer_bytes(r as This).is_some() {
            (!uav).then_some(buffers)
        } else {
            crate::resource::alias_source(r as This)
                .is_none()
                .then(|| Seen::Texture { resource: r, copy: copy_of(r), placed })
        }
    };
    match view {
        View::Constant(..) => (!uav).then_some(buffers),
        View::Sampler(..) | View::ConsoleSampler(..) => Some(buffers),
        View::Shader(r, _) | View::Unordered(r, _, _) => texture(*r, false),
        View::Placed(..) => match crate::descriptor::host::placed_texture(view) {
            Some(View::Shader(r, _) | View::Unordered(r, _, _)) => texture(r, true),
            Some(_) => None,
            None => (!uav).then_some(buffers),
        },
    }
}

/// Whether what `seen` depended on still stands for the slot at `handle`.
fn stands(seen: Seen, handle: u64) -> bool {
    match seen {
        Seen::Buffers(generation) => copies_made() == generation,
        Seen::Texture { resource, copy, placed } => crate::descriptor::view(handle).is_some_and(|view| {
            let views = |v: &View| matches!(v, View::Shader(r, _) | View::Unordered(r, _, _) if *r == resource);
            let same = if placed {
                crate::descriptor::host::placed_texture(&view).is_some_and(|p| views(&p))
            } else {
                views(&view)
            };
            same && copy_of(resource) == copy
        }),
    }
}

impl Memo {
    /// Before a table entry (the title slot `handle` read as `kind`) is
    /// prepared in `list`; `slot`: its host heap and slot, if it has one.
    pub(super) fn begin(
        &self,
        list: &HostList,
        slot: Option<(&crate::descriptor::host::HostHeap, u32)>,
        handle: u64,
        kind: ResourceType,
    ) -> Entry {
        let mode = mode();
        if mode == 0 {
            return Entry::Off;
        }
        // SAFETY: a descriptor slot of a live title heap (the caller found
        // its heap).
        let bytes = unsafe { (handle as *const [u64; 4]).read_unaligned() };
        let key = (handle, kind);
        count(LOOKUPS);
        let hit = match self.0.get(&key) {
            None => {
                count(FIRST);
                false
            }
            Some(r) if r.bytes != bytes => {
                count(REWRITTEN);
                false
            }
            Some(r) => {
                let epoch = match r.seen {
                    Seen::Buffers(_) => list.deferred_epoch(),
                    Seen::Texture { .. } => list.epoch(),
                };
                if r.epoch != epoch {
                    count(EPOCH);
                    false
                } else if slot.is_some_and(|(host, index)| !host.texture_slot_made_from(index, handle))
                    || !stands(r.seen, handle)
                {
                    count(CHANGED);
                    false
                } else {
                    true
                }
            }
        };
        if hit {
            count(HITS);
            if mode == 1 {
                return Entry::Skip;
            }
        }
        let verify = hit.then(|| (list.epoch(), list.deferred_epoch(), list.deferred_steps()));
        Entry::Do { key, bytes, verify }
    }

    /// After the entry `entry` was prepared, its view `view` (if any).
    pub(super) fn end(&mut self, list: &HostList, entry: Entry, view: Option<&View>) {
        let Entry::Do { key, bytes, verify } = entry else {
            return;
        };
        if let Some((epoch, deferred_epoch, steps)) = verify {
            let changed = [
                (list.epoch() != epoch || list.deferred_epoch() != deferred_epoch, VERIFY_EPOCH, "changed the list's epochs"),
                (list.deferred_steps() != steps, VERIFY_STEPS, "added steps to the list"),
            ];
            for (_, counter, what) in changed.into_iter().filter(|c| c.0) {
                count(counter);
                static SAID: AtomicU64 = AtomicU64::new(0);
                if SAID.fetch_add(1, Ordering::Relaxed) < 20 {
                    tracing::warn!(handle = format_args!("{:#x}", key.0), kind = ?key.1, ?view, "table memo (verify): a remembered entry {what}");
                }
            }
        }
        if self.0.len() >= 1 << 16 {
            self.0.clear();
        }
        match view.and_then(|v| seen_for(v, key.1)) {
            Some(seen) => {
                let epoch = match seen {
                    Seen::Buffers(_) => list.deferred_epoch(),
                    Seen::Texture { .. } => list.epoch(),
                };
                self.0.insert(key, Remembered { bytes, epoch, seen });
            }
            None => {
                count(UNREMEMBERED);
                self.0.remove(&key);
            }
        }
    }
}

pub(super) const LOOKUPS: usize = 0;
pub(super) const HITS: usize = 1;
pub(super) const FIRST: usize = 2;
pub(super) const REWRITTEN: usize = 3;
pub(super) const EPOCH: usize = 4;
pub(super) const CHANGED: usize = 5;
pub(super) const UNREMEMBERED: usize = 6;
pub(super) const VERIFY_EPOCH: usize = 7;
pub(super) const VERIFY_STEPS: usize = 8;
const COUNTERS: usize = 9;

/// Since the last frame line, summed over the threads ([`count`]).
static COUNTED: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

thread_local! {
    /// This thread's counts not yet added to [`COUNTED`], and how many.
    static LOCAL: std::cell::RefCell<([u64; COUNTERS], u32)> = const { std::cell::RefCell::new(([0; COUNTERS], 0)) };
}

fn count(i: usize) {
    LOCAL.with_borrow_mut(|(c, n)| {
        c[i] += 1;
        *n += 1;
        if *n >= 4096 {
            flush(c, n);
        }
    });
}

fn flush(c: &mut [u64; COUNTERS], n: &mut u32) {
    *n = 0;
    for (i, v) in c.iter_mut().enumerate() {
        let v = std::mem::take(v);
        if v != 0 {
            COUNTED[i].fetch_add(v, Ordering::Relaxed);
        }
    }
}

/// Adds this thread's counts now (a list closing, tests).
pub(super) fn flush_here() {
    LOCAL.with_borrow_mut(|(c, n)| flush(c, n));
}

#[cfg(test)]
pub(super) fn counted_here() -> [u64; COUNTERS] {
    LOCAL.with_borrow(|(c, _)| *c)
}

/// The 5-second line, with the switch on (`frames` since the last one).
/// The counts of title threads that recorded fewer than 4,096 lookups since
/// their last flush come in a later line.
pub(crate) fn log(frames: f64) {
    let mode = mode();
    if mode == 0 {
        return;
    }
    flush_here();
    let v: [f64; COUNTERS] = std::array::from_fn(|i| COUNTED[i].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    let share = 100.0 * v[HITS] / v[LOOKUPS].max(1.0);
    let verify = if mode == 2 {
        format!(
            "; verify (hits prepared anyway): {:.1} changed the list's epochs, {:.1} added steps (both must be 0)",
            v[VERIFY_EPOCH], v[VERIFY_STEPS]
        )
    } else {
        String::new()
    };
    tracing::info!(
        "table memo (VICTORY_TABLE_MEMO={mode}) per frame: {:.0} table entries looked up, {:.0} remembered ({share:.1}%, \
         {}); not remembered: {:.0} first in the list, {:.0} slot rewritten, {:.0} list changed, {:.0} copy or slot \
         changed, {:.0} prepared every time{verify}",
        v[LOOKUPS],
        v[HITS],
        if mode == 1 { "skipped" } else { "prepared anyway" },
        v[FIRST],
        v[REWRITTEN],
        v[EPOCH],
        v[CHANGED],
        v[UNREMEMBERED],
    );
}

#[cfg(test)]
mod tests {
    use super::mode_rule;

    #[test]
    fn the_memo_is_off_unless_asked_for() {
        for off in [None, Some(""), Some("0"), Some("on"), Some("3")] {
            assert_eq!(mode_rule(off), 0, "{off:?}");
        }
        assert_eq!(mode_rule(Some("1")), 1);
        assert_eq!(mode_rule(Some(" 1 ")), 1);
        assert_eq!(mode_rule(Some("2")), 2);
        assert_eq!(mode_rule(Some("verify")), 2);
    }
}
