//! Views shaders reach through the resource heap itself (decision 0007,
//! "Heap-indexed views"). Grand Theft Auto VI is largely bindless: its
//! shaders load descriptors at indices they take from root constants or
//! memory (`HostShader::heap`), so no draw names the views it reads. Every
//! view the title writes into a shader-visible heap gets its host view when
//! it is written (`descriptor::host`), and each heap keeps here what its
//! slots view that title memory feeds: ranges of host buffer copies, and
//! textures whose contents the title writes. A submission whose lists run
//! such shaders first brings all of them up to date (what changed since,
//! `crate::watch`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::census_fast::{Fast, Hold, Tickets, Unordered};
use crate::census_maps::{Counts, Hashing};
use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// What a title heap slot views that title memory feeds.
#[derive(Clone)]
pub(crate) enum Seen {
    /// `len` bytes at `address` of a buffer's host copy; unordered access.
    Buffer(Arc<Shadow>, u64, u64, bool),
    /// A title texture (resource object) whose contents the title writes.
    Texture(usize),
    /// A view left without a host view (textures the title placed itself),
    /// by kind (diagnostics: `KINDS`), with a placed texture's address and
    /// console bytes.
    Left(#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] (usize, u64, u64)),
}

/// Kinds of views left without a host view, for the census.
#[cfg(feature = "diagnostics")]
const KINDS: [&str; 4] = ["buffer", "unordered buffer", "placed texture", "other"];

/// A heap's slots that view title memory, and the distinct buffer ranges
/// and textures among them (with how many slots view each).
pub(crate) struct Viewed {
    inner: Mutex<Inner>,
    /// mac: with `VICTORY_CENSUS_FAST=1` (`crate::census_fast`): the kept
    /// answer of [`unordered_writes`] and its generation.
    fast: Option<Fast<Shadow>>,
    /// mac: with `VICTORY_CENSUS_BATCH=1` (and the fast parts): a heap sync
    /// pass holds the updates of its slots ([`Pass`]).
    batch: bool,
}

impl Default for Viewed {
    fn default() -> Self {
        Viewed::new(crate::mac_switches::census_fast())
    }
}

impl Viewed {
    /// A census, with the fast parts (`fast`) or as before.
    pub(crate) fn new(fast: bool) -> Self {
        Viewed::with_batch(fast, fast && crate::mac_switches::census_batch())
    }

    /// A census, with the fast parts (`fast`) and passes holding updates
    /// (`batch`, only with them).
    pub(crate) fn with_batch(fast: bool, batch: bool) -> Self {
        Viewed { inner: Mutex::new(Inner::new(fast)), fast: fast.then(Fast::default), batch: fast && batch }
    }
}

pub(crate) type Range = (usize, u64, u64, bool);

/// A merged range: copy, start, bytes, unordered access, the copy's bit in
/// the change tracker (`crate::watch::clean_bit`).
type Hull = (Arc<Shadow>, u64, u64, bool, Option<u32>);

struct Inner {
    /// These two in the form `VICTORY_QUICK_SYNC` chooses (`crate::census_maps`).
    slots: HashMap<u32, Seen, Hashing>,
    ranges: Counts<Range, (u32, Arc<Shadow>)>,
    textures: BTreeMap<usize, u32>,
    /// `ranges` merged per copy and kind ([`Viewed::ranges`]), kept current
    /// as ranges come and go (each list running heap shaders asks for all
    /// of them; rebuilt whole after every change, they cost 13-27 ms a frame,
    /// run perf4).
    hulls: HashMap<(usize, bool), Hull>,
    /// Per copy and kind (unordered access): the copy, and how many ranges
    /// start and end at each offset, so that the hulls are made from the
    /// copies, not from every range (the title rewrites slots between most
    /// lists, and remerging the world's 130,000 ranges each time cost 0.15
    /// s per frame).
    ends: HashMap<(usize, bool), Ends>,
    /// mac: with `VICTORY_CENSUS_FAST=1`: the unordered ranges of `ranges`
    /// alone, changed where `ranges` gains or loses one ([`Inner::count`]).
    unordered: Option<Unordered<Shadow>>,
    /// mac: with `VICTORY_CENSUS_FAST=1`: per slot, the ticket of the making
    /// its entry is from ([`Inner::enter`]).
    tickets: Tickets,
}

impl Inner {
    fn new(fast: bool) -> Self {
        let [quick, _] = crate::mac_switches::quick_sync(crate::shader_cache::under_wine);
        Inner {
            slots: HashMap::with_hasher(Hashing::new(quick)),
            ranges: Counts::new(quick),
            textures: BTreeMap::new(),
            hulls: HashMap::new(),
            ends: HashMap::new(),
            unordered: fast.then(Unordered::new),
            tickets: Tickets::default(),
        }
    }
}

/// A copy, and the counts of its ranges' starts and ends.
type Ends = (Arc<Shadow>, BTreeMap<u64, u32>, BTreeMap<u64, u32>);

/// One more (`add`) or one fewer of `at` in `counts`.
fn tally(counts: &mut BTreeMap<u64, u32>, at: u64, add: bool) {
    let n = counts.entry(at).or_insert(0);
    *n = if add { *n + 1 } else { n.saturating_sub(1) };
    if *n == 0 {
        counts.remove(&at);
    }
}

impl Inner {
    /// One more (`add`) or one fewer slot views what `seen` reads. Whether
    /// the census's unordered ranges changed (mac: told only with
    /// `VICTORY_CENSUS_FAST=1`; false without it).
    fn count(&mut self, seen: &Seen, add: bool) -> bool {
        let step = |n: &mut u32| if add { *n += 1 } else { *n = n.saturating_sub(1) };
        let mut unordered_changed = false;
        match seen {
            Seen::Buffer(shadow, address, len, uav) => {
                let key = (Arc::as_ptr(shadow) as usize, *address, address + len, *uav);
                let e = self.ranges.entry_or(key, || (0, shadow.clone()));
                let was = e.0;
                step(&mut e.0);
                let now = e.0;
                if now == 0 {
                    self.ranges.remove(&key);
                }
                // mac: an unordered range came or went (`crate::census_fast`).
                if (was == 0) != (now == 0)
                    && key.3
                    && let Some(unordered) = &mut self.unordered
                {
                    if now > 0 {
                        unordered.came(key, shadow);
                    } else {
                        unordered.went(&key);
                    }
                    unordered_changed = true;
                }
                // A range came or went: its copy's hull may change.
                if (was == 0) != (now == 0) {
                    let ends = self
                        .ends
                        .entry((key.0, key.3))
                        .or_insert_with(|| (shadow.clone(), BTreeMap::new(), BTreeMap::new()));
                    tally(&mut ends.1, key.1, now > 0);
                    tally(&mut ends.2, key.2, now > 0);
                    let hull = match (ends.1.keys().next(), ends.2.keys().next_back()) {
                        (Some(&from), Some(&to)) => Some((shadow.clone(), from, to - from, key.3, shadow.dirty_bit())),
                        _ => None,
                    };
                    match hull {
                        Some(hull) => {
                            self.hulls.insert((key.0, key.3), hull);
                        }
                        None => {
                            self.ends.remove(&(key.0, key.3));
                            self.hulls.remove(&(key.0, key.3));
                        }
                    }
                }
            }
            Seen::Texture(r) => {
                let n = self.textures.entry(*r).or_insert(0);
                step(n);
                if *n == 0 {
                    self.textures.remove(r);
                }
            }
            Seen::Left(_) => {}
        }
        unordered_changed
    }

    /// mac (`VICTORY_CENSUS_FAST=1`): slot `index` holds `seen`, as of the
    /// making that drew `ticket`, unless a later making of the slot is in
    /// already (`crate::census_fast::Tickets`).
    fn enter(&mut self, fast: &Fast<Shadow>, index: u32, ticket: u64, seen: Option<Seen>) {
        if !self.tickets.newer(index, ticket) {
            return;
        }
        let mut changed = false;
        if let Some(old) = self.slots.remove(&index) {
            changed |= self.count(&old, false);
        }
        if let Some(seen) = seen {
            changed |= self.count(&seen, true);
            self.slots.insert(index, seen);
        }
        // Told before the census lock goes.
        if changed {
            fast.changed();
        }
    }
}

/// mac: the census updates of a heap sync pass made the quick way
/// (`crate::descriptor::host`, `sync_quick`), entered under one hold of
/// the census lock [`crate::census_fast::BATCH`] at a time instead of one
/// lock a slot (`VICTORY_CENSUS_FAST=1`); without the switch each goes in
/// at once, as before. Each update carries the ticket of its slot's making,
/// so a slot made again meanwhile by another thread (entered at once, with
/// a later ticket) keeps that making's entry. Entered at the latest when
/// the batch goes: the pass's own thread reads the census after it (the
/// submission's refreshes) only once the sync has returned.
pub(crate) struct Batch<'a> {
    viewed: &'a Viewed,
    updates: Vec<(u32, u64, Option<Seen>)>,
}

impl Batch<'_> {
    /// Where the pass makes slot `index` (under its lock): what
    /// [`Viewed::set_seen`] enters, now (no switch) or with the batch.
    pub(crate) fn set_seen(&mut self, index: u32, seen: Option<Seen>) {
        match &self.viewed.fast {
            Some(fast) => self.updates.push((index, fast.ticket(), seen)),
            None => self.viewed.set_seen(index, seen),
        }
    }

    /// With no slot lock held: the updates so far, if there are a batch's.
    pub(crate) fn enter_if_full(&mut self) {
        if self.updates.len() >= crate::census_fast::BATCH {
            self.enter();
        }
    }

    /// With no slot lock held: the updates so far, under one hold of the
    /// census lock (waits counted with `VICTORY_HEAP_LOCKS`, as
    /// [`Viewed::set_seen`]'s).
    pub(crate) fn enter(&mut self) {
        let Some(fast) = &self.viewed.fast else {
            return;
        };
        if self.updates.is_empty() {
            return;
        }
        let mut inner = crate::descriptor::slots::locked(&self.viewed.inner, crate::descriptor::slots::CENSUS);
        for (index, ticket, seen) in self.updates.drain(..) {
            inner.enter(fast, index, ticket, seen);
        }
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        self.enter();
    }
}

/// A census update as a pass holds it: the slot, its making's ticket, what
/// it views.
type Update = (u32, u64, Option<Seen>);

thread_local! {
    /// mac (`VICTORY_CENSUS_BATCH=1`): the updates this thread holds for
    /// the heap sync pass it is in, if it is in one ([`Pass`]).
    static HELD: std::cell::RefCell<Option<Hold<Update>>> = const { std::cell::RefCell::new(None) };
}

/// mac: while this lives, the census updates of the slots this thread makes
/// in `viewed` through [`Viewed::set_seen`] (`HostHeap::mirror`,
/// `HostHeap::make_null`) are held ([`crate::census_fast::Hold`]) and entered
/// under one hold of the census lock [`crate::census_fast::BATCH`] at a time
/// ([`Pass::enter_if_full`]), the rest when the outermost pass goes, before
/// the pass's function returns: readers ordered after it see every slot it
/// made. Texture views still go in at once ([`forget`] must find them), and
/// so do other heaps' and other threads' updates. Without the switch (or
/// inside another pass of the thread) it holds nothing.
pub(crate) struct Pass<'a> {
    viewed: &'a Viewed,
    /// This pass installed the thread's hold (it is the outermost one).
    own: bool,
}

impl Pass<'_> {
    /// With no slot lock held: a batch's held updates, if there are a
    /// batch's (for this census).
    pub(crate) fn enter_if_full(&self) {
        self.enter(false);
    }

    fn enter(&self, all: bool) {
        let Some(fast) = &self.viewed.fast else {
            return;
        };
        let owner = std::ptr::from_ref(self.viewed) as usize;
        let updates = HELD.with(|h| match h.borrow_mut().as_mut() {
            Some(hold) if hold.is_for(owner) => hold.take(all),
            _ => Vec::new(),
        });
        if updates.is_empty() {
            return;
        }
        let n = updates.len();
        let mut inner = crate::descriptor::slots::locked(&self.viewed.inner, crate::descriptor::slots::CENSUS);
        for (index, ticket, seen) in updates {
            inner.enter(fast, index, ticket, seen);
        }
        drop(inner);
        crate::census_fast::entered(n);
    }
}

impl Drop for Pass<'_> {
    fn drop(&mut self) {
        if self.own {
            self.enter(true);
            HELD.with(|h| *h.borrow_mut() = None);
        }
    }
}

impl Viewed {
    /// mac: a heap sync pass over this census's heap ([`Pass`]).
    pub(crate) fn pass(&self) -> Pass<'_> {
        let owner = std::ptr::from_ref(self) as usize;
        let own = self.batch
            && HELD.with(|h| {
                let mut h = h.borrow_mut();
                let free = h.is_none();
                if free {
                    *h = Some(Hold::new(owner));
                }
                free
            });
        Pass { viewed: self, own }
    }

    /// mac: `seen` for slot `index`, made with `ticket`, held for the pass
    /// this thread is in over this census; given back if it is not in one,
    /// or if it is a texture view.
    fn hold(&self, index: u32, ticket: u64, seen: Option<Seen>) -> Result<(), Option<Seen>> {
        if !self.batch {
            return Err(seen);
        }
        let owner = std::ptr::from_ref(self) as usize;
        HELD.with(|h| match h.borrow_mut().as_mut() {
            Some(hold) if !matches!(seen, Some(Seen::Texture(_))) => {
                hold.hold(owner, (index, ticket, seen)).map_err(|(_, _, seen)| seen)
            }
            Some(_) => {
                crate::census_fast::at_once();
                Err(seen)
            }
            None => Err(seen),
        })
    }

    /// Title slot `index` now holds `view` (none: no view), made on the
    /// host (`made`) or left without a host view.
    pub(crate) fn set(&self, index: u32, view: Option<&View>, made: bool) {
        let seen = view.and_then(|v| if made { seen(v) } else { Some(Seen::Left(left_kind(v))) });
        self.set_seen(index, seen);
    }

    /// [`Viewed::set`] with what the slot's view reads already worked out
    /// (`crate::descriptor::host`, "quick").
    pub(crate) fn set_seen(&self, index: u32, seen: Option<Seen>) {
        // mac: with `VICTORY_CENSUS_FAST=1` in the order of the slot's
        // makings, by a ticket drawn here (callers hold the slot's lock), as
        // a heap sync's batch enters them ([`Batch`]).
        if let Some(fast) = &self.fast {
            let ticket = fast.ticket();
            // mac: held for the heap sync pass this thread is in
            // (`VICTORY_CENSUS_BATCH=1`, [`Pass`]).
            let Err(seen) = self.hold(index, ticket, seen) else {
                return;
            };
            let mut inner = crate::descriptor::slots::locked(&self.inner, crate::descriptor::slots::CENSUS);
            inner.enter(fast, index, ticket, seen);
            return;
        }
        // mac: a wait for it is counted with `VICTORY_HEAP_LOCKS`.
        let mut inner = crate::descriptor::slots::locked(&self.inner, crate::descriptor::slots::CENSUS);
        if let Some(old) = inner.slots.remove(&index) {
            inner.count(&old, false);
        }
        if let Some(seen) = seen {
            inner.count(&seen, true);
            inner.slots.insert(index, seen);
        }
    }

    /// mac: census updates entered together (`VICTORY_CENSUS_FAST=1`).
    pub(crate) fn batch(&self) -> Batch<'_> {
        Batch { viewed: self, updates: Vec::new() }
    }

    /// The range slot `index` views of a buffer copy (the copy, the title
    /// address, the bytes, unordered access), if it views one (tests).
    #[cfg(test)]
    pub(crate) fn range_at(&self, index: u32) -> Option<Range> {
        match self.inner.lock().unwrap_or_else(|e| e.into_inner()).slots.get(&index) {
            Some(Seen::Buffer(shadow, address, len, uav)) => Some((Arc::as_ptr(shadow) as usize, *address, *len, *uav)),
            _ => None,
        }
    }

    /// The buffer ranges views reach, merged per host copy and kind
    /// (shader resource, unordered access): copy, start, bytes.
    #[cfg(any(test, feature = "diagnostics"))]
    fn ranges(&self) -> Vec<Hull> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).hulls.values().cloned().collect()
    }

    /// Snapshot of distinct title texture resources currently viewed by this
    /// heap, for the background heap worker.
    pub(crate) fn texture_resources(&self) -> Vec<usize> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .textures
            .keys()
            .copied()
            .collect()
    }

    /// Whether any title-written texture is currently viewed by this heap.
    pub(crate) fn has_textures(&self) -> bool {
        !self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .textures
            .is_empty()
    }
}

impl Seen {
    /// What [`seen`] finds for a made shader view of `len` bytes (not none)
    /// at title address `address`, whose host copy is `shadow`.
    pub(crate) fn shader_buffer(shadow: Arc<Shadow>, address: u64, len: u64) -> Self {
        let len = len.min(shadow.left(address));
        Seen::Buffer(shadow, address, len, false)
    }
}

/// What a made view reads that title memory feeds.
fn seen(view: &View) -> Option<Seen> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let View::Shader(resource, _) = view
        && let Some(texture) = crate::resource::made_copy(*resource as This)
    {
        return texture.title_written().then_some(Seen::Texture(*resource));
    }
    // The copy the host view reads.
    let (shadow, address, len) = match *view {
        View::Constant(address, size) => {
            let len = u64::from(size);
            ((len > 0).then(|| crate::shadow::find(address, len)).flatten()?, address, len)
        }
        _ => crate::descriptor::host::view_copy(view)?,
    };
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    // mac: `VICTORY_ALIAS_PHYSICAL=verify`: the view `=gpu` would put in
    // the slot when it is made.
    if crate::alias_physical::verifying() {
        let layout = match *view {
            View::Constant(address, size) => Some((address, u64::from(size), 4)),
            _ => crate::descriptor::host::view_layout(view),
        };
        if let Some((address, len, element)) = layout {
            let bind = if uav { crate::alias_verify::Bind::SlotUav } else { crate::alias_verify::Bind::Slot };
            crate::alias_verify::table(bind, address, len, element);
        }
    }
    Some(Seen::Buffer(shadow, address, len, uav))
}

fn left_kind(view: &View) -> (usize, u64, u64) {
    match view {
        View::Placed(_, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (2, address, crate::resource::placed_size(resource).unwrap_or(0))
        }
        View::Unordered(..) | View::Placed(true, ..) => (1, 0, 0),
        View::Shader(..) | View::Placed(false, ..) | View::Constant(..) => (0, 0, 0),
        _ => (3, 0, 0),
    }
}

/// At submission, before anything else of it: the textures the heaps of its
/// heap-indexing lists view, and `textures` (staging and host texture of
/// those its lists read through tables, `crate::texture::upload`), are
/// brought up to date, those whose title memory changed copied into their
/// host textures by a list of ours, which the submission runs first (and
/// leaves them readable by shaders): each at most once.
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
///
/// mac: `before` is told what is to be brought up before any of its title
/// memory is read (flights, `VICTORY_FLIGHTS_WITH=alias`); it says how to
/// read them ([`Reads`]): now, or some or all only after the submission
/// left is finished (`alias,uploads`, `alias,uploads2`,
/// [`Uploads::land_late`]).
pub(crate) fn uploads_after(
    heaps: &[Arc<Viewed>],
    textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
    before: &mut dyn FnMut(&[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)]) -> Reads,
) -> Option<Uploads> {
    // mac: two, used in turn with flights' `alias`, else the first only: a
    // submission prepared while the one before still runs takes the list
    // and staging that one is not using (`crate::texture::upload_slot`).
    static LISTS: [Mutex<Holder>; 2] = [const { Mutex::new(Holder(None)) }; 2];
    let slot = crate::texture::upload_slot();
    let mut guard = LISTS[slot].lock().unwrap_or_else(|e| e.into_inner());
    crate::texture::recycle_in(slot);
    match &guard.0 {
        // Closed by the last submission of the slot, which has run.
        Some(list) => list.reset(),
        // Made open.
        None => {
            let list = crate::host::host()?.new_list().ok()?;
            list.set_ours();
            guard.0 = Some(list);
        }
    }
    let list = guard.0.as_ref()?;
    // What to bring up: the textures lists read through tables, and those
    // the heaps view that shaders read lately. Collected under the heaps'
    // locks and brought up without them, several at a time: reading title
    // memory, detiling and making staging buffers took half the submitting
    // thread in a new shot, and title threads setting views waited on the
    // lock meanwhile (run evict5).
    let mut work: Vec<(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)> = Vec::new();
    let mut staged = std::collections::HashSet::new();
    for (staging, copy) in textures {
        // A host copy wrote it in a submission made before this one (it
        // was not when the list was recorded).
        if copy.host_written.load(Relaxed) {
            continue;
        }
        if staged.insert(Arc::as_ptr(staging)) {
            work.push((staging.clone(), copy.clone()));
        }
    }
    let worker_on = crate::heap_worker::on();
    let mut done = std::collections::HashSet::new();
    if !worker_on {
        for viewed in heaps {
            let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
            for &r in inner.textures.keys() {
                if !done.insert(r) {
                    continue;
                }
                // Not made (given up, or never read): nothing to bring up.
                let Some(texture) = crate::resource::made_copy(r as This) else {
                    continue;
                };
                // Only what shaders read lately (the eviction pass's stamps):
                // the heaps view thousands of textures a frame never reads, and
                // bringing each up whenever its pages change re-uploaded about
                // 4 GB in 30-46 world frames (runs evict2, noevict1).
                // shortcut: a texture read again after a gap is brought up after
                // the submission that read it, which can see bytes one refresh
                // old (a slot the title rewrites is made fresh anyway); upload on
                // the mark itself if that shows.
                if !texture.used_recently() {
                    continue;
                }
                // Its contents are the host's since a host copy wrote it.
                if texture.host_written.load(Relaxed) {
                    continue;
                }
                if let Some(staging) = crate::texture::staging(r as This, &texture.texture)
                    && staged.insert(Arc::as_ptr(&staging))
                {
                    work.push((staging, texture));
                }
            }
        }
    }
    // mac: before their title memory is read; those it names (E2b of
    // flights) are kept for after the submission left is finished
    // ([`Uploads::land_late`]): the list notes their states now, for its
    // prelude, and records their copies then.
    let mut late = Late::default();
    // (E2c: the state each texture read beside it was in, for after.)
    let mut now_states = Vec::new();
    let later = |item: (Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>), late: &mut Late| {
        let (first, back) = crate::texture::upload_later(list, &item.1);
        if let Some(first) = first {
            // SAFETY: an open list; the barrier names a live texture.
            unsafe { list.list.ResourceBarrier(&[first]) };
        }
        late.back.extend(back);
        late.work.push(item);
    };
    match before(&work) {
        Reads::Now => {}
        Reads::Keep(mut kept) if !kept.is_empty() => {
            kept.sort_unstable();
            kept.dedup();
            let mut now = Vec::with_capacity(work.len() - kept.len());
            let mut kept = kept.into_iter().peekable();
            for (i, item) in std::mem::take(&mut work).into_iter().enumerate() {
                if kept.next_if_eq(&i).is_some() {
                    later(item, &mut late);
                } else {
                    now.push(item);
                }
            }
            work = now;
        }
        Reads::Keep(_) => {}
        // mac (E2c, `uploads2`): sorted by what each texture's tracker says
        // and where its memory is (`crate::flights::Beside`), with nothing
        // asked of the one left: the changed ones read now unless a page is
        // in an aliased region or they are not tracked (those after its
        // finish); every other one is looked at again after its finish,
        // with the state it is in now.
        Reads::Again => {
            let started = std::time::Instant::now();
            let textures = work.len();
            late.uploads2 = true;
            late.marks = crate::watch::texture_marks();
            let (mut changed, mut kept) = (0, [0usize; 2]);
            let mut now = Vec::new();
            for item in std::mem::take(&mut work) {
                match item.0.beside() {
                    crate::flights::Beside::Later(why) => {
                        changed += 1;
                        kept[why as usize] += 1;
                        later(item, &mut late);
                    }
                    crate::flights::Beside::Now => {
                        changed += 1;
                        now_states.push(item.1.state.load(Relaxed));
                        now.push(item);
                    }
                    crate::flights::Beside::Unchanged => {
                        let state = item.1.state.load(Relaxed);
                        late.again.push(Again { item, state, read: false });
                    }
                }
            }
            work = now;
            crate::host::unwaited::flights::checked(textures, changed, kept, work.len(), started.elapsed());
        }
    }
    let mut recorded = !late.work.is_empty() || !late.again.is_empty();
    // The textures copied into stay alive in `work` until recorded.
    let mut readable = Vec::new();
    for ((staging, copy), upload) in work.iter().zip(crate::texture::refresh_all(&work)) {
        if let Some(upload) = upload {
            readable.extend(crate::texture::record_upload(list, copy, staging, &upload));
            recorded = true;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
        }
    }
    let ready = if worker_on {
        let ready = crate::heap_worker::take_ready();
        for item in ready.iter_uploads() {
            let still_made = crate::resource::made_copy(item.resource as This)
                .is_some_and(|c| Arc::ptr_eq(&c, &item.copy));
            if !still_made {
                crate::heap_worker::STATS[5].fetch_add(1, Relaxed);
                continue;
            }
            if let Some(p) = &item.publish {
                let key_matches =
                    crate::descriptor::view_and_key(p.handle).is_some_and(|(_, k)| k == p.key);
                if !key_matches {
                    crate::heap_worker::STATS[4].fetch_add(1, Relaxed);
                    if item.upload.is_some()
                        && !item.copy.host_written.load(Relaxed)
                        && !item.copy.is_gpu_written()
                    {
                        item.staging.invalidate();
                    }
                    continue;
                }
            }
            if let Some(upload) = &item.upload {
                if item.copy.host_written.load(Relaxed) || item.copy.is_gpu_written() {
                    crate::heap_worker::STATS[5].fetch_add(1, Relaxed);
                } else if staged.insert(Arc::as_ptr(&item.staging)) {
                    readable.extend(crate::texture::record_upload(
                        list,
                        &item.copy,
                        &item.staging,
                        upload,
                    ));
                    item.staging.mark_uploaded_once();
                    recorded = true;
                    #[cfg(feature = "diagnostics")]
                    UPLOADED.fetch_add(1, Relaxed);
                }
            }
            if let Some(p) = &item.publish
                && let Some(host) = crate::descriptor::heap(p.heap as This).made_host()
            {
                let shown = host.publish(p.index, p.handle, p.key);
                crate::heap_worker::STATS[if shown { 3 } else { 4 }].fetch_add(1, Relaxed);
            }
        }
        if !heaps.is_empty() {
            crate::descriptor::kick_worker_heaps();
        }
        (!ready.is_empty() && recorded).then_some(ready)
    } else {
        None
    };
    if !readable.is_empty() {
        // SAFETY: an open list; the barriers name live textures.
        unsafe { list.list.ResourceBarrier(&readable) };
    }
    // mac (E2c): those read beside it are looked at again after its finish
    // too, where its landing may have told their trackers since.
    if late.uploads2 {
        late.again.extend(work.into_iter().zip(now_states).map(|(item, state)| Again { item, state, read: true }));
        recorded |= !late.again.is_empty();
    }
    // mac: with textures kept for later, closed once they are recorded.
    let pending = !late.work.is_empty() || !late.again.is_empty();
    if !pending {
        list.close();
    }
    // A list recording nothing is not run (its reset next time is fine).
    recorded.then(|| {
        crate::texture::slot_given(slot);
        Uploads {
            list: guard,
            worker: Mutex::new(ready),
            pending: pending.into(),
            late: Mutex::new(late),
        }
    })
}

/// [`uploads_after`] with nothing told first (tests).
#[cfg(test)]
pub(crate) fn uploads(
    heaps: &[Arc<Viewed>],
    textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
) -> Option<Uploads> {
    uploads_after(heaps, textures, &mut |_| Reads::Now)
}

/// The upload list, used under its lock only.
pub(crate) struct Holder(Option<HostList>);
// SAFETY: D3D12 objects are free-threaded; the list is used by one
// submission at a time, under the lock.
unsafe impl Send for Holder {}

/// mac: how [`uploads_after`] reads the textures of a submission prepared
/// beside the one left (flights' `alias`, `crate::host::unwaited::flights`).
pub(crate) enum Reads {
    /// All now (nothing runs beside, or it was finished first).
    Now,
    /// Now, but those named (indices), kept for after its finish (E2b,
    /// `uploads`).
    Keep(Vec<usize>),
    /// Sorted by the textures' own trackers, those not read now looked at
    /// again after its finish (E2c, `uploads2`, `crate::flights::Beside`).
    Again,
}

/// mac (E2b of flights): textures an upload list brings up only once the
/// submission left is finished, and the barriers that leave them readable.
#[derive(Default)]
struct Late {
    work: Vec<(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)>,
    back: Vec<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER>,
    /// mac (E2c, `uploads2`): the check's other textures, looked at again
    /// after the finish; the count of texture-page marks at the check
    /// (`crate::watch::texture_marks`).
    uploads2: bool,
    again: Vec<Again>,
    marks: u64,
}

/// mac (E2c): a texture of the check not kept: whether it was read beside
/// the one left, and the state it was in before the submission's preludes
/// (what the upload list, run first, finds it in on the GPU).
struct Again {
    item: (Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>),
    state: u32,
    read: bool,
}

/// The upload list of a submission, ours until it has run.
pub(crate) struct Uploads {
    list: std::sync::MutexGuard<'static, Holder>,
    worker: Mutex<Option<crate::heap_worker::ReadyDrain>>,
    /// mac: textures kept for after the submission left is finished: the
    /// list stays open until [`Uploads::land_late`] records them.
    pending: std::sync::atomic::AtomicBool,
    late: Mutex<Late>,
}

impl Uploads {
    pub(crate) fn list(&self) -> &HostList {
        self.list.0.as_ref().expect("made before it is handed out")
    }

    /// mac (E2b of flights, `VICTORY_FLIGHTS_WITH=alias,uploads`): once the
    /// submission left is finished, and before the list goes to the GPU:
    /// the textures kept for then are brought up from title memory as it
    /// is now (its shaders' writes and its landing in), their copies and
    /// the barriers that leave them readable recorded, and the list closed.
    /// At once and nothing more if none are kept, or after the first call.
    pub(crate) fn land_late(&self) {
        if !self.pending.swap(false, Relaxed) {
            return;
        }
        let started = std::time::Instant::now();
        let late = std::mem::take(&mut *self.late.lock().unwrap_or_else(|e| e.into_inner()));
        let list = self.list();
        if late.uploads2 {
            return land2(list, late, started);
        }
        let mut uploaded = 0;
        for ((staging, copy), upload) in late.work.iter().zip(crate::texture::refresh_all(&late.work)) {
            if let Some((buffer, at, subresources)) = upload {
                crate::texture::record_copies(list, copy, staging, (&buffer, at), &subresources);
                uploaded += 1;
                #[cfg(feature = "diagnostics")]
                UPLOADED.fetch_add(1, Relaxed);
            }
        }
        if !late.back.is_empty() {
            // SAFETY: an open list; the barriers name live textures.
            unsafe { list.list.ResourceBarrier(&late.back) };
        }
        list.close();
        crate::host::unwaited::flights::landed_late(late.work.len(), uploaded, started.elapsed());
    }

    /// Whether it holds uploads the background heap worker staged.
    pub(crate) fn has_worker(&self) -> bool {
        self.worker.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Returns worker-owned staging chunks immediately once the submission's
    /// GPU fence has completed.
    pub(crate) fn release_worker(&self) {
        drop(self.worker.lock().unwrap_or_else(|e| e.into_inner()).take());
    }
}

/// mac (E2c of flights, `uploads2`): [`Uploads::land_late`] once the
/// submission left is finished. The kept textures are read now, as E2b's
/// are. Every other texture of the check is brought up if its tracker says
/// it changed since the check (the one left's landing marks every tracker
/// over what it writes back; asked only if a texture page was marked at
/// all since, `crate::watch::texture_marks`): through the list's own
/// states where it moves the texture already, else from the state it was
/// in before the submission ([`crate::texture::record_upload_from`]).
fn land2(list: &HostList, late: Late, started: std::time::Instant) {
    let mut landed = crate::host::unwaited::flights::Landed2 { kept: late.work.len(), ..Default::default() };
    for ((staging, copy), upload) in late.work.iter().zip(crate::texture::refresh_all(&late.work)) {
        if let Some((buffer, at, subresources)) = upload {
            crate::texture::record_copies(list, copy, staging, (&buffer, at), &subresources);
            landed.kept_uploaded += 1;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
        }
    }
    if !late.back.is_empty() {
        // SAFETY: an open list; the barriers name live textures.
        unsafe { list.list.ResourceBarrier(&late.back) };
    }
    if crate::watch::texture_marks() == late.marks {
        landed.skipped = late.again.len();
    } else {
        landed.asked = late.again.len();
        let told: Vec<&Again> = late.again.iter().filter(|a| a.item.0.maybe_changed()).collect();
        let items: Vec<_> = told.iter().map(|a| a.item.clone()).collect();
        for ((again, (staging, copy)), upload) in told.iter().zip(&items).zip(crate::texture::refresh_all(&items)) {
            landed.told[usize::from(again.read)] += 1;
            let Some(upload) = upload else { continue };
            landed.uploaded += 1;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
            if list.moves(copy) {
                if let Some(readable) = crate::texture::record_upload(list, copy, staging, &upload) {
                    // SAFETY: an open list; the barrier names a live texture.
                    unsafe { list.list.ResourceBarrier(&[readable]) };
                }
            } else {
                crate::texture::record_upload_from(list, copy, staging, &upload, again.state);
            }
        }
    }
    list.close();
    crate::host::unwaited::flights::landed2(&landed, started.elapsed());
}

impl Drop for Uploads {
    /// mac: textures kept for later and never landed (not a path
    /// `crate::host::Host::run` takes): the list is closed with them left
    /// readable, so that it can be reset; they are brought up next time.
    fn drop(&mut self) {
        if !self.pending.swap(false, Relaxed) {
            return;
        }
        tracing::warn!("texture uploads kept for after the submission before were not made: their list is closed without them");
        let late = std::mem::take(&mut *self.late.lock().unwrap_or_else(|e| e.into_inner()));
        for (staging, _) in &late.work {
            staging.invalidate();
        }
        let list = self.list();
        if !late.back.is_empty() {
            // SAFETY: an open list; the barriers name live textures.
            unsafe { list.list.ResourceBarrier(&late.back) };
        }
        list.close();
    }
}

/// Per submission: brings the buffer ranges a heap's views reach up to date
/// (in the submission's batch of refreshes): the pages the title changed
/// since their copies took them.
///
/// shortcut: what shaders write through the heap's unordered access views
/// stays in the host copies (later draws read it there) and does not go
/// back to title memory, so the CPU does not see it, and a page the title
/// also writes is copied over from title memory, losing the GPU's bytes
/// there. Copying back the whole unordered ranges after every submission
/// (about 1 GB of them in the world) cost seconds per frame; the upgrade
/// is shaders marking the pages they write (as `crate::memory` does).
///
/// Copies the title has not written since they took it are passed by: a
/// world submission's heaps reach ~9,000 copies, a few of them written.
///
/// mac: while the submission before still runs beside this one (flights),
/// also the clean copies its landing may tell (`crate::flights::asked`).
pub(crate) fn refresh(viewed: &Viewed, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let beside = crate::host::unwaited::flights::landing();
    let beside = beside.as_deref().map(crate::host::unwaited::flights::Shared::landing);
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (shadow, address, len, uav, bit) in inner.hulls.values() {
        if crate::flights::asked(crate::watch::clean_bit(*bit), beside, *address as usize, *len as usize) {
            let (address, len) = if *uav && shadow.marked() {
                shadow.page_range(*address, *len)
            } else {
                (*address, *len)
            };
            add(shadow, address as usize, len as usize);
        }
    }
}

/// The copies the heap views for unordered access: what shaders that take
/// views from the heap itself may write with no list declaring it
/// (`crate::host::unwaited`, for a submission still running).
pub(crate) fn written(viewed: &Viewed, mut each: impl FnMut(&Arc<Shadow>)) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (shadow, _, _, uav, _) in inner.hulls.values() {
        if *uav {
            each(shadow);
        }
    }
}

/// mac: what the heap's views for unordered access reach, not the whole
/// copies they are in ([`written`]): the union of each copy's unordered
/// ranges (title addresses, `[from, to)`), or the copy's hull of them past
/// [`UNION_MAX`] ranges. A shader writing through a heap view reaches only
/// its range (`crate::descriptor::host`'s `buffer_uav` makes the host view
/// over exactly that range of the copy). For the texture uploads beside a
/// submission still running (`VICTORY_FLIGHTS_WITH=alias,uploads`).
pub(crate) fn written_ranges(viewed: &Viewed, mut each: impl FnMut(u64, u64)) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (&(_, uav), (_, starts, ends)) in &inner.ends {
        if !uav {
            continue;
        }
        if starts.len() > UNION_MAX {
            if let (Some(&from), Some(&to)) = (starts.keys().next(), ends.keys().next_back()) {
                each(from, to);
            }
        } else {
            crate::flights::union_of(starts, ends, &mut each);
        }
    }
}

/// Past this many unordered ranges in one copy, [`written_ranges`] gives
/// their hull (the walk is done under the heap's lock, per submission).
const UNION_MAX: usize = 4096;

/// The buffer range (title address, bytes) of the view currently in
/// `index`, changed or not (diagnostics: what a table draw reads there).
#[cfg(feature = "diagnostics")]
pub(crate) fn slot_range(viewed: &Viewed, index: u32) -> Option<(usize, usize)> {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    match inner.slots.get(&index) {
        Some(Seen::Buffer(_, address, len, _)) => Some((*address as usize, *len as usize)),
        _ => None,
    }
}

/// Per submission: the buffer range of the view currently in `index`. A
/// table draw may have been recorded with another view in that slot.
pub(crate) fn refresh_slot(
    viewed: &Viewed,
    index: u32,
    mut add: impl FnMut(&Arc<Shadow>, usize, usize),
) {
    // mac: as in [`refresh`], asked before the lock.
    let beside = crate::host::unwaited::flights::landing();
    let beside = beside.as_deref().map(crate::host::unwaited::flights::Shared::landing);
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Seen::Buffer(shadow, address, len, _)) = inner.slots.get(&index) else {
        return;
    };
    if crate::flights::asked(crate::watch::clean_bit(shadow.dirty_bit()), beside, *address as usize, *len as usize) {
        add(shadow, *address as usize, *len as usize);
    }
}

/// Heap UAV ranges that may need to refresh a widened SRV after a later
/// buffer barrier. Computed heap indices do not name ordinary write steps.
pub(crate) fn widened_writes(viewed: &Viewed) -> Vec<(Arc<Shadow>, usize, usize)> {
    let mut out = Vec::new();
    widened_writes_into(viewed, &mut out);
    out
}

/// [`widened_writes`] added to `out` (with `VICTORY_WIDENED_IN_PLACE=1` the
/// list's buffer, kept from barrier to barrier: the world's heaps have tens
/// of thousands). From VIctory-macOS #520.
pub(crate) fn widened_writes_into(viewed: &Viewed, out: &mut Vec<(Arc<Shadow>, usize, usize)>) {
    if !crate::under_wine() {
        return;
    }
    out.extend(
        viewed
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ranges
            .iter()
            .filter(|((_, address, end, uav), _)| *uav && address < end)
            .map(|((_, address, end, _), (_, shadow))| {
                (shadow.clone(), *address as usize, (end - address) as usize)
            }),
    );
}

/// mac: with `VICTORY_CENSUS_FAST=1`, [`widened_writes`] without walking
/// every range under the census lock (`crate::census_fast`): the answer
/// kept since the heap's unordered ranges last changed, with no census lock
/// at all, or one made from those ranges alone. The same ranges (in another
/// order where the census is in hash maps, whose order is none anyway).
/// None without the switch.
pub(crate) fn unordered_writes(viewed: &Viewed) -> Option<Arc<[(Arc<Shadow>, usize, usize)]>> {
    let fast = viewed.fast.as_ref()?;
    if !crate::under_wine() {
        return Some(Arc::new([]));
    }
    if let Some(kept) = fast.kept() {
        return Some(kept);
    }
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    inner.unordered.as_ref().map(|unordered| fast.answer(unordered))
}

/// mac: with the frame line (`frames` since the last one): the census
/// updates heap sync passes held (`VICTORY_CENSUS_BATCH=1`, [`Pass`]), the
/// census lock takes that entered them and the takes that saved (one a held
/// update before). Nothing without the switch.
pub(crate) fn log_held(frames: f64) {
    if !crate::mac_switches::census_batch() {
        return;
    }
    let [held, takes, saved, once] = crate::census_fast::take_held(frames);
    tracing::info!(
        "census batch (VICTORY_CENSUS_BATCH=1): {held:.0} census updates/frame of mirror and make_null held by heap \
         sync passes, entered in {takes:.0} census lock takes ({saved:.0} takes/frame saved); {once:.0} texture \
         views/frame entered at once beside a pass"
    );
}

/// Draws and dispatches run with heap-indexing shaders, those of them
/// writing through the heap, textures uploaded for heaps (diagnostics).
static HEAP_DRAWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "diagnostics")]
static UPLOADED: AtomicU64 = AtomicU64::new(0);

/// A draw or dispatch runs a shader that indexes the heap (and writes
/// through it).
pub(crate) fn note_draw(writes: bool) {
    HEAP_DRAWS[0].fetch_add(1, Relaxed);
    if writes {
        HEAP_DRAWS[1].fetch_add(1, Relaxed);
        if !HEAP_WRITTEN.swap(true, Relaxed) {
            REMAKE.store(true, Relaxed);
        }
    }
}

/// A shader wrote through a heap's views (ever): textures viewed for
/// unordered access there may hold what the GPU wrote, which no state move
/// shows (`crate::resource::evictable`).
static HEAP_WRITTEN: AtomicBool = AtomicBool::new(false);
/// The unordered views left null are to be made again before the next
/// submission ([`take_remake`]).
static REMAKE: AtomicBool = AtomicBool::new(false);

/// Whether any shader wrote through a heap yet.
pub(crate) fn heap_written() -> bool {
    HEAP_WRITTEN.load(Relaxed)
}

/// Whether the null unordered views are to be made again now (once).
pub(crate) fn take_remake() -> bool {
    REMAKE.swap(false, Relaxed)
}

/// Every view heap's registry.
static HEAPS: Mutex<Vec<std::sync::Weak<Viewed>>> = Mutex::new(Vec::new());

pub(crate) fn register(viewed: &Arc<Viewed>) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.retain(|h| h.strong_count() > 0);
    heaps.push(Arc::downgrade(viewed));
}

/// The title released `resource`: slots viewing it as a texture view
/// nothing to upload any more (its memory may be gone).
pub(crate) fn forget(resource: usize) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.textures.remove(&resource).is_some() {
            inner.slots.retain(|_, seen| !matches!(seen, Seen::Texture(r) if *r == resource));
        }
    }
}

/// Logs, per frame over `frames`, the heap-indexed draws, and what the
/// view heaps' slots view (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [draws, writes] = [0, 1].map(|i| HEAP_DRAWS[i].swap(0, Relaxed) as f64 / frames);
    let uploaded = UPLOADED.swap(0, Relaxed) as f64 / frames;
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut left = [0u32; KINDS.len()];
        let (mut buffers, mut textures) = (0u32, 0u32);
        // Placed textures left null: the largest at each address.
        let mut placed = HashMap::new();
        for seen in inner.slots.values() {
            match seen {
                Seen::Buffer(..) => buffers += 1,
                Seen::Texture(_) => textures += 1,
                Seen::Left((k, address, bytes)) => {
                    left[*k] += 1;
                    if *k == 2 {
                        let b = placed.entry(*address).or_insert(0u64);
                        *b = (*b).max(*bytes);
                    }
                }
            }
        }
        let placed_mb = placed.values().sum::<u64>() as f64 / 1e6;
        let bytes: u64 = inner.ranges.keys().map(|&(_, a, b, _)| b - a).sum();
        drop(inner);
        let hulls = viewed.ranges();
        let uav_hulls: u64 = hulls.iter().filter(|r| r.3).map(|r| r.2).sum();
        let all_hulls: u64 = hulls.iter().map(|r| r.2).sum();
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let copies = {
            let mut c: Vec<usize> = inner.ranges.keys().map(|k| k.0).collect();
            // The ranges come in order only from the B-tree form.
            c.sort_unstable();
            c.dedup();
            c.len()
        };
        let left: Vec<String> = KINDS.iter().zip(left).map(|(k, n)| format!("{k} {n}")).collect();
        let all = crate::shadow::all();
        let all_bytes: u64 = all.iter().map(|s| s.range().1).sum();
        tracing::info!(
            "heap views: {draws:.0} draws/frame index the heap ({writes:.0} writing through it), \
             {uploaded:.0} textures uploaded/frame; slots: {buffers} buffer views ({} ranges, \
             {:.1} MB, {copies} copies), {textures} title-written textures ({} distinct); \
             left null: {} (placed textures at {} addresses, {placed_mb:.0} MB); \
             host copies of title memory: {} ({:.1} MB); hulls {:.1} MB, unordered {:.1} MB",
            inner.ranges.len(),
            bytes as f64 / 1e6,
            inner.textures.len(),
            left.join(", "),
            placed.len(),
            all.len(),
            all_bytes as f64 / 1e6,
            all_hulls as f64 / 1e6,
            uav_hulls as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// Reads slots 0 (raw buffer) and 2 (texture) of the heap and writes
    /// slot 1 (raw unordered buffer): through a table over the heap's
    /// start, the same slots a shader indexing the heap reads.
    const SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x % 4, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, src.Load(id.x * 4) + texel);
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        use crate::compute::{Param, srv, uav};
        let params = &[Param::Table(&[srv(0, 1), uav(0, 1), srv(1, 1)])];
        crate::compute::pipeline(SOURCE, s!("heap"), 0, params).unwrap()
    }

    /// A title resource from a description in the console's layout (56
    /// bytes: dimension, alignment, width, height, depth and mips, format,
    /// samples, layout, flags).
    fn resource(dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// A title resource at `address` (title memory of its own), as the
    /// title places buffers and textures in its heaps.
    fn resource_at(address: usize, dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address as u64,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    const LINEAR: u32 = 0x100;

    /// A copy's hull is its live ranges' first start to last end, kept as
    /// slots come and go; a copy no slot views any more has none.
    #[test]
    fn hulls_follow_the_ranges() {
        let memory = vec![0u8; 0x1000];
        let at = memory.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 0x1000);
        let Some(copy) = crate::shadow::find(at, 0x1000) else {
            crate::shadow::unregister(at as usize, at);
            return; // no host GPU here
        };
        let viewed = Viewed::default();
        let set = |slot: u32, range: Option<(u64, u64)>| {
            let mut inner = viewed.inner.lock().unwrap();
            if let Some(old) = inner.slots.remove(&slot) {
                inner.count(&old, false);
            }
            if let Some((from, len)) = range {
                let seen = Seen::Buffer(copy.clone(), at + from, len, false);
                inner.count(&seen, true);
                inner.slots.insert(slot, seen);
            }
        };
        let hull = || viewed.ranges().iter().map(|h| (h.1 - at, h.2)).collect::<Vec<_>>();
        set(0, Some((0x100, 0x100)));
        set(1, Some((0x800, 0x200)));
        set(2, Some((0x100, 0x100)));
        assert_eq!(hull(), [(0x100, 0x900)]);
        set(1, None);
        assert_eq!(hull(), [(0x100, 0x100)], "the far range gone");
        set(0, None);
        assert_eq!(hull(), [(0x100, 0x100)], "one slot still views it");
        set(2, None);
        assert!(hull().is_empty());
        crate::shadow::unregister(at as usize, at);
    }

    /// Three host copies of test memory, registered (none: no host GPU
    /// here), and what unregisters them.
    fn three_copies(memory: &[u8]) -> (Option<Vec<Arc<Shadow>>>, impl Fn() + use<>) {
        let at = memory.as_ptr() as u64;
        for k in 0..3u64 {
            crate::shadow::register((at + 0x1000 * k) as usize, at + 0x1000 * k, 0x1000);
        }
        let release = move || {
            for k in 0..3u64 {
                crate::shadow::unregister((at + 0x1000 * k) as usize, at + 0x1000 * k);
            }
        };
        let copies: Option<Vec<_>> = (0..3u64).map(|k| crate::shadow::find(at + 0x1000 * k, 0x1000)).collect();
        (copies, release)
    }

    /// A census's answers as plain numbers, sorted: the ranges unordered
    /// views write, the hulls, each of `slots` slots' range.
    type Plain = (Vec<(usize, usize, usize)>, Vec<(usize, u64, u64, bool)>, Vec<Option<Range>>);

    fn plain(writes: &[(Arc<Shadow>, usize, usize)], viewed: &Viewed, slots: u32) -> Plain {
        let mut w: Vec<_> = writes.iter().map(|(s, a, l)| (Arc::as_ptr(s) as usize, *a, *l)).collect();
        w.sort_unstable();
        let mut h: Vec<_> = viewed.ranges().iter().map(|h| (Arc::as_ptr(&h.0) as usize, h.1, h.2, h.3)).collect();
        h.sort_unstable();
        (w, h, (0..slots).map(|s| viewed.range_at(s)).collect())
    }

    /// mac: a census with `VICTORY_CENSUS_FAST`'s parts answers as one
    /// without, after slots rewritten at random with views of three copies
    /// (shader and unordered views, the same range again, empty ranges,
    /// none), half of them entered at once and half in a batch entered
    /// later (as a heap sync pass's): the ranges unordered views write (kept
    /// or made anew), the hulls, each slot's range.
    #[test]
    fn the_fast_census_answers_as_the_old_one() {
        let memory = vec![0u8; 0x3000];
        let at = memory.as_ptr() as u64;
        let (Some(copies), release) = three_copies(&memory) else {
            return; // no host GPU here
        };
        let (old, fast) = (Viewed::new(false), Viewed::new(true));
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut kept = 0;
        let mut batch = fast.batch();
        for step in 0..20_000u32 {
            let slot = (next() % 200) as u32;
            let seen = (step % 7 != 0).then(|| {
                let k = next() % 3;
                let from = at + 0x1000 * k + (next() % 64) * 16;
                let len = if next() % 40 == 0 { 0 } else { 16 + next() % 256 };
                Seen::Buffer(copies[k as usize].clone(), from, len, next() % 3 == 0)
            });
            old.set_seen(slot, seen.clone());
            if next() % 2 == 0 {
                batch.set_seen(slot, seen);
            } else {
                fast.set_seen(slot, seen);
            }
            if next() % 8 == 0 {
                batch.enter_if_full();
                batch.enter();
            }
            if step % 3 == 0 {
                batch.enter();
                kept += usize::from(fast.fast.as_ref().expect("the fast parts").kept().is_some());
                let answer = unordered_writes(&fast).expect("the fast parts");
                let slots = if step % 101 == 0 { 200 } else { 0 };
                assert_eq!(plain(&answer, &fast, slots), plain(&widened_writes(&old), &old, slots), "step {step}");
            }
        }
        drop(batch);
        assert!(unordered_writes(&old).is_none(), "nothing without the switch");
        assert!(!crate::under_wine() || kept > 100, "kept answers are tested: {kept}");
        drop((old, fast));
        release();
    }

    /// mac: a pass's batch entered after a later making of one of its slots
    /// (another thread's, entered at once) leaves that slot as the later
    /// making left it; an update of a batch made after one entered at once
    /// wins, and a batch is entered when it goes. Without the switch a
    /// batch enters each update at once.
    #[test]
    fn a_batch_entered_late_leaves_a_later_making_in() {
        let memory = vec![0u8; 0x3000];
        let at = memory.as_ptr() as u64;
        let (Some(copies), release) = three_copies(&memory) else {
            return; // no host GPU here
        };
        let view = |k: usize, uav: bool| Some(Seen::Buffer(copies[k].clone(), at + 0x1000 * k as u64, 0x100, uav));
        let range = |k: usize, uav: bool| Some((Arc::as_ptr(&copies[k]) as usize, at + 0x1000 * k as u64, 0x100, uav));
        let (old, fast) = (Viewed::new(false), Viewed::new(true));
        for viewed in [&old, &fast] {
            viewed.set_seen(0, view(0, true));
        }
        // A pass makes slots 0 and 1, then another thread makes slot 0 again
        // before the pass's batch is entered.
        let mut batch = fast.batch();
        batch.set_seen(0, view(1, false));
        batch.set_seen(1, view(1, true));
        assert_eq!(fast.range_at(0), range(0, true), "not entered yet");
        fast.set_seen(0, view(2, true));
        batch.enter();
        drop(batch);
        for (slot, seen) in [(0, view(1, false)), (1, view(1, true)), (0, view(2, true))] {
            old.set_seen(slot, seen);
        }
        assert_eq!(fast.range_at(0), range(2, true), "the later making stays");
        assert_eq!(fast.range_at(1), range(1, true), "the pass's other slot is in");
        let answer = |v: &Viewed| match unordered_writes(v) {
            Some(writes) => plain(&writes, v, 2),
            None => plain(&widened_writes(v), v, 2),
        };
        assert_eq!(answer(&fast), answer(&old), "the counts follow the entries");
        // A batch's update made after an update entered at once wins, when
        // the batch goes.
        fast.set_seen(1, view(0, false));
        let mut batch = fast.batch();
        batch.set_seen(1, None);
        batch.set_seen(0, None);
        assert_eq!(fast.range_at(1), range(0, false));
        drop(batch);
        assert_eq!((fast.range_at(0), fast.range_at(1)), (None, None));
        assert!(answer(&fast).0.is_empty() && answer(&fast).1.is_empty());
        // Without the switch: at once.
        let mut batch = old.batch();
        batch.set_seen(0, None);
        assert_eq!(old.range_at(0), None, "entered at once");
        drop(batch);
        drop((old, fast));
        release();
    }

    /// mac (`VICTORY_CENSUS_BATCH`): what a slot's census entry is: a buffer
    /// range, a texture, a view left without a host view, or none.
    fn entry(viewed: &Viewed, index: u32) -> Option<(u8, Option<Range>)> {
        let kind = match viewed.inner.lock().unwrap().slots.get(&index)? {
            Seen::Buffer(..) => 0,
            Seen::Texture(_) => 1,
            Seen::Left(_) => 2,
        };
        Some((kind, viewed.range_at(index)))
    }

    /// mac: a pass (`VICTORY_CENSUS_BATCH=1`) holds the updates its thread
    /// makes for its census until a batch is full or it goes, and a reader
    /// after it finds them all: texture views, other heaps' updates and
    /// other threads' go in at once; a nested pass enters nothing when it
    /// goes; an update held past another thread's later making of the slot
    /// leaves that making in. Without the switch a pass holds nothing.
    #[test]
    fn a_pass_holds_its_slots_until_it_goes() {
        let memory = vec![0u8; 0x3000];
        let at = memory.as_ptr() as u64;
        let (Some(copies), release) = three_copies(&memory) else {
            return; // no host GPU here
        };
        let view = |k: usize, uav: bool| Some(Seen::Buffer(copies[k].clone(), at + 0x1000 * k as u64, 0x100, uav));
        let range = |k: usize, uav: bool| Some((0u8, Some((Arc::as_ptr(&copies[k]) as usize, at + 0x1000 * k as u64, 0x100, uav))));
        let (viewed, other) = (Viewed::with_batch(true, true), Viewed::with_batch(true, true));
        viewed.set_seen(3, view(0, false));
        let _ = crate::census_fast::take_held(1.0);
        {
            let pass = viewed.pass();
            viewed.set_seen(0, view(1, true));
            viewed.set_seen(3, None);
            viewed.set_seen(4, Some(Seen::Left((1, 0, 0))));
            assert_eq!((entry(&viewed, 0), entry(&viewed, 3), entry(&viewed, 4)), (None, range(0, false), None), "held");
            viewed.set_seen(5, Some(Seen::Texture(0x1234)));
            assert_eq!(entry(&viewed, 5), Some((1, None)), "a texture view goes in at once");
            other.set_seen(0, view(2, false));
            assert_eq!(entry(&other, 0), range(2, false), "another heap's update goes in at once");
            // Another thread (in no pass) makes slot 6 after this one did.
            viewed.set_seen(6, view(0, true));
            std::thread::scope(|s| {
                s.spawn(|| viewed.set_seen(6, view(2, true)));
            });
            assert_eq!(entry(&viewed, 6), range(2, true), "the other thread's at once");
            {
                let inner = viewed.pass();
                viewed.set_seen(1, view(2, false));
                drop(inner);
            }
            assert_eq!(entry(&viewed, 1), None, "a nested pass enters nothing when it goes");
            pass.enter_if_full();
            assert_eq!(entry(&viewed, 0), None, "less than a batch stays held");
            for slot in 100..100 + crate::census_fast::BATCH as u32 {
                viewed.set_seen(slot, view(0, false));
            }
            pass.enter_if_full();
            assert_eq!(entry(&viewed, 0), range(1, true), "a full batch went in");
            assert_eq!(entry(&viewed, 100), range(0, false));
            viewed.set_seen(2, view(1, false));
            assert_eq!(entry(&viewed, 2), None);
        }
        // The pass went: everything is in, and slot 6 keeps the later making.
        let want = [(0, range(1, true)), (1, range(2, false)), (2, range(1, false)), (3, None), (4, Some((2, None))), (6, range(2, true))];
        for (slot, entry_want) in want {
            assert_eq!(entry(&viewed, slot), entry_want, "slot {slot}");
        }
        let [held, takes, saved, once] = crate::census_fast::take_held(1.0);
        let n = 6.0 + crate::census_fast::BATCH as f64;
        assert!(held >= n && takes >= 2.0 && saved >= n - 2.0 && once >= 1.0, "{held} {takes} {saved} {once}");
        // The counts follow the entries, as in a census without passes.
        let old = Viewed::new(false);
        for (slot, seen) in [(3, view(0, false)), (0, view(1, true)), (3, None), (6, view(2, true)), (1, view(2, false)), (2, view(1, false))] {
            old.set_seen(slot, seen);
        }
        for slot in 100..100 + crate::census_fast::BATCH as u32 {
            old.set_seen(slot, view(0, false));
        }
        let slots = 100 + crate::census_fast::BATCH as u32;
        let answer = |v: &Viewed| match unordered_writes(v) {
            Some(writes) => plain(&writes, v, slots),
            None => plain(&widened_writes(v), v, slots),
        };
        assert_eq!(answer(&viewed), answer(&old));
        // Without the switch: at once.
        let plain_census = Viewed::with_batch(true, false);
        let pass = plain_census.pass();
        plain_census.set_seen(0, view(0, true));
        assert_eq!(entry(&plain_census, 0), range(0, true), "no pass without the switch");
        drop(pass);
        drop((viewed, other, old, plain_census));
        release();
    }

    /// The title writes a buffer and a texture, and views of them and of an
    /// output buffer into its heap; a shader reaching them only through the
    /// heap (no table entry names them) reads what the title wrote at
    /// submission, every round (the title rewrites both each time, as Bink
    /// rewrites its planes every frame). The output buffer is not read back
    /// into title memory (`refresh`'s shortcut): it is read from its host
    /// copy.
    #[test]
    fn heap_views_are_current() {
        let (src, dst) = (resource(1, 256, 1, 0, 1), resource(1, 256, 1, 0, 1));
        let tex = resource(3, 4, 1, 28, LINEAR);
        let write = |base: u32, texel: u8| {
            let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
            let (tex_bytes, tex_desc) = crate::resource::texture_memory(tex).unwrap();
            let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
            // SAFETY: the resources' own memory, 256 bytes and the texture's.
            unsafe {
                let words = std::slice::from_raw_parts_mut(src_bytes.cast::<u32>(), 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes(tex_bytes.cast_mut(), texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    /// The same with the buffer and the texture in one view of physical
    /// pages and the title writing them through another (the title fills its
    /// heaps through a pool view): every round must reach the shader.
    #[test]
    fn heap_views_follow_writes_through_another_mapping() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let src = resource_at(a, 1, 256, 1, 0, 1);
        let tex = resource_at(a + PAGE_BYTES, 3, 4, 1, 28, LINEAR);
        let dst = resource(1, 256, 1, 0, 1);
        let (_, tex_desc) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
        let write = |base: u32, texel: u8| {
            // SAFETY: the second view of the buffer's and the texture's pages.
            unsafe {
                let words = std::slice::from_raw_parts_mut(b as *mut u32, 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes((b + PAGE_BYTES) as *mut u8, texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    const PAGE_BYTES: usize = crate::watch::PAGE;

    /// Three rounds of: record a dispatch reading slots 0 and 2 and writing
    /// slot 1, let the title `write` (after recording, before submitting),
    /// submit, check the output's host copy.
    fn rounds(src: This, dst: This, tex: This, write: impl Fn(u32, u8)) {
        let _serial = crate::heap_worker::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        // A shader-visible view heap of 8 slots, in memory of its own.
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        // Raw views of 64 dwords: format R32_TYPELESS, buffer dimension,
        // the default mapping, then first element, count, stride, flags.
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_srv(null, tex, std::ptr::null(), slot(2));
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (round, (base, texel)) in [(1000u32, 0x40u8), (5000, 0x10), (9000, 0x22)].into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.defer_heap(&heap_host.viewed);
            list.close();
            // The title writes after recording, before submitting.
            write(base, texel);
            host.execute(&[&list]);
            let words = out.read_host(dst_bytes as u64, 64);
            for (i, &v) in words.iter().enumerate() {
                assert_eq!(v, base + i as u32 + u32::from(texel), "round {round}, dword {i}");
            }
        }
    }

    /// With `VICTORY_HEAP_WORKER=1`:
    /// - a placed bindless texture `A` stays null until the worker stages it
    ///   and `Host::execute` publishes it atomically ahead of the dispatch;
    /// - a bindless texture `B` whose `host_written` flips before submission
    ///   has its staged CPU upload skipped so host-written contents are kept;
    /// - a table-bound texture `C` (`Step::Texture`) uploads synchronously in
    ///   the very first submission without waiting for the worker.
    #[test]
    fn worker_upload_publishes_atomically_and_skips_host_written() {
        // mac: the rule these asserts are about, whatever the suite's
        // environment (`VICTORY_PLACED_AT_ONCE`).
        let _rule = crate::mac_switches::tests::PlacedHere::new(crate::mac_switches::PlacedAtOnce::Off);
        let Some(host) = crate::host::host() else {
            return;
        };
        let _worker = crate::heap_worker::EnableForTest::new();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 16, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        let heap_host = title_heap.host().expect("host view heap");

        let pool = crate::watch::tests::pool(2);
        let addr_a = pool as u64;
        let addr_b = (pool + PAGE_BYTES) as u64;
        // SAFETY: two dedicated pool pages.
        unsafe {
            std::ptr::write_bytes(addr_a as *mut u8, 0x33, PAGE_BYTES);
            std::ptr::write_bytes(addr_b as *mut u8, 0x77, PAGE_BYTES);
        }
        let placed_res = [3u32, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, LINEAR, 0, 0];
        let mut desc_a = [0u32; 16];
        (desc_a[0], desc_a[1], desc_a[2], desc_a[5]) = (28, 4, 0x1688, 1);
        (desc_a[10], desc_a[11]) = (addr_a as u32, (addr_a >> 32) as u32);
        let mut desc_b = desc_a;
        (desc_b[10], desc_b[11]) = (addr_b as u32, (addr_b >> 32) as u32);

        let src = resource(1, 256, 1, 0, 1);
        let dst = resource(1, 256, 1, 0, 1);
        let tex_c = resource(3, 4, 1, 28, LINEAR);
        let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let (c_bytes, c_desc) = crate::resource::texture_memory(tex_c).unwrap();
        let c_size = crate::resource::texture_layout(&c_desc).unwrap().planes[0].layout.size as usize;
        // SAFETY: owned buffers/texture memory.
        unsafe {
            std::ptr::write_bytes(src_bytes, 0, 256);
            std::ptr::write_bytes(c_bytes.cast_mut(), 0x55, c_size);
        }

        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_placed_srv(null, &placed_res, &desc_a, slot(2));
        crate::descriptor::create_placed_srv(null, &placed_res, &desc_b, slot(3));
        assert!(!heap_host.kept(2) && !heap_host.kept(3), "placed views stay null before worker");

        // Stage A and B on the worker pass, then flip B to `host_written` before submission.
        _worker.step_heap(title_heap);
        assert!(crate::heap_worker::has_ready());
        assert!(!heap_host.kept(2), "not published until Host::execute");
        let res_b = crate::resource::texture_at(addr_b, &placed_res).unwrap();
        let copy_b = crate::resource::made_copy(res_b).unwrap();
        let uploads_b_before = crate::texture::uploads(res_b);
        copy_b.host_written.store(true, Relaxed);

        // Also attach table-bound texture C synchronously on this list.
        let copy_c = crate::resource::host_copy(tex_c).unwrap();
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        crate::texture::upload(&list, tex_c, &copy_c.texture);
        // SAFETY: open list, live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(4, 1, 1);
        }
        list.defer_heap(&heap_host.viewed);
        list.close();
        host.execute(&[&list]);

        // A was published and read 0x33 in that exact submission.
        assert!(heap_host.kept(2), "slot 2 published at submission");
        let out = crate::shadow::find(dst_bytes as u64, 256).unwrap();
        let words = out.read_host(dst_bytes as u64, 4);
        assert_eq!(words, [0x33, 0x33, 0x33, 0x33], "shader saw A's uploaded texels");
        // B's CPU upload was skipped because `host_written` flipped before submission.
        assert_eq!(
            crate::texture::uploads(res_b),
            uploads_b_before,
            "host_written texture upload was skipped"
        );
        // Table-bound texture C uploaded synchronously in the first submission.
        assert_eq!(
            crate::texture::uploads(tex_c),
            1,
            "table-bound texture uploaded synchronously"
        );

        crate::object::release_ref(heap);
        crate::object::release_ref(src);
        crate::object::release_ref(dst);
        crate::object::release_ref(tex_c);
        crate::watch::tests::free(pool);
    }
}
