//! mac: the plain arithmetic of the flights (`crate::host::unwaited::flights`,
//! `VICTORY_FLIGHTS=1`) and of their frame lines: plain std, so its tests
//! run everywhere.

/// Title memory as ranges `[from, to)`: after [`Spans::join`], in order,
/// none touching the next.
#[derive(Debug, Default, Clone)]
pub(crate) struct Spans(Vec<(usize, usize)>);

impl Spans {
    /// Adds `[from, to)` (nothing if empty); [`Spans::join`] after the last.
    pub(crate) fn add(&mut self, from: usize, to: usize) {
        if from < to {
            self.0.push((from, to));
        }
    }

    /// Puts the ranges in order, those that overlap or touch as one.
    pub(crate) fn join(&mut self) {
        self.0.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(self.0.len());
        for &(from, to) in &self.0 {
            match out.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => out.push((from, to)),
            }
        }
        self.0 = out;
    }

    /// Whether `[from, from + len)` shares a byte with one of them (joined).
    pub(crate) fn touches(&self, from: usize, len: usize) -> bool {
        let to = from.saturating_add(len);
        if from >= to {
            return false;
        }
        let at = self.0.partition_point(|r| r.1 <= from);
        self.0.get(at).is_some_and(|r| r.0 < to)
    }

    /// Whether they hold nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The ranges, in order (joined).
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0.iter().copied()
    }

    #[cfg(test)]
    fn ranges(&self) -> &[(usize, usize)] {
        &self.0
    }
}

/// What the landing of the submission left (its finish) may change in title
/// memory, as the next submission's preparation beside it sees it.
#[derive(Debug, Default)]
pub(crate) struct Landing {
    /// It runs programs that use title memory: all of it.
    pub(crate) every: bool,
    /// Title memory it may write or lands in (joined).
    pub(crate) spans: Spans,
    /// The pages that show a page of `spans` at another address (joined): a
    /// write or a landing there marks the trackers here too.
    pub(crate) others: Spans,
}

impl Landing {
    /// Whether `[at, at + len)` is in what it may write or lands in.
    pub(crate) fn reaches(&self, at: usize, len: usize) -> bool {
        self.every || self.spans.touches(at, len)
    }

    /// Whether a tracker over `[at, at + len)` may be told at its landing
    /// that the memory changed: what it reaches, and its pages' other
    /// addresses.
    pub(crate) fn tells(&self, at: usize, len: usize) -> bool {
        self.reaches(at, len) || self.others.touches(at, len)
    }
}

/// mac: whether a submission's ask to bring `[at, at + len)` of a host copy
/// up to date goes into its refreshes. A copy the title wrote since the copy
/// took it does (`clean` false). A clean one does only while the submission
/// before still runs beside it (`beside`) and its landing may tell the copy
/// ([`Landing::tells`]): then the ask is held and made once that one is
/// finished, when its copy-back has told the copy what changed. Before, a
/// clean copy asked for before that finish was left out, and the next
/// submission read it one submission old.
pub(crate) fn asked(clean: bool, beside: Option<&Landing>, at: usize, len: usize) -> bool {
    !clean || beside.is_some_and(|l| l.tells(at, len))
}

/// What the `bytes` (1 to 8) at `address` will hold once `writes` are made
/// in their order (each: where, the value it stores if that is known now,
/// how many of its low bytes it stores, 4 or 8, little-endian):
///
/// - `None`: no write reaches those bytes, memory decides;
/// - `Some(Some(value))`: the last write that reaches them stores all of
///   them, and its value is known;
/// - `Some(None)`: a write reaches them that cannot be told now (a time
///   stamp, or one that stores only a part of them and leaves the rest to
///   memory or to an earlier write).
pub(crate) fn after_writes(writes: &[(usize, Option<u64>, usize)], address: usize, bytes: usize) -> Option<Option<u64>> {
    let end = address.checked_add(bytes)?;
    if bytes == 0 || bytes > 8 {
        return None;
    }
    let mut seen = None;
    for &(at, value, n) in writes {
        let Some(stop) = at.checked_add(n) else { continue };
        if stop <= address || end <= at {
            continue;
        }
        seen = Some(match value {
            Some(v) if at <= address && end <= stop => {
                // The low `n` bytes of `v`, then the ones asked for.
                let stored = if n >= 8 { v } else { v & ((1u64 << (8 * n)) - 1) };
                let shifted = stored >> (8 * (address - at));
                Some(if bytes >= 8 { shifted } else { shifted & ((1u64 << (8 * bytes)) - 1) })
            }
            _ => None,
        });
    }
    seen
}

/// mac: the names `VICTORY_FLIGHTS_WITH=<list>` takes, each building on the
/// ones before it (`notes/overlap-stages-design.md` §6), but `uploads`
/// (E2b, a refinement of `alias`), which needs `alias` only ([`NEEDS`]).
pub(crate) const WITH: [&str; 7] = ["alias", "writes", "copies", "ring", "relay", "present", "uploads"];

/// The names each of [`WITH`] needs taken first (bits).
const NEEDS: [u32; 7] = [0, 0b1, 0b11, 0b111, 0b1111, 0b1_1111, 0b1];

/// mac: `VICTORY_FLIGHTS_WITH=alias,uploads` (E2b of
/// `notes/overlap-stages-design.md`): what the texture uploads of the
/// submission prepared beside the one left may not read before that one is
/// finished, in whole pages (the grain of the write marks, of the trackers
/// and of the links between addresses). In place of E2's rule (whole copies
/// its heaps view for unordered access, and any page shown at another
/// address whenever it writes anything): the ranges it declared its shaders
/// write and its landing lands in (read-backs, texture copy-backs) with its
/// bottom-of-pipe writes; the ranges its heaps' unordered views reach, not
/// the whole copies they are in; and the other addresses of those pages
/// only. With title-memory programs: everything.
#[derive(Debug, Default)]
pub(crate) struct UploadReach {
    /// It runs programs that use title memory (they write at addresses
    /// they compute): every texture.
    pub(crate) every: bool,
    /// What its lists declared and what it lands in, its bottom-of-pipe
    /// writes (pages, joined).
    pub(crate) declared: Spans,
    /// What its heaps view for unordered access (pages, joined).
    pub(crate) heaps: Spans,
    /// The pages that show a page of those two at another address (joined).
    pub(crate) others: Spans,
}

/// The reasons [`UploadReach::meets`] gives, as indices.
pub(crate) const MET_EVERY: usize = 0;
pub(crate) const MET_DECLARED: usize = 1;
pub(crate) const MET_HEAPS: usize = 2;
pub(crate) const MET_OTHER: usize = 3;

impl UploadReach {
    /// Whether a texture over title memory `[at, at + len)` (what its
    /// refresh may read) is to be read only once the one left is finished,
    /// and why ([`MET_EVERY`] to [`MET_OTHER`]); none for an empty range.
    pub(crate) fn meets(&self, at: usize, len: usize) -> Option<usize> {
        if len == 0 {
            None
        } else if self.every {
            Some(MET_EVERY)
        } else if self.declared.touches(at, len) {
            Some(MET_DECLARED)
        } else if self.heaps.touches(at, len) {
            Some(MET_HEAPS)
        } else if self.others.touches(at, len) {
            Some(MET_OTHER)
        } else {
            None
        }
    }

    /// Whether it may write nothing at all (no texture meets it).
    pub(crate) fn nothing(&self) -> bool {
        !self.every && self.declared.is_empty() && self.heaps.is_empty()
    }
}

impl Spans {
    /// [`Spans::add`] of `[from, to)` widened to whole pages of `page`
    /// bytes (a power of two).
    pub(crate) fn add_pages(&mut self, from: usize, to: usize, page: usize) {
        if from < to {
            self.add(from & !(page - 1), to.checked_next_multiple_of(page).unwrap_or(usize::MAX));
        }
    }
}

/// mac: the union of ranges `[start, end)` given as the counts of their
/// starts and of their ends (`crate::bindless`'s per-copy census), each run
/// of it to `each` in order; ranges that touch are one run.
pub(crate) fn union_of(
    starts: &std::collections::BTreeMap<u64, u32>,
    ends: &std::collections::BTreeMap<u64, u32>,
    mut each: impl FnMut(u64, u64),
) {
    let (mut s, mut e) = (starts.iter().peekable(), ends.iter().peekable());
    let (mut depth, mut open) = (0i64, 0u64);
    loop {
        // At one place, starts before ends: touching ranges stay one run.
        let start_first = match (s.peek(), e.peek()) {
            (Some(&(&a, _)), Some(&(&b, _))) => a <= b,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if start_first {
            let (&at, &n) = s.next().expect("peeked");
            if depth == 0 {
                open = at;
            }
            depth += i64::from(n);
        } else {
            let (&at, &n) = e.next().expect("peeked");
            depth -= i64::from(n);
            if depth <= 0 {
                if open < at {
                    each(open, at);
                }
                depth = 0;
            }
        }
    }
}

/// `VICTORY_FLIGHTS_WITH`'s rule: of the comma list `value`, the names taken
/// (bit `i` for [`WITH`]`[i]`), and a line for each one refused: unknown, not
/// in this build (`built`, the bits it has), asked without every name it
/// needs ([`NEEDS`]: the ones before it), or refused by `refuse(i)` for a
/// reason of its own.
pub(crate) fn with_names(value: &str, built: u32, refuse: impl Fn(usize) -> Option<&'static str>) -> (u32, Vec<String>) {
    let mut asked = 0u32;
    let mut refused = Vec::new();
    for name in value.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        match WITH.iter().position(|w| w.eq_ignore_ascii_case(name)) {
            Some(i) => asked |= 1 << i,
            None => refused.push(format!("{name}: not a name it takes ({})", WITH.join(","))),
        }
    }
    let mut taken = 0u32;
    for (i, name) in WITH.iter().enumerate().filter(|&(i, _)| asked & 1 << i != 0) {
        let why = if built & 1 << i == 0 {
            Some("not in this build".to_owned())
        } else if taken & NEEDS[i] != NEEDS[i] {
            let needed: Vec<&str> = (0..WITH.len()).filter(|&k| NEEDS[i] & 1 << k != 0).map(|k| WITH[k]).collect();
            Some(format!("it needs {} first", needed.join(",")))
        } else {
            refuse(i).map(str::to_owned)
        };
        match why {
            Some(why) => refused.push(format!("{name}: {why}")),
            None => taken |= 1 << i,
        }
    }
    (taken, refused)
}

/// mac: E1 of `notes/overlap-stages-design.md` (lines only): per thread that
/// calls us, the microseconds inside our queue calls (the outermost one),
/// waiting for the held-work lock and waiting for the submission lock, since
/// the last frame line. This counting runs on every queue call whatever the
/// switches (only the line needs `VICTORY_ASYNC_SUBMIT=1`;
/// `crate::host::unwaited::lines` has its cost).
///
/// A thread takes one of [`threads::SLOTS`] rows at its first count (under
/// a lock, once) and gives it back when it ends; what it counted since the
/// last line is then kept under its id for that line, and the row starts
/// again at zero for the next thread. A count itself is a thread-local read
/// and an atomic add, as before. The threads past the rows at once count
/// together in one more row, with how many shared it (a thread there stays
/// there), as do threads that end past [`threads::ENDED`] since the last
/// line (so that nothing grows while no line is made).
pub(crate) mod threads {
    use std::cell::Cell;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    /// Threads with a row of their own at once.
    pub(crate) const SLOTS: usize = 16;

    /// Threads that ended since the last line kept under their own ids.
    pub(crate) const ENDED: usize = 64;

    /// The row the threads past [`SLOTS`] share.
    const SHARED: usize = SLOTS;

    /// No row yet (or given back).
    const NONE: usize = usize::MAX;

    /// What a thread's time went on.
    #[derive(Clone, Copy)]
    pub(crate) enum Spent {
        Inside = 0,
        Held = 1,
        Lock = 2,
    }

    /// Per row, microseconds by [`Spent`]: [`SLOTS`] rows of one thread
    /// each, then [`SHARED`].
    static US: [[AtomicU64; 3]; SLOTS + 1] = [const { [const { AtomicU64::new(0) }; 3] }; SLOTS + 1];

    /// Who holds the rows: taken when a thread takes or gives back its row,
    /// and by [`take`]; never on a count.
    static ROWS: Mutex<Rows> =
        Mutex::new(Rows { ids: [None; SLOTS], ended: Vec::new(), shared_now: 0, shared_ended: 0 });

    struct Rows {
        /// Per row, the id of the thread that holds it.
        ids: [Option<u64>; SLOTS],
        /// Threads that ended since the last take: id, microseconds.
        ended: Vec<(u64, [u64; 3])>,
        /// Threads counting in [`SHARED`] now,
        shared_now: usize,
        /// and those that counted there and ended since the last take.
        shared_ended: usize,
    }

    fn rows() -> std::sync::MutexGuard<'static, Rows> {
        ROWS.lock().unwrap_or_else(|e| e.into_inner())
    }

    thread_local! {
        /// The thread's row, read on every count (nothing to drop, so the
        /// read costs what it did).
        static SLOT: Cell<usize> = const { Cell::new(NONE) };
        /// Gives the row back when the thread ends (touched once, when the
        /// thread takes its row).
        static GIVE_BACK: GiveBack = const { GiveBack(Cell::new(NONE)) };
        static DEPTH: Cell<u32> = const { Cell::new(0) };
    }

    /// Gives the thread's row (`.0`) back when the thread ends.
    struct GiveBack(Cell<usize>);

    impl Drop for GiveBack {
        fn drop(&mut self) {
            // A count after this (from another local's drop) takes a row
            // again, and as it can no longer give one back, shares one.
            let _ = SLOT.try_with(|s| s.set(NONE));
            let row = self.0.get();
            let mut rows = rows();
            if row == SHARED {
                rows.shared_now = rows.shared_now.saturating_sub(1);
                rows.shared_ended += 1;
            } else if let Some(id) = rows.ids.get_mut(row).and_then(Option::take) {
                // Its row starts again at zero; what it counted stays its own.
                let us = US[row].each_ref().map(|u| u.swap(0, Relaxed));
                if us != [0; 3] {
                    if rows.ended.len() < ENDED {
                        rows.ended.push((id, us));
                    } else {
                        for (k, u) in us.into_iter().enumerate() {
                            US[SHARED][k].fetch_add(u, Relaxed);
                        }
                        rows.shared_ended += 1;
                    }
                }
            }
        }
    }

    /// The row for the thread's first count: a free one, or the shared one.
    #[cold]
    fn claim() -> usize {
        let id = id();
        let mut rows = rows();
        // Its own row only if it can be given back (not from the drop of
        // another of the thread's locals once this one is gone).
        match rows.ids.iter().position(Option::is_none) {
            Some(i) if GIVE_BACK.try_with(|g| g.0.set(i)).is_ok() => {
                rows.ids[i] = Some(id);
                i
            }
            _ if GIVE_BACK.try_with(|g| g.0.set(SHARED)).is_ok() => {
                rows.shared_now += 1;
                SHARED
            }
            _ => {
                rows.shared_ended += 1;
                SHARED
            }
        }
    }

    /// The thread's id as the table shows it (the Windows thread id).
    pub(crate) fn id() -> u64 {
        #[cfg(windows)]
        // SAFETY: a plain query.
        return u64::from(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() });
        #[cfg(not(windows))]
        {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            thread_local!(static ID: u64 = NEXT.fetch_add(1, Relaxed));
            ID.with(|i| *i)
        }
    }

    fn slot() -> usize {
        SLOT.with(|s| {
            if s.get() == NONE {
                s.set(claim());
            }
            s.get()
        })
    }

    pub(crate) fn add(spent: Spent, us: u64) {
        US[slot()][spent as usize].fetch_add(us, Relaxed);
    }

    /// Times the thread's outermost call of ours while it lives.
    pub(crate) struct Inside(Option<Instant>);

    impl Inside {
        pub(crate) fn enter() -> Inside {
            let outermost = DEPTH.with(|d| {
                d.set(d.get() + 1);
                d.get() == 1
            });
            Inside(outermost.then(Instant::now))
        }
    }

    impl Drop for Inside {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get() - 1));
            if let Some(at) = self.0 {
                add(Spent::Inside, at.elapsed().as_micros() as u64);
            }
        }
    }

    /// What [`take`] found.
    pub(crate) struct Taken {
        /// Per thread with anything, those that ended included: its id and
        /// the microseconds inside, held, lock.
        pub(crate) threads: Vec<(u64, [u64; 3])>,
        /// The threads past the rows together, if they counted anything:
        /// how many (there now, or there and ended since the last take) and
        /// their microseconds.
        pub(crate) shared: Option<(usize, [u64; 3])>,
    }

    /// Since the last take (then zero).
    pub(crate) fn take() -> Taken {
        let mut rows = rows();
        let mut threads = std::mem::take(&mut rows.ended);
        for (row, id) in rows.ids.iter().enumerate() {
            let Some(id) = *id else { continue };
            let us = US[row].each_ref().map(|u| u.swap(0, Relaxed));
            if us != [0; 3] {
                threads.push((id, us));
            }
        }
        let us = US[SHARED].each_ref().map(|u| u.swap(0, Relaxed));
        let many = rows.shared_now + std::mem::take(&mut rows.shared_ended);
        Taken { threads, shared: (us != [0; 3]).then_some((many.max(1), us)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clean copy asked for beside the submission left stays in the
    /// refreshes where that one's landing may tell it (what it writes or
    /// lands in, and those pages' other addresses), so that it is made after
    /// the finish; elsewhere, and with nothing beside, it is left out as
    /// before. A copy the title wrote is always asked for.
    #[test]
    fn a_clean_copy_the_landing_may_tell_is_kept_for_after_it() {
        let mut landing = Landing::default();
        landing.spans.add(0x1_0000, 0x2_0000);
        landing.others.add(0x9_0000, 0xa_0000);
        landing.spans.join();
        landing.others.join();
        // Clean: kept only where the landing may tell it.
        assert!(asked(true, Some(&landing), 0x1_8000, 0x100), "in what it lands in");
        assert!(asked(true, Some(&landing), 0x9_ff00, 0x1000), "a page shown there too");
        assert!(!asked(true, Some(&landing), 0x2_0000, 0x1000), "next to it");
        assert!(!asked(true, Some(&landing), 0x1_8000, 0), "nothing asked");
        assert!(!asked(true, None, 0x1_8000, 0x100), "nothing beside: as before");
        // Written by the title: always.
        assert!(asked(false, None, 0x5_0000, 0x10));
        assert!(asked(false, Some(&landing), 0x5_0000, 0x10));
        // Title-memory programs: all of it.
        let every = Landing { every: true, ..Landing::default() };
        assert!(asked(true, Some(&every), 0x5_0000, 0x10));
    }

    /// `VICTORY_FLIGHTS_WITH`: names in their order, each needing those
    /// before it; unknown, unbuilt and otherwise refused names named.
    #[test]
    fn flights_with_takes_each_name_after_the_ones_before_it() {
        let none = |_: usize| None;
        assert_eq!(with_names("", 1, none), (0, vec![]));
        assert_eq!(with_names(" alias , ", 1, none), (1, vec![]));
        assert_eq!(with_names("ALIAS", 1, none).0, 1);
        let (taken, refused) = with_names("alias,writes", 1, none);
        assert_eq!(taken, 1);
        assert_eq!(refused, ["writes: not in this build"]);
        let (taken, refused) = with_names("writes", 0b11, none);
        assert_eq!(taken, 0);
        assert_eq!(refused, ["writes: it needs alias first"]);
        assert_eq!(with_names("writes,alias", 0b11, none), (0b11, vec![]));
        let (taken, refused) = with_names("alias,copies", 0b111, none);
        assert_eq!(taken, 1);
        assert_eq!(refused, ["copies: it needs alias,writes first"]);
        let (taken, refused) = with_names("alias,bogus", 1, none);
        assert_eq!(taken, 1);
        assert!(refused[0].starts_with("bogus: not a name"), "{refused:?}");
        let (taken, refused) = with_names("alias,writes", 0b11, |i| (i == 0).then_some("the heap worker"));
        assert_eq!(taken, 0);
        assert_eq!(refused, ["alias: the heap worker", "writes: it needs alias first"]);
        // E2b's `uploads` needs `alias` only, wherever it is in the list.
        let built = 0b100_0011;
        assert_eq!(with_names("alias,uploads", built, none), (0b100_0001, vec![]));
        assert_eq!(with_names("uploads,writes,alias", built, none), (0b100_0011, vec![]));
        let (taken, refused) = with_names("uploads", built, none);
        assert_eq!(taken, 0);
        assert_eq!(refused, ["uploads: it needs alias first"]);
        let (taken, refused) = with_names("alias,uploads", 0b11, none);
        assert_eq!(taken, 1);
        assert_eq!(refused, ["uploads: not in this build"]);
    }

    /// E2b: an upload beside the one left waits for it only where it may
    /// write or land, in whole pages: what it declared or lands in, what its
    /// heaps' unordered views reach, those pages' other addresses; with
    /// title-memory programs, everything; elsewhere, nothing.
    #[test]
    fn an_upload_meets_the_one_left_only_where_it_may_write() {
        const PAGE: usize = 0x1_0000;
        let mut reach = UploadReach::default();
        assert!(reach.nothing());
        // A 0x100-byte declaration and bottom-of-pipe word: their pages.
        reach.declared.add_pages(0x10_0040, 0x10_0140, PAGE);
        reach.declared.add_pages(0x30_fff8, 0x31_0000, PAGE);
        // An unordered view over two pages and a byte.
        reach.heaps.add_pages(0x50_0000, 0x52_0001, PAGE);
        // A page of those shown at another address.
        reach.others.add_pages(0x90_0000, 0x91_0000, PAGE);
        for s in [&mut reach.declared, &mut reach.heaps, &mut reach.others] {
            s.join();
        }
        assert!(!reach.nothing());
        assert_eq!(reach.declared.ranges(), &[(0x10_0000, 0x11_0000), (0x30_0000, 0x31_0000)]);
        assert_eq!(reach.heaps.ranges(), &[(0x50_0000, 0x53_0000)]);
        // A texture sharing a page with a declaration (not its bytes) meets it.
        assert_eq!(reach.meets(0x10_8000, 0x1000), Some(MET_DECLARED));
        assert_eq!(reach.meets(0x30_0000, 0x10), Some(MET_DECLARED));
        assert_eq!(reach.meets(0x52_ff00, 0x8000), Some(MET_HEAPS));
        assert_eq!(reach.meets(0x8f_0000, 0x1_0001), Some(MET_OTHER));
        // Next to them, and empty: not.
        assert_eq!(reach.meets(0x11_0000, 0x1_0000), None);
        assert_eq!(reach.meets(0x53_0000, 0x3d_0000), None, "up to the other address's page");
        assert_eq!(reach.meets(0x10_8000, 0), None);
        // Title-memory programs: all of it.
        let every = UploadReach { every: true, ..UploadReach::default() };
        assert!(!every.nothing());
        assert_eq!(every.meets(0x7000_0000, 0x10), Some(MET_EVERY));
        assert_eq!(every.meets(0x7000_0000, 0), None);
        // Pages at the top of memory do not wrap.
        let mut top = Spans::default();
        top.add_pages(usize::MAX - 0x10, usize::MAX, PAGE);
        top.join();
        assert!(top.touches(usize::MAX - 1, 1));
    }

    /// The census keeps, per copy, the counts of its ranges' starts and
    /// ends: their union is found from those alone, touching and nested
    /// ranges one run, the same range twice once.
    #[test]
    fn the_union_of_ranges_comes_from_their_starts_and_ends() {
        let union = |ranges: &[(u64, u64)]| {
            let (mut starts, mut ends) = (std::collections::BTreeMap::new(), std::collections::BTreeMap::new());
            for &(a, b) in ranges {
                *starts.entry(a).or_insert(0u32) += 1;
                *ends.entry(b).or_insert(0u32) += 1;
            }
            let mut out = Vec::new();
            union_of(&starts, &ends, |a, b| out.push((a, b)));
            out
        };
        assert_eq!(union(&[]), []);
        assert_eq!(union(&[(10, 20)]), [(10, 20)]);
        assert_eq!(union(&[(10, 20), (20, 30)]), [(10, 30)], "touching");
        assert_eq!(union(&[(10, 40), (15, 20), (25, 30)]), [(10, 40)], "nested");
        assert_eq!(union(&[(10, 20), (10, 20), (30, 40)]), [(10, 20), (30, 40)], "twice, and apart");
        assert_eq!(union(&[(30, 40), (10, 35), (50, 60), (55, 70)]), [(10, 40), (50, 70)]);
        // Against a plain merge of many random ranges.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..50 {
            let ranges: Vec<(u64, u64)> = (0..40)
                .map(|_| {
                    let a = next() % 1000;
                    (a, a + 1 + next() % 60)
                })
                .collect();
            let mut plain = Spans::default();
            ranges.iter().for_each(|&(a, b)| plain.add(a as usize, b as usize));
            plain.join();
            let want: Vec<(u64, u64)> = plain.iter().map(|(a, b)| (a as u64, b as u64)).collect();
            assert_eq!(union(&ranges), want, "{ranges:?}");
        }
    }

    /// The tests of E1's table, one at a time (a take takes what all counted).
    static THREADS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn threads_alone() -> std::sync::MutexGuard<'static, ()> {
        THREADS.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// E1's table: each thread its own slot, only the outermost call timed,
    /// the waits added as they come, zero after a take.
    #[test]
    fn each_thread_counts_its_outermost_call_and_its_waits() {
        use threads::{Inside, Spent, add, id, take};
        let _alone = threads_alone();
        let ids: Vec<u64> = (0..2u64)
            .map(|k| {
                std::thread::spawn(move || {
                    let outer = Inside::enter();
                    {
                        let _inner = Inside::enter();
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                    drop(outer);
                    add(Spent::Held, 1000 * (k + 1));
                    add(Spent::Lock, 7);
                    id()
                })
                .join()
                .unwrap()
            })
            .collect();
        assert_ne!(ids[0], ids[1]);
        let seen = take().threads;
        for (k, me) in ids.iter().enumerate() {
            let &(_, [inside, held, lock]) = seen.iter().find(|(i, _)| i == me).expect("the thread's slot");
            assert!((20_000..1_000_000).contains(&inside), "once, not twice: {inside}");
            assert_eq!((held, lock), (1000 * (k as u64 + 1), 7));
        }
        assert!(take().threads.iter().all(|(i, _)| !ids.contains(i)), "zero after a take");
    }

    /// A thread gives its row back when it ends: more threads than rows,
    /// counting one after another (each ended before the next starts), each
    /// keep their own figures under their own ids, none starting from what
    /// the one before left in the row.
    #[test]
    fn threads_one_after_another_past_the_rows_each_keep_their_own_figures() {
        use threads::{SLOTS, Spent, add, id, take};
        let _alone = threads_alone();
        take();
        let ids: Vec<u64> = (0..2 * SLOTS as u64 + 1)
            .map(|k| {
                std::thread::spawn(move || {
                    add(Spent::Held, 1000 + k);
                    add(Spent::Lock, k + 1);
                    id()
                })
                .join()
                .unwrap()
            })
            .collect();
        let seen = take();
        for (k, &me) in ids.iter().enumerate() {
            let k = k as u64;
            assert!(seen.threads.contains(&(me, [0, 1000 + k, k + 1])), "thread {k} on its own: {:?}", seen.threads);
        }
        // (Under Wine, the host tests' threads count too.)
        if cfg!(not(windows)) {
            assert_eq!(seen.threads.len(), ids.len());
            assert!(seen.shared.is_none(), "{:?}", seen.shared);
        }
    }

    /// The threads past the rows at once count together in one more row,
    /// with how many shared it; once they end, the rows are free again.
    #[test]
    fn threads_past_the_rows_at_once_count_together_with_how_many() {
        use std::sync::{Arc, Barrier};
        use threads::{SLOTS, Spent, add, id, take};
        let _alone = threads_alone();
        take();
        let n = SLOTS + 4;
        let (counted, taken) = (Arc::new(Barrier::new(n + 1)), Arc::new(Barrier::new(n + 1)));
        let live: Vec<_> = (0..n as u64)
            .map(|k| {
                let (counted, taken) = (counted.clone(), taken.clone());
                std::thread::spawn(move || {
                    add(Spent::Held, 1);
                    add(Spent::Lock, k + 1);
                    counted.wait();
                    taken.wait();
                    id()
                })
            })
            .collect();
        // All of them live, each having counted.
        counted.wait();
        let seen = take();
        taken.wait();
        let ids: Vec<u64> = live.into_iter().map(|t| t.join().unwrap()).collect();
        let own: Vec<usize> = (0..n).filter(|&k| seen.threads.contains(&(ids[k], [0, 1, k as u64 + 1]))).collect();
        let rest = n - own.len();
        let rest_lock: u64 = (0..n).filter(|k| !own.contains(k)).map(|k| k as u64 + 1).sum();
        let (shared, us) = seen.shared.expect("the threads past the rows, together");
        assert!(own.len() <= SLOTS, "{own:?}");
        if cfg!(windows) {
            // Under Wine, the host tests' threads may hold rows or share it too.
            assert!(shared >= rest && us[1] >= rest as u64 && us[2] >= rest_lock, "{shared} {us:?}");
        } else {
            assert_eq!(own.len(), SLOTS, "a row each");
            assert_eq!((shared, us), (n - SLOTS, [0, (n - SLOTS) as u64, rest_lock]));
        }
        // They ended: a thread after them has a row of its own.
        let after = std::thread::spawn(|| {
            add(Spent::Held, 5);
            id()
        })
        .join()
        .unwrap();
        let seen = take();
        assert!(seen.threads.contains(&(after, [0, 5, 0])), "{:?}", seen.threads);
        if cfg!(not(windows)) {
            assert!(seen.shared.is_none(), "{:?}", seen.shared);
        }
    }

    #[test]
    fn spans_join_where_they_touch_and_find_what_they_share() {
        let mut s = Spans::default();
        s.add(100, 200);
        s.add(300, 400);
        s.add(200, 250);
        s.add(500, 500);
        s.add(350, 360);
        s.join();
        assert_eq!(s.ranges(), &[(100, 250), (300, 400)]);
        assert!(s.touches(249, 1));
        assert!(!s.touches(250, 50));
        assert!(s.touches(250, 51));
        assert!(!s.touches(0, 100));
        assert!(s.touches(0, 101));
        assert!(!s.touches(400, 1000));
        assert!(!s.touches(120, 0));
        assert!(!Spans::default().touches(0, usize::MAX));
    }

    #[test]
    fn a_wait_reads_what_the_last_write_over_it_stores() {
        // Nothing there: memory decides.
        assert_eq!(after_writes(&[(0x100, Some(7), 8)], 0x200, 4), None);
        // Whole: an 8-byte value read as 8 and as its two halves.
        let w = [(0x100, Some(0x1111_2222_3333_4444), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(0x1111_2222_3333_4444)));
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(0x3333_4444)));
        assert_eq!(after_writes(&w, 0x104, 4), Some(Some(0x1111_2222)));
        // A 4-byte write stores the low half of its value only.
        let w = [(0x100, Some(0xdead_beef_0000_0005), 4)];
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(5)));
        // Read wider than it stores: the rest is memory's, not known.
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        // A time stamp: not known.
        assert_eq!(after_writes(&[(0x100, None, 8)], 0x100, 8), Some(None));
        // In order: the later write decides, known or not.
        let w = [(0x100, Some(1), 8), (0x100, Some(2), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(2)));
        let w = [(0x100, Some(1), 8), (0x100, None, 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        let w = [(0x100, None, 8), (0x100, Some(3), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(3)));
        // A later write over a part of it: not known.
        let w = [(0x100, Some(1), 8), (0x104, Some(9), 4)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        assert_eq!(after_writes(&w, 0x104, 4), Some(Some(9)));
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(1)));
        // Nothing asked, or more than a value holds.
        assert_eq!(after_writes(&[(0x100, Some(1), 8)], 0x100, 0), None);
        assert_eq!(after_writes(&[(0x100, Some(1), 8)], 0x100, 9), None);
    }
}
