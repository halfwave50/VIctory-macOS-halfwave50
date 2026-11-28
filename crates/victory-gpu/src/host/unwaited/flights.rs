//! mac: flights (`VICTORY_FLIGHTS=1`; off by default, and nothing without
//! `VICTORY_ASYNC_SUBMIT=1`): the title's next submission is prepared while
//! the one left to the host GPU still runs.
//!
//! After hyphennn's flights (fork commit 578d459, `VICTORY_LATE_WAIT`): a
//! submission is sent and not waited for, and the next is prepared while
//! the GPU runs it; what it owes is settled only where something could see
//! it. There several flights are in the air and settled together; here one
//! submission stays on the GPU at a time, as with the switch off, because
//! this line keeps per host copy, not per submission, what the copy-back
//! needs (snapshots, pending ranges, the mark bitmaps, the trackers' "taken"
//! pages; `docs/unwaited.md` of 64bitMango's line §7.1 lists them), and
//! because our host copies are buffers the GPU reads in place: a title copy
//! the next submission makes on the CPU would change bytes the one left
//! still reads (on the console such a copy is ordered behind it on the
//! GPU). What moves is the finish ([`Host::finish_left`]): from before the
//! next submission's preparation to the first of
//!
//! - its GPU work found done (asked without waiting, before each list and
//!   every [`POLL`] steps: then it is finished at once, and its fences move);
//! - a step of the next one that could see or disturb what it owes
//!   ([`Why`]): the eviction pass when it is due; unordered views made
//!   again; the heap sync without the rounds ahead
//!   (`VICTORY_ASYNC_HEAP_SYNC=1` writes heap slots while the GPU runs
//!   already); the texture uploads while its own upload list is on the GPU
//!   (one list and one staging pool, reset per submission); a title copy or
//!   move; a step of ours, but one that writes only the list's own memory
//!   or our flags (`Step::Own`) from title memory it cannot write; a
//!   declaration of what shaders write (snapshots,
//!   arming); a title-memory program's tables; a fill from title memory it
//!   may write or lands in; refreshes of a copy it may write (the prewarm's
//!   rules, [`Host::prewarm`]), or of the title memory it lands in (those
//!   of the other copies are made first);
//! - just before the next one goes to the GPU.
//!
//! Refreshes made before that are made as the prewarm makes them
//! ([`crate::shadow::Shadow::update_early`]): the title's bytes, which the
//! running lists read from pages their own preparation took (the console's
//! rule: a title does not change what a submitted list reads before it
//! knows the list ran). The heap sync writes slots the running lists may
//! read, as the rounds ahead do. A view heap made meanwhile (`Host::quiet`,
//! VIctory-macOS #31) waits for the GPU first.
//!
//! `VICTORY_FLIGHTS_WITH=alias` (E2 of `notes/overlap-stages-design.md`)
//! keeps title buffers zero-copy (`crate::alias`), where the shaders of the
//! submission left write title memory itself while the next is prepared.
//! The next one's texture uploads then use the upload list and staging the
//! one left does not (two, in turn), and finish it first when a texture's
//! memory could meet what it writes or lands ([`Host::beside_texture_reads`]);
//! a fill and a step of ours that read title memory follow the same rule.
//! With `writes` too (E3), a declaration of what shaders write in an aliased
//! copy is made beside it: such a copy keeps no snapshot, and its declared
//! range is noted for its copy-back when its own submission is finished
//! (`Host::finish`), not when it is prepared, where the finish of the one
//! left would take it.
//!
//! With `uploads` too (`VICTORY_FLIGHTS_WITH=alias,uploads`, E2b), the
//! texture uploads never finish it: a texture whose memory it may write or
//! land in (in pages: its declared and landing ranges and bottom-of-pipe
//! writes, the ranges its heaps' unordered views reach, those pages' other
//! addresses; `crate::flights::UploadReach`) is kept out of the reads and
//! brought up into the same upload list as soon as it is finished, for
//! whatever reason, at the latest just before the next goes to the GPU
//! (`crate::bindless::Uploads::land_late`); the others are read beside it.
//!
//! With `uploads2` instead (`VICTORY_FLIGHTS_WITH=alias,uploads2`, E2c) the
//! uploads do not finish it either, at a cost proportional to what changed:
//! nothing is asked of what it may write. A texture changed since its last
//! upload is read beside it, unless a page of it is in an aliased region
//! (written in place while it runs) or it is not tracked: those are read
//! once, after its finish. After its finish every other texture of the
//! check is brought up if its tracker was told of a change meanwhile (its
//! landing marks every tracker over what it writes back), in the same
//! upload list, with barriers from the state it had before this
//! submission (`crate::flights::Beside`). Asks for clean host copies its
//! landing may tell are set aside, and only those whose copies were told
//! are made after its finish (`crate::flights::Stash`).

use super::*;

/// Turns the switch on for the tests, with [`super::tests::FORCED`].
#[cfg(test)]
pub(in crate::host) static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the title's next submission is prepared while the one left to
/// the host GPU still runs (`VICTORY_FLIGHTS=1`, with `VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(Relaxed) {
        return super::on();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    super::on()
        && *ON.get_or_init(|| {
            let on = crate::switch::on("VICTORY_FLIGHTS");
            if on {
                tracing::info!(
                    "VICTORY_FLIGHTS: the title's next submission is prepared while the one left to the host GPU \
                     runs, and that one is finished at the first step that could see what it owes, at the latest \
                     before the next goes to the GPU; held waits its pending writes meet count as met, and Signals \
                     go behind it"
                );
            }
            on
        })
}

/// Turns `alias` on for the tests (with [`FORCED`]).
#[cfg(test)]
pub(in crate::host) static FORCED_ALIAS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns `writes` on for the tests (with [`FORCED_ALIAS`]).
#[cfg(test)]
pub(in crate::host) static FORCED_WRITES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns `uploads` on for the tests (with [`FORCED_ALIAS`]).
#[cfg(test)]
pub(in crate::host) static FORCED_UPLOADS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns `uploads2` on for the tests (with [`FORCED_ALIAS`]).
#[cfg(test)]
pub(in crate::host) static FORCED_UPLOADS2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The names of `VICTORY_FLIGHTS_WITH` this build has (bits of
/// `crate::flights::WITH`): `alias`, `writes`, `uploads`, `uploads2`.
const BUILT: u32 = 0b1100_0011;

/// `uploads`' bit in `crate::flights::WITH`.
const UPLOADS_BIT: u32 = 1 << 6;

/// `uploads2`' bit in `crate::flights::WITH`.
const UPLOADS2_BIT: u32 = 1 << 7;

/// mac: `VICTORY_FLIGHTS_WITH=<list>` (`notes/overlap-stages-design.md` §6),
/// read once: the overlap line's increments taken, as bits of
/// `crate::flights::WITH`. Nothing without the switch on; each name needs
/// the ones before it; one line for what is taken and one for each refusal.
/// `VICTORY_AB` cannot flip it (they carry state across submissions).
fn with() -> u32 {
    static WITH: OnceLock<u32> = OnceLock::new();
    *WITH.get_or_init(|| {
        let value = std::env::var("VICTORY_FLIGHTS_WITH").unwrap_or_default();
        if value.trim().is_empty() {
            return 0;
        }
        if !on() {
            tracing::warn!("VICTORY_FLIGHTS_WITH={value} does nothing without VICTORY_FLIGHTS=1 and VICTORY_ASYNC_SUBMIT=1");
            return 0;
        }
        let (taken, refused) = crate::flights::with_names(&value, BUILT, |i| {
            (i == 0 && crate::heap_worker::on()).then_some(
                "the background heap worker (VICTORY_HEAP_WORKER=1) reads texture memory at any time, where the \
                 check before the uploads cannot see it",
            )
        });
        for why in refused {
            tracing::warn!("VICTORY_FLIGHTS_WITH refused {why}");
        }
        if taken & 1 != 0 {
            tracing::info!(
                "VICTORY_FLIGHTS_WITH=alias: title buffers stay aliased with flights (VICTORY_ALIAS_BUFFERS); the \
                 texture uploads of a submission prepared while the one before runs take an upload list and staging \
                 of their own (two, in turn), and finish that one first where a texture's memory, or the memory a \
                 fill or a step of ours reads, is in what it may write or (if it writes anything) is shown at \
                 another address too"
            );
        }
        if taken & 2 != 0 {
            tracing::info!(
                "VICTORY_FLIGHTS_WITH=writes: a declaration of what shaders write in an aliased copy does not finish \
                 the submission left (no snapshot to keep, nothing to flush); its range is noted for its copy-back \
                 when its own submission is finished"
            );
        }
        if taken & UPLOADS_BIT != 0 {
            tracing::info!(
                "VICTORY_FLIGHTS_WITH=uploads: the texture uploads of a submission prepared beside the one before do \
                 not finish that one; a texture whose memory it may write or land in (by page: what it declared and \
                 lands in, what its heaps' unordered views reach, those pages' other addresses; all with \
                 title-memory programs) is brought up into the same upload list as soon as it is finished, at the \
                 latest before the next goes to the GPU; the others are read beside it"
            );
        }
        if taken & UPLOADS2_BIT != 0 {
            tracing::info!(
                "VICTORY_FLIGHTS_WITH=uploads2: the texture uploads of a submission prepared beside the one before do \
                 not finish that one, at a cost proportional to what changed: a texture changed since its last \
                 upload is read beside it, unless it is in an aliased region or not tracked (read once after its \
                 finish); after its finish the check's other textures whose trackers its landing told are brought \
                 up into the same upload list; asks for clean copies its landing may tell are set aside and made \
                 only where it told them"
            );
        }
        taken
    })
}

/// mac: whether the texture uploads beside the submission left never
/// finish it (`VICTORY_FLIGHTS_WITH=alias,uploads`, E2b of
/// `notes/overlap-stages-design.md`, or `alias,uploads2`, E2c).
pub(crate) fn with_uploads() -> bool {
    #[cfg(test)]
    if FORCED_UPLOADS.load(Relaxed) || FORCED_UPLOADS2.load(Relaxed) {
        return with_alias();
    }
    on() && with() & (UPLOADS_BIT | UPLOADS2_BIT) != 0
}

/// mac: whether that is E2c's rule (`VICTORY_FLIGHTS_WITH=alias,uploads2`):
/// textures read beside it only where they changed and the GPU cannot be
/// writing them, the others brought up after its finish where its landing
/// told their trackers; clean copies' asks set aside
/// (`crate::flights::Beside`, `crate::flights::Stash`).
pub(crate) fn with_uploads2() -> bool {
    #[cfg(test)]
    if FORCED_UPLOADS2.load(Relaxed) {
        return with_alias();
    }
    on() && with() & UPLOADS2_BIT != 0
}

/// E2b's rule (`uploads`, not `uploads2`): the footprint works out its
/// [`Footprint::upload`].
fn with_reach() -> bool {
    with_uploads() && !with_uploads2()
}

/// Whether flights take zero-copy title buffers (`VICTORY_FLIGHTS_WITH=alias`).
pub(crate) fn with_alias() -> bool {
    #[cfg(test)]
    if FORCED_ALIAS.load(Relaxed) {
        return on();
    }
    on() && with() & 1 != 0
}

/// mac: whether a declaration of what shaders write in an aliased copy is
/// made beside the submission left and noted at its own finish
/// (`VICTORY_FLIGHTS_WITH=alias,writes`, E3 of
/// `notes/overlap-stages-design.md`).
pub(crate) fn with_writes() -> bool {
    #[cfg(test)]
    if FORCED_WRITES.load(Relaxed) {
        return with_alias();
    }
    on() && with() & 2 != 0
}

/// The GPU is asked whether it has run the submission left every this many
/// steps of the next one (and before each of its lists).
const POLL: u32 = 64;

/// Why the submission left was finished while the next was prepared.
#[derive(Clone, Copy)]
pub(in crate::host) enum Why {
    /// The GPU had run it.
    Done,
    /// The eviction pass was due (it gives up what the last submission used).
    Evict,
    /// Unordered views were made again.
    Remake,
    /// The heap sync, without the rounds ahead.
    Heaps,
    /// The texture uploads, with its upload list on the GPU.
    Uploads,
    /// A title copy or move.
    Copy,
    /// A step of ours.
    Host,
    /// A declaration of what shaders write.
    Write,
    /// A title-memory program's tables.
    Memory,
    /// A fill from title memory it may write.
    Fill,
    /// Refreshes of what it may write.
    Flush,
    /// The next one goes to the GPU.
    Give,
    /// A view heap was made (`Host::quiet`): it stood the GPU.
    Quiet,
}

const WHYS: usize = 13;

/// Since the last frame line: submissions prepared beside the one left;
/// microseconds of their preparation done while it was on the GPU.
static BESIDE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// By [`Why`]: finishes, and the microseconds each waited for the GPU.
static FINISHED_BY: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
static WAITED_BY: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
/// Refreshes the early form refused all the same (0 if the rules hold).
static REFUSED: AtomicU64 = AtomicU64::new(0);
/// E1 (`notes/overlap-stages-design.md`): later parts of a copy-order split
/// that finished it before they were prepared, and the microseconds they
/// waited for the GPU (counted under "the title's next submission" only);
/// microseconds of landing (finishing it once the GPU was done) by those
/// and by [`Why`]; of the finishes by a declaration and by a title copy,
/// those on aliased copies and between aliased buffers, and their waits.
static SPLIT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static LANDING: AtomicU64 = AtomicU64::new(0);
static ON_ALIASED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// With `VICTORY_FLIGHTS_WITH=alias`: texture uploads made while it ran;
/// of them, those that finished it first as a texture's memory is in what it
/// may write, and as one is shown at another address too.
static UPLOADS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// With `VICTORY_FLIGHTS_WITH=writes`: declarations in aliased copies made
/// while it ran, which no longer finish it.
static WRITES: AtomicU64 = AtomicU64::new(0);

/// With `VICTORY_FLIGHTS_WITH=alias,uploads` (E2b), since the last frame
/// line, of the upload checks beside it (indices below): textures looked at
/// and of them changed since their last upload (their trackers say so);
/// checks that kept textures for after its finish, and textures kept by
/// reason (`crate::flights::MET_EVERY` to `MET_OTHER`); new looks at the
/// other addresses (a link was made since the last); one check in
/// [`SAMPLE`] looked at more closely, its textures, of them in aliased
/// memory and shown at another address too.
static CHECKS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const LOOKED: usize = 0;
const CHANGED: usize = 1;
const KEPT_CHECKS: usize = 2;
const KEPT_BY: usize = 3;
const ANEW: usize = 7;
const SAMPLED: usize = 8;
const SAMPLED_TEXTURES: usize = 9;
const SAMPLED_ALIASED: usize = 10;
const SAMPLED_SHOWN: usize = 11;
const SAMPLE: u64 = 8;
static SAMPLING: AtomicU64 = AtomicU64::new(0);
/// Of those kept: landings ([`landed_late`]), textures brought up then and
/// of them uploaded (changed), microseconds; landings by the [`Why`] of the
/// finish they followed ([`LAST_WHY`]).
static LATE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static LATE_BY: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
/// The [`Why`] of the last finish beside a submission prepared.
static LAST_WHY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(Why::Give as usize);

/// With `uploads` or `uploads2`, since the last frame line: microseconds of
/// the upload checks beside it, a clock pair around each whole check (E2b:
/// its look at every texture, and the footprint work that asks for; E2c:
/// the sort of the check's textures, [`checked`]); footprints worked out
/// and their microseconds (every flights arm works them out; counted here).
static CHECK_US: AtomicU64 = AtomicU64::new(0);
static FOOTPRINTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// With `uploads2` (E2c), since the last frame line, indices below: checks
/// beside it, the textures in them; of those, changed since their last
/// upload (looked at), kept for after its finish (changed in an aliased
/// region, not tracked), read beside it; after its finish: landings of
/// checks, their trackers asked again, textures not asked as no texture
/// page was marked since the check, brought up as told (unchanged at the
/// check, read beside it), uploaded of those, kept ones uploaded, the
/// landings' microseconds; clean copies' asks set aside, made after its
/// finish, the microseconds that took, and asks lost (never made: 0).
static U2: [AtomicU64; 19] = [const { AtomicU64::new(0) }; 19];
const U2_CHECKS: usize = 0;
const U2_TEXTURES: usize = 1;
const U2_CHANGED: usize = 2;
const U2_KEPT: usize = 3;
const U2_NOW: usize = 5;
const U2_LANDINGS: usize = 6;
const U2_ASKED: usize = 7;
const U2_SKIPPED: usize = 8;
const U2_TOLD: usize = 9;
const U2_UPLOADED: usize = 11;
const U2_KEPT_UPLOADED: usize = 12;
const U2_LATE_US: usize = 13;
const U2_STASHED: usize = 14;
const U2_MADE: usize = 15;
const U2_REPLAY_US: usize = 16;
const U2_LOST: usize = 17;
const U2_KEPT_BROUGHT: usize = 18;

/// mac (E2c): an upload check beside the submission left sorted `textures`:
/// `changed` of them changed since their last upload, `kept` of those kept
/// for after its finish by [`crate::flights::Kept`], `now` read beside it,
/// in `took` (the whole check, the sort and the noting of states).
pub(crate) fn checked(textures: usize, changed: usize, kept: [usize; 2], now: usize, took: Duration) {
    U2[U2_CHECKS].fetch_add(1, Relaxed);
    U2[U2_TEXTURES].fetch_add(textures as u64, Relaxed);
    U2[U2_CHANGED].fetch_add(changed as u64, Relaxed);
    for (k, n) in kept.iter().enumerate() {
        U2[U2_KEPT + k].fetch_add(*n as u64, Relaxed);
    }
    U2[U2_NOW].fetch_add(now as u64, Relaxed);
    CHECK_US.fetch_add(took.as_micros() as u64, Relaxed);
}

/// What an E2c landing after the finish did ([`landed2`]).
#[derive(Default)]
pub(crate) struct Landed2 {
    /// Textures whose trackers were asked again, and those not asked (no
    /// texture page marked since the check).
    pub(crate) asked: usize,
    pub(crate) skipped: usize,
    /// Brought up as told: unchanged at the check, read beside it; of all
    /// those, uploaded (their refresh found bytes to copy).
    pub(crate) told: [usize; 2],
    pub(crate) uploaded: usize,
    /// The kept ones brought up, and of them uploaded.
    pub(crate) kept: usize,
    pub(crate) kept_uploaded: usize,
}

/// mac (E2c): the textures of a check were brought up after the finish of
/// the submission left (`crate::bindless::Uploads::land_late`), in `took`;
/// counted by the finish they followed.
pub(crate) fn landed2(l: &Landed2, took: Duration) {
    U2[U2_LANDINGS].fetch_add(1, Relaxed);
    U2[U2_ASKED].fetch_add(l.asked as u64, Relaxed);
    U2[U2_SKIPPED].fetch_add(l.skipped as u64, Relaxed);
    for (k, n) in l.told.iter().enumerate() {
        U2[U2_TOLD + k].fetch_add(*n as u64, Relaxed);
    }
    U2[U2_UPLOADED].fetch_add(l.uploaded as u64, Relaxed);
    U2[U2_KEPT_BROUGHT].fetch_add(l.kept as u64, Relaxed);
    U2[U2_KEPT_UPLOADED].fetch_add(l.kept_uploaded as u64, Relaxed);
    U2[U2_LATE_US].fetch_add(took.as_micros() as u64, Relaxed);
    LATE_BY[LAST_WHY.load(Relaxed).min(WHYS - 1)].fetch_add(1, Relaxed);
}

/// mac (E2c): an ask for a clean copy set aside (`crate::host::Batch`).
pub(crate) fn stashed() {
    U2[U2_STASHED].fetch_add(1, Relaxed);
}

/// mac (E2c): asks set aside were looked at again after the finish:
/// `made` of them went into the refreshes, in `took`.
pub(crate) fn replayed(made: usize, took: Duration) {
    U2[U2_MADE].fetch_add(made as u64, Relaxed);
    U2[U2_REPLAY_US].fetch_add(took.as_micros() as u64, Relaxed);
}

/// mac (E2c): asks set aside that were never looked at again (0 if the
/// rules hold: every path finishes the one left before its last flush).
pub(crate) fn lost(asks: usize) {
    U2[U2_LOST].fetch_add(asks as u64, Relaxed);
}

/// mac: the upload check took `took` (E2b's, with `uploads`).
pub(crate) fn checked_reach(took: Duration) {
    CHECK_US.fetch_add(took.as_micros() as u64, Relaxed);
}

/// What the submission left may write or will write once it is finished,
/// read from it under the submission lock when first needed.
pub(in crate::host) struct Footprint {
    /// Title memory its landing may change (`crate::flights::Landing`):
    /// with programs that use title memory (they write copies at addresses
    /// they compute) every copy and all of it; else its declared ranges,
    /// the copies below whole, its read-backs and texture copy-backs, its
    /// bottom-of-pipe writes and the fences behind it; and the other
    /// addresses of those pages.
    landing: crate::flights::Landing,
    /// Copies its heaps view for unordered access (their shaders may write
    /// them with no list declaring it).
    written: std::collections::HashSet<usize>,
    /// Title memory whose copies hold shader writes kept from it
    /// (`crate::memory::kept`): a refresh there drops them.
    kept: crate::flights::Spans,
    /// mac: with `VICTORY_FLIGHTS_WITH=alias,uploads` (E2b): what the
    /// texture uploads beside it may not read before it is finished, and
    /// the count of links between addresses (`crate::watch::links_made`)
    /// its other addresses were found at.
    upload: Option<Mutex<(u64, crate::flights::UploadReach)>>,
}

/// The pages that show a page of what `reach` names at another address.
fn others_of(reach: &crate::flights::UploadReach) -> crate::flights::Spans {
    let mut pages = Vec::new();
    crate::watch::shown_elsewhere(reach.declared.iter().chain(reach.heaps.iter()), &mut pages);
    let mut others = crate::flights::Spans::default();
    for page in pages {
        others.add(page, page + crate::watch::PAGE);
    }
    others.join();
    others
}

impl Footprint {
    /// E2b's [`Footprint::upload`] of `left`: the pages of its declared and
    /// landing ranges and bottom-of-pipe writes, of what its heaps' views
    /// for unordered access reach (`crate::bindless::written_ranges`), and
    /// their other addresses; everything with title-memory programs.
    fn upload_of(left: &Left) -> Mutex<(u64, crate::flights::UploadReach)> {
        let mut reach = crate::flights::UploadReach { every: left.memory, ..Default::default() };
        let links = crate::watch::links_made();
        if !left.memory {
            let page = crate::watch::PAGE;
            for &(at, end) in &left.declared {
                reach.declared.add_pages(at, end, page);
            }
            for &(at, n) in &left.writes {
                reach.declared.add_pages(at as usize, (at + n) as usize, page);
            }
            for heap in &left.ran.heaps {
                crate::bindless::written_ranges(heap, |from, to| reach.heaps.add_pages(from as usize, to as usize, page));
            }
            reach.declared.join();
            reach.heaps.join();
            reach.others = others_of(&reach);
        }
        Mutex::new((links, reach))
    }

    fn of(left: &Left) -> Footprint {
        let mut written = std::collections::HashSet::new();
        let mut spans = crate::flights::Spans::default();
        for heap in &left.ran.heaps {
            crate::bindless::written(heap, |copy| {
                if written.insert(std::sync::Arc::as_ptr(copy) as usize) {
                    let (start, len) = copy.range();
                    spans.add(start as usize, (start + len) as usize);
                }
            });
        }
        for &(at, end) in &left.declared {
            spans.add(at, end);
        }
        for &(at, n) in &left.writes {
            spans.add(at as usize, (at + n) as usize);
        }
        spans.join();
        let mut pages = Vec::new();
        crate::watch::shown_elsewhere(spans.iter(), &mut pages);
        let mut others = crate::flights::Spans::default();
        for page in pages {
            others.add(page, page + crate::watch::PAGE);
        }
        others.join();
        let mut kept = crate::flights::Spans::default();
        for (at, n) in crate::memory::kept() {
            kept.add(at, at + n);
        }
        kept.join();
        // mac: E2b only; E2c asks nothing of it for the uploads.
        let upload = with_reach().then(|| Footprint::upload_of(left));
        Footprint { landing: crate::flights::Landing { every: left.memory, spans, others }, written, kept, upload }
    }

    /// Whether title memory `[at, at + len)` may change when it is finished.
    fn reaches(&self, at: usize, len: usize) -> bool {
        self.landing.reaches(at, len)
    }

    /// Whether a read of title memory `[at, at + len)` while it runs could
    /// meet what it writes or lands: 1 if the memory is in what it may write
    /// ([`Footprint::reaches`]); with `VICTORY_FLIGHTS_WITH=alias`, 2 if it
    /// writes anything at all and a page of the memory is shown at another
    /// address too (`crate::watch::aliased`), where its writes are not
    /// named by the addresses here.
    fn meets(&self, at: usize, len: usize) -> Option<usize> {
        if self.reaches(at, len) {
            Some(1)
        } else if len > 0 && with_alias() && !self.landing.spans.is_empty() && crate::watch::aliased(at, len) {
            Some(2)
        } else {
            None
        }
    }

    /// Whether `copy` may not be refreshed before it is finished (the
    /// prewarm's rules, and title memory its landing may tell the copy of:
    /// what it lands in, and those pages' other addresses): `ranges` are
    /// what is to be refreshed.
    fn holds(&self, copy: &std::sync::Arc<crate::shadow::Shadow>, ranges: &crate::ranges::Ranges) -> bool {
        if self.landing.every || !copy.idle() || self.written.contains(&(std::sync::Arc::as_ptr(copy) as usize)) {
            return true;
        }
        let (start, len) = copy.range();
        self.kept.touches(start as usize, len as usize) || ranges.iter().any(|(from, to)| self.landing.tells(from, to - from))
    }
}

/// The footprint of the submission left, shared by the preparation beside
/// it ([`Beside`]) and that preparation's refreshes ([`landing`]); worked
/// out when first asked for, and done with once it is finished.
pub(crate) struct Shared {
    done: std::sync::atomic::AtomicBool,
    footprint: OnceLock<Footprint>,
}

impl Shared {
    fn footprint(&self) -> &Footprint {
        self.footprint.get_or_init(|| {
            // mac: timed for the uploads' part of the flights line.
            let started = Instant::now();
            let footprint = LEFT.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or_else(
                || Footprint { landing: crate::flights::Landing { every: true, ..Default::default() }, ..Footprint::empty() },
                Footprint::of,
            );
            FOOTPRINTS[0].fetch_add(1, Relaxed);
            FOOTPRINTS[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
            footprint
        })
    }

    /// What its landing may change (`crate::flights::asked`).
    pub(crate) fn landing(&self) -> &crate::flights::Landing {
        &self.footprint().landing
    }
}

thread_local! {
    /// The submission left while this thread prepares the next one beside
    /// it (under the submission lock): set by [`Host::submissions_beside`],
    /// let go with its [`Beside`].
    static NOW: std::cell::RefCell<Option<std::sync::Arc<Shared>>> = const { std::cell::RefCell::new(None) };
}

/// mac: while this thread prepares a submission beside the one left and
/// that one is not finished yet: its footprint, worked out now, so that
/// asks for clean copies its landing may tell are kept for after it
/// (`crate::flights::asked`; the stale reads it ends are in
/// `notes/overlap-stages-design.md`, "E2: the run"). Asked before a heap's
/// lock is taken: working it out takes the heaps' locks.
pub(crate) fn landing() -> Option<std::sync::Arc<Shared>> {
    let now = NOW.with(|n| n.borrow().clone()).filter(|s| !s.done.load(Acquire))?;
    now.footprint();
    Some(now)
}

/// [`landing`] lent to `f` (none when nothing runs beside), without taking
/// a hold on it: for each ask of the refreshes. Worked out here if it was
/// not yet, so not under a heap's lock (the walks of the heaps ask
/// [`landing`] before they take theirs).
pub(crate) fn with_landing<R>(f: impl FnOnce(Option<&crate::flights::Landing>) -> R) -> R {
    NOW.with(|n| {
        let now = n.borrow();
        f(now.as_ref().filter(|s| !s.done.load(Acquire)).map(|s| s.landing()))
    })
}

/// mac (E2c): whether nothing runs beside the submission this thread
/// prepares any more (none left, or it was finished), without working out
/// its footprint (asked under a heap's lock: working it out takes them).
pub(crate) fn beside_done() -> bool {
    NOW.with(|n| n.borrow().as_ref().is_none_or(|s| s.done.load(Acquire)))
}

/// An ask for a clean copy kept for after the finish (the frame line).
pub(crate) fn kept_clean() {
    KEPT.fetch_add(1, Relaxed);
}

/// Since the last frame line: asks for clean copies kept for after the
/// finish of the submission left ([`landing`]).
static KEPT: AtomicU64 = AtomicU64::new(0);

/// The submission left while the title's next is prepared under the
/// submission lock ([`Host::submissions_beside`]).
pub(in crate::host) struct Beside {
    /// Its fence value, until it is finished.
    value: Option<u64>,
    started: Instant,
    steps: u32,
    /// Its footprint, shared with the refreshes of the one prepared
    /// ([`landing`]).
    shared: std::sync::Arc<Shared>,
    /// Time spent finishing it (its wait and what it owes): not the next
    /// one's preparation.
    pub(in crate::host) finishing: Duration,
}

impl Beside {
    /// Whether the submission left is still not finished.
    pub(in crate::host) fn running(&self) -> bool {
        self.value.is_some()
    }

    fn footprint(&mut self) -> &Footprint {
        self.shared.footprint()
    }
}

impl Drop for Beside {
    fn drop(&mut self) {
        self.shared.done.store(true, Release);
        NOW.with(|n| n.borrow_mut().take());
    }
}

impl Footprint {
    fn empty() -> Footprint {
        Footprint {
            landing: Default::default(),
            written: Default::default(),
            kept: Default::default(),
            upload: with_reach()
                .then(|| Mutex::new((0, crate::flights::UploadReach { every: true, ..Default::default() }))),
        }
    }
}

impl Host {
    /// [`Host::submissions`] for the title's lists `next` with the switch
    /// on: the submission left (if any) is not finished but returned
    /// running, to be finished by the next calls here, unless `next` is the
    /// second part of a split piece (`VICTORY_COPY_ORDER=split`: it starts
    /// with a title copy) or the GPU was stood for it already.
    pub(in crate::host) fn submissions_beside(
        &self,
        next: &[&HostList],
    ) -> (std::sync::MutexGuard<'_, u64>, Option<Beside>) {
        let mut submitted = super::lines::lock_submitted(&self.submitted);
        // Not a list it runs too (a list submitted twice): its steps are
        // looked at under the list's lock, which finishing it takes.
        let running = (VALUE.load(Acquire) != 0)
            .then(|| {
                LEFT.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .filter(|l| l.signalled && !l.stood)
                    .filter(|l| !l.lists.iter().any(|h| next.iter().any(|n| std::ptr::eq(&**h, *n))))
                    .map(|l| l.value)
            })
            .flatten();
        let split = running.is_some() && next.iter().any(|l| l.deferred().after_copy);
        if split {
            // [`Host::settle_held`] (one is left, and `next` is not empty),
            // its wait and landing counted.
            if prewarms() {
                self.prewarm(next);
            }
            let at = Instant::now();
            let waited = self.finish_left(&mut submitted, By::Next);
            SPLIT[0].fetch_add(1, Relaxed);
            SPLIT[1].fetch_add(waited.as_micros() as u64, Relaxed);
            LANDING.fetch_add(at.elapsed().saturating_sub(waited).as_micros() as u64, Relaxed);
            return (submitted, None);
        }
        let Some(value) = running else {
            self.settle_held(&mut submitted, By::Next, next);
            return (submitted, None);
        };
        BESIDE[0].fetch_add(1, Relaxed);
        let shared = std::sync::Arc::new(Shared { done: Default::default(), footprint: OnceLock::new() });
        NOW.with(|n| *n.borrow_mut() = Some(shared.clone()));
        let beside = Beside {
            value: Some(value),
            started: Instant::now(),
            steps: 0,
            shared,
            finishing: Duration::ZERO,
        };
        (submitted, Some(beside))
    }

    /// Finishes the submission left, if `beside` still runs it ([`Why`]):
    /// how long that waited for the GPU.
    pub(in crate::host) fn finish_beside(&self, submitted: &mut u64, beside: &mut Option<Beside>, why: Why) -> Duration {
        let Some(b) = beside.as_mut() else { return Duration::ZERO };
        let Some(value) = b.value.take() else { return Duration::ZERO };
        b.shared.done.store(true, Release);
        LAST_WHY.store(why as usize, Relaxed);
        let at = Instant::now();
        BESIDE[1].fetch_add((at - b.started).as_micros() as u64, Relaxed);
        let mut waited = Duration::ZERO;
        // Only finished under the lock we hold: still the one left.
        if VALUE.load(Acquire) == value {
            waited = self.finish_left(submitted, By::Next);
            FINISHED_BY[why as usize].fetch_add(1, Relaxed);
            WAITED_BY[why as usize].fetch_add(waited.as_micros() as u64, Relaxed);
        }
        b.finishing += at.elapsed();
        LANDING.fetch_add(at.elapsed().saturating_sub(waited).as_micros() as u64, Relaxed);
        waited
    }

    /// Finishes the submission left if the GPU has run it.
    pub(in crate::host) fn poll_beside(&self, submitted: &mut u64, beside: &mut Option<Beside>) {
        let Some(value) = beside.as_ref().and_then(|b| b.value) else { return };
        // SAFETY: a plain query of our fence.
        if unsafe { self.fence.GetCompletedValue() } >= value {
            self.finish_beside(submitted, beside, Why::Done);
        }
    }

    /// Before `step` of the next submission's lists: finishes the
    /// submission left if the step could see or disturb what it owes.
    pub(in crate::host) fn beside_step(&self, submitted: &mut u64, beside: &mut Option<Beside>, step: &Step) {
        let Some(b) = beside.as_mut().filter(|b| b.running()) else { return };
        b.steps += 1;
        let why = match step {
            Step::Copy(..) | Step::Move(..) => Some(Why::Copy),
            Step::Host(_) => Some(Why::Host),
            &Step::Own(_, at, len) => (len > 0 && b.footprint().meets(at, len).is_some()).then_some(Why::Host),
            // mac: `VICTORY_FLIGHTS_WITH=writes`: in an aliased copy nothing
            // of the one left is touched (`Host::finish` notes the range).
            Step::GpuWrite(copy, ..) if copy.is_aliased() && with_writes() => {
                WRITES.fetch_add(1, Relaxed);
                None
            }
            Step::GpuWrite(..) => Some(Why::Write),
            Step::Memory(_) => Some(Why::Memory),
            &Step::Fill(_, src, len) => b.footprint().meets(src, len).is_some().then_some(Why::Fill),
            Step::Refresh(..) | Step::Slot(..) | Step::Heap(_) | Step::Texture(..) => None,
        };
        match why {
            Some(why) => {
                // E1: of these, the finishes stage 3 would leave out (a
                // declaration on an aliased copy, a title copy between
                // aliased buffers).
                let aliased = match step {
                    Step::GpuWrite(copy, ..) => copy.is_aliased().then_some(0),
                    &Step::Copy(dst, src, len) => {
                        let inside = |at: usize| crate::shadow::find_aliased(at as u64, (at + len) as u64).is_some();
                        (inside(dst) && inside(src)).then_some(2)
                    }
                    _ => None,
                };
                let waited = self.finish_beside(submitted, beside, why);
                if let Some(k) = aliased {
                    ON_ALIASED[k].fetch_add(1, Relaxed);
                    ON_ALIASED[k + 1].fetch_add(waited.as_micros() as u64, Relaxed);
                }
            }
            None if b.steps.is_multiple_of(POLL) => self.poll_beside(submitted, beside),
            None => {}
        }
    }

    /// Before the next submission's refreshes are made, if the submission
    /// left still runs: takes out of `refreshes` those of copies it may
    /// write or lands in (returned, to be made once it is finished); the
    /// rest may be made now, early ([`crate::shadow::Shadow::update_early`]).
    /// `None` (nothing taken) if nothing runs beside.
    pub(in crate::host) fn beside_refreshes(
        &self,
        beside: &mut Option<Beside>,
        refreshes: &mut Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    ) -> Option<Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>> {
        let b = beside.as_mut().filter(|b| b.running())?;
        let footprint = b.footprint();
        let (held, early): (Vec<_>, Vec<_>) =
            std::mem::take(refreshes).into_iter().partition(|(copy, ranges)| footprint.holds(copy, ranges));
        *refreshes = early;
        Some(held)
    }

    /// [`Host::quiet`] on the thread that prepares the next submission: a
    /// view heap is to be made and filled, and the submission left may
    /// still run (VIctory-macOS #31): the GPU is waited for first. Under
    /// the submission lock.
    pub(in crate::host) fn stand_beside(&self) {
        if VALUE.load(Acquire) != 0 {
            FINISHED_BY[Why::Quiet as usize].fetch_add(1, Relaxed);
            self.stand();
        }
    }

    /// Whether the texture uploads would reset what the submission left
    /// runs: its upload list, and the staging pool its uploads read from
    /// (`crate::bindless::uploads_after`, `crate::texture::recycle_in`).
    /// mac: never with `VICTORY_FLIGHTS_WITH=alias`, where the next
    /// submission's uploads take the other list and staging
    /// (`crate::texture::upload_slot`).
    pub(in crate::host) fn beside_uploads(&self, beside: &Option<Beside>) -> bool {
        beside.as_ref().is_some_and(Beside::running)
            && !with_alias()
            && LEFT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_none_or(|l| l.uploads)
    }

    /// mac: with `VICTORY_FLIGHTS_WITH=alias`, before the texture uploads
    /// read title memory (`work`: what they bring up, whether it changed or
    /// not): the submission left is finished first if one of the textures'
    /// memory could meet what it writes or lands ([`Footprint::meets`]).
    /// Their reads then never meet its shaders' writes in aliased memory,
    /// and a texture its copy-back tells is told before they look, as when
    /// it is finished before the next is prepared.
    ///
    /// With `alias,uploads` (E2b) it is not finished: the indices returned
    /// are those of `work` to bring up only once it is
    /// (`crate::bindless::Uploads::land_late`), which [`kept_for_later`]
    /// picks. With `alias,uploads2` (E2c) it is not finished either, and
    /// nothing is asked of it: the check sorts its textures itself
    /// (`crate::bindless::Reads::Again`).
    pub(in crate::host) fn beside_texture_reads(
        &self,
        submitted: &mut u64,
        beside: &mut Option<Beside>,
        work: &[(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>)],
    ) -> crate::bindless::Reads {
        use crate::bindless::Reads;
        if !with_alias() || work.is_empty() {
            return Reads::Now;
        }
        let Some(b) = beside.as_mut().filter(|b| b.running()) else { return Reads::Now };
        UPLOADS[0].fetch_add(1, Relaxed);
        if with_uploads2() {
            return Reads::Again;
        }
        let started = Instant::now();
        let footprint = b.footprint();
        if let Some(upload) = &footprint.upload {
            let kept = kept_for_later(upload, work);
            checked_reach(started.elapsed());
            return Reads::Keep(kept);
        }
        let met = work.iter().find_map(|(staging, _)| {
            let (at, len) = staging.title_range();
            footprint.meets(at, len)
        });
        if let Some(k) = met {
            UPLOADS[k].fetch_add(1, Relaxed);
            self.finish_beside(submitted, beside, Why::Uploads);
        }
        Reads::Now
    }
}

/// mac: E2b (`VICTORY_FLIGHTS_WITH=alias,uploads`): of `work`, the textures
/// to bring up only once the submission left is finished: those whose title
/// memory (what a refresh of it may read) shares a page with what it may
/// write or land in, or with another address of such a page
/// ([`crate::flights::UploadReach::meets`]). The other addresses are looked
/// up again if a link between addresses was made since (a mapping made
/// meanwhile). Counted for the flights line.
fn kept_for_later(
    upload: &Mutex<(u64, crate::flights::UploadReach)>,
    work: &[(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>)],
) -> Vec<usize> {
    let mut held = upload.lock().unwrap_or_else(|e| e.into_inner());
    let (links, reach) = &mut *held;
    let now = crate::watch::links_made();
    if now != *links && !reach.every {
        reach.others = others_of(reach);
        *links = now;
        CHECKS[ANEW].fetch_add(1, Relaxed);
    }
    let mut kept = Vec::new();
    let mut by = [0u64; 4];
    if !reach.nothing() {
        for (i, (staging, _)) in work.iter().enumerate() {
            let (at, len) = staging.title_range();
            if let Some(k) = reach.meets(at, len) {
                by[k] += 1;
                kept.push(i);
            }
        }
    }
    drop(held);
    let changed = work.iter().filter(|(staging, _)| staging.maybe_changed()).count();
    CHECKS[LOOKED].fetch_add(work.len() as u64, Relaxed);
    CHECKS[CHANGED].fetch_add(changed as u64, Relaxed);
    if !kept.is_empty() {
        CHECKS[KEPT_CHECKS].fetch_add(1, Relaxed);
    }
    for (k, n) in by.iter().enumerate() {
        CHECKS[KEPT_BY + k].fetch_add(*n, Relaxed);
    }
    if SAMPLING.fetch_add(1, Relaxed).is_multiple_of(SAMPLE) {
        let (mut aliased, mut shown) = (0, 0);
        for (staging, _) in work {
            let (at, len) = staging.title_range();
            aliased += u64::from(crate::shadow::find_aliased(at as u64, (at + len) as u64).is_some());
            shown += u64::from(len > 0 && crate::watch::aliased(at, len));
        }
        CHECKS[SAMPLED].fetch_add(1, Relaxed);
        CHECKS[SAMPLED_TEXTURES].fetch_add(work.len() as u64, Relaxed);
        CHECKS[SAMPLED_ALIASED].fetch_add(aliased, Relaxed);
        CHECKS[SAMPLED_SHOWN].fetch_add(shown, Relaxed);
    }
    kept
}

/// mac: E2b: textures kept for after the finish of the submission left
/// were brought up (`crate::bindless::Uploads::land_late`): `textures`,
/// `uploaded` of them changed, in `took`. Counted by the finish they
/// followed.
pub(crate) fn landed_late(textures: usize, uploaded: usize, took: Duration) {
    LATE[0].fetch_add(1, Relaxed);
    LATE[1].fetch_add(textures as u64, Relaxed);
    LATE[2].fetch_add(uploaded as u64, Relaxed);
    LATE[3].fetch_add(took.as_micros() as u64, Relaxed);
    LATE_BY[LAST_WHY.load(Relaxed).min(WHYS - 1)].fetch_add(1, Relaxed);
}

/// An early refresh refused (the copy stopped being idle): made the
/// ordinary way, and counted.
pub(in crate::host) fn refused() {
    REFUSED.fetch_add(1, Relaxed);
}

/// The frame line of the flights (`frames` since the last one).
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let beside = BESIDE[0].swap(0, Relaxed) as f64 / frames;
    let beside_ms = BESIDE[1].swap(0, Relaxed) as f64 / 1000.0 / frames;
    let by: [f64; WHYS] = std::array::from_fn(|i| FINISHED_BY[i].swap(0, Relaxed) as f64 / frames);
    let ms: [f64; WHYS] = std::array::from_fn(|i| WAITED_BY[i].swap(0, Relaxed) as f64 / 1000.0 / frames);
    let refused = REFUSED.swap(0, Relaxed) as f64 / frames;
    let met = MET.swap(0, Relaxed) as f64 / frames;
    let [split, split_ms] = [0, 1].map(|i| SPLIT[i].swap(0, Relaxed) as f64 / if i == 0 { frames } else { 1000.0 * frames });
    let landing = LANDING.swap(0, Relaxed) as f64 / 1000.0 / frames;
    let kept = KEPT.swap(0, Relaxed) as f64 / frames;
    let [write, write_ms, copy, copy_ms] =
        [0, 1, 2, 3].map(|i| ON_ALIASED[i].swap(0, Relaxed) as f64 / if i % 2 == 0 { frames } else { 1000.0 * frames });
    tracing::info!(
        "flights: {beside:.1}/frame prepared while the submission before ran on the host GPU, {beside_ms:.1} ms/frame \
         of preparing beside it; it was finished by (its GPU work done, eviction, unordered views made again, heap \
         sync, texture uploads, a title copy, a step of ours, a shader-write declaration, a title-memory program, a \
         fill from what it may write, refreshes of what it may write, the next going to the GPU, a view heap made): \
         {by:.1?} per frame, after waiting {ms:.1?} ms/frame for the GPU; {refused:.1} early refreshes refused (0 if \
         the rules hold); {met:.1} held-wait looks/frame answered from a pending write; split parts: {split:.1}/frame \
         later parts of a copy-order split finished it before they were prepared, after waiting {split_ms:.1} \
         ms/frame (only under the title's next submission above); landing it after its wait took {landing:.1} \
         ms/frame of these finishes; of them, shader-write declarations on aliased copies {write:.1}/frame \
         ({write_ms:.1} ms waited) and title copies between aliased buffers {copy:.1}/frame ({copy_ms:.1} ms); \
         {kept:.0}/frame asks for copies the title had not written kept for after its finish (its landing may tell \
         them){}",
        uploads_line(frames)
    );
}

/// The flights line's part for `VICTORY_FLIGHTS_WITH=alias` (empty without).
fn uploads_line(frames: f64) -> String {
    if !with_alias() {
        return String::new();
    }
    let [made, reached, twice] = UPLOADS.each_ref().map(|n| n.swap(0, Relaxed) as f64 / frames);
    let writes = WRITES.swap(0, Relaxed) as f64 / frames;
    format!(
        "; VICTORY_FLIGHTS_WITH=alias: {made:.1}/frame texture uploads made while it ran, {reached:.1} of them \
         finished it first as a texture's memory is in what it may write, {twice:.1} as a texture's memory is shown \
         at another address too{}{}",
        if with_writes() {
            format!("; writes: {writes:.1}/frame declarations in aliased copies made while it ran, not finishing it")
        } else {
            String::new()
        },
        if with_uploads2() {
            uploads2_line(frames)
        } else if with_uploads() {
            kept_line(frames)
        } else {
            String::new()
        }
    )
}

/// The ms a frame of the upload checks and the footprints worked out (how
/// many a frame, ms), for both E2b's and E2c's parts.
fn clocks(frames: f64) -> (f64, f64, f64) {
    let check = CHECK_US.swap(0, Relaxed) as f64 / 1000.0 / frames;
    let [n, us] = FOOTPRINTS.each_ref().map(|c| c.swap(0, Relaxed) as f64 / frames);
    (check, n, us / 1000.0)
}

/// The flights line's part for `VICTORY_FLIGHTS_WITH=alias,uploads2` (E2c).
fn uploads2_line(frames: f64) -> String {
    let c: [u64; 19] = std::array::from_fn(|i| U2[i].swap(0, Relaxed));
    let by: [f64; WHYS] = std::array::from_fn(|i| LATE_BY[i].swap(0, Relaxed) as f64 / frames);
    let f = |n: u64| n as f64 / frames;
    let ms = |us: u64| us as f64 / 1000.0 / frames;
    let (check_ms, footprints, footprint_ms) = clocks(frames);
    // Looked at: the changed ones, and those unchanged at the check that
    // were brought up after (those read beside it and told again are among
    // the changed).
    format!(
        "; uploads2: of those checks ({checks:.1}/frame, {textures:.0} textures/frame in them), {looked:.0} \
         textures/frame looked at ({changed:.0} changed since their last upload, sorted at the check; {told_clean:.1} \
         unchanged then, brought up after its finish as their trackers were told since); kept for after its finish \
         {kept:.1} (changed in an aliased region {in_place:.1}, not tracked whole {untracked:.1}), read beside it \
         {now:.1}; \
         {landings:.1}/frame landings after the finish by (the 13 above) {by:.1?}: trackers asked again {asked:.0}, \
         {skipped:.0} not asked as no texture page was marked since the check, brought up as told {told_clean:.1} \
         unchanged at the check and {told_again:.1} read beside it ({uploaded:.1} of them uploaded), kept ones \
         brought up {kept_brought:.1} ({kept_uploaded:.1} uploaded); clocks: the checks {check_ms:.2} ms/frame, the \
         landings after the finish {late_ms:.2} ms/frame, footprints worked out {footprints:.1}/frame in \
         {footprint_ms:.2} ms; {stashed:.0}/frame asks for clean copies set aside, {made:.0} of them made after its \
         finish (told) in {replay_ms:.2} ms/frame, {lost:.0} lost (0 if the rules hold)",
        checks = f(c[U2_CHECKS]),
        textures = f(c[U2_TEXTURES]),
        looked = f(c[U2_CHANGED] + c[U2_TOLD]),
        changed = f(c[U2_CHANGED]),
        kept = f(c[U2_KEPT] + c[U2_KEPT + 1]),
        in_place = f(c[U2_KEPT + crate::flights::Kept::InPlace as usize]),
        untracked = f(c[U2_KEPT + crate::flights::Kept::Untracked as usize]),
        now = f(c[U2_NOW]),
        landings = f(c[U2_LANDINGS]),
        asked = f(c[U2_ASKED]),
        skipped = f(c[U2_SKIPPED]),
        told_clean = f(c[U2_TOLD]),
        told_again = f(c[U2_TOLD + 1]),
        uploaded = f(c[U2_UPLOADED]),
        kept_brought = f(c[U2_KEPT_BROUGHT]),
        kept_uploaded = f(c[U2_KEPT_UPLOADED]),
        late_ms = ms(c[U2_LATE_US]),
        stashed = f(c[U2_STASHED]),
        made = f(c[U2_MADE]),
        replay_ms = ms(c[U2_REPLAY_US]),
        lost = f(c[U2_LOST]),
    )
}

/// E2c's counts so far (tests): changed at the checks, kept by
/// [`crate::flights::Kept`], read beside it, brought up as told (unchanged
/// at the check, read beside it), asks set aside and made.
#[cfg(test)]
pub(super) fn uploads2_counts() -> [u64; 8] {
    let c = |i: usize| U2[i].load(Relaxed);
    [c(U2_CHANGED), c(U2_KEPT), c(U2_KEPT + 1), c(U2_NOW), c(U2_TOLD), c(U2_TOLD + 1), c(U2_STASHED), c(U2_MADE)]
}

/// The flights line's part for `VICTORY_FLIGHTS_WITH=alias,uploads` (E2b).
fn kept_line(frames: f64) -> String {
    let c: [u64; 12] = std::array::from_fn(|i| CHECKS[i].swap(0, Relaxed));
    let late: [u64; 4] = std::array::from_fn(|i| LATE[i].swap(0, Relaxed));
    let by: [f64; WHYS] = std::array::from_fn(|i| LATE_BY[i].swap(0, Relaxed) as f64 / frames);
    let f = |n: u64| n as f64 / frames;
    let kept: u64 = c[KEPT_BY..KEPT_BY + 4].iter().sum();
    let per = |n: u64| if c[SAMPLED] == 0 { 0.0 } else { n as f64 / c[SAMPLED] as f64 };
    // mac: the clocks E2c added, for comparison (counts only).
    let (check_ms, footprints, footprint_ms) = clocks(frames);
    format!(
        "; uploads: of those checks, {looked:.0} textures/frame looked at ({changed:.0} changed since their last \
         upload); {met:.1}/frame kept {kept:.1} textures/frame for after its finish instead of finishing it (as it \
         runs title-memory programs {e:.1}, in a page it declared or lands in {d:.1}, in a page its heaps' unordered \
         views reach {h:.1}, at another address of such a page {o:.1}); {landings:.1}/frame landings of them, after \
         the finish by (the 13 above) {by:.1?}, brought up {brought:.1} textures/frame, {uploaded:.1} of them \
         changed, in {late_ms:.1} ms/frame; {anew:.1}/frame other addresses looked up again; one check in {SAMPLE} \
         ({sampled:.1}/frame): {t:.0} textures a check, {a:.0} in aliased memory, {s:.0} shown at another address \
         too; clocks: the checks {check_ms:.2} ms/frame, footprints worked out {footprints:.1}/frame in \
         {footprint_ms:.2} ms",
        looked = f(c[LOOKED]),
        changed = f(c[CHANGED]),
        met = f(c[KEPT_CHECKS]),
        kept = f(kept),
        e = f(c[KEPT_BY + crate::flights::MET_EVERY]),
        d = f(c[KEPT_BY + crate::flights::MET_DECLARED]),
        h = f(c[KEPT_BY + crate::flights::MET_HEAPS]),
        o = f(c[KEPT_BY + crate::flights::MET_OTHER]),
        landings = f(late[0]),
        brought = f(late[1]),
        uploaded = f(late[2]),
        late_ms = late[3] as f64 / 1000.0 / frames,
        anew = f(c[ANEW]),
        sampled = f(c[SAMPLED]),
        t = per(c[SAMPLED_TEXTURES]),
        a = per(c[SAMPLED_ALIASED]),
        s = per(c[SAMPLED_SHOWN]),
    )
}

/// Textures kept for after the finish so far, by reason
/// (`crate::flights::MET_EVERY` to `MET_OTHER`; tests).
#[cfg(test)]
pub(super) fn kept_for(k: usize) -> u64 {
    CHECKS[KEPT_BY + k].load(Relaxed)
}

/// Kept textures brought up after a finish so far, and of them uploaded
/// (tests).
#[cfg(test)]
pub(super) fn landed() -> (u64, u64) {
    (LATE[1].load(Relaxed), LATE[2].load(Relaxed))
}

/// Finishes so far by every [`Why`] (tests; the frame line takes them).
#[cfg(test)]
pub(super) fn finished_beside() -> u64 {
    FINISHED_BY.iter().map(|n| n.load(Relaxed)).sum()
}

/// Finishes so far for `why` (tests).
#[cfg(test)]
pub(super) fn finished_for(why: Why) -> u64 {
    FINISHED_BY[why as usize].load(Relaxed)
}

/// Since the last frame line: looks at a held wait answered from the
/// pending writes of the submission left ([`super::pending_value`]).
pub(super) static MET: AtomicU64 = AtomicU64::new(0);
