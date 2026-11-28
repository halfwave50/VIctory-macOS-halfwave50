//! mac: E1 of `notes/overlap-stages-design.md`: lines only, so that the arms
//! of an overlap A/B can be read side by side. Nothing counted here changes
//! what runs or when. The line is made only with `VICTORY_ASYNC_SUBMIT=1`
//! (flights or not), but the counting and timing run in every run, whatever
//! the switches:
//!
//! - bottom-of-pipe writes made, and those into aliased memory (a later
//!   submission in the air could read them on the GPU before we write them):
//!   per write an atomic add and, while aliased copies exist, a look-up in
//!   them (`crate::shadow::find_aliased`: a read lock and a map search);
//! - shader-write ranges finished, and those on aliased copies (stage 3's
//!   declarations that need no finish): per range one or two atomic adds;
//! - per thread calling us: the time inside our queue calls, and the waits
//!   for the held-work lock and the submission lock in it
//!   (`crate::flights::threads`). Per ExecuteCommandLists, Signal, Wait
//!   that waits and PresentX: four clock reads (the call's start and end,
//!   the held-work lock's wait), two atomic adds and four thread-local
//!   reads. Per other take of the held-work lock (a CPU Signal, a page
//!   mapping, the frame line) and per take of the submission lock (each
//!   host submission, each finish of one left to the host GPU): two clock
//!   reads and an atomic add.
//!
//! Under Wine a clock read (`Instant::now`, QueryPerformanceCounter) is a
//! call into Wine's Unix side (NtQueryPerformanceCounter, then
//! mach_continuous_time), not a read inside the process. At some 20 title
//! submissions a frame (the held-work line of the 28 November runs) that is
//! some 200 to 300 clock reads a frame (the Signals a frame are a guess),
//! at a cost not measured: they would reach 1% of a 100 ms frame only at
//! some 4 µs each, so they are not switched off with the line.

use super::*;

/// Since the last frame line: words made, into aliased memory; ranges
/// declared, on aliased copies.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// When the last line was made.
static SINCE: Mutex<Option<Instant>> = Mutex::new(None);

/// A bottom-of-pipe write of `bytes` at `at` was made.
pub(in crate::host) fn word(at: usize, bytes: usize) {
    COUNTS[0].fetch_add(1, Relaxed);
    if crate::shadow::find_aliased(at as u64, (at + bytes) as u64).is_some() {
        COUNTS[1].fetch_add(1, Relaxed);
    }
}

/// A finished submission declared that shaders write a range of `copy`.
pub(in crate::host) fn declared(copy: &crate::shadow::Shadow) {
    COUNTS[2].fetch_add(1, Relaxed);
    if copy.is_aliased() {
        COUNTS[3].fetch_add(1, Relaxed);
    }
}

/// The submission lock, its wait counted for the thread.
pub(super) fn lock_submitted(lock: &Mutex<u64>) -> std::sync::MutexGuard<'_, u64> {
    let at = Instant::now();
    let held = lock.lock().unwrap_or_else(|e| e.into_inner());
    crate::flights::threads::add(crate::flights::threads::Spent::Lock, at.elapsed().as_micros() as u64);
    held
}

/// The line (`frames` since the last one).
pub(super) fn log(frames: f64) {
    let [words, aliased_words, ranges, aliased_ranges] = COUNTS.each_ref().map(|n| n.swap(0, Relaxed) as f64 / frames);
    let now = Instant::now();
    let since = SINCE.lock().unwrap_or_else(|e| e.into_inner()).replace(now);
    let frame = since.map_or(0.0, |s| (now - s).as_secs_f64() * 1000.0 / frames);
    let crate::flights::threads::Taken { threads: mut taken, shared } = crate::flights::threads::take();
    taken.sort_by_key(|t| std::cmp::Reverse(t.1[0]));
    let ms = |us: [u64; 3]| us.map(|u| u as f64 / 1000.0 / frames);
    let mut threads: Vec<String> = taken
        .iter()
        .map(|&(id, us)| {
            let [inside, held, lock] = ms(us);
            format!("{id}: {inside:.1}/{held:.1}/{lock:.1}")
        })
        .collect();
    if let Some((many, us)) = shared {
        let [inside, held, lock] = ms(us);
        let slots = crate::flights::threads::SLOTS;
        threads.push(format!("{many} threads past {slots}, together: {inside:.1}/{held:.1}/{lock:.1}"));
    }
    tracing::info!(
        "overlap lines: {words:.1}/frame bottom-of-pipe writes made, {aliased_words:.1} of them into aliased memory; \
         {ranges:.1}/frame shader-write ranges finished, {aliased_ranges:.1} of them on aliased copies; threads \
         calling us by Windows thread id (those that ended since the last line too; past {} at once, the rest in \
         one entry, \"N threads past {}, together\"), ms/frame inside our queue calls (ExecuteCommandLists, Signal, \
         Wait, PresentX) / waiting for the held-work lock / waiting for the submission lock: [{}]; {frame:.1} \
         ms/frame by the clock (outside our calls, a thread spent that less its first figure)",
        crate::flights::threads::SLOTS,
        crate::flights::threads::SLOTS,
        threads.join(", ")
    );
}
