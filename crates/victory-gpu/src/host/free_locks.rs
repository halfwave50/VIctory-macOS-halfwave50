//! mac: `VICTORY_FREE_LOCKS=1` (off by default): the lock words of emulated
//! 64-bit image atomics (`victory_shader::emit`, "Atomic locks") are freed
//! after every submission in which lanes gave up waiting for a lock, not
//! only after one the host GPU ended before its mark ([`Host::lost`]).
//!
//! Why (halfwave50/VIctory-macOS-halfwave50 issue #7, an M1 Pro tester's
//! idea and trace): D3DMetal can end one command buffer of a submission
//! (macOS ends a compute command buffer of about 500 ms when WindowServer
//! needs the GPU, `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`)
//! while the submission still reaches its mark. Then [`Host::lost`] does not
//! run, and the locks the ended shader held stay taken: every later lane
//! that needs one of them waits out its bound
//! (`victory_shader::emit::ATOMIC_LOCK_SPINS`), gives up and goes on
//! without it, in every later submission. We are not told about such a
//! command buffer (D3DMetal writes nothing to the log, and the mark is
//! reached), but a lane that meets such a lock always gives up, so the
//! give-up count the host reads after each submission is the sign. A
//! submission that has run holds no lock (each lane frees its own before
//! its shader ends), and only one submission is on the host GPU at a time
//! (`host::unwaited`), so a lock word still taken after it was left by an
//! ended shader, and freeing all of them is exact.
//!
//! With the switch, a line every 5 s gives the counts since the last one:
//! submissions ended before their mark (their locks [`Host::lost`] frees in
//! any case), submissions with lanes that gave up and those lanes, and the
//! lock words found taken and freed after them. Taken words there mean a
//! command buffer was ended inside a submission that reached its mark.

use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;

use super::*;

/// Whether locks are freed after submissions with give-ups
/// (`VICTORY_FREE_LOCKS=1`), read once.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_FREE_LOCKS");
        if on {
            tracing::info!(
                "VICTORY_FREE_LOCKS=1: the locks of emulated 64-bit atomics are freed after every submission \
                 in which lanes gave up waiting for one"
            );
        }
        on
    })
}

/// The counts of the 5-s line.
#[derive(Debug, Default, PartialEq)]
struct Counts {
    /// Submissions in which lanes gave up waiting for a lock.
    with_give_ups: u64,
    /// The lanes that gave up.
    gave_up: u64,
    /// Submissions after which taken lock words were found and freed.
    with_taken: u64,
    /// The lock words freed.
    freed: u64,
}

impl Counts {
    /// After a submission in which `gave_up` lanes gave up: with the switch
    /// (`on`), `free` frees the lock words and says how many were taken.
    fn after(&mut self, on: bool, gave_up: u64, free: impl FnOnce() -> usize) {
        if !on || gave_up == 0 {
            return;
        }
        self.with_give_ups += 1;
        self.gave_up += gave_up;
        let freed = free() as u64;
        if freed > 0 {
            self.with_taken += 1;
            self.freed += freed;
        }
    }
}

/// The counts since the last line, when that line was, and the
/// submissions ended before their mark ([`LOST`]) by then.
struct Window {
    counts: Counts,
    since: Instant,
    lost: u64,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// After a submission has run (`Host::finish`), with the lanes that gave up
/// waiting for a lock in it, `gave_up` (`crate::shadow::take_gave_up`):
/// with the switch, the locks are freed if any did, and every 5 s a line
/// gives the counts. Under the submission lock; nothing without the switch.
pub(super) fn after_submission(gave_up: u64) {
    if !on() {
        return;
    }
    let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    let lost = LOST[0].load(Relaxed);
    let w = window.get_or_insert_with(|| Window {
        counts: Counts::default(),
        since: Instant::now(),
        lost,
    });
    w.counts
        .after(true, gave_up, crate::shadow::free_atomic_locks);
    if w.since.elapsed() < Duration::from_secs(5) {
        return;
    }
    let Counts {
        with_give_ups,
        gave_up,
        with_taken,
        freed,
    } = std::mem::take(&mut w.counts);
    tracing::info!(
        ended = lost - w.lost,
        with_give_ups,
        gave_up,
        with_taken,
        freed,
        "free locks (5 s): submissions ended before their mark, submissions with lanes that gave up \
         waiting for a 64-bit atomic's lock, those lanes, submissions after which lock words were left \
         taken (an ended command buffer), the lock words freed"
    );
    w.since = Instant::now();
    w.lost = lost;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without give-ups nothing is freed or counted, and without the switch
    /// nothing at all.
    #[test]
    fn only_give_ups_with_the_switch_free_locks() {
        let mut counts = Counts::default();
        counts.after(true, 0, || panic!("freed without give-ups"));
        counts.after(false, 5, || panic!("freed without the switch"));
        assert_eq!(counts, Counts::default());
        counts.after(true, 3, || 0);
        assert_eq!(
            counts,
            Counts {
                with_give_ups: 1,
                gave_up: 3,
                with_taken: 0,
                freed: 0
            }
        );
    }

    /// A shader the host GPU ended in a submission that still reached its
    /// mark left lock word 7 taken; a lane of the next submission waited out
    /// its bound on it and gave up. After that submission the lock is free
    /// again and the count is taken; without the switch the lock stays
    /// taken. On the real bitmaps, under the submission lock.
    #[test]
    fn a_lock_an_ended_shader_left_taken_is_freed_after_the_give_ups() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        // (The test lock first, as a submission takes them: not while a
        // test of `unwaited` has the submissions to itself.)
        let _shared = super::unwaited::shared();
        let _submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        use crate::shadow::tests::{add_gave_up, lock_word, set_lock_word};
        // Any count and locks a test before left.
        let _ = crate::shadow::take_gave_up();
        crate::shadow::free_atomic_locks();
        for memory in [false, true] {
            let mut counts = Counts::default();
            // The ended shader's lock, and the lane that met it.
            set_lock_word(memory, 7, 1);
            add_gave_up(memory, 1);
            counts.after(
                false,
                crate::shadow::take_gave_up(),
                crate::shadow::free_atomic_locks,
            );
            assert_eq!(
                lock_word(memory, 7),
                1,
                "without the switch the lock stays taken"
            );
            assert_eq!(counts, Counts::default());
            // The next submission: a lane met it again.
            add_gave_up(memory, 1);
            counts.after(
                true,
                crate::shadow::take_gave_up(),
                crate::shadow::free_atomic_locks,
            );
            assert_eq!(lock_word(memory, 7), 0, "freed after the give-up");
            assert_eq!(
                counts,
                Counts {
                    with_give_ups: 1,
                    gave_up: 1,
                    with_taken: 1,
                    freed: 1
                }
            );
            assert_eq!(crate::shadow::take_gave_up(), 0, "the count was taken");
            // The submission after: no lane waits, nothing to free.
            counts.after(
                true,
                crate::shadow::take_gave_up(),
                crate::shadow::free_atomic_locks,
            );
            assert_eq!(counts.with_give_ups, 1);
        }
    }
}
