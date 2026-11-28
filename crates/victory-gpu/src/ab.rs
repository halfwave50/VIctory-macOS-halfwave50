//! mac: one switch flipped while the game runs
//! (`VICTORY_AB=<SWITCH>[=<value B>]:<seconds>`, `notes/ab-toggle.md`), so
//! that one run measures both sides of it in the same title mode. Off when
//! unset: a switch then reads nothing here but one atomic load ([`side`]).
//! Plain std, so its tests run everywhere.
//!
//! Side A is the switch as the environment sets it; side B the opposite, or
//! with `=<value B>` that value, read by the switch's own rule. The side
//! flips at the present (`crate::queue`), after the submission left to the
//! host GPU is finished, every `<seconds>` of wall time, from the first
//! frame that ran `VICTORY_AB_AFTER` draws on the host (default 500, the
//! world's first frame as the frame recorder and the captures count it).
//! A frame's lines (`ab=` on "held work this frame" and "title memory this
//! frame", and the frame line) name the side its work ran on.
//!
//! Switches read while a submission is prepared or copied back read the
//! side of the moment ([`flag`], [`value`]); each such read decides a whole
//! step (a batch, a range, a heap pass, a texture's refresh, a copy-back),
//! so a flip during another thread's submission leaves no step half in one
//! mode. Switches read while the title records a list ([`flag_at`],
//! [`value_at`]) read the side its title list took at its reset
//! (`crate::list`, `crate::host::HostList::ab`). Only switches whose two
//! sides can follow each other between frames are accepted ([`SWITCHES`]):
//! none that changes shader translation, a pipeline, how resources are laid
//! out or made, or the async set. A name that is not one of them is
//! refused with an error line, and nothing flips.

// The present that flips is only built for Windows.
#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// The switches a toggle may name, and `none`, which flips sides and
/// changes nothing (the A/A check).
pub(crate) const SWITCHES: [&str; 17] = [
    "VICTORY_SYNC_EPOCH",
    "VICTORY_FLUSH_OVERLAPPING",
    "VICTORY_COPY_BACK_SPLIT",
    "VICTORY_COPY_BACK_SPANS",
    "VICTORY_PARALLEL_COPIES",
    "VICTORY_TEXTURE_ONCE_PER_FRAME",
    "VICTORY_UNMAPPED_SPLIT",
    "VICTORY_DIRTY_ROWS",
    "VICTORY_SNAPSHOT_SPANS",
    "VICTORY_QUICK_SYNC",
    "VICTORY_PLACED_AT_ONCE",
    "VICTORY_BATCH_CLAMPS",
    "VICTORY_DISPATCH_CLAMPS",
    "VICTORY_COPY_ORDER",
    "VICTORY_WIDENED_IN_PLACE",
    "VICTORY_RAY_SKIP",
    NOTHING,
];

/// The name of the toggle that changes nothing.
pub(crate) const NOTHING: &str = "none";

/// Switches asked for often that a toggle cannot flip, and why.
const REFUSED: [(&str, &str); 6] = [
    (
        "VICTORY_ASYNC_",
        "the async set (cached by several dependents, a thread started once)",
    ),
    (
        "VICTORY_FLIGHTS",
        "the async set (cached by several dependents, a thread started once)",
    ),
    (
        "VICTORY_SHADER_OPT",
        "it changes shader translation (separate runs, in blocks)",
    ),
    (
        "VICTORY_UAV_SIZE_LEVEL0",
        "it changes shader translation (separate runs, in blocks)",
    ),
    (
        "VICTORY_HALVES_",
        "it changes shader translation (separate runs, in blocks)",
    ),
    (
        "VICTORY_BVH_",
        "it changes shader translation (separate runs, in blocks)",
    ),
];

/// The draws a frame runs on the host from which flipping starts when
/// `VICTORY_AB_AFTER` does not say: the world's first frame
/// (`crate::capture`, `replay::plain::WORLD_DRAWS`).
const WORLD_DRAWS: u64 = 500;

/// What `VICTORY_AB` asks for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Spec {
    /// The switch, with its `VICTORY_` prefix.
    pub(crate) name: String,
    /// Side B's value (`=<value B>`), else the opposite of side A.
    pub(crate) value: Option<String>,
    /// Seconds of wall time a side lasts.
    pub(crate) seconds: f64,
}

/// `VICTORY_AB`'s rule: `<SWITCH>[=<value B>]:<seconds>`; the `VICTORY_`
/// prefix may be left out, and the value may hold `:` (the seconds are
/// after the last). The switch must be one of [`SWITCHES`].
pub(crate) fn parse(text: &str) -> Result<Spec, String> {
    let text = text.trim();
    let (head, seconds) = text.rsplit_once(':').ok_or_else(|| {
        format!("VICTORY_AB={text}: no seconds (VICTORY_AB=<SWITCH>[=<value B>]:<seconds>)")
    })?;
    let seconds: f64 = seconds
        .trim()
        .parse()
        .ok()
        .filter(|s: &f64| s.is_finite() && *s >= 0.5)
        .ok_or_else(|| format!("VICTORY_AB={text}: the seconds must be a number of 0.5 or more"))?;
    let (name, value) = match head.split_once('=') {
        Some((name, value)) => (name.trim(), Some(value.trim().to_owned())),
        None => (head.trim(), None),
    };
    if name.is_empty() {
        return Err(format!("VICTORY_AB={text}: no switch"));
    }
    let name = if name == NOTHING || name.starts_with("VICTORY_") {
        name.to_owned()
    } else {
        format!("VICTORY_{name}")
    };
    if name == NOTHING && value.is_some() {
        return Err(format!("VICTORY_AB={text}: `{NOTHING}` takes no value"));
    }
    if !SWITCHES.contains(&name.as_str()) {
        let why = REFUSED
            .iter()
            .find(|(prefix, _)| name.starts_with(prefix))
            .map_or(
                "it is not one of the switches whose sides can follow each other between frames",
                |(_, why)| why,
            );
        return Err(format!(
            "VICTORY_AB={text}: {name} cannot be flipped while the game runs: {why}; nothing flips \
             (accepted: {})",
            SWITCHES.join(", ")
        ));
    }
    Ok(Spec {
        name,
        value,
        seconds,
    })
}

/// `VICTORY_AB_AFTER`'s rule: a number of draws, else [`WORLD_DRAWS`].
fn after(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(WORLD_DRAWS)
}

/// When the side flips: from the first frame that ran `after` draws, every
/// `seconds`.
#[derive(Debug)]
struct Clock {
    seconds: f64,
    after: u64,
    /// Since when the side has been what it is (none: not started).
    since: Option<f64>,
    /// Side B.
    b: bool,
}

impl Clock {
    fn new(seconds: f64, after: u64) -> Self {
        Clock {
            seconds,
            after,
            since: None,
            b: false,
        }
    }

    /// At a present `now` seconds into the run, of a frame that ran
    /// `draws` draws: the side from the next frame on, if it flips.
    fn step(&mut self, now: f64, draws: u64) -> Option<bool> {
        match self.since {
            None => {
                if draws >= self.after {
                    self.since = Some(now);
                }
                None
            }
            Some(since) if now - since >= self.seconds => {
                self.since = Some(now);
                self.b = !self.b;
                Some(self.b)
            }
            Some(_) => None,
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: a toggle and its side on this thread, whatever the
    /// environment says ([`force`]).
    static FORCED: std::cell::Cell<Option<(&'static Spec, bool)>> = const { std::cell::Cell::new(None) };
}

/// Tests: toggle `spec` on side `b` on this thread from now on (none: the
/// environment's again).
#[cfg(test)]
pub(crate) fn force(spec: Option<Spec>, b: bool) {
    FORCED.with(|f| f.set(spec.map(|s| (&*Box::leak(Box::new(s)), b))));
}

/// The toggle asked for, read once (none when unset or refused).
fn toggle() -> Option<&'static Spec> {
    #[cfg(test)]
    if let Some((spec, _)) = FORCED.with(std::cell::Cell::get) {
        return Some(spec);
    }
    static TOGGLE: OnceLock<Option<Spec>> = OnceLock::new();
    TOGGLE
        .get_or_init(|| {
            let text = std::env::var("VICTORY_AB")
                .ok()
                .filter(|t| !t.trim().is_empty())?;
            match parse(&text) {
                Ok(spec) => Some(spec),
                Err(why) => {
                    tracing::error!("{why}");
                    None
                }
            }
        })
        .as_ref()
}

/// The side work is done on now: B (true) once the toggle flipped to it.
static SIDE: AtomicBool = AtomicBool::new(false);
/// The side of the frame presented last, which its lines name.
static SHOWN: AtomicBool = AtomicBool::new(false);
/// The side flipped at the last present (the frame line ends its window).
static FLIPPED: AtomicBool = AtomicBool::new(false);
/// The toggle was refused at the first present: nothing flips.
static REFUSED_NOW: AtomicBool = AtomicBool::new(false);

/// The side now (false: A). Without a toggle always A: the one atomic
/// load a switch pays.
pub(crate) fn side() -> bool {
    #[cfg(test)]
    if let Some((_, b)) = FORCED.with(std::cell::Cell::get) {
        return b;
    }
    SIDE.load(Relaxed)
}

/// Side B's value of switch `name` when `b` (a side read before, see
/// [`side`]) is B and the toggle flips `name`: `Some(None)` for the
/// opposite of side A, `Some(Some(v))` for `=<v>`; `None` on side A.
pub(crate) fn at(name: &str, b: bool) -> Option<Option<&'static str>> {
    if !b {
        return None;
    }
    at_in(toggle(), name)
}

/// [`at`]'s rule on side B, for the toggle `spec`.
fn at_in<'a>(spec: Option<&'a Spec>, name: &str) -> Option<Option<&'a str>> {
    spec.filter(|t| t.name == name).map(|t| t.value.as_deref())
}

/// What a switch reads: `a` on side A (`over` none), else the `opposite`
/// of it, or `rule` of side B's value.
fn pick<T>(
    over: Option<Option<&str>>,
    a: T,
    opposite: impl FnOnce(&T) -> T,
    rule: impl FnOnce(&str) -> T,
) -> T {
    match over {
        None => a,
        Some(None) => opposite(&a),
        Some(Some(v)) => rule(v),
    }
}

/// Whether the toggle asked for names switch `name` (what is known only
/// while a part is on, such as `VICTORY_RAY_SKIP`'s programs, is then
/// learnt for both sides).
pub(crate) fn names(name: &str) -> bool {
    toggle().is_some_and(|t| t.name == name)
}

/// An on/off switch read now: `a` (side A, as the environment set it), or
/// on side B the opposite, or `rule` of side B's value.
pub(crate) fn flag(name: &str, a: bool, rule: impl FnOnce(&str) -> bool) -> bool {
    flag_at(name, side(), a, rule)
}

/// [`flag`] on side `b` (a title list's, taken at its reset).
pub(crate) fn flag_at(name: &str, b: bool, a: bool, rule: impl FnOnce(&str) -> bool) -> bool {
    pick(at(name, b), a, |a| !a, rule)
}

/// A switch with values read now: `a` on side A, else `opposite` of it or
/// `rule` of side B's value.
pub(crate) fn value<T>(
    name: &str,
    a: T,
    opposite: impl FnOnce(&T) -> T,
    rule: impl FnOnce(&str) -> T,
) -> T {
    value_at(name, side(), a, opposite, rule)
}

/// [`value`] on side `b`.
pub(crate) fn value_at<T>(
    name: &str,
    b: bool,
    a: T,
    opposite: impl FnOnce(&T) -> T,
    rule: impl FnOnce(&str) -> T,
) -> T {
    pick(at(name, b), a, opposite, rule)
}

/// The side the frame presented last ran on, for its lines; none without
/// a toggle (or a refused one).
pub(crate) fn label() -> Option<&'static str> {
    if toggle().is_none() || REFUSED_NOW.load(Relaxed) {
        return None;
    }
    Some(if SHOWN.load(Relaxed) { "B" } else { "A" })
}

/// Whether the side flipped at the last present, once (the frame line's
/// window ends there).
pub(crate) fn take_flip() -> bool {
    FLIPPED.load(Relaxed) && FLIPPED.swap(false, Relaxed)
}

/// At the present of `frame`, after what ran is finished: the frame's
/// side is noted for its lines, and the side may flip for the frames
/// after. `drawn`: the draws run on the host so far. `blocked`: why this
/// run cannot flip the switch (its side B value given), asked once at the
/// first present.
pub(crate) fn present(
    frame: u64,
    drawn: impl FnOnce() -> u64,
    blocked: impl FnOnce(&str, Option<&str>) -> Option<String>,
) {
    let Some(spec) = toggle() else { return };
    if REFUSED_NOW.load(Relaxed) {
        return;
    }
    let b = SIDE.load(Relaxed);
    SHOWN.store(b, Relaxed);
    struct Run {
        started: std::time::Instant,
        clock: Clock,
        drawn: Option<u64>,
    }
    static RUN: Mutex<Option<Run>> = Mutex::new(None);
    let mut guard = RUN.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        if let Some(why) = blocked(&spec.name, spec.value.as_deref()) {
            tracing::error!("VICTORY_AB: {why}; nothing flips");
            REFUSED_NOW.store(true, Relaxed);
            return;
        }
        let after = after(std::env::var("VICTORY_AB_AFTER").ok().as_deref());
        let set = std::env::var(&spec.name).ok();
        tracing::info!(
            switch = spec.name,
            seconds = spec.seconds,
            after,
            "VICTORY_AB: {} flips every {} s from the first frame of {after} draws or more: side A as the environment \
             sets it ({}), side B {}",
            spec.name,
            spec.seconds,
            set.as_deref()
                .map_or("unset".to_owned(), |v| format!("={v}")),
            spec.value
                .as_deref()
                .map_or("the opposite".to_owned(), |v| format!("={v}")),
        );
        if spec.value.is_some()
            && spec.value.as_deref().map(str::trim) == set.as_deref().map(str::trim)
        {
            tracing::warn!(
                "VICTORY_AB: side B's value is the one the environment sets: both sides are the same (an A/A run)"
            );
        }
        *guard = Some(Run {
            started: std::time::Instant::now(),
            clock: Clock::new(spec.seconds, after),
            drawn: None,
        });
    }
    let Some(run) = guard.as_mut() else { return };
    let now = drawn();
    // The first present counts nothing: its draws may be a whole loading's.
    let draws = run.drawn.map_or(0, |last| now.saturating_sub(last));
    run.drawn = Some(now);
    let armed = run.clock.since.is_some();
    let flip = run.clock.step(run.started.elapsed().as_secs_f64(), draws);
    if !armed && run.clock.since.is_some() {
        tracing::info!(
            frame,
            draws,
            "VICTORY_AB: started, side A for {} s",
            spec.seconds
        );
    }
    if let Some(b) = flip {
        SIDE.store(b, Relaxed);
        FLIPPED.store(true, Relaxed);
        let side = if b { "B" } else { "A" };
        tracing::info!(
            frame,
            side,
            switch = spec.name,
            "VICTORY_AB: the frames after this one run on side {side}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{Clock, NOTHING, SWITCHES, Spec, after, parse};

    fn spec(name: &str, value: Option<&str>, seconds: f64) -> Spec {
        Spec {
            name: name.to_owned(),
            value: value.map(str::to_owned),
            seconds,
        }
    }

    #[test]
    fn a_toggle_names_a_switch_and_its_seconds() {
        assert_eq!(
            parse("VICTORY_FLUSH_OVERLAPPING:10"),
            Ok(spec("VICTORY_FLUSH_OVERLAPPING", None, 10.0))
        );
        assert_eq!(
            parse(" FLUSH_OVERLAPPING : 7.5 "),
            Ok(spec("VICTORY_FLUSH_OVERLAPPING", None, 7.5)),
            "prefix left out"
        );
        assert_eq!(
            parse("VICTORY_PLACED_AT_ONCE=1:10"),
            Ok(spec("VICTORY_PLACED_AT_ONCE", Some("1"), 10.0))
        );
        assert_eq!(
            parse("VICTORY_COPY_ORDER=split:4096:12"),
            Ok(spec("VICTORY_COPY_ORDER", Some("split:4096"), 12.0)),
            "a value with a colon: the seconds are after the last"
        );
        assert_eq!(parse("none:10"), Ok(spec(NOTHING, None, 10.0)));
    }

    #[test]
    fn a_toggle_without_seconds_or_switch_is_refused() {
        for bad in [
            "VICTORY_FLUSH_OVERLAPPING",
            "VICTORY_FLUSH_OVERLAPPING:",
            "VICTORY_FLUSH_OVERLAPPING:x",
            ":10",
            "=1:10",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        for seconds in ["0", "0.2", "-3", "inf", "NaN"] {
            assert!(
                parse(&format!("VICTORY_DIRTY_ROWS:{seconds}")).is_err(),
                "{seconds}"
            );
        }
        assert!(parse("none=1:10").is_err(), "the A/A toggle takes no value");
    }

    #[test]
    fn only_switches_whose_sides_can_follow_each_other_are_taken() {
        for name in SWITCHES {
            assert!(parse(&format!("{name}:10")).is_ok(), "{name}");
        }
        let async_set = parse("VICTORY_ASYNC_SUBMIT:10").unwrap_err();
        assert!(
            async_set.contains("async set") && async_set.contains("nothing flips"),
            "{async_set}"
        );
        let shaders = parse("SHADER_OPT=0:10").unwrap_err();
        assert!(shaders.contains("shader translation"), "{shaders}");
        let other = parse("VICTORY_CENSUS_FAST:10").unwrap_err();
        assert!(
            other.contains("cannot be flipped") && other.contains("VICTORY_SYNC_EPOCH"),
            "{other}"
        );
        assert!(parse("VICTORY_GPU_SPLIT_UAV_SRV_PRELUDE:10").is_err());
    }

    #[test]
    fn flipping_starts_at_the_first_frame_with_enough_draws() {
        assert_eq!(after(None), 500);
        assert_eq!(after(Some(" 4000 ")), 4000);
        assert_eq!(after(Some("many")), 500);
        let mut clock = Clock::new(10.0, 500);
        // Menus: no flip however long they last.
        for t in 0..100 {
            assert_eq!(clock.step(f64::from(t), 120), None);
        }
        // The world's first frame starts side A's first 10 s.
        assert_eq!(clock.step(100.0, 4950), None);
        assert_eq!(
            clock.step(105.0, 10),
            None,
            "a later frame with few draws does not stop it"
        );
        assert_eq!(clock.step(109.9, 4950), None);
        assert_eq!(clock.step(110.0, 4950), Some(true), "side B after 10 s");
        assert_eq!(clock.step(115.0, 4950), None);
        assert_eq!(clock.step(120.3, 4950), Some(false), "side A again");
        // Each side lasts at least its seconds from the present it began at.
        assert_eq!(clock.step(130.2, 4950), None);
        assert_eq!(clock.step(130.4, 4950), Some(true));
    }

    #[test]
    fn a_long_frame_flips_once() {
        let mut clock = Clock::new(2.0, 0);
        assert_eq!(clock.step(0.0, 0), None);
        assert_eq!(
            clock.step(7.0, 0),
            Some(true),
            "one flip, however many periods the frame took"
        );
        assert_eq!(clock.step(8.0, 0), None);
        assert_eq!(clock.step(9.0, 0), Some(false));
    }

    #[test]
    fn side_b_is_the_opposite_or_the_value_given_and_only_for_the_named_switch() {
        use super::{at_in, pick};
        let opposite = spec("VICTORY_BATCH_CLAMPS", None, 10.0);
        let given = spec("VICTORY_BATCH_CLAMPS", Some("hoist"), 10.0);
        assert_eq!(at_in(Some(&opposite), "VICTORY_BATCH_CLAMPS"), Some(None));
        assert_eq!(
            at_in(Some(&given), "VICTORY_BATCH_CLAMPS"),
            Some(Some("hoist"))
        );
        assert_eq!(
            at_in(Some(&given), "VICTORY_DISPATCH_CLAMPS"),
            None,
            "another switch keeps side A"
        );
        assert_eq!(at_in(None, "VICTORY_BATCH_CLAMPS"), None);
        let rule = |v: &str| match v {
            "1" => [true, true],
            "hoist" => [true, false],
            _ => [false, false],
        };
        let flip = |a: &[bool; 2]| if a[0] { [false, false] } else { [true, true] };
        assert_eq!(
            pick(None, [true, true], flip, rule),
            [true, true],
            "side A: as set"
        );
        assert_eq!(pick(Some(None), [true, true], flip, rule), [false, false]);
        assert_eq!(pick(Some(None), [false, false], flip, rule), [true, true]);
        assert_eq!(
            pick(Some(Some("hoist")), [true, true], flip, rule),
            [true, false]
        );
        // An on/off switch: the opposite, or its own rule of the value.
        assert!(!pick(Some(None), true, |a| !a, |v| v == "1"));
        assert!(
            pick(Some(Some("1")), true, |a| !a, |v| v == "1"),
            "=1 on a switch set to 1: the same (A/A)"
        );
        let none = spec(NOTHING, None, 10.0);
        for name in SWITCHES {
            if name != NOTHING {
                assert_eq!(
                    at_in(Some(&none), name),
                    None,
                    "`none` changes no switch: {name}"
                );
            }
        }
    }

    #[test]
    fn without_a_toggle_every_switch_reads_side_a() {
        // The tests never set VICTORY_AB.
        if std::env::var_os("VICTORY_AB").is_some() {
            return;
        }
        assert!(!super::side());
        assert_eq!(
            super::at("VICTORY_DIRTY_ROWS", true),
            None,
            "no toggle: no side B even if asked"
        );
        assert!(!super::names("VICTORY_DIRTY_ROWS"));
        assert!(super::flag("VICTORY_DIRTY_ROWS", true, |_| false));
        assert!(!super::flag_at("VICTORY_DIRTY_ROWS", true, false, |_| true));
        assert_eq!(
            super::value_at("VICTORY_COPY_ORDER", true, Some(7), |_| None, |_| Some(1)),
            Some(7)
        );
        assert_eq!(super::label(), None);
        assert!(!super::take_flip());
        let mut asked = false;
        super::present(
            1,
            || 10_000,
            |_, _| {
                asked = true;
                None
            },
        );
        assert!(!asked && !super::side(), "nothing runs at the present");
    }
}
