//! mac: `VICTORY_DEPTH_TOUCH_LOG=1` (off by default), log only.
//!
//! Three console list calls the host never sees could change a depth
//! target between the title's letterbox bar fill and its depth of field
//! (`notes/picture-next.md`, "Why the bar fill writes depth", candidate 1):
//! `DiscardResource` (0x1b8), the unknown `(u64)` method 0x470 of the state
//! flush, and the pixel shader depth order slot. They are ignored before the
//! host, so a capture does not hold them. With the switch on, each is logged
//! once per presented frame per (call, resource): a discard only when its
//! resource is a depth texture or not one of ours; the other two always, with
//! their raw arguments, since what they name is not known. The resource is
//! named as captures name it (`#N`) while recording, and always by its
//! description, title object and host texture address (`@0x...`, the host
//! texture's name). Nothing else changes; off, each call is one cached load.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use crate::object::This;

const NAME: &str = "VICTORY_DEPTH_TOUCH_LOG";

/// At most this many lines per call kind and frame; a frame with more logs
/// one line saying so.
const PER_FRAME: usize = 64;

/// Whether the switch is on, read once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on(NAME);
        if on {
            tracing::info!(
                "{NAME}=1: DiscardResource of depth targets, list method 0x470 and the pixel shader depth order \
                 calls are logged once per frame per resource; nothing else changes"
            );
        }
        on
    })
}

/// The calls logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    Discard,
    Method470,
    DepthOrder,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Discard => "DiscardResource (0x1b8)",
            Kind::Method470 => "method 0x470",
            Kind::DepthOrder => "pixel shader depth order",
        }
    }
}

/// What to do with a call.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// First time this frame: log it.
    Log,
    /// The kind's lines for this frame are used up: say so once.
    Full,
    /// Logged already this frame, or after [`Verdict::Full`].
    Quiet,
}

/// The (kind, key) pairs seen in the current frame.
#[derive(Default)]
struct Seen {
    frame: u64,
    keys: HashSet<(Kind, u64)>,
    per_kind: HashMap<Kind, usize>,
}

impl Seen {
    fn verdict(&mut self, frame: u64, kind: Kind, key: u64) -> Verdict {
        if frame != self.frame {
            self.frame = frame;
            self.keys.clear();
            self.per_kind.clear();
        }
        if !self.keys.insert((kind, key)) {
            return Verdict::Quiet;
        }
        let n = self.per_kind.entry(kind).or_default();
        *n += 1;
        match (*n).cmp(&PER_FRAME) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => Verdict::Log,
            std::cmp::Ordering::Greater if *n == PER_FRAME + 1 => Verdict::Full,
            std::cmp::Ordering::Greater => Verdict::Quiet,
        }
    }
}

fn verdict(kind: Kind, key: u64) -> (u64, Verdict) {
    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);
    let frame = crate::queue::presented();
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    (
        frame,
        seen.get_or_insert_with(Seen::default)
            .verdict(frame, kind, key),
    )
}

/// A title resource as the log names it, and whether it is a depth texture;
/// `None` if `p` is not one of our resources.
#[cfg(windows)]
fn describe(p: This) -> Option<(String, bool)> {
    use windows::core::Interface;
    if !crate::resource::is_resource(p) {
        return None;
    }
    let (dimension, width, height, format, flags) = crate::resource::brief(p);
    let host = crate::resource::made_copy(p).map(|h| h.texture.as_raw() as usize);
    let id = host.and_then(crate::replay::record::id_of);
    let mut text = String::new();
    if let Some(id) = id {
        text.push_str(&format!("#{id} "));
    }
    text.push_str(&format!(
        "dim {dimension} {width}x{height} f{format} flags {flags:#x} title {:#x}",
        p as usize
    ));
    match host {
        Some(at) => text.push_str(&format!(" host @{at:#x}")),
        None => text.push_str(" host none"),
    }
    // D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
    Some((text, flags & 0x2 != 0))
}

#[cfg(not(windows))]
fn describe(_: This) -> Option<(String, bool)> {
    None
}

fn log(kind: Kind, key: u64, what: impl FnOnce() -> String) {
    let (frame, verdict) = verdict(kind, key);
    match verdict {
        Verdict::Log => tracing::info!(frame, call = kind.name(), "{NAME}: {}", what()),
        Verdict::Full => tracing::info!(
            frame,
            call = kind.name(),
            "{NAME}: more than {PER_FRAME} this frame; the rest of this frame not logged"
        ),
        Verdict::Quiet => {}
    }
}

/// `DiscardResource(resource, region)`: logged when the resource is a depth
/// texture or not one we know.
pub(crate) fn discard(list: This, resource: This, region: *const u32) {
    let described = describe(resource);
    if described.as_ref().is_some_and(|(_, depth)| !depth) {
        return;
    }
    // D3D12_DISCARD_REGION: NumRects, pRects, FirstSubresource, NumSubresources.
    let region = (!region.is_null()).then(|| {
        // SAFETY: the title passes a whole region when it passes one.
        unsafe {
            let rects = region.read_unaligned();
            let subs = region
                .cast::<u8>()
                .add(16)
                .cast::<[u32; 2]>()
                .read_unaligned();
            (rects, subs)
        }
    });
    log(Kind::Discard, resource as u64, || {
        let what = described.map_or_else(
            || format!("unknown object {:#x}", resource as usize),
            |(t, _)| t,
        );
        let region = region.map_or_else(
            || "whole resource".to_owned(),
            |(rects, [first, n])| format!("{rects} rects, subresources {first}+{n}"),
        );
        format!("{what}; {region}; list {:#x}", list as usize)
    });
}

/// A call whose arguments are not known (`kind` 0x470 or the depth order):
/// its raw words, and the resource the first one names if it names one.
pub(crate) fn raw(kind: Kind, list: This, first: u64, second: u64) {
    log(kind, first, || {
        let named =
            describe(first as usize as This).map_or_else(String::new, |(t, _)| format!(" = {t}"));
        format!(
            "arg0 {first:#x}{named}, arg1 {second:#x}; list {:#x}",
            list as usize
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_per_frame_per_kind_and_key() {
        let mut s = Seen::default();
        assert_eq!(s.verdict(5, Kind::Discard, 240), Verdict::Log);
        assert_eq!(s.verdict(5, Kind::Discard, 240), Verdict::Quiet);
        assert_eq!(s.verdict(5, Kind::Discard, 241), Verdict::Log);
        // Same key, another kind.
        assert_eq!(s.verdict(5, Kind::Method470, 240), Verdict::Log);
        assert_eq!(s.verdict(5, Kind::DepthOrder, 240), Verdict::Log);
        // Next frame: again.
        assert_eq!(s.verdict(6, Kind::Discard, 240), Verdict::Log);
        assert_eq!(s.verdict(6, Kind::Discard, 240), Verdict::Quiet);
    }

    #[test]
    fn a_full_frame_says_so_once() {
        let mut s = Seen::default();
        for key in 0..PER_FRAME as u64 {
            assert_eq!(s.verdict(1, Kind::Method470, key), Verdict::Log);
        }
        assert_eq!(s.verdict(1, Kind::Method470, 1000), Verdict::Full);
        assert_eq!(s.verdict(1, Kind::Method470, 1001), Verdict::Quiet);
        // Other kinds have their own lines.
        assert_eq!(s.verdict(1, Kind::Discard, 1000), Verdict::Log);
        assert_eq!(s.verdict(2, Kind::Method470, 1000), Verdict::Log);
    }
}
