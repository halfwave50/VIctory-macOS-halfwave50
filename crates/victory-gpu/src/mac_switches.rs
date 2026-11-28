//! The Mac line's translation switches (`victory_shader::emit::Options`,
//! `lower::mac` there): read once from the environment, and part of the
//! shader cache key ([`key`], used by `shader_cache`), so translations made
//! with and without one never mix. The rules are here, outside the
//! Windows-only shader cache, so that their tests run on any host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::OnceLock;

/// Whether pixel programs that ask for the depth test before the shader
/// get DXIL's early depth-stencil flag
/// (`victory_shader::emit::Options::early_depth`): by default under Wine
/// (`wine`), where the title's see-through pre-pass is the cost of a frame
/// and hidden fragments ran it. `VICTORY_EARLY_DEPTH=0` or `1` overrides.
pub(crate) fn early_depth(wine: bool) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_EARLY_DEPTH").ok().as_deref(), wine);
        tracing::info!(
            on,
            "VICTORY_EARLY_DEPTH: pixel programs that ask for the depth test first get the early depth-stencil flag"
        );
        on
    })
}

/// Which lanes stay out of the lock loop of a 64-bit image atomic
/// (`victory_shader::emit::Options::atomic_lock_helpers` and
/// `atomic_lock_keys`; they act only where atomic locks are on): by default
/// under Wine (`wine`) both the helper lanes of pixel programs and the lanes
/// whose `umax` key is below the texel's. `VICTORY_ATOMIC_LOCK_WRITERS=0`:
/// every lane takes the lock, as before; `helpers`: only helper lanes stay
/// out; `1`: both, anywhere.
pub(crate) fn lock_writers(wine: bool) -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_ATOMIC_LOCK_WRITERS").ok();
        let [helpers, keys] = writers(value.as_deref(), wine);
        tracing::info!(
            helpers,
            keys,
            "VICTORY_ATOMIC_LOCK_WRITERS: lanes that cannot change the texel stay out of the lock loop of 64-bit image atomics"
        );
        [helpers, keys]
    })
}

/// Which parts of the quicker heap sync are on (`crate::descriptor::sync_heaps`
/// makes 25,000 to 40,000 host descriptor slots again in a world frame, on
/// the submitting thread): the heap census in hash maps
/// (`crate::census_maps`), and buffer shader views made from one lookup
/// (`crate::descriptor::host`, "quick"). By default both under Wine (`wine`
/// says whether; asked once), where a frame's CPU part is the cost.
/// `VICTORY_QUICK_SYNC=0`: neither, as before; `maps` or `views`: that part
/// alone; `1`: both, anywhere. Neither changes what a slot shows.
///
/// mac: `VICTORY_AB=VICTORY_QUICK_SYNC` (`crate::ab`) flips the views part
/// only (read once a heap sync pass); the maps are how a census is made.
pub(crate) fn quick_sync(wine: impl Fn() -> bool) -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    let [maps, views] = *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_QUICK_SYNC").ok();
        let [maps, views] = quick(value.as_deref(), wine());
        tracing::info!(
            maps,
            views,
            "VICTORY_QUICK_SYNC: the heap census in hash maps, buffer shader views of a heap sync made from one lookup"
        );
        [maps, views]
    });
    [maps, crate::ab::flag("VICTORY_QUICK_SYNC", views, |v| quick(Some(v), wine())[1])]
}

/// [`quick_sync`]'s rule: census maps, views.
fn quick(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("maps") => [true, false],
        Some("views") => [false, true],
        value => [setting(value, default); 2],
    }
}

/// Whether a compute group of one console wave that shares LDS in a way
/// barriers at its common blocks cannot order by themselves gets the finer
/// order of `victory_shader::emit::lower::lds_order`
/// (`victory_shader::emit::Options::lds_order`; `notes/cross-half-programs.md`,
/// "Shared LDS"): only with `VICTORY_HALVES_LDS_ORDER=1`. Exact where it
/// applies; a program it cannot order stays refused, or runs unsynced with
/// `VICTORY_HALVES_SHARED_LDS=1`.
pub(crate) fn lds_order() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_LDS_ORDER").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_LDS_ORDER=1: one-wave64 groups that share LDS get group barriers where every lane runs, when that orders the LDS exactly"
            );
        }
        on
    })
}

/// Whether a lane read across the two 32-lane halves of a console wave64,
/// in a block every lane runs once, is handed over through group-shared
/// memory at a group barrier (`victory_shader::emit::lower::exchange`,
/// `victory_shader::emit::Options::halves_exchange`;
/// `notes/ray-tracing-plan.md`): only with `VICTORY_HALVES_EXCHANGE=1`.
/// Exact where it applies; a read it does not cover stays refused, or is
/// answered by the reader's half with `VICTORY_HALVES_LOCAL=1`.
pub(crate) fn exchange() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_EXCHANGE").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_EXCHANGE=1: lane reads across 32-lane halves are handed over through group-shared memory where every lane runs"
            );
        }
        on
    })
}

/// Whether, in a program in halves with half-local reads
/// (`VICTORY_HALVES_LOCAL=1`), a read across the halves is handed over the
/// way VIctory-macOS does it (its #309, #360 and #376;
/// `victory_shader::emit::Options::halves_across`) before it is answered
/// from the reader's own half: only with `VICTORY_HALVES_ACROSS=1`. This
/// line's own exchange (`VICTORY_HALVES_EXCHANGE=1`) still comes first.
/// Unset, a translation with half-local reads is what it was before that
/// hand-over was merged; without half-local reads the hand-over is always
/// made, as in VIctory-macOS, where the read would otherwise be refused.
pub(crate) fn across() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_ACROSS").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_ACROSS=1: reads across 32-lane halves are handed over as in VIctory-macOS where every lane of the group runs together, before half-local reads answer them"
            );
        }
        on
    })
}

/// Whether a compute group of one console wave that shares LDS across the
/// rounds of a loop, with lanes that leave the program before it, gets
/// group barriers where every thread that has not ended meets
/// (`victory_shader::emit::lower::lds_rounds`,
/// `victory_shader::emit::Options::lds_rounds`; `notes/replay-determinism.md`,
/// "HairOnly, 27 November"): only with `VICTORY_HALVES_LDS_ROUNDS=1`. The
/// title's hair shadow masks (`0a5ecfee6b79e13b`) then no longer race
/// between the halves. Exact for the LDS where it applies; a program it
/// does not order stays as it was (refused, or raced with
/// `VICTORY_HALVES_SHARED_LDS=1`). It rests on a thread that has ended not
/// holding up a group barrier, measured on D3DMetal.
pub(crate) fn lds_rounds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_LDS_ROUNDS").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_LDS_ROUNDS=1: one-wave64 groups that share LDS across loop rounds get group barriers where every thread that has not ended meets, when that orders the LDS exactly"
            );
        }
        on
    })
}

/// Whether the halves' walk also opens regions holding whole-wave code, and
/// a first-lane read under a narrowed EXEC is handed over across the halves
/// (`victory_shader::emit::Options::halves_whole_wave`; `notes/far-ray.md`):
/// only with `VICTORY_HALVES_WHOLE_WAVE=1`, and only where
/// `VICTORY_HALVES_EXCHANGE=1` and `VICTORY_HALVES_WALK=1` are on. The
/// title's ray bin offsets (`sr_tracing:CS_GenerateRayBinOffsets`,
/// `9d1d228b02e56a96`) and shader bin offsets
/// (`shader_binning_utils:CS_GenerateShaderBinOffsets`, `8dd8d739047fe557`)
/// then count over the whole console wave instead of each half alone, and
/// one-wave groups whose LDS raced between the halves behind an early-out
/// or `if` (`sr_tracing:CS_RayBudgeting_Cull_RayBinning`, `03e3e26783a5799b`,
/// the ray pre-passes, particle emitters, reductions) get barriers instead.
/// Exact where it applies; a program it does not make exact stays as it was.
pub(crate) fn whole_wave() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_WHOLE_WAVE").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_WHOLE_WAVE=1: the halves' walk opens regions with whole-wave code, and first-lane reads under a narrowed EXEC are handed over, where that is exact"
            );
        }
        on
    })
}

/// Whether the program `container` keeps the LDS part of
/// `VICTORY_HALVES_WHOLE_WAVE=1` (the walk opening the code round unsynced
/// one-wave LDS, which barriers then order; `notes/far-ray.md`):
/// `VICTORY_HALVES_WHOLE_WAVE_LDS` unset or empty is every program (as
/// before the list came), `0` none, a comma list of 16-digit shader hashes
/// those programs only. The list to test (28 November) is the ray chain:
/// `03e3e26783a5799b` (`CS_RayBudgeting_Cull_RayBinning`) and the six
/// `CS_RayPrePass_*` (`fd8393c7f4e0e3da`, `7168dd781ca7efb9`,
/// `d2b169b82f67ec5b`, `edb8e66237fc38e6`, `049459e0c78e0d50`,
/// `11276d0c8bbe180a`): runs with the whole part showed a growing GPU cost
/// in three of five boat runs.
pub(crate) fn whole_wave_lds_for(container: &[u8]) -> bool {
    match whole_wave_lds_rule() {
        None => true,
        Some(only) => only.contains(&blake3::hash(container).to_hex()[..16].to_owned()),
    }
}

fn whole_wave_lds_rule() -> &'static Option<Vec<String>> {
    static RULE: OnceLock<Option<Vec<String>>> = OnceLock::new();
    RULE.get_or_init(|| {
        let rule = whole_wave_lds(std::env::var("VICTORY_HALVES_WHOLE_WAVE_LDS").ok().as_deref());
        if let Some(only) = &rule {
            tracing::info!(
                programs = ?only,
                "VICTORY_HALVES_WHOLE_WAVE_LDS: the whole-wave walk orders one-wave LDS in these programs only"
            );
        }
        rule
    })
}

/// `VICTORY_HALVES_WHOLE_WAVE_LDS`'s rule: `None` is every program (unset,
/// empty or unreadable: today's behaviour), `Some` the programs listed
/// (`0`: none).
fn whole_wave_lds(value: Option<&str>) -> Option<Vec<String>> {
    let value = value.unwrap_or_default().trim();
    if value.is_empty() {
        return None;
    }
    if value == "0" {
        return Some(Vec::new());
    }
    let hashes: Vec<String> = value.split(',').map(|h| h.trim().to_ascii_lowercase()).collect();
    hashes
        .iter()
        .all(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        .then_some(hashes)
}

/// Whether a size query on a writable image asks for level 0 in every
/// program (`victory_shader::emit::Options::uav_size_level0`): only with
/// `VICTORY_UAV_SIZE_LEVEL0=1`. D3DMetal answers the undefined level DXC
/// writes with a size of 0 (measured: `uav_size_tests` in `victory-shader`),
/// so the title's 35 programs that compute with such a size are wrong
/// without it (`notes/uav-size.md`). Off by default until the picture with
/// it is checked. A list of shader hashes instead of `1` turns it on for
/// those programs only ([`uav_size_level0_for`]).
pub(crate) fn uav_size_level0() -> bool {
    uav_size_rule().0
}

/// Whether the program in `container` gets the level-0 size query: with
/// `VICTORY_UAV_SIZE_LEVEL0=1`, or when the variable lists its hash (16
/// digits, as `shader translated` lines print it; several with commas), to
/// try the change on one program at a time.
pub(crate) fn uav_size_level0_for(container: &[u8]) -> bool {
    let (all, only) = uav_size_rule();
    *all || (!only.is_empty() && only.contains(&blake3::hash(container).to_hex()[..16].to_owned()))
}

fn uav_size_rule() -> &'static (bool, Vec<String>) {
    static RULE: OnceLock<(bool, Vec<String>)> = OnceLock::new();
    RULE.get_or_init(|| {
        let rule = uav_size(std::env::var("VICTORY_UAV_SIZE_LEVEL0").ok().as_deref());
        if rule.0 {
            tracing::info!("VICTORY_UAV_SIZE_LEVEL0=1: size queries on writable images ask for level 0");
        } else if !rule.1.is_empty() {
            tracing::info!(
                programs = ?rule.1,
                "VICTORY_UAV_SIZE_LEVEL0: size queries on writable images ask for level 0 in these programs only"
            );
        }
        rule
    })
}

/// `VICTORY_UAV_SIZE_LEVEL0`'s rule: `1` is every program; a list of
/// 16-digit shader hashes is those programs; anything else is off.
fn uav_size(value: Option<&str>) -> (bool, Vec<String>) {
    let value = value.unwrap_or_default().trim();
    if value == "1" {
        return (true, Vec::new());
    }
    let hashes: Vec<String> = value.split(',').map(|h| h.trim().to_ascii_lowercase()).collect();
    if hashes.iter().all(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit())) {
        (false, hashes)
    } else {
        (false, Vec::new())
    }
}

/// Whether a console wave64 program whose hand-overs across the halves
/// sit in code only some lanes run is translated with those skips left
/// out, every lane walking the skipped code with its EXEC bit clear
/// (`victory_shader::emit::walk`, `victory_shader::emit::Options::halves_walk`;
/// `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs"): only with
/// `VICTORY_HALVES_WALK=1`, and only together with
/// `VICTORY_HALVES_EXCHANGE=1` (alone it changes nothing, and is then left
/// out of the cache key too). Exact where it applies; a program it does
/// not make exact is translated as without it.
pub(crate) fn walk() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_WALK").ok().as_deref(), false);
        if on && exchange() {
            tracing::info!(
                "VICTORY_HALVES_WALK=1: every lane walks the skips around a hand-over across 32-lane halves, where that makes the program exact"
            );
        } else if on {
            tracing::warn!("VICTORY_HALVES_WALK=1 does nothing without VICTORY_HALVES_EXCHANGE=1");
        }
        on && exchange()
    })
}

/// Whether a title copy recorded in a list keeps its order with the
/// commands before it that read what it changes, and the longest read
/// range looked at: only with `VICTORY_COPY_ORDER=split` (every range) or
/// `split:<bytes>` (ranges of at most so many bytes).
///
/// On the console a list's copies (`CopyBufferRegion`, buffer
/// `CopyResource`, `CopyMemoryToMemoryX`) and its draws and dispatches run
/// in their order. Here every copy of a host submission is done on the CPU
/// before its first GPU command, into the one host copy of each buffer, so
/// a command recorded before such a copy ran with the bytes meant for the
/// commands after it (the title's TAA passes worked on a quarter of the
/// screen, `notes/uav-size.md`, "Copy order"). With the switch the title
/// list is cut at such a copy (`crate::list`) and a queue's piece is split
/// there into two host submissions (`crate::host::copy_order_parts`): the
/// first has run on the GPU before the copy is made. `0`, `keep` and
/// anything else leave it off (the diagnostics of `crate::host` read the
/// same variable).
///
/// This is side A of `VICTORY_AB` (`crate::ab`); what a list records reads
/// its title list's side ([`copy_order_split_at`]).
pub(crate) fn copy_order_split() -> Option<usize> {
    #[cfg(test)]
    if tests::FORCED_COPY_ORDER.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        return Some(usize::MAX);
    }
    static RULE: OnceLock<Option<usize>> = OnceLock::new();
    *RULE.get_or_init(|| {
        let rule = copy_order(std::env::var("VICTORY_COPY_ORDER").ok().as_deref());
        match rule {
            Some(usize::MAX) => tracing::info!(
                "VICTORY_COPY_ORDER=split: a title copy that changes what earlier commands of its submission read starts a new host submission"
            ),
            Some(bytes) => tracing::info!(
                bytes,
                "VICTORY_COPY_ORDER=split:<bytes>: a title copy that changes a range of at most so many bytes that earlier commands of its submission read starts a new host submission"
            ),
            None => {}
        }
        rule
    })
}

/// mac: [`copy_order_split`] on `VICTORY_AB` side `b` (`crate::ab`): side
/// B is off where A splits, else every range, or `=<value>` by the rule.
pub(crate) fn copy_order_split_at(b: bool) -> Option<usize> {
    let a = copy_order_split();
    crate::ab::value_at(
        "VICTORY_COPY_ORDER",
        b,
        a,
        |a| if a.is_some() { None } else { Some(usize::MAX) },
        |v| copy_order(Some(v)),
    )
}

/// `VICTORY_COPY_ORDER`'s rule for the split: `split` is every read range,
/// `split:<bytes>` those of at most `bytes` (1 or more); anything else is
/// off.
fn copy_order(value: Option<&str>) -> Option<usize> {
    match value?.trim().strip_prefix("split")? {
        "" => Some(usize::MAX),
        bound => bound.strip_prefix(':')?.parse().ok().filter(|&bytes| bytes > 0),
    }
}

/// How the clamps of indirect draws are recorded (`crate::indirect`):
/// whether the clamps of a run of indirect draws go to the host list before
/// the one that draws, and whether a run's clamps are then one dispatch.
/// Both only with `VICTORY_BATCH_CLAMPS=1`; `hoist` moves the clamps and
/// leaves each a dispatch of its own (to tell the two steps apart in a
/// run). Off by default: each clamp is a dispatch of ours just before its
/// draw, about 4,200 a world frame, each between two draws.
///
/// This is side A of `VICTORY_AB` (`crate::ab`); lists read their title
/// list's side ([`batch_clamps_at`]).
pub(crate) fn batch_clamps() -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let [hoist, batch] = clamps(std::env::var("VICTORY_BATCH_CLAMPS").ok().as_deref());
        if hoist {
            tracing::info!(
                batch,
                "VICTORY_BATCH_CLAMPS: the clamps of a run of indirect draws go to the host list before it (batch: as one dispatch a run)"
            );
        }
        [hoist, batch]
    })
}

/// mac: [`batch_clamps`] on `VICTORY_AB` side `b` (`crate::ab`), a title
/// list's from its reset: side B is off where A hoists, else both, or
/// `=<value>` by the rule.
pub(crate) fn batch_clamps_at(b: bool) -> [bool; 2] {
    crate::ab::value_at(
        "VICTORY_BATCH_CLAMPS",
        b,
        batch_clamps(),
        |a| if a[0] { [false, false] } else { [true, true] },
        |v| clamps(Some(v)),
    )
}

/// `VICTORY_BATCH_CLAMPS`'s rule: hoisted, batched. `1` is both, `hoist`
/// the first alone; anything else is off.
fn clamps(value: Option<&str>) -> [bool; 2] {
    match value.map(str::trim) {
        Some("1") => [true, true],
        Some("hoist") => [true, false],
        _ => [false, false],
    }
}

/// Whether the clamps of indirect dispatches go to the host list before
/// too, as those of draws do with `VICTORY_BATCH_CLAMPS` (`crate::indirect`,
/// `crate::list`): `VICTORY_DISPATCH_CLAMPS=1`, and only where
/// `VICTORY_BATCH_CLAMPS` hoists (`1` batches them, `hoist` leaves each a
/// dispatch of its own). A run of them ends at the title's next barrier,
/// pipeline sync, copy, clear or immediate write, not at its dispatches.
/// Off by default: each indirect dispatch has a clamp of ours just before
/// it, about 960 a frame at the prison.
///
/// mac: on `VICTORY_AB` side `b` (`crate::ab`; false is the switch as set),
/// a title list's from its reset: asked for on that side (the opposite of
/// side A's asking, or `=<value>`), and only where that side hoists
/// ([`batch_clamps_at`]: a toggle of the draws' clamps takes these along).
pub(crate) fn dispatch_clamps_at(b: bool) -> bool {
    static ASKED: OnceLock<bool> = OnceLock::new();
    let asked = *ASKED.get_or_init(|| {
        let asked = dispatch_rule(std::env::var("VICTORY_DISPATCH_CLAMPS").ok().as_deref());
        if asked && batch_clamps()[0] {
            tracing::info!(
                batch = batch_clamps()[1],
                "VICTORY_DISPATCH_CLAMPS: the clamps of a run of indirect dispatches go to the host list before it too (a run ends at the title's next barrier)"
            );
        } else if asked {
            tracing::warn!("VICTORY_DISPATCH_CLAMPS=1 does nothing without VICTORY_BATCH_CLAMPS=1 or hoist");
        }
        asked
    });
    crate::ab::flag_at("VICTORY_DISPATCH_CLAMPS", b, asked, |v| dispatch_rule(Some(v))) && batch_clamps_at(b)[0]
}

/// Whether every BVH node test misses without reading the node
/// (`victory_shader::emit::Options::bvh_miss`): `VICTORY_BVH_MISS=1`, as in
/// trustno1-coder 24a479d3. Off by default. This line holds the structure
/// builder back, so the title's reflection rays walk memory that was never
/// built: at the boat, whole 8x8 tiles of the reflections came back dark
/// once the denoiser's history reached the lighting (`VICTORY_PLACED_TWIN`;
/// run 08-32-25 of 27 November without the ray binning pass had none).
pub(crate) fn bvh_miss() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_BVH_MISS").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("VICTORY_BVH_MISS=1: every BVH node test misses without reading the node (the title's rays find nothing)");
        }
        on
    })
}

/// Whether a vector buffer load through a descriptor a program builds
/// around its own constant table reads the table
/// (`victory_shader::emit::Options::code_table_loads`):
/// `VICTORY_CODE_TABLE_LOADS=1`, from LakshanSivakumar/Native-Victory-macOS
/// 578eb743. Off by default: such a load is a missing value, as before.
/// There it left the display LUT one value and the world flat grey.
pub(crate) fn code_table_loads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_CODE_TABLE_LOADS").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("VICTORY_CODE_TABLE_LOADS=1: buffer loads through a descriptor around the code's constant table read the table");
        }
        on
    })
}

/// Whether a read of title memory that crosses onto the next 64 KiB page
/// reads that page's copy (`victory_shader::emit::Options::cross_page_reads`):
/// `VICTORY_CROSS_PAGE_READS=1`, from VIctory-macOS PR #595 (5311f00c, open
/// when taken). Off by default: the dwords past the first page then read
/// through its copy, 0 where another copy follows, as before.
pub(crate) fn cross_page_reads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_CROSS_PAGE_READS").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("VICTORY_CROSS_PAGE_READS=1: a title-memory read that crosses onto the next page reads that page's copy");
        }
        on
    })
}

/// Whether the dwords of one console load are read four a call
/// (`victory_shader::emit::Options::wide_loads`): bit 64 of
/// `VICTORY_SHADER_OPT`, as in 64bitMango 759c4d6b and
/// fxceless/victory-fork (a number, decimal or `0x` hex; its other bits are
/// the fork's and not read here). Off by default. 64bitMango measured 19
/// to 29 ms less GPU time of about 150 in a recorded frame on the same Wine
/// and D3DMetal route.
pub(crate) fn wide_loads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = shader_opt_wide(std::env::var("VICTORY_SHADER_OPT").ok().as_deref());
        if on {
            tracing::info!("VICTORY_SHADER_OPT=64: the dwords of one console load are read four a call where that is exact");
        }
        on
    })
}

/// `VICTORY_SHADER_OPT`'s bit 64; off when unset or unreadable.
fn shader_opt_wide(value: Option<&str>) -> bool {
    let bits = value.map(str::trim).and_then(|v| match v.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => v.parse().ok(),
    });
    bits.is_some_and(|b: u32| b & 64 != 0)
}

#[cfg(test)]
mod wide_load_tests {
    use super::shader_opt_wide;

    #[test]
    fn the_shader_opt_bit_is_read_or_off() {
        assert!(!shader_opt_wide(None));
        assert!(!shader_opt_wide(Some("0")));
        assert!(shader_opt_wide(Some(" 64 ")));
        assert!(shader_opt_wide(Some("0x40")));
        assert!(shader_opt_wide(Some("127")));
        assert!(!shader_opt_wide(Some("31")), "the fork's other bits are not this one");
        assert!(!shader_opt_wide(Some("on")));
    }
}

/// `VICTORY_DISPATCH_CLAMPS`'s rule: on only for `1`.
fn dispatch_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// Whether view heaps' censuses (`crate::bindless::Viewed`) take their lock
/// less (`crate::census_fast`): the ranges a heap's unordered views write
/// are kept apart and their answer kept until they change, so a list's
/// heap-writing draws no longer walk every range of the census under its
/// lock at each barrier, and a heap sync enters the slots of a pass a batch
/// at a time, not one lock a slot. Only with `VICTORY_CENSUS_FAST=1`, read
/// once (at the first view heap). What a slot shows, what a submission
/// refreshes and what widened views get stay as they are.
pub(crate) fn census_fast() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = census_rule(std::env::var("VICTORY_CENSUS_FAST").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_CENSUS_FAST=1: view heap censuses keep their unordered ranges apart with a kept answer, and heap syncs enter their slots in batches"
            );
        }
        on
    })
}

/// `VICTORY_CENSUS_FAST`'s rule: on only for `1`.
fn census_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// From ScarlettMac 4921b7e6: whether a submission's refresh of a buffer
/// range none of whose pages the title wrote since the host copy took them
/// is left out (`VICTORY_SYNC_EPOCH=1`, `crate::shadow::Shadow::synced`):
/// `Shadow::update` would copy nothing there, and asking costs the copy's
/// locks and a place in the submission's batch. Off (unset, `0`, or
/// anything else): every request is refreshed, as before. Nothing with
/// `VICTORY_FLIGHTS=1`, where refreshes are asked for while the submission
/// before still runs and its copy-back can still change title memory.
///
/// mac: `VICTORY_AB=VICTORY_SYNC_EPOCH` (`crate::ab`) flips it per range
/// asked; refused with `VICTORY_FLIGHTS=1` ([`ab_blocked`]).
pub(crate) fn sync_epoch(flights: impl FnOnce() -> bool) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let a = *ON.get_or_init(|| {
        let asked = epoch(std::env::var("VICTORY_SYNC_EPOCH").ok().as_deref());
        let on = asked && !flights();
        if on {
            tracing::info!(
                "VICTORY_SYNC_EPOCH=1: a refresh of a buffer range no page of changed since the copy took it is left out"
            );
        } else if asked {
            tracing::warn!("VICTORY_SYNC_EPOCH=1 does nothing with VICTORY_FLIGHTS=1");
        }
        on
    });
    crate::ab::flag("VICTORY_SYNC_EPOCH", a, |v| epoch(Some(v)))
}

/// [`sync_epoch`]'s rule: `1` turns it on; every other value leaves it off.
fn epoch(value: Option<&str>) -> bool {
    setting(value, false)
}

/// From ScarlettMac 4921b7e6: `VICTORY_TEXTURE_WORKERS`'s rule
/// (`crate::texture`'s worker threads): a count of 2 or more, at most the
/// CPUs, asks for that many; anything else (and unset) leaves the quarter
/// of the CPUs, four at most, as before. `None` where that is fewer than
/// two: the work runs on the submitting thread.
pub(crate) fn texture_workers(value: Option<&str>, cpus: usize) -> Option<usize> {
    let threads = match value.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(asked) if asked >= 2 => asked.min(cpus.max(1)),
        _ => (cpus / 4).clamp(1, 4),
    };
    (threads >= 2).then_some(threads)
}

/// Whether a heap sync pass also holds the census updates of the slots
/// `HostHeap::mirror` and `HostHeap::make_null` make in it, and enters them
/// a batch at a time (`crate::bindless::Pass`, `crate::census_fast::Hold`)
/// instead of one census lock a slot. Only with `VICTORY_CENSUS_BATCH=1`
/// and `VICTORY_CENSUS_FAST=1` (whose tickets keep a slot's last making),
/// read once (at the first view heap). Every held update is entered before
/// the pass returns.
pub(crate) fn census_batch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = census_rule(std::env::var("VICTORY_CENSUS_BATCH").ok().as_deref());
        let on = asked && census_fast();
        if on {
            tracing::info!(
                "VICTORY_CENSUS_BATCH=1: heap sync passes hold the census updates of mirror and make_null too, \
                 entered 256 at a time and before the pass returns"
            );
        } else if asked {
            tracing::warn!("VICTORY_CENSUS_BATCH=1 does nothing without VICTORY_CENSUS_FAST=1");
        }
        on
    })
}

/// Whether block-compressed host textures are made castable only to their
/// sRGB partner (`crate::resource::castable_metal`, VIctory-macOS #499):
/// Metal views block-compressed memory only in its own compression and
/// aborts the process (SIGABRT `0x80000101`) on a view in another; Series
/// 1.0.2.1 reuses BC5 memory as BC7 in the world. Such memory then gets its
/// own host texture, as on hosts that refuse castable lists, and a copy that
/// asked for a same-family cast (BC5 UNORM as SNORM) no longer finds it in the
/// list. Only with `VICTORY_METAL_BC_CASTS=1` and under Wine (`wine`, asked
/// only then), read once (at the first colour host texture).
pub(crate) fn metal_bc_casts(wine: impl FnOnce() -> bool) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = bc_casts_rule(std::env::var("VICTORY_METAL_BC_CASTS").ok().as_deref());
        let on = asked && wine();
        if on {
            tracing::info!(
                "VICTORY_METAL_BC_CASTS=1: block-compressed host textures are castable only to their sRGB partner; \
                 memory viewed in another compression gets its own host texture"
            );
        } else if asked {
            tracing::warn!("VICTORY_METAL_BC_CASTS=1 does nothing outside Wine");
        }
        on
    })
}

/// Whether a list's buffer barriers take the UAV writes since the last one
/// into its widened history in place (`crate::host::widened_in_place`,
/// VIctory-macOS #520): the old way copies the whole history and the new
/// writes into a list made for `crate::widened::sync` at every barrier that
/// has widened RGB32 views to refresh (about 90,000 writes and 2 MB in
/// hackgods' world frames, 50 to 90 ms a frame of growing and freeing heap
/// blocks), and makes the list of a barrier's expanded writes again each
/// time. In place, the buffers stay with the list, and keep their room from
/// one use to the next. The same writes reach `sync` in the same order.
/// Only with `VICTORY_WIDENED_IN_PLACE=1`, read once (at the first barrier).
///
/// This is side A of `VICTORY_AB` (`crate::ab`); a host list reads its
/// title list's side from its reset to its close ([`widened_in_place_at`]).
pub(crate) fn widened_in_place() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = in_place_rule(std::env::var("VICTORY_WIDENED_IN_PLACE").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_WIDENED_IN_PLACE=1: buffer barriers take new UAV writes into the widened history without copying it"
            );
        }
        on
    })
}

/// mac: [`widened_in_place`] on `VICTORY_AB` side `b` (`crate::ab`).
pub(crate) fn widened_in_place_at(b: bool) -> bool {
    crate::ab::flag_at("VICTORY_WIDENED_IN_PLACE", b, widened_in_place(), |v| in_place_rule(Some(v)))
}

/// From VIctory-macOS #436 (82587a26): whether a write snapshot keeps only
/// the bytes each declared GPU write newly covers
/// (`crate::snapshot_spans::Spans`), instead of merging overlapping
/// declarations into one segment copied again whenever it grows. Only for
/// unmarked writes (marked ones keep no snapshot). Only with
/// `VICTORY_SNAPSHOT_SPANS=1`, read once.
///
/// mac: `VICTORY_AB=VICTORY_SNAPSHOT_SPANS` (`crate::ab`): a snapshot keeps
/// the form its first capture took until copy-back empties it
/// (`crate::shadow::Shadow::writing`), so a flip changes the next ones.
pub(crate) fn snapshot_spans() -> bool {
    #[cfg(test)]
    if tests::SPANS_HERE.with(std::cell::Cell::get) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    let a = *ON.get_or_init(|| {
        let on = spans_rule(std::env::var("VICTORY_SNAPSHOT_SPANS").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_SNAPSHOT_SPANS=1: write snapshots copy only the bytes no earlier declaration of the submission took (VIctory-macOS #436)"
            );
        }
        on
    });
    crate::ab::flag("VICTORY_SNAPSHOT_SPANS", a, |v| spans_rule(Some(v)))
}

/// `VICTORY_SNAPSHOT_SPANS`'s rule: on only for `1`.
fn spans_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// Whether marked and snapshot copy-backs scan 8 bytes at a time, batch kernel
/// reads and guard locks, partition unaliased pages onto parallel workers, and
/// write back only coalesced changed byte spans (`crate::copy_back_spans`,
/// `crate::shadow::copy_back_fast`). Off unless `VICTORY_COPY_BACK_SPANS=1`.
///
/// mac: `VICTORY_AB=VICTORY_COPY_BACK_SPANS` (`crate::ab`): read per copy-back
/// call, and both sides leave `Shadow` and page guards in the same state.
pub(crate) fn copy_back_spans() -> bool {
    #[cfg(test)]
    if tests::COPY_BACK_SPANS_HERE.with(std::cell::Cell::get) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    let a = *ON.get_or_init(|| {
        let on = copy_back_spans_rule(std::env::var("VICTORY_COPY_BACK_SPANS").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_COPY_BACK_SPANS=1: marked and snapshot copy-backs batch reads and guard locks, split unaliased pages, and write back only changed byte spans"
            );
        }
        on
    });
    crate::ab::flag("VICTORY_COPY_BACK_SPANS", a, |v| copy_back_spans_rule(Some(v)))
}

/// `VICTORY_COPY_BACK_SPANS`'s rule: on only for `1`.
fn copy_back_spans_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// From VIctory-macOS #452 (18f33b86): whether a title copy made while a
/// submission is prepared (not into memory shown at two addresses) makes
/// only the pending refreshes whose queued ranges reach its window, leaving
/// the others queued until the submission's flush (`crate::host`,
/// `Batch::title_copied`). Off: every title copy makes all pending
/// refreshes first, as before. Only with `VICTORY_FLUSH_OVERLAPPING=1`, read
/// once.
///
/// mac: `VICTORY_AB=VICTORY_FLUSH_OVERLAPPING` (`crate::ab`): a submission's
/// batch takes the side once, when it is made.
pub(crate) fn flush_overlapping() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let a = *ON.get_or_init(|| {
        let on = overlapping_rule(std::env::var("VICTORY_FLUSH_OVERLAPPING").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_FLUSH_OVERLAPPING=1: a title copy makes only the pending refreshes it can change (VIctory-macOS #452)"
            );
        }
        on
    });
    crate::ab::flag("VICTORY_FLUSH_OVERLAPPING", a, |v| overlapping_rule(Some(v)))
}

/// `VICTORY_FLUSH_OVERLAPPING`'s rule: on only for `1`.
fn overlapping_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// `VICTORY_METAL_BC_CASTS`'s rule: on only for `1`.
fn bc_casts_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// `VICTORY_WIDENED_IN_PLACE`'s rule: on only for `1`.
fn in_place_rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// What a heap sync does with a slot it finds rewritten with a view of a
/// placed texture (`VICTORY_PLACED_AT_ONCE`, notes/flicker.md). Without it
/// the slot is left null until a shader has read it: one submission of
/// zeros for the draws that pick it by index, an object black for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacedAtOnce {
    /// Unset, empty or `0`: as before.
    Off,
    /// `1`: the slot is made at once when the texture's host texture is
    /// made already (making the view then makes no texture).
    Made,
    /// `keep`: as `1`, and a slot that held a host view keeps it until a
    /// shader has read the slot (the old texture for that submission
    /// instead of zeros; the title moving a texture leaves the same texels).
    Keep,
    /// `all`: every such slot is made at once, its texture made if it is
    /// not yet (the host textures the null rule spares: memory).
    All,
    /// `read`: as `keep`, and a slot a shader read through the heap in the
    /// last `VICTORY_PLACED_READ_FRAMES` frames is made at once as with
    /// `all` (`crate::placed_read`).
    Read,
}

impl PlacedAtOnce {
    /// Whether a view whose host texture is made is made at once.
    pub(crate) fn made(self) -> bool {
        self != PlacedAtOnce::Off
    }

    /// Whether a slot left for later keeps the host view it held (`keep`,
    /// and `read` for the slots it does not make at once).
    pub(crate) fn keeps(self) -> bool {
        matches!(self, PlacedAtOnce::Keep | PlacedAtOnce::Read)
    }
}

/// `VICTORY_PLACED_AT_ONCE`'s rule.
fn placed_rule(value: Option<&str>) -> PlacedAtOnce {
    match value.map(str::trim) {
        Some("1") => PlacedAtOnce::Made,
        Some("keep") => PlacedAtOnce::Keep,
        Some("all") => PlacedAtOnce::All,
        Some("read") => PlacedAtOnce::Read,
        _ => PlacedAtOnce::Off,
    }
}

/// `VICTORY_PLACED_AT_ONCE`, read once ([`PlacedAtOnce`]).
///
/// mac: `VICTORY_AB=VICTORY_PLACED_AT_ONCE` (`crate::ab`): side B is off
/// where A is on, else `all`, or `=<value>`; read once a slot a heap sync
/// makes. A slot made either way stays valid the other way: a view made at
/// once stays until the slot is rewritten, a slot left null is made when a
/// shader reads it.
pub(crate) fn placed_at_once() -> PlacedAtOnce {
    #[cfg(test)]
    if let Some(mode) = tests::PLACED_HERE.with(std::cell::Cell::get) {
        return mode;
    }
    static ON: OnceLock<PlacedAtOnce> = OnceLock::new();
    let a = *ON.get_or_init(|| {
        let mode = placed_rule(std::env::var("VICTORY_PLACED_AT_ONCE").ok().as_deref());
        match mode {
            PlacedAtOnce::Off => {}
            PlacedAtOnce::Made => tracing::info!(
                "VICTORY_PLACED_AT_ONCE=1: heap slots rewritten with views of placed textures whose host texture is made are made at once, not left null until read"
            ),
            PlacedAtOnce::Keep => tracing::info!(
                "VICTORY_PLACED_AT_ONCE=keep: as =1, and a slot that showed a view keeps it until a shader reads the slot, not null"
            ),
            PlacedAtOnce::All => tracing::info!(
                "VICTORY_PLACED_AT_ONCE=all: every heap slot rewritten with a view of a placed texture is made at once, its texture made if need be"
            ),
            // Its line is `crate::placed_read`'s, with the window.
            PlacedAtOnce::Read => {}
        }
        mode
    });
    crate::ab::value(
        "VICTORY_PLACED_AT_ONCE",
        a,
        |a| if a.made() { PlacedAtOnce::Off } else { PlacedAtOnce::All },
        |v| placed_rule(Some(v)),
    )
}

/// Which views a record names whole instead of through the view table
/// (`crate::descriptor::inline`, after fxceless's fork):
/// `VICTORY_VIEW_INLINE=1`, constant buffer views; `all`, also views of
/// placed memory. Unset or `0`: every view is a table entry, as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ViewInline {
    Off,
    Constants,
    All,
}

/// `VICTORY_VIEW_INLINE`'s rule.
fn view_inline_rule(value: Option<&str>) -> ViewInline {
    match value.map(str::trim) {
        Some("1") => ViewInline::Constants,
        Some("all") => ViewInline::All,
        _ => ViewInline::Off,
    }
}

/// `VICTORY_VIEW_INLINE`, read once ([`ViewInline`]).
pub(crate) fn view_inline() -> ViewInline {
    static ON: OnceLock<ViewInline> = OnceLock::new();
    *ON.get_or_init(|| {
        let mode = view_inline_rule(std::env::var("VICTORY_VIEW_INLINE").ok().as_deref());
        match mode {
            ViewInline::Off => {}
            ViewInline::Constants => tracing::info!(
                "VICTORY_VIEW_INLINE=1: constant buffer views are named whole in their records, not entries of the view table"
            ),
            ViewInline::All => tracing::info!(
                "VICTORY_VIEW_INLINE=all: constant buffer views and views of placed memory are named whole in their records, not entries of the view table"
            ),
        }
        mode
    })
}

#[cfg(test)]
mod view_inline_tests {
    use super::{ViewInline, view_inline_rule};

    #[test]
    fn views_are_entries_unless_the_switch_says_otherwise() {
        assert_eq!(view_inline_rule(None), ViewInline::Off);
        assert_eq!(view_inline_rule(Some("0")), ViewInline::Off);
        assert_eq!(view_inline_rule(Some("yes")), ViewInline::Off);
        assert_eq!(view_inline_rule(Some("1")), ViewInline::Constants);
        assert_eq!(view_inline_rule(Some(" all ")), ViewInline::All);
    }
}

#[cfg(test)]
mod census_tests {
    use super::{bc_casts_rule, census_rule, in_place_rule};

    #[test]
    fn metal_bc_casts_are_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("2")] {
            assert!(!bc_casts_rule(off), "{off:?}");
        }
        assert!(bc_casts_rule(Some("1")) && bc_casts_rule(Some(" 1 ")));
    }

    #[test]
    fn widened_in_place_is_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("2")] {
            assert!(!in_place_rule(off), "{off:?}");
        }
        assert!(in_place_rule(Some("1")) && in_place_rule(Some(" 1 ")));
    }

    #[test]
    fn the_census_switch_is_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("true"), Some("2")] {
            assert!(!census_rule(off), "{off:?}");
        }
        assert!(census_rule(Some("1")));
        assert!(census_rule(Some(" 1 ")));
    }

    #[test]
    fn placed_views_at_once_are_off_by_default() {
        use super::{PlacedAtOnce, placed_rule};
        for off in [None, Some(""), Some("0"), Some("yes"), Some("true"), Some("2"), Some("Keep")] {
            assert_eq!(placed_rule(off), PlacedAtOnce::Off, "{off:?}");
            assert!(!placed_rule(off).made());
        }
        assert_eq!(placed_rule(Some(" 1 ")), PlacedAtOnce::Made);
        assert_eq!(placed_rule(Some("keep")), PlacedAtOnce::Keep);
        assert_eq!(placed_rule(Some("all")), PlacedAtOnce::All);
        assert_eq!(placed_rule(Some(" read ")), PlacedAtOnce::Read);
        assert!(
            [PlacedAtOnce::Made, PlacedAtOnce::Keep, PlacedAtOnce::All, PlacedAtOnce::Read].iter().all(|m| m.made())
        );
        // `read` keeps a slot's old view where it does not make it at once.
        assert_eq!(
            [PlacedAtOnce::Off, PlacedAtOnce::Made, PlacedAtOnce::Keep, PlacedAtOnce::All, PlacedAtOnce::Read]
                .map(PlacedAtOnce::keeps),
            [false, false, true, false, true]
        );
        // Read once; the tests never set it.
        if std::env::var_os("VICTORY_PLACED_AT_ONCE").is_none() {
            assert_eq!(super::placed_at_once(), PlacedAtOnce::Off);
        }
    }

    #[test]
    fn the_census_batch_is_off_by_default() {
        // Read once; the tests never set it.
        if std::env::var_os("VICTORY_CENSUS_BATCH").is_none() {
            assert!(!super::census_batch());
        }
    }
}

#[cfg(test)]
mod clamp_tests {
    use super::{clamps, dispatch_rule};

    #[test]
    fn the_clamp_switch_is_off_unless_it_is_1_or_hoist() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("batch"), Some("2")] {
            assert_eq!(clamps(off), [false, false], "{off:?}");
        }
        assert_eq!(clamps(Some("1")), [true, true]);
        assert_eq!(clamps(Some(" 1 ")), [true, true]);
        assert_eq!(clamps(Some("hoist")), [true, false]);
    }

    #[test]
    fn the_dispatch_clamp_switch_is_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("hoist"), Some("2")] {
            assert!(!dispatch_rule(off), "{off:?}");
        }
        assert!(dispatch_rule(Some("1")));
        assert!(dispatch_rule(Some(" 1 ")));
    }
}

/// mac: `VICTORY_AB` (`crate::ab`): side B of each switch read here. The
/// tests set none of these switches, so side A is each one's default.
#[cfg(test)]
mod ab_tests {
    use super::*;

    /// Runs `f` with `VICTORY_AB=<toggle>` on side `b` on this thread.
    fn with(toggle: &str, b: bool, f: impl FnOnce()) {
        crate::ab::force(Some(crate::ab::parse(toggle).unwrap()), b);
        f();
        crate::ab::force(None, false);
    }

    fn unset(names: &[&str]) -> bool {
        names.iter().all(|n| std::env::var_os(n).is_none())
    }

    #[test]
    fn on_off_switches_read_the_opposite_or_the_value_on_side_b() {
        if !unset(&["VICTORY_FLUSH_OVERLAPPING", "VICTORY_SNAPSHOT_SPANS", "VICTORY_COPY_BACK_SPANS", "VICTORY_SYNC_EPOCH", "VICTORY_WIDENED_IN_PLACE"]) {
            return;
        }
        assert!(!flush_overlapping() && !snapshot_spans() && !copy_back_spans() && !sync_epoch(|| false) && !widened_in_place());
        with("VICTORY_FLUSH_OVERLAPPING:10", true, || assert!(flush_overlapping()));
        with("VICTORY_FLUSH_OVERLAPPING:10", false, || assert!(!flush_overlapping(), "side A: as set"));
        with("VICTORY_FLUSH_OVERLAPPING=0:10", true, || assert!(!flush_overlapping(), "=0 on side B: off (A/A)"));
        with("VICTORY_SNAPSHOT_SPANS:10", true, || {
            assert!(snapshot_spans());
            assert!(!flush_overlapping(), "another switch keeps side A");
        });
        with("VICTORY_COPY_BACK_SPANS:10", true, || {
            assert!(copy_back_spans());
            assert!(!snapshot_spans(), "another switch keeps side A");
        });
        with("VICTORY_SYNC_EPOCH:10", true, || assert!(sync_epoch(|| false)));
        // A list's side, not the side now.
        with("VICTORY_WIDENED_IN_PLACE:10", false, || {
            assert!(widened_in_place_at(true) && !widened_in_place_at(false));
        });
    }

    #[test]
    fn quick_sync_flips_only_its_views() {
        if !unset(&["VICTORY_QUICK_SYNC"]) {
            return;
        }
        let a = quick_sync(|| true);
        with("VICTORY_QUICK_SYNC:10", true, || assert_eq!(quick_sync(|| true), [a[0], !a[1]]));
        with("VICTORY_QUICK_SYNC=views:10", true, || assert_eq!(quick_sync(|| true), [a[0], true]));
        with("VICTORY_QUICK_SYNC=0:10", true, || assert_eq!(quick_sync(|| true), [a[0], false]));
    }

    #[test]
    fn placed_views_are_made_at_once_on_side_b_where_side_a_leaves_them() {
        if !unset(&["VICTORY_PLACED_AT_ONCE"]) {
            return;
        }
        assert_eq!(placed_at_once(), PlacedAtOnce::Off);
        with("VICTORY_PLACED_AT_ONCE:10", true, || assert_eq!(placed_at_once(), PlacedAtOnce::All));
        with("VICTORY_PLACED_AT_ONCE=keep:10", true, || assert_eq!(placed_at_once(), PlacedAtOnce::Keep));
        with("VICTORY_PLACED_AT_ONCE=read:10", true, || assert_eq!(placed_at_once(), PlacedAtOnce::Read));
    }

    #[test]
    fn clamps_flip_per_list_and_take_the_dispatch_clamps_along() {
        if !unset(&["VICTORY_BATCH_CLAMPS", "VICTORY_DISPATCH_CLAMPS"]) {
            return;
        }
        assert_eq!(batch_clamps_at(false), [false, false]);
        with("VICTORY_BATCH_CLAMPS:10", false, || {
            assert_eq!(batch_clamps_at(true), [true, true], "a list on side B");
            assert_eq!(batch_clamps_at(false), [false, false], "a list on side A");
            assert!(!dispatch_clamps_at(true), "dispatch clamps not asked for: off on both sides");
        });
        with("VICTORY_BATCH_CLAMPS=hoist:10", false, || assert_eq!(batch_clamps_at(true), [true, false]));
        // Dispatch clamps asked for on side B only hoist where side B hoists.
        with("VICTORY_DISPATCH_CLAMPS:10", false, || {
            assert!(!dispatch_clamps_at(true), "side A does not hoist the draws' clamps");
        });
    }

    #[test]
    fn the_copy_order_split_flips_per_list() {
        if !unset(&["VICTORY_COPY_ORDER"]) || tests::FORCED_COPY_ORDER.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            return;
        }
        assert_eq!(copy_order_split_at(false), None);
        with("VICTORY_COPY_ORDER:10", false, || {
            // (Another test may force the split on meanwhile: then both are.)
            if tests::FORCED_COPY_ORDER.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                assert_eq!(copy_order_split_at(true), Some(usize::MAX));
                assert_eq!(copy_order_split_at(false), None);
            }
        });
        with("VICTORY_COPY_ORDER=split:4096:10", false, || {
            if tests::FORCED_COPY_ORDER.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                assert_eq!(copy_order_split_at(true), Some(4096));
            }
        });
    }
}

/// [`lock_writers`]' rule: helper lanes, low keys.
fn writers(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("helpers") => [true, false],
        value => [setting(value, default); 2],
    }
}

/// `VICTORY_ASYNC_HEAP_SYNC`'s rule (`crate::descriptor::ahead`): off when
/// unset, empty or `0`; else on, and the milliseconds between a round of
/// the heap sync and the next while the GPU still runs: 8 for `1` (or
/// anything that is no number), a larger number for itself.
pub(crate) fn heap_sync_pause(value: Option<&str>) -> Option<u64> {
    match value {
        None | Some("" | "0") => None,
        Some(v) => Some(v.parse::<u64>().ok().filter(|&ms| ms > 1).unwrap_or(8)),
    }
}

/// `VICTORY_HEAP_LOCKS`'s rule (`crate::descriptor::slots`): whether a host
/// heap's slots get locks of their own, and whether lock waits and made
/// slots are counted. Unset, empty or `0`: neither, as before; `count`:
/// the one lock as before, counted; anything else: both.
pub(crate) fn heap_locks(value: Option<&str>) -> [bool; 2] {
    match value {
        None | Some("" | "0") => [false, false],
        Some("count") => [false, true],
        Some(_) => [true, true],
    }
}

/// `VICTORY_HEAP_SYNC_THREADS`'s rule (`crate::descriptor::pool`): the
/// threads a heap sync makes a pass's changed slots on, its own among them;
/// 1 is the syncing thread alone, as before. Unset, no number, 0 or 1: 1;
/// a larger number: itself, `most` at most. Always 1 unless the heap's
/// slots have locks of their own (`split`, `VICTORY_HEAP_LOCKS=1`).
pub(crate) fn heap_sync_threads(value: Option<&str>, split: bool, most: usize) -> usize {
    let asked = value.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1);
    if split { asked.clamp(1, most.max(1)) } else { 1 }
}

/// mac: what `VICTORY_ALIAS_PHYSICAL` asks of the title's physical pages
/// (stage 2, `notes/alias-physical-design.md`; `crate::alias::physical_heap`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AliasPhysical {
    /// Unset, empty or `0`: nothing, not even the counts (`crate::physical_runs`);
    /// each section is a pagefile section, as before.
    Off,
    /// `count`, or a value that names nothing built: stage 2's counts alone.
    Count,
    /// `cpu`: each section is backed by a CPU-visible D3D12 heap and every
    /// title view of its pages is remapped onto it (VIctory-macOS #275's
    /// section heap): the heap backing alone, with the counts; the GPU still
    /// reads the pages through host copies.
    Cpu,
    /// `gpu`: `cpu`, and one reserved buffer per piece of each placeholder
    /// reservation, tiled from the heaps as the title maps its pages
    /// (`crate::alias_physical`). A range that is one unbroken run of one
    /// section's pages is a view of a placed buffer over that section's heap
    /// (canonical: every title address of a page reaches one GPU range);
    /// other ranges in one piece are views of the piece.
    Gpu,
    /// `gpu-tiles`: `gpu` without the section views: every range in one piece
    /// is a view of the piece (I4's order, for comparison).
    GpuTiles,
    /// `gpu-single`: `gpu-tiles`, but a range with a page shown at two or
    /// more addresses keeps its copy (`notes/alias-physical-design.md` §3).
    GpuSingle,
    /// `verify`: `gpu`'s heaps, pieces, tile flushes and views, but nothing
    /// is drawn through the views: every lookup keeps its host copy, as with
    /// the switch off, and each view `gpu` would have bound is checked on the
    /// CPU against the title's bytes (`crate::alias_verify`,
    /// `notes/alias-physical-debug.md`).
    Verify,
    /// `verify-tiles`: `verify` in `gpu-tiles`' order (no section views).
    VerifyTiles,
    /// `gpu-sections`: `gpu`'s views of sections alone: no reserved pieces,
    /// no tile mappings, no sink and no flush. A range that is one run of one
    /// section's pages is a view of the placed buffer over the section's heap
    /// (every title address of a page reaches that one GPU range); every other
    /// range keeps its host copy, as with the switch off
    /// (`notes/alias-physical-debug.md`, "One range per page").
    GpuSections,
    /// `verify-sections`: `verify` for `gpu-sections` (views of sections only,
    /// checked, never drawn through).
    VerifySections,
    /// `gpu-ranges-old`: `gpu` with I4b's rule for views beside copies (a
    /// lookup over tiles views of the other kind hold keeps a copy, and
    /// nothing else), for comparison with the one-range rule.
    GpuRangesOld,
    /// `verify-ranges-old`: `verify` with I4b's rule, as `gpu-ranges-old`.
    VerifyRangesOld,
}

impl AliasPhysical {
    /// Whether the title's sections get heaps.
    pub(crate) fn heaps(self) -> bool {
        matches!(
            self,
            AliasPhysical::Cpu
                | AliasPhysical::Gpu
                | AliasPhysical::GpuTiles
                | AliasPhysical::GpuSingle
                | AliasPhysical::Verify
                | AliasPhysical::VerifyTiles
                | AliasPhysical::GpuSections
                | AliasPhysical::VerifySections
                | AliasPhysical::GpuRangesOld
                | AliasPhysical::VerifyRangesOld
        )
    }

    /// Whether the GPU side is on (`crate::alias_physical`): the GPU reads the
    /// pages through views of the sections' heaps and (but `gpu-sections`)
    /// of the reservations' pieces, or (`verify`) their views are checked
    /// and not used.
    pub(crate) fn gpu(self) -> bool {
        matches!(
            self,
            AliasPhysical::Gpu
                | AliasPhysical::GpuTiles
                | AliasPhysical::GpuSingle
                | AliasPhysical::Verify
                | AliasPhysical::VerifyTiles
                | AliasPhysical::GpuSections
                | AliasPhysical::VerifySections
                | AliasPhysical::GpuRangesOld
                | AliasPhysical::VerifyRangesOld
        )
    }

    /// Whether the reservations get pieces, reserved buffers tiled from the
    /// heaps as the title maps its pages (all GPU modes but `gpu-sections`
    /// and `verify-sections`, which never map a tile).
    pub(crate) fn pieces(self) -> bool {
        self.gpu() && !matches!(self, AliasPhysical::GpuSections | AliasPhysical::VerifySections)
    }

    /// Whether runs of a section's pages are views of the section's heap
    /// (`=gpu`, canonical views first; `=verify` checks them).
    pub(crate) fn canonical(self) -> bool {
        matches!(
            self,
            AliasPhysical::Gpu
                | AliasPhysical::Verify
                | AliasPhysical::GpuSections
                | AliasPhysical::VerifySections
                | AliasPhysical::GpuRangesOld
                | AliasPhysical::VerifyRangesOld
        )
    }

    /// Whether a heap tile is reached through one GPU range at most while
    /// views reach it (`crate::one_range`): all canonical modes but
    /// `gpu-ranges-old` and `verify-ranges-old`, which keep I4b's rule.
    pub(crate) fn one_range(self) -> bool {
        self.canonical() && !matches!(self, AliasPhysical::GpuRangesOld | AliasPhysical::VerifyRangesOld)
    }

    /// Whether the views are only checked, never drawn through (`verify`,
    /// `verify-tiles`, `verify-sections`, `verify-ranges-old`).
    pub(crate) fn verify(self) -> bool {
        matches!(
            self,
            AliasPhysical::Verify
                | AliasPhysical::VerifyTiles
                | AliasPhysical::VerifySections
                | AliasPhysical::VerifyRangesOld
        )
    }

    /// The switch as written.
    pub(crate) fn name(self) -> &'static str {
        match self {
            AliasPhysical::Off => "VICTORY_ALIAS_PHYSICAL unset",
            AliasPhysical::Count => "VICTORY_ALIAS_PHYSICAL=count",
            AliasPhysical::Cpu => "VICTORY_ALIAS_PHYSICAL=cpu",
            AliasPhysical::Gpu => "VICTORY_ALIAS_PHYSICAL=gpu",
            AliasPhysical::GpuTiles => "VICTORY_ALIAS_PHYSICAL=gpu-tiles",
            AliasPhysical::GpuSingle => "VICTORY_ALIAS_PHYSICAL=gpu-single",
            AliasPhysical::Verify => "VICTORY_ALIAS_PHYSICAL=verify",
            AliasPhysical::VerifyTiles => "VICTORY_ALIAS_PHYSICAL=verify-tiles",
            AliasPhysical::GpuSections => "VICTORY_ALIAS_PHYSICAL=gpu-sections",
            AliasPhysical::VerifySections => "VICTORY_ALIAS_PHYSICAL=verify-sections",
            AliasPhysical::GpuRangesOld => "VICTORY_ALIAS_PHYSICAL=gpu-ranges-old",
            AliasPhysical::VerifyRangesOld => "VICTORY_ALIAS_PHYSICAL=verify-ranges-old",
        }
    }
}

/// mac: `VICTORY_WITHHELD_COPIES` (`notes/alias-physical-debug.md`, "Withheld
/// writes"): which ranges written by programs using title memory stop being
/// written through an alias (`VICTORY_ALIAS_BUFFERS`) or a view
/// (`VICTORY_ALIAS_PHYSICAL=gpu`) and get a host copy instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WithheldCopies {
    /// Unset, empty or `0`: none, as before (withheld writes through an
    /// alias or a view are in title memory and stay).
    Off,
    /// `1`: a range in which a withheld write landed in title memory, from
    /// then on (carved at the submission's finish).
    Evidence,
    /// `all`: every range such a program binds for unordered access through
    /// a root view, before it is bound, whether its writes are withheld or
    /// not (for comparison).
    All,
}

/// [`WithheldCopies`], read once.
pub(crate) fn withheld_copies() -> WithheldCopies {
    static RULE: OnceLock<WithheldCopies> = OnceLock::new();
    *RULE.get_or_init(|| {
        let value = std::env::var("VICTORY_WITHHELD_COPIES").ok();
        let rule = withheld_copies_rule(value.as_deref());
        if rule != WithheldCopies::Off {
            tracing::info!(
                "VICTORY_WITHHELD_COPIES={}: {} gets a host copy, never an alias or a view, so that what \
                 programs using title memory write while it is withheld stays out of title memory; frame line \
                 'withheld writes'",
                value.as_deref().unwrap_or_default().trim(),
                if rule == WithheldCopies::All {
                    "every range such a program binds for unordered access through a root view"
                } else {
                    "a range in which a withheld write landed in title memory, from then on,"
                }
            );
        }
        rule
    })
}

fn withheld_copies_rule(value: Option<&str>) -> WithheldCopies {
    match value.map(str::trim) {
        Some("1") => WithheldCopies::Evidence,
        Some("all") => WithheldCopies::All,
        _ => WithheldCopies::Off,
    }
}

/// [`AliasPhysical`], read once (at the title's first `xmem` call that asks).
/// Any other value gives the counts alone, with a warning.
pub(crate) fn alias_physical() -> AliasPhysical {
    static RULE: OnceLock<AliasPhysical> = OnceLock::new();
    *RULE.get_or_init(|| {
        let value = std::env::var("VICTORY_ALIAS_PHYSICAL").ok();
        let (rule, known) = alias_physical_rule(value.as_deref());
        if !known {
            tracing::warn!(
                "VICTORY_ALIAS_PHYSICAL={}: not one of count, cpu, gpu, gpu-tiles, gpu-single, gpu-sections, gpu-ranges-old, \
                 verify, verify-tiles, verify-sections and verify-ranges-old; stage 2's counts alone",
                value.as_deref().unwrap_or_default()
            );
        }
        if rule != AliasPhysical::Off {
            tracing::info!(
                "VICTORY_ALIAS_PHYSICAL: stage 2's counts of the title's physical pages, in the physical pages line"
            );
        }
        rule
    })
}

/// [`alias_physical`]'s rule, and whether the value names something built.
fn alias_physical_rule(value: Option<&str>) -> (AliasPhysical, bool) {
    match value.map(str::trim) {
        None | Some("" | "0") => (AliasPhysical::Off, true),
        Some("count") => (AliasPhysical::Count, true),
        Some("cpu") => (AliasPhysical::Cpu, true),
        Some("gpu") => (AliasPhysical::Gpu, true),
        Some("gpu-tiles") => (AliasPhysical::GpuTiles, true),
        Some("gpu-single") => (AliasPhysical::GpuSingle, true),
        Some("verify") => (AliasPhysical::Verify, true),
        Some("verify-tiles") => (AliasPhysical::VerifyTiles, true),
        Some("gpu-sections") => (AliasPhysical::GpuSections, true),
        Some("verify-sections") => (AliasPhysical::VerifySections, true),
        Some("gpu-ranges-old") => (AliasPhysical::GpuRangesOld, true),
        Some("verify-ranges-old") => (AliasPhysical::VerifyRangesOld, true),
        Some(_) => (AliasPhysical::Count, false),
    }
}

/// A switch's state: `0` off, `1` on, anything else (and unset) `default`.
fn setting(value: Option<&str>, default: bool) -> bool {
    match value {
        Some("0") => false,
        Some("1") => true,
        _ => default,
    }
}

/// mac: why `VICTORY_AB` (`crate::ab`) cannot flip switch `name` in this
/// run (side B's `value`, if given), asked at the first present: a side
/// the switch's own rules refuse would measure nothing, or a picture the
/// line never runs with.
#[cfg(windows)]
pub(crate) fn ab_blocked(name: &str, value: Option<&str>) -> Option<String> {
    let _ = value;
    match name {
        "VICTORY_SYNC_EPOCH" if crate::host::unwaited::flights::on() => {
            Some("VICTORY_SYNC_EPOCH does nothing with VICTORY_FLIGHTS=1".to_owned())
        }
        "VICTORY_DISPATCH_CLAMPS" if !batch_clamps()[0] => {
            Some("VICTORY_DISPATCH_CLAMPS does nothing without VICTORY_BATCH_CLAMPS=1 or hoist".to_owned())
        }
        // `crate::alias` turns the aliasing on only with the split: off on
        // side B, commands before a title copy into aliased memory would
        // read its new bytes.
        "VICTORY_COPY_ORDER" if crate::alias::enabled() && copy_order_split_at(true).is_none() => Some(
            "VICTORY_ALIAS_BUFFERS is on, which needs VICTORY_COPY_ORDER=split on both sides \
             (side B may be =split:<bytes>)"
                .to_owned(),
        ),
        _ => None,
    }
}

/// The cache key `build` for translations made with these settings. A
/// setting that is off adds nothing, so the entries made without it stay
/// the ones read.
pub(crate) fn key(
    build: &[u8; 32],
    early_depth: bool,
    lock_writers: [bool; 2],
    lds_order: bool,
    exchange: bool,
    uav_size_level0: bool,
    walk: bool,
) -> [u8; 32] {
    if !early_depth
        && lock_writers == [false; 2]
        && !lds_order
        && !exchange
        && !uav_size_level0
        && !walk
    {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    if early_depth {
        h.update(b"early depth");
    }
    if lock_writers[0] {
        h.update(b"atomic lock helpers");
    }
    if lock_writers[1] {
        h.update(b"atomic lock keys");
    }
    if lds_order {
        h.update(b"lds order");
    }
    if exchange {
        h.update(b"halves exchange");
    }
    if uav_size_level0 {
        h.update(b"uav size level 0");
    }
    if walk {
        h.update(b"halves walk");
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{
        AliasPhysical, WithheldCopies, alias_physical_rule, withheld_copies_rule, copy_back_spans_rule, copy_order, epoch, heap_locks, heap_sync_pause,
        heap_sync_threads, key, overlapping_rule, quick, setting, spans_rule, texture_workers,
        uav_size, whole_wave_lds, writers,
    };

    #[test]
    fn alias_physical_counts_only_when_set() {
        assert!(AliasPhysical::Gpu.heaps() && AliasPhysical::Gpu.gpu());
        assert!(AliasPhysical::Gpu.canonical() && !AliasPhysical::Gpu.verify());
        for tiles in [AliasPhysical::GpuTiles, AliasPhysical::GpuSingle] {
            assert!(tiles.heaps() && tiles.gpu() && !tiles.canonical() && !tiles.verify(), "{tiles:?}");
        }
        // Verify modes run the GPU side's bookkeeping and only check its views.
        let v = AliasPhysical::Verify;
        assert!(v.heaps() && v.gpu() && v.canonical() && v.verify());
        let v = AliasPhysical::VerifyTiles;
        assert!(v.heaps() && v.gpu() && !v.canonical() && v.verify());
        assert!(!AliasPhysical::Cpu.verify() && !AliasPhysical::Count.verify());
        // Sections alone: no pieces, no tiles.
        for s in [AliasPhysical::GpuSections, AliasPhysical::VerifySections] {
            assert!(s.heaps() && s.gpu() && s.canonical() && !s.pieces(), "{s:?}");
        }
        assert!(AliasPhysical::VerifySections.verify() && !AliasPhysical::GpuSections.verify());
        for pieces in [
            AliasPhysical::Gpu,
            AliasPhysical::Verify,
            AliasPhysical::GpuTiles,
            AliasPhysical::GpuSingle,
            AliasPhysical::VerifyTiles,
        ] {
            assert!(pieces.pieces(), "{pieces:?}");
        }
        assert!(!AliasPhysical::Cpu.pieces() && !AliasPhysical::Off.pieces());
        // The one-range rule: every canonical mode but the old ones.
        for one in [
            AliasPhysical::Gpu,
            AliasPhysical::Verify,
            AliasPhysical::GpuSections,
            AliasPhysical::VerifySections,
        ] {
            assert!(one.one_range(), "{one:?}");
        }
        for old in [AliasPhysical::GpuRangesOld, AliasPhysical::VerifyRangesOld] {
            assert!(old.gpu() && old.canonical() && old.pieces() && !old.one_range(), "{old:?}");
        }
        assert!(AliasPhysical::VerifyRangesOld.verify() && !AliasPhysical::GpuRangesOld.verify());
        for tiles in [AliasPhysical::GpuTiles, AliasPhysical::GpuSingle, AliasPhysical::VerifyTiles] {
            assert!(!tiles.one_range(), "{tiles:?}");
        }
        assert!(AliasPhysical::Cpu.heaps() && !AliasPhysical::Cpu.gpu());
        assert!(!AliasPhysical::Count.heaps() && !AliasPhysical::Off.heaps());
        for off in [None, Some(""), Some("0")] {
            assert_eq!(
                alias_physical_rule(off),
                (AliasPhysical::Off, true),
                "{off:?}"
            );
        }
        for (value, rule) in [
            ("count", AliasPhysical::Count),
            (" count ", AliasPhysical::Count),
            ("cpu", AliasPhysical::Cpu),
            (" cpu ", AliasPhysical::Cpu),
            ("gpu", AliasPhysical::Gpu),
            ("gpu-tiles", AliasPhysical::GpuTiles),
            ("gpu-single", AliasPhysical::GpuSingle),
            ("verify", AliasPhysical::Verify),
            (" verify-tiles", AliasPhysical::VerifyTiles),
            ("gpu-sections", AliasPhysical::GpuSections),
            ("verify-sections ", AliasPhysical::VerifySections),
            ("gpu-ranges-old", AliasPhysical::GpuRangesOld),
            ("verify-ranges-old", AliasPhysical::VerifyRangesOld),
        ] {
            assert_eq!(alias_physical_rule(Some(value)), (rule, true), "{value:?}");
        }
        // Any other value: the counts, with a warning.
        for other in [Some("1"), Some("GPU"), Some("CPU"), Some("yes")] {
            assert_eq!(
                alias_physical_rule(other),
                (AliasPhysical::Count, false),
                "{other:?}"
            );
        }
    }

    #[test]
    fn withheld_copies_by_evidence_or_for_all() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("ALL")] {
            assert_eq!(withheld_copies_rule(off), WithheldCopies::Off, "{off:?}");
        }
        assert_eq!(withheld_copies_rule(Some("1")), WithheldCopies::Evidence);
        assert_eq!(withheld_copies_rule(Some(" 1 ")), WithheldCopies::Evidence);
        assert_eq!(withheld_copies_rule(Some("all")), WithheldCopies::All);
    }

    thread_local! {
        /// `snapshot_spans` is on for this thread's snapshots ([`SpansHere`]).
        pub(crate) static SPANS_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// While it lives, `snapshot_spans` is on for snapshots this thread
    /// starts, as with `VICTORY_SNAPSHOT_SPANS=1` (other threads' tests keep
    /// the default form).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) struct SpansHere;

    impl SpansHere {
        #[cfg_attr(not(windows), allow(dead_code))]
        pub(crate) fn new() -> Self {
            SPANS_HERE.with(|on| on.set(true));
            SpansHere
        }
    }

    impl Drop for SpansHere {
        fn drop(&mut self) {
            SPANS_HERE.with(|on| on.set(false));
        }
    }

    thread_local! {
        /// `copy_back_spans` is on for this thread's copy-backs ([`CopyBackSpansHere`]).
        pub(crate) static COPY_BACK_SPANS_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) struct CopyBackSpansHere;

    impl CopyBackSpansHere {
        #[cfg_attr(not(windows), allow(dead_code))]
        pub(crate) fn new() -> Self {
            COPY_BACK_SPANS_HERE.with(|on| on.set(true));
            CopyBackSpansHere
        }
    }

    impl Drop for CopyBackSpansHere {
        fn drop(&mut self) {
            COPY_BACK_SPANS_HERE.with(|on| on.set(false));
        }
    }

    #[test]
    fn copy_back_spans_are_on_only_when_asked_for() {
        assert!(copy_back_spans_rule(Some("1")));
        assert!(copy_back_spans_rule(Some(" 1 ")));
        for off in [None, Some(""), Some("0"), Some("2"), Some("on")] {
            assert!(!copy_back_spans_rule(off), "{off:?}");
        }
    }

    thread_local! {
        /// `placed_at_once` on this thread, whatever the environment says
        /// ([`PlacedHere`]).
        pub(crate) static PLACED_HERE: std::cell::Cell<Option<super::PlacedAtOnce>> =
            const { std::cell::Cell::new(None) };
    }

    /// While it lives, `placed_at_once` is `mode` on this thread: for tests
    /// that assert what a heap sync does with a placed view under one rule,
    /// so that `VICTORY_PLACED_AT_ONCE` set for the suite (or `=read`'s
    /// history, one per process, filled by other tests) does not change it.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) struct PlacedHere(Option<super::PlacedAtOnce>);

    impl PlacedHere {
        #[cfg_attr(not(windows), allow(dead_code))]
        pub(crate) fn new(mode: super::PlacedAtOnce) -> Self {
            PlacedHere(PLACED_HERE.with(|here| here.replace(Some(mode))))
        }
    }

    impl Drop for PlacedHere {
        fn drop(&mut self) {
            PLACED_HERE.with(|here| here.set(self.0));
        }
    }

    #[test]
    fn a_placed_rule_pinned_here_holds_on_this_thread_only() {
        use super::{PlacedAtOnce, placed_at_once};
        let outside = placed_at_once();
        {
            let _read = PlacedHere::new(PlacedAtOnce::Read);
            assert_eq!(placed_at_once(), PlacedAtOnce::Read);
            {
                let _off = PlacedHere::new(PlacedAtOnce::Off);
                assert_eq!(placed_at_once(), PlacedAtOnce::Off);
            }
            assert_eq!(placed_at_once(), PlacedAtOnce::Read, "the outer pin is back");
            assert_eq!(std::thread::spawn(placed_at_once).join().unwrap(), outside, "other threads keep theirs");
        }
        assert_eq!(placed_at_once(), outside);
    }

    #[test]
    fn flushing_only_overlapping_refreshes_is_on_only_when_asked_for() {
        assert!(overlapping_rule(Some("1")));
        for off in [None, Some(""), Some("0"), Some("2"), Some("yes")] {
            assert!(!overlapping_rule(off), "{off:?}");
        }
    }

    #[test]
    fn snapshot_spans_are_on_only_when_asked_for() {
        assert!(spans_rule(Some("1")));
        assert!(spans_rule(Some(" 1 ")));
        for off in [None, Some(""), Some("0"), Some("2"), Some("on")] {
            assert!(!spans_rule(off), "{off:?}");
        }
    }

    #[test]
    fn texture_workers_are_a_quarter_of_the_cpus_unless_asked_for() {
        for off in [None, Some(""), Some("0"), Some("1"), Some("many")] {
            assert_eq!(texture_workers(off, 18), Some(4), "{off:?}");
            assert_eq!(texture_workers(off, 14), Some(3), "{off:?}");
            assert_eq!(texture_workers(off, 8), Some(2), "{off:?}");
            assert_eq!(texture_workers(off, 4), None, "{off:?}");
            assert_eq!(texture_workers(off, 1), None, "{off:?}");
        }
        assert_eq!(texture_workers(Some("6"), 18), Some(6));
        assert_eq!(texture_workers(Some(" 6 "), 14), Some(6));
        assert_eq!(texture_workers(Some("64"), 14), Some(14));
        assert_eq!(texture_workers(Some("2"), 18), Some(2));
        // Two asked for on a one-CPU machine: none, as before.
        assert_eq!(texture_workers(Some("2"), 1), None);
    }

    #[test]
    fn the_epoch_skip_is_on_only_when_asked_for() {
        assert!(epoch(Some("1")));
        // Unset, `0`, or a word: every request refreshes, as the bench needs.
        for off in [None, Some(""), Some("0"), Some("2"), Some("on"), Some("yes")] {
            assert!(!epoch(off), "{off:?}");
        }
    }

    /// How many tests have the copy-order split forced on right now.
    pub(crate) static FORCED_COPY_ORDER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// While it lives, `copy_order_split` is on for the process, as with
    /// `VICTORY_COPY_ORDER=split` (other tests then only note what their
    /// lists read).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) struct ForcedCopyOrder;

    #[cfg_attr(not(windows), allow(dead_code))]
    impl ForcedCopyOrder {
        pub(crate) fn on() -> ForcedCopyOrder {
            FORCED_COPY_ORDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ForcedCopyOrder
        }
    }

    impl Drop for ForcedCopyOrder {
        fn drop(&mut self) {
            FORCED_COPY_ORDER.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn the_copy_order_switch_is_split_with_an_optional_bound() {
        for off in [None, Some(""), Some("0"), Some("1"), Some("keep"), Some("split:"), Some("split:0"), Some("split:x"), Some("splits")] {
            assert_eq!(copy_order(off), None, "{off:?}");
        }
        assert_eq!(copy_order(Some("split")), Some(usize::MAX));
        assert_eq!(copy_order(Some(" split ")), Some(usize::MAX));
        assert_eq!(copy_order(Some("split:256")), Some(256));
        assert_eq!(copy_order(Some("split:65536")), Some(65536));
    }

    #[test]
    fn the_size_query_switch_is_all_programs_or_a_list_of_hashes() {
        assert_eq!(uav_size(None), (false, vec![]));
        assert_eq!(uav_size(Some("")), (false, vec![]));
        assert_eq!(uav_size(Some("0")), (false, vec![]));
        assert_eq!(uav_size(Some("1")), (true, vec![]));
        assert_eq!(uav_size(Some("yes")), (false, vec![]));
        let two = vec!["07f451a14b13fecb".to_owned(), "11eeb27c2d15980d".to_owned()];
        assert_eq!(uav_size(Some("07F451A14B13FECB, 11eeb27c2d15980d")), (false, two));
        // A list with a wrong entry is no list: nothing is turned on.
        assert_eq!(uav_size(Some("07f451a14b13fecb,1")), (false, vec![]));
    }

    #[test]
    fn the_whole_wave_lds_list_is_every_program_unless_it_names_some() {
        assert_eq!(whole_wave_lds(None), None);
        assert_eq!(whole_wave_lds(Some("")), None);
        assert_eq!(whole_wave_lds(Some(" ")), None);
        assert_eq!(whole_wave_lds(Some("0")), Some(vec![]));
        let two = vec!["03e3e26783a5799b".to_owned(), "fd8393c7f4e0e3da".to_owned()];
        assert_eq!(whole_wave_lds(Some("03E3E26783A5799B, fd8393c7f4e0e3da")), Some(two));
        // A list with a wrong entry is no list: every program, as unset.
        assert_eq!(whole_wave_lds(Some("03e3e26783a5799b,1")), None);
    }

    #[test]
    fn the_heap_sync_ahead_is_off_unless_asked_and_takes_its_pause() {
        assert_eq!(heap_sync_pause(None), None);
        assert_eq!(heap_sync_pause(Some("")), None);
        assert_eq!(heap_sync_pause(Some("0")), None);
        assert_eq!(heap_sync_pause(Some("1")), Some(8));
        assert_eq!(heap_sync_pause(Some("yes")), Some(8));
        assert_eq!(heap_sync_pause(Some("4")), Some(4));
        assert_eq!(heap_sync_pause(Some("1000")), Some(1000));
    }

    #[test]
    fn the_heap_locks_are_one_and_uncounted_unless_asked() {
        assert_eq!(heap_locks(None), [false, false]);
        assert_eq!(heap_locks(Some("")), [false, false]);
        assert_eq!(heap_locks(Some("0")), [false, false]);
        assert_eq!(heap_locks(Some("count")), [false, true]);
        assert_eq!(heap_locks(Some("1")), [true, true]);
        assert_eq!(heap_locks(Some("yes")), [true, true]);
    }

    #[test]
    fn a_heap_sync_has_one_thread_unless_asked_and_the_locks_are_split() {
        for value in [None, Some(""), Some("0"), Some("1"), Some("yes"), Some("-3")] {
            assert_eq!(heap_sync_threads(value, true, 8), 1, "{value:?}");
        }
        assert_eq!(heap_sync_threads(Some("2"), true, 8), 2);
        assert_eq!(heap_sync_threads(Some(" 4 "), true, 8), 4);
        assert_eq!(heap_sync_threads(Some("64"), true, 8), 8);
        for value in [None, Some("1"), Some("4"), Some("64")] {
            assert_eq!(heap_sync_threads(value, false, 8), 1, "{value:?} without the split locks");
        }
    }

    #[test]
    fn the_quick_sync_switch_names_its_parts() {
        for default in [false, true] {
            assert_eq!(quick(Some("0"), default), [false, false]);
            assert_eq!(quick(Some("maps"), default), [true, false]);
            assert_eq!(quick(Some("views"), default), [false, true]);
            assert_eq!(quick(Some("1"), default), [true, true]);
            assert_eq!(quick(None, default), [default, default]);
            assert_eq!(quick(Some("both"), default), [default, default]);
        }
    }

    #[test]
    fn a_switch_is_off_at_0_on_at_1_and_else_the_default() {
        for default in [false, true] {
            assert!(!setting(Some("0"), default));
            assert!(setting(Some("1"), default));
            assert_eq!(setting(None, default), default);
            assert_eq!(setting(Some(""), default), default);
            assert_eq!(setting(Some("yes"), default), default);
        }
    }

    #[test]
    fn the_lock_writers_switch_has_three_states() {
        for default in [false, true] {
            assert_eq!(writers(Some("0"), default), [false, false]);
            assert_eq!(writers(Some("helpers"), default), [true, false]);
            assert_eq!(writers(Some("1"), default), [true, true]);
            assert_eq!(writers(None, default), [default, default]);
            assert_eq!(writers(Some("keys"), default), [default, default]);
        }
    }

    /// Every combination of the settings is a key of its own; all off is
    /// the key without them; early depth alone is what it was before the
    /// lock settings came, the three of them without the LDS order what
    /// they were before it came, and all four without the exchange what
    /// they were before that came; and every key without the walk is what
    /// it was before the walk came.
    #[test]
    fn each_combination_of_settings_has_cache_entries_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(
            key(&build, false, [false, false], false, false, false, false),
            build
        );
        let of = |build: &[u8; 32], n: u8| {
            key(
                build,
                n & 1 != 0,
                [n & 2 != 0, n & 4 != 0],
                n & 8 != 0,
                n & 16 != 0,
                n & 32 != 0,
                n & 64 != 0,
            )
        };
        let mut keys = std::collections::HashSet::new();
        for n in 0..128u8 {
            let k = of(&build, n);
            assert_eq!(k, of(&build, n));
            keys.insert(k);
            // Over another build (other settings, another translator).
            assert!(n == 0 || k != of(&[8u8; 32], n));
        }
        assert_eq!(keys.len(), 128);
        // The exchange and the level-0 size query, as before the walk.
        let mut two = blake3::Hasher::new();
        two.update(&build);
        two.update(b"halves exchange");
        two.update(b"uav size level 0");
        assert_eq!(of(&build, 48), *two.finalize().as_bytes());
        two = blake3::Hasher::new();
        two.update(&build);
        two.update(b"halves exchange");
        two.update(b"halves walk");
        assert_eq!(of(&build, 80), *two.finalize().as_bytes());
        let mut early = blake3::Hasher::new();
        early.update(&build);
        early.update(b"early depth");
        assert_eq!(
            key(&build, true, [false, false], false, false, false, false),
            *early.finalize().as_bytes()
        );
        let mut four = blake3::Hasher::new();
        four.update(&build);
        four.update(b"early depth");
        four.update(b"atomic lock helpers");
        four.update(b"atomic lock keys");
        four.update(b"lds order");
        assert_eq!(of(&build, 15), *four.finalize().as_bytes());
        let mut three = blake3::Hasher::new();
        three.update(&build);
        three.update(b"early depth");
        three.update(b"atomic lock helpers");
        three.update(b"atomic lock keys");
        assert_eq!(of(&build, 7), *three.finalize().as_bytes());
    }
}
