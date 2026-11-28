//! mac: `VICTORY_NO_DEPTH_WRITE=<pixel shader hash>,...` (off by default).
//!
//! A graphics pipeline whose pixel shader is listed gets `DepthWriteMask`
//! ZERO; its depth test, depth function, stencil and everything else stay
//! as the title gave them. Made for the letterbox strip
//! (`notes/picture-next.md`, "Why the bar fill writes depth"): the title's
//! bar fill (VS `58d926085fca4999` + PS `edc1cad8ffb79bc9`) writes depth 1.0
//! (nearest, reversed Z) into the letterbox rows before the depth of field,
//! whose tile pass turns that into the largest blur and spreads it one tile
//! into the picture. A test of that cause, not a fix: the same pixel shader
//! draws other things (the final bars, without a depth target).
//!
//! Read once ([`hashes`]); unset or empty, [`apply`] returns at its first
//! line, once per pipeline the title creates and nothing per draw.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_DEPTH_STENCIL_DESC, D3D12_DEPTH_WRITE_MASK, D3D12_DEPTH_WRITE_MASK_ZERO,
};

use super::Shader;
use super::host::GraphicsState;

const NAME: &str = "VICTORY_NO_DEPTH_WRITE";

/// The 16-digit shader hashes of `value` (comma-separated, as the pipeline
/// lines print them), lower case; anything else is warned about and left out.
fn parse(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(hash, "{NAME} expects 16-digit pixel shader hashes");
                None
            }
        })
        .collect()
}

/// The listed pixel shader hashes, read once.
fn hashes() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let value = std::env::var(NAME).unwrap_or_default();
        let hashes = parse(&value);
        if !value.trim().is_empty() {
            tracing::info!(
                "{NAME}: pipelines with any of {} pixel shader hashes ({}) get no depth write; \
                 depth test and stencil unchanged",
                hashes.len(),
                hashes.join(",")
            );
        }
        hashes
    })
}

/// Reads the switch now (the host device is ready), so that its line is
/// logged at start and not with the title's first pipeline.
pub(crate) fn note() {
    let _ = hashes();
}

/// The pure rule: `depth` with its write mask zeroed when `ps` (the pipeline's
/// pixel shader hash) is in `hashes` and the mask is not zero already; the
/// old mask if it changed. Nothing else of `depth` changes.
fn transform(
    depth: &mut D3D12_DEPTH_STENCIL_DESC,
    ps: Option<&str>,
    hashes: &[String],
) -> Option<D3D12_DEPTH_WRITE_MASK> {
    let ps = ps?;
    if !hashes.iter().any(|h| h == ps) || depth.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ZERO {
        return None;
    }
    let old = depth.DepthWriteMask;
    depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    Some(old)
}

/// Applies the switch to a pipeline the title creates from a stream
/// (`read_stream`): logged once per changed pipeline (by its shaders and old
/// mask) at INFO.
pub(super) fn apply(shaders: &[Arc<Shader>], graphics: &mut GraphicsState) {
    let hashes = hashes();
    if hashes.is_empty() {
        return;
    }
    let hash_of = |stage: &str| {
        shaders
            .iter()
            .find(|s| s.stage == stage)
            .map(|s| s.hash.as_str())
    };
    let ps = hash_of("pixel");
    let Some(old) = transform(graphics.depth_mut(), ps, hashes) else {
        return;
    };
    let vs = hash_of("vertex")
        .or_else(|| hash_of("mesh"))
        .unwrap_or("none");
    let ps = ps.unwrap_or("none");
    static LOGGED: Mutex<Option<HashSet<(String, String, i32)>>> = Mutex::new(None);
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged
        .get_or_insert_with(HashSet::new)
        .insert((ps.to_owned(), vs.to_owned(), old.0))
    {
        let d = graphics.depth_mut();
        tracing::info!(
            ps,
            vs,
            old_write_mask = old.0,
            depth_enable = d.DepthEnable.as_bool(),
            depth_func = d.DepthFunc.0,
            stencil_enable = d.StencilEnable.as_bool(),
            "{NAME}: pipeline gets no depth write"
        );
    }
}

/// The replayer's form (`victory replay`, `crate::replay::play`): whether the
/// switch names the pixel shader of a recorded host pipeline, by its name
/// (`[vertex <hash>, pixel <hash>]`, as host pipelines are named).
pub(crate) fn replay_listed(name: &str) -> bool {
    listed_by_name(name, hashes())
}

fn listed_by_name(name: &str, hashes: &[String]) -> bool {
    hashes.iter().any(|h| name.contains(&format!("pixel {h}")))
}

/// The replayer's edit of a recorded depth-stencil subobject (types 11, 21
/// and 26; all three start with `DepthEnable`, `DepthWriteMask`): the write
/// mask zeroed; the old mask if it changed.
pub(crate) fn zero_write_mask(kind: i32, data: &mut [u8]) -> Option<u32> {
    if !matches!(kind, 11 | 21 | 26) {
        return None;
    }
    let mask = data.get_mut(4..8)?;
    let old = u32::from_le_bytes(mask.try_into().ok()?);
    if old == 0 {
        return None;
    }
    mask.copy_from_slice(&0u32.to_le_bytes());
    Some(old)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        D3D12_DEPTH_STENCILOP_DESC, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_STENCIL_OP_KEEP,
        D3D12_STENCIL_OP_REPLACE,
    };

    #[test]
    fn parses_full_hashes_only() {
        let got = parse(" EDC1CAD8FFB79BC9, ,58d926085fca4999,xyz,1234,edc1cad8ffb79bc9a");
        assert_eq!(got, ["edc1cad8ffb79bc9", "58d926085fca4999"]);
        assert!(parse("").is_empty());
        assert!(parse(" , ").is_empty());
    }

    fn bar_fill() -> D3D12_DEPTH_STENCIL_DESC {
        // The bar fill's state in capture-3 (PSO 12411), with stencil set
        // to values that must survive.
        D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL,
            StencilEnable: true.into(),
            StencilReadMask: 0xf0,
            StencilWriteMask: 0x0f,
            FrontFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_REPLACE,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            },
            BackFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_REPLACE,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL,
            },
        }
    }

    #[test]
    fn a_listed_pixel_shader_loses_only_the_depth_write() {
        let list = parse("edc1cad8ffb79bc9");
        let mut d = bar_fill();
        assert_eq!(
            transform(&mut d, Some("edc1cad8ffb79bc9"), &list),
            Some(D3D12_DEPTH_WRITE_MASK_ALL)
        );
        let want = D3D12_DEPTH_STENCIL_DESC {
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            ..bar_fill()
        };
        assert_eq!(d, want);
        // Already zero: no change to report.
        assert_eq!(transform(&mut d, Some("edc1cad8ffb79bc9"), &list), None);
        assert_eq!(d, want);
    }

    #[test]
    fn an_unlisted_pipeline_is_unchanged() {
        let list = parse("edc1cad8ffb79bc9");
        for ps in [Some("58d926085fca4999"), Some("456f02b52910aabc"), None] {
            let mut d = bar_fill();
            assert_eq!(transform(&mut d, ps, &list), None);
            assert_eq!(d, bar_fill());
        }
        let mut d = bar_fill();
        assert_eq!(transform(&mut d, Some("edc1cad8ffb79bc9"), &[]), None);
        assert_eq!(d, bar_fill());
    }

    #[test]
    fn the_replayer_edits_only_listed_pixel_shaders_write_masks() {
        let list = parse("edc1cad8ffb79bc9");
        assert!(listed_by_name(
            "[vertex 58d926085fca4999, pixel edc1cad8ffb79bc9]",
            &list
        ));
        assert!(!listed_by_name(
            "[vertex edc1cad8ffb79bc9, pixel 58d926085fca4999]",
            &list
        ));
        assert!(!listed_by_name(
            "[vertex 58d926085fca4999, pixel edc1cad8ffb79bc9]",
            &[]
        ));
        let d = bar_fill();
        // SAFETY: a plain struct read as its bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&d).cast::<u8>(),
                size_of::<D3D12_DEPTH_STENCIL_DESC>(),
            )
        };
        let mut data = bytes.to_vec();
        assert_eq!(
            zero_write_mask(11, &mut data),
            Some(D3D12_DEPTH_WRITE_MASK_ALL.0 as u32)
        );
        // SAFETY: as many bytes as the struct, read unaligned.
        let edited = unsafe {
            data.as_ptr()
                .cast::<D3D12_DEPTH_STENCIL_DESC>()
                .read_unaligned()
        };
        let mut want = bar_fill();
        want.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        assert_eq!(edited, want);
        assert_eq!(zero_write_mask(11, &mut data), None);
        let mut other = bytes.to_vec();
        assert_eq!(zero_write_mask(8, &mut other), None);
        assert_eq!(other, bytes);
    }
}
