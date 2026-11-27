//! Buffer loads through a descriptor a program builds around its own
//! constant table (`lower::code_table`, `Mac::code_table_loads`,
//! `Options::code_table_loads`; from LakshanSivakumar/Native-Victory-macOS
//! 578eb743, whose GPU test `addr_tests::a_buffer_load_through_a_
//! descriptor_around_the_code_table_reads_it_on_the_gpu` this follows).
//!
//! A hand-built GFX10 program of 32 lanes: `s_getpc_b64` plus the table's
//! offset, 16 bytes in dword 2, control word `0x31016fac` (OOB_SELECT 3),
//! and `buffer_load_dword ... offen` at byte `4 * (lane & 7)`. Lane `i`
//! reads dword `i & 7` of the table `[11, 22, 33, 44]`: the four past its
//! 16 bytes read 0. Each lane stores what it read to dword `i` of a raw UAV.

use super::lower::{self, Mac};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, range, resource};

const S_ENDPGM: u32 = 0xBF81_0000;
/// The table the program appends to its code.
const TABLE: [u32; 4] = [11, 22, 33, 44];

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u32) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// The program (code bytes): its root parameters are four root constants
/// (`s0..s3`, unused) and a table (`s4`) whose first entry is the raw UAV.
fn code() -> Vec<u8> {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load_dwordx4(8, 4)); // s[8:11]: the result UAV
    w.push(0x3402_0082); // v_lshlrev_b32 v1, 2, v0: dword i of the UAV
    let pc = w.len();
    w.push(0xBE8C_1F00); // s_getpc_b64 s[12:13]: the next instruction's byte
    w.extend([0x800C_FF0C, 0]); // s_add_u32 s12, s12, <the table's offset>
    w.push(0x820D_800D); // s_addc_u32 s13, s13, 0
    w.push(s_mov_b32(14, 0x90)); // s14 = 16: the table's bytes
    w.extend([s_mov_b32(15, 0xFF), 0x3101_6FAC]); // s15: raw, OOB_SELECT 3
    w.push(0x3604_0087); // v_and_b32 v2, 7, v0
    w.push(0x3404_0482); // v_lshlrev_b32 v2, 2, v2
    w.extend([0xE030_1000, 0x8003_0302]); // buffer_load_dword v3, v2, s[12:15], 0 offen
    w.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    w.extend([0xE070_1000, 0x8002_0301]); // buffer_store_dword v3, v1, s[8:11], 0 offen
    w.push(S_ENDPGM);
    let table = 4 * w.len() as u32;
    w[pc + 2] = table - 4 * (pc as u32 + 1);
    w.extend(TABLE);
    w.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn native() -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [32, 1, 1],
        }),
        code: code(),
    }
}

/// The program's lowering with or without the setting, or why it stopped.
fn lowered(native: &NativeProgram, code_table_loads: bool) -> Result<lower::Lowered, String> {
    let (p, st) =
        range::build(&native.code, native.wave64, native.vgprs).map_err(|e| e.to_string())?;
    let st = st.map_err(|e| e.to_string())?;
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(|e| e.to_string())?;
    let io = io::raise(&p, native).map_err(|e| e.to_string())?;
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Mac {
            code_table_loads,
            ..Mac::default()
        },
    })
    .map_err(|e| e.to_string())
}

/// Without the setting the load is a missing value (here the store that
/// needs it stops the program); with it the program translates. Each way
/// the load is counted.
#[test]
fn a_buffer_load_of_the_code_table_needs_the_setting_and_is_counted() {
    let program = native();
    let _ = lower::take_code_table_loads();
    let refused = match lowered(&program, false) {
        Ok(_) => panic!("translated without the setting"),
        Err(why) => why,
    };
    assert!(
        refused.contains("a descriptor built around a table in the code"),
        "{refused}"
    );
    assert!(
        lower::take_code_table_loads() > 0,
        "the load is counted without the setting"
    );
    let made = lowered(&program, true).expect("translated with the setting");
    assert!(!made.dxil.is_empty());
    assert!(
        lower::take_code_table_loads() > 0,
        "the load is counted with the setting"
    );
    assert_eq!(
        lower::take_code_table_loads(),
        0,
        "the count starts again at 0"
    );
}

/// On the host GPU (and WARP): lane `i` reads dword `i & 7` of the table,
/// and the four past its 16 bytes read 0.
#[cfg(windows)]
#[test]
fn a_buffer_load_of_the_code_table_reads_it_on_the_gpu() {
    use super::host::{Dispatch, Host, Root, View};
    use super::{HostShader, validator};
    let program = native();
    let l = lowered(&program, true).expect("translated with the setting");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_srv_bytes: l.root_srv_bytes,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..32)
        .map(|i| [11, 22, 33, 44, 0, 0, 0, 0][i % 8])
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = 0;
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
