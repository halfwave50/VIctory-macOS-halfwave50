//! Mac ([`Mac::code_table_loads`], `emit::Options::code_table_loads`; from
//! LakshanSivakumar/Native-Victory-macOS 578eb743): vector buffer loads
//! through a descriptor the program builds around its own constant table.
//!
//! The console compiler appends constant tables to the code (the spline
//! coefficients of an ACES output transform, say). It reads them with scalar
//! loads, global loads, or, per lane, with `buffer_load_dword* ... offen`
//! through a raw descriptor it builds from `s_getpc_b64`: dword 0-1 the
//! table's address, dword 2 its byte count, dword 3 `0x31016fac`
//! (`OOB_SELECT` 3). The translator read the first two kinds and refused
//! the third ("a descriptor built around a table in the code", from
//! [`Lower::handle`]). A load has no side effect, so that refusal is kept
//! as a missing value and the program goes on; it is reported only where
//! an instruction with an effect needs the value, and not at all where the
//! value only steers control flow. On the native port the world's 64^3
//! display LUT (`1b1365c6711630be`) came out as one value (0.2278 in all
//! 262,144 texels), and the tone map drew a flat grey world. Their corpus:
//! 115 of 21,600 programs newly translated, 206 outputs changed, all
//! through this load.
//!
//! With the setting, such a load reads the table's dwords, and 0 for a
//! dword whose end passes the byte count (RDNA2 ISA table 35, mode 3;
//! SOFFSET moves the address, not the bound).
//!
//! Without it, the translation is what it was. Either way each such load
//! is counted ([`take_code_table_loads`]), so that `translate --list` and
//! the host's translation log name the programs that have one.

use super::*;

thread_local! {
    /// Buffer loads through a descriptor around the code's table that the
    /// translations this thread made since [`take_code_table_loads`] met.
    static CODE_TABLE_LOADS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The count since the last call, and zeroes it. A program lowered more
/// than once (a retry) counts its loads each time: only "none" or "some"
/// is meant.
pub(crate) fn take_code_table_loads() -> u32 {
    CODE_TABLE_LOADS.with(|n| n.replace(0))
}

impl Lower<'_> {
    /// Counts a buffer access through a descriptor built around the code's
    /// table; whether [`Lower::code_buffer_load`] reads it.
    pub(super) fn code_table_load(&self) -> bool {
        CODE_TABLE_LOADS.with(|n| n.set(n.get() + 1));
        self.x.mac.code_table_loads
    }

    /// `buffer_load_dword*` through a descriptor the program builds around
    /// its constant table (`records` bytes in dword 2, no stride): the
    /// table's dwords, and 0 for a dword past `records`, as the console's
    /// range check gives. `byte` is the access's byte in the code (the
    /// descriptor's base included), `in_view` the part of it the range check
    /// covers (all but SOFFSET).
    // shortcut: every OOB_SELECT mode is taken as a check of each dword's
    // end (mode 3, the one the console compiler's tables use); an index
    // (`idxen`) is refused.
    pub(super) fn code_buffer_load(
        &mut self,
        inst: &Inst,
        (byte, in_view): (V, V),
        records: u32,
        indexed: bool,
        guard: Option<V>,
    ) -> Result<(), Why> {
        const OTHER: Why = "an access other than a dword load through a descriptor built around a table in the code";
        let count = match inst.name().strip_prefix("buffer_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x3") => 3,
            Some("x4") => 4,
            _ => return Err(OTHER),
        };
        if indexed {
            return Err("an indexed load through a descriptor built around a table in the code");
        }
        let Fields::Buffer {
            data: Some(Operand::Vector { reg, .. }),
            ..
        } = &inst.fields
        else {
            return Err("a buffer result");
        };
        let reg = *reg;
        let (zero, limit) = (self.s.m.i32(0), self.s.m.i32(records));
        for k in 0..count {
            let c = self.cur;
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, byte, four);
            let x = self.code_dword(at)?;
            let c = self.cur;
            let end = self.s.m.i32(4 * k + 4);
            let end = self.s.m.bin(c, Bin::Add, in_view, end);
            let fits = self.s.m.cmp(c, pred::IULE, end, limit);
            let x = self.s.m.select(c, fits, x, zero);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
