//! Buffer instructions (`buffer_*`, `tbuffer_*`): loads, stores and atomics
//! through a buffer descriptor, with the typed loads' format conversion.
//! Moved out of `lower.rs` as they were (it neared the policy's file size
//! limit); no change in behaviour.

use super::*;

impl Lower<'_> {
    /// A 32-bit typed load through independently annotated float and integer
    /// handles. The view's format is dynamic, so the overloads must be in
    /// mutually exclusive branches: executing `bufferLoad.i32` on a float view
    /// (or the reverse) is not a bitcast. Each result channel is an `i32` phi;
    /// the float path bitcasts only after the float load has taken place.
    fn typed_buffer_load(
        &mut self,
        float_h: V,
        integer_h: V,
        index: V,
        integer: V,
        count: u32,
    ) -> Vec<V> {
        let head = self.cur;
        let (integer_block, float_block, join) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(head, Term::CondBr(integer, integer_block, float_block));
        let iu = self.s.m.undef(self.s.t.i32);
        let mut integer_values = Vec::new();
        self.cur = integer_block;
        let r = self.s.op(
            integer_block,
            "bufferLoad.i32",
            Attr::ReadOnly,
            self.s.t.res_ret_i32,
            68,
            &[integer_h, index, iu],
        );
        for ch in 0..count {
            integer_values.push(self.s.m.extract(integer_block, r, ch));
        }
        self.s.m.terminate(integer_block, Term::Br(join));
        let mut float_values = Vec::new();
        self.cur = float_block;
        let r = self.s.op(
            float_block,
            "bufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            68,
            &[float_h, index, iu],
        );
        for ch in 0..count {
            let x = self.s.m.extract(float_block, r, ch);
            float_values.push(self.as_int(x));
        }
        self.s.m.terminate(float_block, Term::Br(join));
        self.cur = join;
        integer_values
            .into_iter()
            .zip(float_values)
            .map(|(integer, float)| {
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, vec![(integer, integer_block), (float, float_block)]);
                phi
            })
            .collect()
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.raw_buffer_load(c, &[h, at, u, mask, align]);
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            // A normalized format's ends are exactly 1.0 and -1.0. The host
            // may divide through a reciprocal and land just below 1.0: RDNA
            // 4 loaded 16_16_UNORM's 65535 as the half 0x3BFF, not 0x3C00.
            let x = match numeric {
                Numeric::Unorm => {
                    let top = (1u64 << w) - 1;
                    let q = if w <= 16 {
                        let q = self.norm_quotient(field, w);
                        self.as_float(q)
                    } else {
                        let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                        let max = self.s.m.f32((top as f32).to_bits());
                        self.s.m.bin(c, Bin::SDiv, f, max)
                    };
                    let (high, one) = (self.s.m.i32(top as u32), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::EQ, field, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let top = (1u32 << (w - 1)) - 1;
                    let q = if w <= 17 {
                        // The magnitude's quotient, signed (codes at and
                        // below -top are -1.0 below).
                        let (zero, sign) = (self.s.m.i32(0), self.s.m.i32(0x8000_0000));
                        let neg = self.s.m.cmp(c, pred::SLT, s, zero);
                        let minus = self.s.m.bin(c, Bin::Sub, zero, s);
                        let m = self.s.m.select(c, neg, minus, s);
                        let q = self.norm_quotient(m, w - 1);
                        let sign = self.s.m.select(c, neg, sign, zero);
                        let q = self.s.m.bin(c, Bin::Or, q, sign);
                        self.as_float(q)
                    } else {
                        let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                        let max = self.s.m.f32((top as f32).to_bits());
                        self.s.m.bin(c, Bin::SDiv, f, max)
                    };
                    // The lowest value is -1.0 too, as is the one above it.
                    let (high, one) = (self.s.m.i32(top), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::SGE, s, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    let low = self.s.m.i32(top.wrapping_neg());
                    let minus_one = self.s.m.f32((-1.0f32).to_bits());
                    let at_bottom = self.s.m.cmp(c, pred::SLE, s, low);
                    let q = self.s.m.select(c, at_bottom, minus_one, q);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self.f16_to_f32(c, h);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            // Console cache coherence: gl0 writes back and invalidates the
            // per-CU shader L0, gl1 invalidates GL1, so one wavefront's
            // writes become visible to others. D3D12 already guarantees UAV
            // coherence across wavefronts and dispatches, DXIL has no
            // equivalent instruction, and this project's host-visible
            // writeback comes from command-stream GpuWrite ranges rather
            // than shader cache hints. Ordering is kept by the surrounding
            // s_waitcnt/s_barrier, which the translator does lower.
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            Some("ubyte" | "sbyte" | "ushort" | "sshort") if !store => (false, 1),
            Some("byte" | "short") if store => (false, 1),
            _ => return Err(name),
        };
        // Loads or stores of a byte or a half: its width and whether signed.
        let sub = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
        {
            Some("ubyte" | "byte") => Some((8, false)),
            Some("sbyte") => Some((8, true)),
            Some("ushort" | "short") => Some((16, false)),
            Some("sshort") => Some((16, true)),
            _ => None,
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        let in_view = off;
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // Mac (`Mac::code_table_loads`): a descriptor built around the
        // code's table (`code_table`).
        if let Source::Built { address, words: [n, _], .. } = src
            && address.param == CODE
            && self.code_table_load()
        {
            return self.code_buffer_load(inst, (off, in_view), *n, index.is_some(), guard);
        }
        // A descriptor built around a root view has the stride the code puts
        // in dword 1 (bits 16-29) and no record limit (dword 2 is -1): an
        // index is a byte offset into the raw root view.
        let index = match (src, index) {
            (Source::Built { address, bits, words }, Some(ix))
                if address.param < self.x.native.layout.len() && !typed =>
            {
                if bits >> 30 != 0 || words[0] != u32::MAX {
                    return Err("a root view indexed with swizzling or a record limit");
                }
                let stride = self.s.m.i32(bits >> 16 & 0x3FFF);
                let byte = self.s.m.bin(c, Bin::Mul, ix, stride);
                off = self.s.m.bin(c, Bin::Add, off, byte);
                None
            }
            (_, ix) => ix,
        };
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let table = used.is_some_and(|u| u.root == RootType::Table);
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            if table {
                // A table's constant buffer is a raw view (scalar loads above).
                let Some(Operand::Vector { reg, .. }) = *data else {
                    return Err("a buffer result");
                };
                let (h, byte) = self.raw_view(site, src, byte, 4 * count, (11, 0), false, guard)?;
                for k in 0..count {
                    let four = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, byte, four);
                    let (mask, al) = (self.s.m.i8(1), self.s.m.i32(4));
                    let u = self.s.m.undef(self.s.t.i32);
                    let r = self.raw_buffer_load(c, &[h, a, u, mask, al]);
                    let x = self.s.m.extract(c, r, 0);
                    let x = self.as_int(x);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // A heap-indexed buffer read by element has no used-resource entry,
        // so no stride: its bytes are addressed as the hardware does, the
        // element times the view's stride (the host's table) plus the offset.
        let index = match index {
            Some(i) if used.is_none() && !typed => {
                let stride = self.view_stride(src)?;
                let rows = self.s.m.bin(c, Bin::Mul, i, stride);
                off = self.s.m.bin(c, Bin::Add, rows, off);
                None
            }
            index => index,
        };
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords, byte) = match index {
                Some(index) => {
                    let stride = Self::structured_stride(used)?;
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    let s = self.s.m.i32(stride);
                    let row = self.s.m.bin(self.cur, Bin::Mul, index, s);
                    let byte = self.s.m.bin(self.cur, Bin::Add, row, off);
                    (h, [index, off, iu], byte)
                }
                None => {
                    let (h, off) = self.raw_view(site, src, off, 4, (11 | u, 0), true, guard)?;
                    (h, [off, iu, iu], off)
                }
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.atomic_bin_op(self.cur, &args)
            };
            let bytes = self.s.m.i32(4);
            self.mark_view_write(h, byte, bytes, None);
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if let (true, Some((width, _))) = (store, sub) {
            // The field of the dword holding it, by two atomics on that dword
            // (clear, then set): other lanes' fields in it are left alone.
            // shortcut: the field is taken not to cross a dword.
            if index.is_some() {
                return Err("a byte or half buffer store by index");
            }
            let d = data.as_ref().ok_or("buffer store data")?;
            let x = self.vgprs(inst, d, 1)?[0];
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let field = self.s.m.i32(if width == 8 { 0xFF } else { 0xFFFF });
            let x = self.s.m.bin(c, Bin::And, x, field);
            let x = self.s.m.bin(c, Bin::Shl, x, sh);
            let mask = self.s.m.bin(c, Bin::Shl, field, sh);
            let all = self.s.m.i32(u32::MAX);
            let keep = self.s.m.bin(c, Bin::Xor, mask, all);
            let (h, at) = self.raw_view(site, src, at, 4, (11 | 0x1000, 0), true, guard)?;
            let begin = self.begin_guarded(guard);
            for (op, v) in [(1, keep), (2, x)] {
                let args = [h, self.s.m.i32(op), at, iu, iu, v];
                self.atomic_bin_op(self.cur, &args);
            }
            let bytes = self.s.m.i32(4);
            self.mark_view_write(h, at, bytes, None);
            self.end_guarded(begin, &[]);
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            let typed_h = if typed {
                Some(self.handle(site, src, false, (10 | u, TYPED_F32_PROPS))?)
            } else {
                None
            };
            if d16 {
                // d16 data is f16 for float and normalized formats, widened
                // for the 32-bit float store; for integer formats (the view's
                // kind, from the view-stride table) the 16-bit integer, zero-
                // or sign-extended (stored through the float-typed handle as
                // loads read: see there).
                let integer = typed_h.and_then(|h| self.integer_view(h));
                let c = self.cur;
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f = self.f16_to_f32(c, x);
                    let mut w = self.as_int(f);
                    if let Some((int, signed)) = integer {
                        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                        let zero_ext = self.s.m.bin(c, Bin::And, x, low);
                        let up = self.s.m.bin(c, Bin::Shl, x, k16);
                        let sign_ext = self.s.m.bin(c, Bin::AShr, up, k16);
                        let n = self.s.m.select(c, signed, sign_ext, zero_ext);
                        w = self.s.m.select(c, int, n, w);
                    }
                    wide.push(w);
                }
                vals = wide;
            }
            if typed && !d16 {
                // As for loads, a dynamic integer view gets a separately
                // annotated handle and a mutually exclusive integer store.
                if count != 4 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                if *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store at an offset");
                }
                let h = typed_h.ok_or("a typed buffer store")?;
                let element = index.unwrap_or(self.s.m.i32(0));
                let (byte, bytes) = if self.x.view_marks {
                    let at = self
                        .s
                        .view_index
                        .get(&h)
                        .copied()
                        .ok_or("a typed buffer store")?;
                    let stride = self.stride_at(at);
                    (self.s.m.bin(self.cur, Bin::Mul, element, stride), stride)
                } else {
                    (element, self.s.m.i32(0))
                };
                let integer = self.integer_view(h).map(|(integer, _)| integer);
                let integer_h = if integer.is_some() {
                    Some(self.handle(site, src, false, (10 | u, TYPED_I32_PROPS))?)
                } else {
                    None
                };
                let outer = self.begin_guarded(guard);
                self.mark_view_write(h, byte, bytes, None);
                if let (Some(integer), Some(integer_h)) = (integer, integer_h) {
                    let head = self.cur;
                    let (integer_block, float_block, join) = (
                        self.s.m.new_block(),
                        self.s.m.new_block(),
                        self.s.m.new_block(),
                    );
                    self.s.m.terminate(head, Term::CondBr(integer, integer_block, float_block));
                    let iu = self.s.m.undef(self.s.t.i32);
                    let mask = self.s.m.i8(0xF);
                    self.cur = integer_block;
                    let mut args = vec![integer_h, element, iu];
                    args.extend_from_slice(&vals);
                    args.push(mask);
                    self.s.op(
                        integer_block,
                        "bufferStore.i32",
                        Attr::NoUnwind,
                        self.s.t.void,
                        69,
                        &args,
                    );
                    self.s.m.terminate(integer_block, Term::Br(join));
                    self.cur = float_block;
                    let mut args = vec![h, element, iu];
                    args.extend(vals.iter().map(|&x| self.as_float(x)));
                    args.push(mask);
                    self.s.op(
                        float_block,
                        "bufferStore.f32",
                        Attr::NoUnwind,
                        self.s.t.void,
                        69,
                        &args,
                    );
                    self.s.m.terminate(float_block, Term::Br(join));
                    self.cur = join;
                } else {
                    let mut args = vec![h, element, iu];
                    args.extend(vals.iter().map(|&x| self.as_float(x)));
                    args.push(self.s.m.i8(0xF));
                    self.s.op(
                        self.cur,
                        "bufferStore.f32",
                        Attr::NoUnwind,
                        self.s.t.void,
                        69,
                        &args,
                    );
                }
                self.end_guarded(outer, &[]);
                return Ok(());
            }
            let mut args = Vec::new();
            // Where the write starts in the view, and how many bytes it may
            // cover (for its marks).
            let (byte, bytes);
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                if *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store at an offset");
                }
                let h = typed_h.ok_or("a typed buffer store")?;
                let element = index.unwrap_or(self.s.m.i32(0));
                args.push(h);
                args.extend([element, iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                // The element's size is the view's stride, not always 16. It
                // is needed at run time only for the write marks.
                (byte, bytes) = if self.x.view_marks {
                    let at = self
                        .s
                        .view_index
                        .get(&h)
                        .copied()
                        .ok_or("a typed buffer store")?;
                    let stride = self.stride_at(at);
                    (self.s.m.bin(self.cur, Bin::Mul, element, stride), stride)
                } else {
                    (element, self.s.m.i32(0))
                };
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = Self::structured_stride(used)?;
                        args.push(self.handle(site, src, false, (12 | u, stride))?);
                        args.extend([index, off]);
                        let s = self.s.m.i32(stride);
                        let row = self.s.m.bin(self.cur, Bin::Mul, index, s);
                        let width = self.s.m.i32(4 * count);
                        (byte, bytes) = (self.s.m.bin(self.cur, Bin::Add, row, off), width);
                    }
                    None => {
                        let n = 4 * count;
                        let (h, off) =
                            self.raw_view(site, src, off, n, (11 | u, 0), true, guard)?;
                        args.extend([h, off, iu]);
                        let width = self.s.m.i32(n);
                        (byte, bytes) = (off, width);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.mark_view_write(args[0], byte, bytes, None);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let (h, off) = self.raw_view(site, src, off, 4 * count, (11 | u, 0), false, guard)?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if let Some((width, signed)) = sub {
            // The dword holding the field, as `global_load_ubyte` and kin.
            // shortcut: the field is taken not to cross a dword.
            if index.is_some() {
                return Err("a byte or half buffer load by index");
            }
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let (h, at) = self.raw_view(site, src, at, 4, (11 | u, 0), false, guard)?;
            let c = self.cur;
            let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
            let r = self.raw_buffer_load(c, &[h, at, iu, mask, align]);
            let w = self.s.m.extract(c, r, 0);
            let w = self.as_int(w);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let x = self.s.m.bin(c, Bin::LShr, w, sh);
            let left = self.s.m.i32(32 - width);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let x = self
                .s
                .m
                .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        if typed && !d16 {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, TYPED_F32_PROPS))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            let xs = if let Some((integer, _)) = self.integer_view(h) {
                let integer_h = self.handle(site, src, false, (10 | u, TYPED_I32_PROPS))?;
                self.typed_buffer_load(h, integer_h, index, integer, count)
            } else {
                let r = self.s.op(
                    self.cur,
                    "bufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    68,
                    &[h, index, iu],
                );
                (0..count)
                    .map(|ch| {
                        let x = self.s.m.extract(self.cur, r, ch);
                        self.as_int(x)
                    })
                    .collect()
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let mut typed_h = None;
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, TYPED_F32_PROPS))?;
            typed_h = Some(h);
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = Self::structured_stride(used)?;
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    self.raw_buffer_load(c, &[h, index, off, mask, align])
                }
                None => {
                    let bytes = 4 * count;
                    let (h, off) =
                        self.raw_view(site, src, off, bytes, (11 | u, 0), false, guard)?;
                    self.raw_buffer_load(self.cur, &[h, off, iu, mask, align])
                }
            }
        };
        let c = self.cur;
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two 16-bit channels per VGPR (an odd last high half 0): f16 for
            // float and normalized formats; for integer formats (the view's
            // kind, from the view-stride table) the integer's low 16 bits.
            // Unlike the 32-bit accesses above, `_d16` keeps the established
            // widened-float path for narrow formats: an integer view is read
            // and written through the float-typed handle, so its channels' bits
            // travel as the driver passes a mismatched component type
            // (unchanged on the RX 6700 XT and WARP, `stride_tests`).
            let integer = typed_h.and_then(|h| self.integer_view(h));
            let c = self.cur;
            let k16 = self.s.m.i32(16);
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let mut h = self.f32_to_f16(c, f);
                    if let Some((int, _)) = integer {
                        let bits = self.as_int(f);
                        let low = self.s.m.i32(0xFFFF);
                        let low = self.s.m.bin(c, Bin::And, bits, low);
                        h = self.s.m.select(c, int, low, h);
                    }
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
