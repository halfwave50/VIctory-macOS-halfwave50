//! An exact answer for a read across the two 32-lane halves of a console
//! wave64 ([`super::halves`]), through thread-group shared memory
//! ([`Mac::exchange`], `emit::Options::halves_exchange`;
//! `notes/ray-tracing-plan.md`).
//!
//! In halves a console wave is two host waves, and what one lane reads of a
//! lane in the other half has no host wave operation: such a read is
//! refused, or with half-local reads answered from the reader's own half,
//! which is not the console's value. Here the lanes hand the value over
//! instead. Every thread of the group stores its own value in a slot of its
//! own, the group meets at a barrier, and every thread then loads the slot
//! of the lane it reads:
//!
//! ```text
//! slots[set][thread] = value        every thread, its own slot
//! barrier (group sync, shared-memory fence)
//! result = slots[set][wave's first thread + lane]
//! ```
//!
//! `thread` is the relabelled flat thread id ([`Lower::relabelled_flat`]),
//! whose low six bits are the console lane and whose other bits name the
//! console wave, so a group of several console waves has all its waves'
//! lanes side by side and each wave reads its own.
//!
//! Where: only in a block every lane runs exactly once
//! ([`lds_order::once`]), where a group barrier is legal and where every
//! thread of the group stores before any loads. A read anywhere else is left
//! to the halves' other rules. With [`Mac::walk`] a half's answer is also
//! handed over in a block the group runs together, every thread once or
//! none ([`super::together`]), and `emit::walk` makes more blocks ones every
//! lane runs by leaving out the skips around them.
//!
//! With [`Mac::whole_wave`] (`emit::Options::halves_whole_wave`, off unless
//! asked for; `notes/far-ray.md`) also:
//!
//! * a lane's value where the group runs together ([`Lower::exchange_together`]),
//!   not only where every lane runs once, with a barrier after the loads
//!   (such a block may come round again) and no reuse of an earlier
//!   hand-over;
//! * blocks every thread runs as often as every other on a loop's body,
//!   none having ended ([`Lower::exchange_met`]): a lane's own `if` inside
//!   the body joins again before the loop goes round, so it changes nothing;
//! * the first lane under a narrowed EXEC, for `v_readfirstlane_b32`
//!   ([`Lower::exchanged_first`]): the console's EXEC there handed over as a
//!   ballot, its lowest lane, and that lane's value handed over. The
//!   title's bin offsets read so the base a bin's atomic returned.
//!
//! What is handed over:
//!
//! * a lane's value, for `v_readlane_b32` ([`Lower::exchanged_lane`]), as
//!   the console compiler ends a prefix sum over the wave's 64 lanes (the
//!   lower half's total, lane 31, added in the upper half): the form of
//!   five programs of the title's ray-tracing structure builder;
//! * lane 0's value, for `v_readfirstlane_b32` where EXEC is every lane
//!   ([`Lower::exchange_first_at`]): the first lane with its EXEC bit set
//!   is then lane 0. (The title's tiled shadow programs let lane 0 load a
//!   tile's header and then read it in every lane this way.) Under a
//!   narrowed EXEC the first lane is not known here and the read is left
//!   as it was;
//! * a half's answer, one dword per host wave
//!   ([`Lower::exchanged_halves`]): its 32 ballot bits, for a lane mask read
//!   as data (the two halves' dwords are the console's mask), and its part
//!   of an integer reduction or its total for a prefix sum (the two parts
//!   combined, or the lower half's total added in the upper half, are the
//!   console's). Each such exchange has slots of its own, stored once.
//!
//! Why the result is the console's:
//!
//! 1. Each host lane is one console lane and holds that lane's value: a
//!    register a lane did not write keeps its old value in the per-lane
//!    form as under the console's EXEC, and `v_readlane_b32` reads the
//!    register whatever EXEC is. So `slots[set][t]` gets what console thread
//!    `t` holds at the instruction.
//! 2. The block is run once by every thread, so each thread stores once,
//!    then waits at the barrier, then loads. The barrier lets no thread on
//!    before all have arrived, and fences shared memory: every load sees
//!    every store of this exchange.
//! 3. Nothing else writes the set meanwhile. The slots are the exchange's
//!    own memory (the program's LDS is another array). Lane exchange `n`
//!    uses set `n & 1`. Its loads lie between its own barrier and the next
//!    exchange's barrier; the stores to the same set are its own, before its
//!    barrier, and those of exchange `n + 2`, after the barrier of exchange
//!    `n + 1`. A barrier divides the group's work into all of before and all
//!    of after, so no store of the set falls among the loads. (With one set
//!    the next exchange's stores would: some thread still loading while
//!    another already stores. The model test shows both.)
//! 4. A second read of the same value right after (no exchange between)
//!    loads from the same set with no new barrier, by 3.
//! 5. A half's answer: every lane of a host wave stores the same dword in
//!    the wave's slot, so the store's winner does not matter; the slots are
//!    that exchange's alone and its block runs once, so nothing stores there
//!    again. The answer is the half's own by the host's wave operation over
//!    the lanes of its wave, all of which are in the block (it runs for
//!    every lane): the same operation a host of 64-lane waves answers for
//!    the whole wave. A ballot may be taken at the end of an earlier block
//!    than the one being lowered (the block that wrote the mask): its
//!    barrier then comes before barriers made earlier, which 3 allows (a
//!    barrier more between two others moves no store among loads).
//!
//! What it rests on, and nothing more: the relabelling (each host wave
//! holds a 32-aligned run of flat thread ids, as every program in halves
//! assumes) and a group barrier that syncs shared memory across the host
//! waves of a group (as [`split_wave_barriers`] and [`lds_order`] assume).
//! No wave operation is asked about other lanes: the host's ballots, lane
//! reads and helper-lane answers are not used.
//!
//! Not exact here, as before: a lane the group does not have (a wave cut
//! short by the group's size) reads a slot nothing stored, where the console
//! reads a register nothing wrote. (A half the group does not have counts
//! as no lanes: no ballot bits, the reduction's identity.) The loop cap
//! ([`LOOP_LIMIT`]) sends a lane out of the program, which would then miss
//! the barrier; the same holds for the barriers of [`split_wave_barriers`].
//! Float reductions are not handed over: a sum's rounding order is the
//! console's own.
//!
//! The exchange through group-shared memory between two barriers is
//! belobug's (Belobug/VIctory-Belobug-AMD 5eac4fa, `Lower::exchange`, two
//! dwords for the halves' ballots) and hyphennn's (fxceless/victory-fork
//! 578d459, `exchange_halves`); both run every lane through every branch
//! to place it anywhere, which hung an NVIDIA card with barriers in loops
//! (e9765a7 there). This form keeps the per-lane control flow, stays out of
//! divergent code, stores per lane instead of per half, and needs one
//! barrier where they need two.

use super::*;

/// D3D12's limit on a compute shader's thread-group shared memory, in bytes.
const SHARED_BYTES: u32 = 32768;

thread_local! {
    /// While [`wanted`] runs a lowering: the IR blocks where it asked for an
    /// exchange and not every lane runs the block exactly once.
    static WANTED: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `lowering` (of one program, on this thread) and returns, with what
/// it returns, the IR blocks where it asked for an exchange that could not
/// be made because only some lanes run the block, or run it more than once:
/// the places `emit::walk` opens to every lane.
pub(crate) fn wanted<T>(lowering: impl FnOnce() -> T) -> (T, Vec<usize>) {
    WANTED.with(|w| *w.borrow_mut() = Some(Vec::new()));
    let made = lowering();
    let blocks = WANTED.with(|w| w.borrow_mut().take()).unwrap_or_default();
    (made, blocks)
}

/// Mac ([`Mac::whole_wave`]): while [`wanted`] runs a lowering of a one-wave
/// group whose LDS no barrier orders yet, the blocks that access LDS, so
/// that `emit::walk` opens the skips round them and the walked form can
/// place its barriers where every lane runs (`split_wave_barriers`).
pub(super) fn want_lds(p: &Program) {
    WANTED.with(|w| {
        if let Some(blocks) = w.borrow_mut().as_mut() {
            for (b, block) in p.blocks.iter().enumerate() {
                let lds = block.insts.iter().any(|i| {
                    let name = i.name();
                    matches!(i.fields, Fields::Ds { .. })
                        && !name.starts_with("ds_swizzle")
                        && !name.contains("permute")
                });
                if lds {
                    blocks.push(b);
                }
            }
        }
    });
}

/// What the exchanges ask of the program alone, not of a lowering's state:
/// kept across the retries of one lowering ([`lower_pruned`]), each of which
/// starts a new [`Lower`] and would work them out again. A phi that does not
/// settle starts the lowering over, and one sun-shadow program the walk opens
/// was lowered 222 times (74 retries in each of three forms), these most of
/// each: kept, it took 23 s instead of 70 natively on an M1 Pro (286 s in a
/// run under Wine). A retry still notes in its own [`State`] when it first
/// asks, so it lowers as it did on its own.
#[derive(Default)]
pub(super) struct Kept {
    once: std::cell::OnceCell<Vec<bool>>,
    together: std::cell::OnceCell<together::Together>,
    met: std::cell::OnceCell<Option<Vec<bool>>>,
}

/// The exchanges of one program.
#[derive(Default)]
pub(super) struct State {
    /// Blocks every lane runs exactly once, when first asked.
    once: Option<Vec<bool>>,
    /// Blocks a thread group runs together ([`together`]), when first
    /// asked.
    together: Option<together::Together>,
    /// The slots: two sets of one dword per thread of the group (rounded
    /// up to whole console waves), group-shared.
    slots: Option<V>,
    /// Exchanges made.
    pub made: u32,
    /// Those of lane values, which use the two sets in turn. Counted apart:
    /// an exchange of a half's answer may be placed before earlier ones, and
    /// must not make two lane exchanges in a row use one set.
    pub lanes: u32,
    /// The value the last exchange handed over, and the first slot of its
    /// set.
    last: Option<(V, u32)>,
    /// With [`Mac::whole_wave`]: blocks every thread of the group runs as
    /// often as every other, none having ended ([`lds_rounds::met_with`]),
    /// when first asked.
    met: Option<Option<Vec<bool>>>,
    /// Bytes of group-shared memory the exchanges have taken.
    bytes: u32,
}

impl Lower<'_> {
    /// The group's threads in whole console waves: the slots of a set.
    fn exchange_slots(&self) -> Option<u32> {
        let threads: u32 = self.s.threads?.iter().product();
        Some(threads.div_ceil(64) * 64)
    }

    /// Takes `bytes` of group-shared memory for an exchange, if they fit
    /// beside the program's LDS and the exchanges before.
    fn exchange_takes(&mut self, bytes: u32) -> bool {
        let fits = self.s.lds_dwords * 4 + self.exchange.bytes + bytes <= SHARED_BYTES;
        if fits {
            self.exchange.bytes += bytes;
        }
        fits
    }

    /// Whether IR block `b` is one every lane runs exactly once, in a
    /// program in halves that is asked to exchange.
    fn exchange_in(&mut self, b: usize) -> bool {
        let p = self.x.p;
        if !(self.halves && self.x.mac.exchange) {
            return false;
        }
        let kept = self.kept;
        let once = (self.exchange.once)
            .get_or_insert_with(|| kept.once.get_or_init(|| lds_order::once(p)).clone())[b];
        if !once {
            WANTED.with(|w| w.borrow_mut().as_mut().map(|blocks| blocks.push(b)));
        }
        once
    }

    /// The group barrier in block `at`.
    fn exchange_barrier(&mut self, at: usize) {
        let cur = std::mem::replace(&mut self.cur, at);
        self.group_barrier();
        self.cur = cur;
    }

    /// What console lane `lane` of the reader's wave holds in `x`, for every
    /// lane of the group, exact across the halves (the module notes); `None`
    /// where the exchange does not apply: not asked for, not in halves, or
    /// not in a block every lane runs exactly once.
    pub(super) fn exchanged_lane(&mut self, x: V, lane: V) -> Option<V> {
        let per_set = self.exchange_slots()?;
        // Mac (`whole_wave`): also in a block the group runs together
        // ([`Lower::exchange_together`], with [`Lower::exchange_met`]), with
        // no reuse of an earlier hand-over (such a block may come round
        // again, or be gone round) and a barrier after the loads.
        let together = !self.exchange_in(self.cur_ir);
        if together && !(self.x.mac.whole_wave && self.exchange_together(self.cur_ir)) {
            return None;
        }
        if together {
            self.exchange.last = None;
        }
        let i32t = self.s.t.i32;
        let slots = match self.exchange.slots {
            Some(g) => g,
            None => {
                if !self.exchange_takes(per_set * 8) {
                    return None;
                }
                let arr = self.s.m.array(2 * per_set, i32t);
                let g = self.s.m.global("exchange", arr, 3);
                self.exchange.slots = Some(g);
                g
            }
        };
        let flat = self.relabelled_flat();
        let c = self.cur;
        let base = match self.exchange.last {
            Some((v, base)) if v == x => base,
            _ => {
                let base = (self.exchange.lanes & 1) * per_set;
                // The thread's own slot; the relabelled id is below the
                // set's size while host waves hold aligned runs of ids, and
                // is held there otherwise so no store leaves the array.
                let n = self.s.m.i32(per_set);
                let inside = self.s.m.cmp(c, pred::IULT, flat, n);
                let zero = self.s.m.i32(0);
                let own = self.s.m.select(c, inside, flat, zero);
                let k = self.s.m.i32(base);
                let own = self.s.m.bin(c, Bin::Add, own, k);
                let at = self.s.m.element(c, slots, own);
                self.s.m.store(c, at, x);
                self.group_barrier();
                self.exchange.made += 1;
                self.exchange.lanes += 1;
                self.exchange.last = Some((x, base));
                base
            }
        };
        // The slot of that lane of the reader's console wave.
        let c = self.cur;
        let (wave, in_wave) = (self.s.m.i32(!63), self.s.m.i32(63));
        let first = self.s.m.bin(c, Bin::And, flat, wave);
        let lane = self.s.m.bin(c, Bin::And, lane, in_wave);
        let from = self.s.m.bin(c, Bin::Or, first, lane);
        let n = self.s.m.i32(per_set);
        let inside = self.s.m.cmp(c, pred::IULT, from, n);
        let zero = self.s.m.i32(0);
        let from = self.s.m.select(c, inside, from, zero);
        let k = self.s.m.i32(base);
        let from = self.s.m.bin(c, Bin::Add, from, k);
        let at = self.s.m.element(c, slots, from);
        let r = self.s.m.load(c, at);
        if together {
            // The next lane exchange may store to this set: none before
            // every thread has loaded (on a cycle, this one next round).
            self.exchange.last = None;
            self.group_barrier();
        }
        Some(r)
    }

    /// Mac (`whole_wave`): what `v_readfirstlane_b32` `inst` reads of `x`
    /// under a narrowed EXEC, exact across the halves: the console's EXEC
    /// at `inst` as a ballot handed over ([`Lower::exchanged_ballot`]), its
    /// lowest lane (lane 0 where it is empty, as the hardware reads), and
    /// that lane's value handed over ([`Lower::exchanged_lane`]). `None`
    /// where either hand-over cannot be placed: not asked for, not in
    /// halves, a block not every thread runs alike, an EXEC not known per
    /// lane.
    pub(super) fn exchanged_first(&mut self, inst: &Inst, x: V) -> Option<V> {
        if !(self.halves && self.x.mac.exchange && self.x.mac.whole_wave) {
            return None;
        }
        let b = self.cur_ir;
        let exec = inst.read(Reg::S(EXEC))?;
        let &m = self.x.st.masks.get(&exec)?;
        // Both hand-overs, or neither.
        let lanes = self.exchange_in(b) || self.exchange_together(b);
        if !(lanes && self.exchange_halves_in(b)) {
            return None;
        }
        let c = self.cur;
        let bit = self.boolean(m, c).ok()?;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let low = self.s.m.extract(c, r, 0);
        let (lo, hi) = self.exchanged_ballot(c, low);
        // The lowest set bit of the 64; lane 0 for none.
        let c = self.cur;
        let (zero, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        let flo = self.unary_bits(c, 32, lo);
        let fhi = self.unary_bits(c, 32, hi);
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        let none_hi = self.s.m.cmp(c, pred::EQ, hi, zero);
        let fhi = self.s.m.select(c, none_hi, zero, fhi);
        let some_lo = self.s.m.cmp(c, pred::NE, lo, zero);
        let first = self.s.m.select(c, some_lo, flo, fhi);
        self.exchanged_lane(x, first)
    }

    /// Whether the first-lane read `inst` (of the block being lowered) reads
    /// lane 0 where the exchange applies: a block every lane runs once, and
    /// EXEC every lane there (no guard), so that the first lane with its
    /// EXEC bit set is lane 0, which every console wave of a group has.
    pub(super) fn exchange_first_at(&mut self, inst: &Inst) -> bool {
        let b = self.cur_ir;
        let i = self.x.p.blocks[b]
            .insts
            .iter()
            .position(|x| std::ptr::eq(x, inst));
        self.exchange_in(b) && i.is_some_and(|i| self.x.st.guards[b][i].is_none())
    }

    /// Whether a half's answer made in IR block `b` can be handed over
    /// ([`Lower::exchanged_halves`]): the block is one every lane runs once
    /// (or, with [`Mac::walk`], one the group runs together) and the slots
    /// fit. Takes the slots.
    pub(super) fn exchange_halves_in(&mut self, b: usize) -> bool {
        match self.exchange_slots() {
            Some(per_set) => {
                (self.exchange_together(b) || self.exchange_in(b))
                    && self.exchange_takes(per_set / 8)
            }
            None => false,
        }
    }

    /// Whether IR block `b` is one a thread group runs together (every
    /// thread as often as every other, or none: [`together`]), in a program
    /// in halves asked to exchange there too ([`Mac::walk`]). A half's
    /// answer has slots of its own, so the module notes' 5 holds for it
    /// there as in a block every lane runs: the group's threads all store,
    /// meet and load, or none does; on a cycle they do so each round
    /// ([`Lower::exchange_again`]).
    fn exchange_together(&mut self, b: usize) -> bool {
        let (x, kept) = (self.x, self.kept);
        let together = self.halves
            && x.mac.exchange
            && x.mac.walk
            && (self.exchange.together)
                .get_or_insert_with(|| kept.together.get_or_init(|| together::blocks(x)).clone())
                .blocks[b];
        together || self.exchange_met(b)
    }

    /// Whether IR block `b` is one every thread of the group runs as often
    /// as every other, with no thread ended on the way
    /// ([`lds_rounds::met_with`]), in a program in halves asked to exchange
    /// there ([`Mac::whole_wave`], with [`Mac::walk`]). That is
    /// [`together`]'s premise for a hand-over, by a finer rule: a branch
    /// whose two ways join again before the block (a lane's own `if` in a
    /// loop's body) does not change how often the block runs. Ended threads
    /// are left out: an ended half would leave its slot unwritten.
    fn exchange_met(&mut self, b: usize) -> bool {
        let (x, kept) = (self.x, self.kept);
        if !(self.halves && x.mac.exchange && x.mac.walk && x.mac.whole_wave) {
            return false;
        }
        // (`together` keeps the cycles for `exchange_again`.)
        (self.exchange.together)
            .get_or_insert_with(|| kept.together.get_or_init(|| together::blocks(x)).clone());
        let met = (self.exchange.met).get_or_insert_with(|| {
            kept.met
                .get_or_init(|| lds_rounds::met_with(x, false))
                .clone()
        });
        met.as_ref().is_some_and(|m| m[b])
    }

    /// Whether the hand-over written in DXIL block `at` is in an IR block on
    /// a cycle, so that the group stores to its slots again next round (a
    /// block run together; one every lane runs once is on none). `at` is
    /// the block being lowered or the last block of the IR block that wrote
    /// the mask ([`Lower::mask_dword`]).
    fn exchange_again(&self, at: usize) -> bool {
        let ir = self.tail.iter().position(|t| *t == Some(at));
        let ir = ir.unwrap_or(self.cur_ir);
        (self.exchange.together.as_ref()).is_some_and(|t| t.looped[ir])
    }

    /// `x`, a dword every lane of a host wave holds alike (the half's
    /// answer), as the lower and the upper half of the reader's console
    /// wave hold it, in DXIL block `at` of an IR block
    /// [`Lower::exchange_halves_in`] accepted; `missing` for an upper half
    /// the group does not have. The module notes, 5.
    pub(super) fn exchanged_halves(&mut self, at: usize, x: V, missing: V) -> (V, V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let waves = threads.div_ceil(64) * 2;
        let i32t = self.s.t.i32;
        let arr = self.s.m.array(waves, i32t);
        let name = format!("exchange{}", self.exchange.made);
        let slots = self.s.m.global(&name, arr, 3);
        self.exchange.made += 1;
        let flat = self.relabelled_flat();
        let (five, n, zero) = (self.s.m.i32(5), self.s.m.i32(waves), self.s.m.i32(0));
        let wave = self.s.m.bin(at, Bin::LShr, flat, five);
        let inside = self.s.m.cmp(at, pred::IULT, wave, n);
        let wave = self.s.m.select(at, inside, wave, zero);
        let own = self.s.m.element(at, slots, wave);
        self.s.m.store(at, own, x);
        self.exchange_barrier(at);
        let (even, one) = (self.s.m.i32(!1), self.s.m.i32(1));
        let lower = self.s.m.bin(at, Bin::And, wave, even);
        let upper = self.s.m.bin(at, Bin::Or, wave, one);
        let from = self.s.m.element(at, slots, lower);
        let lo = self.s.m.load(at, from);
        let from = self.s.m.element(at, slots, upper);
        let hi = self.s.m.load(at, from);
        // The last console wave of a group that ends in its lower half.
        let host_waves = threads.div_ceil(32);
        let hi = if host_waves % 2 == 1 {
            let k = self.s.m.i32(host_waves);
            let there = self.s.m.cmp(at, pred::IULT, upper, k);
            self.s.m.select(at, there, hi, missing)
        } else {
            hi
        };
        // In a loop the next round stores to these slots again: no thread
        // may get there while another still loads this round's.
        if self.exchange_again(at) {
            self.exchange_barrier(at);
        }
        (lo, hi)
    }

    /// The console wave's ballot from a half's 32 bits `lo`
    /// ([`Lower::exchanged_halves`]): the mask's low and high dword.
    pub(super) fn exchanged_ballot(&mut self, at: usize, lo: V) -> (V, V) {
        let none = self.s.m.i32(0);
        self.exchanged_halves(at, lo, none)
    }

    /// An integer reduction over the console wave from the half's own
    /// result `r` (`prefix`: the half's exclusive or inclusive prefix sum
    /// in this lane, with `total` the half's sum), in the current block;
    /// `None` for an operation not handed over (floats).
    pub(super) fn exchanged_reduction(
        &mut self,
        op: super::super::wave::Op,
        r: V,
        prefix_total: Option<V>,
        ident: u32,
    ) -> Option<V> {
        use super::super::wave::Op;
        let c = self.cur;
        let id = self.s.m.i32(ident);
        if let Some(total) = prefix_total {
            // The lanes below a lane of the upper half are the lower half's
            // and its own.
            let zero = self.s.m.i32(0);
            let (below, _) = self.exchanged_halves(c, total, zero);
            let lane = self.console_lane();
            let half = self.s.m.i32(32);
            let upper = self.s.m.cmp(c, pred::IUGE, lane, half);
            let add = self.s.m.select(c, upper, below, zero);
            return Some(self.s.m.bin(c, Bin::Add, r, add));
        }
        let (lo, hi) = self.exchanged_halves(c, r, id);
        let pick = |l: &mut Self, p: u8| {
            let first = l.s.m.cmp(c, p, lo, hi);
            l.s.m.select(c, first, lo, hi)
        };
        Some(match op {
            Op::AddU => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinU => pick(self, pred::IULT),
            Op::MaxU => pick(self, pred::IUGT),
            Op::MinI => pick(self, pred::SLT),
            Op::MaxI => pick(self, pred::SGT),
            Op::AddF | Op::MinF | Op::MaxF | Op::AddH => return None,
        })
    }
}
