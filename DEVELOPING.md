# Developing

How this fork is worked on, kept short. The Mac details are in [docs/macos.md](docs/macos.md), and the design of VIctory itself is in [docs/architecture.md](docs/architecture.md).

## Branch

- `main-mac` is the working line. It updates by itself as work goes on, always as a fast-forward, so it's safe to track.
- It started from VIctory for macOS `main` (25 November 2026). It takes fixes from the Windows forks and merges the Mac port's `main` from time to time.

## Rules the code follows

- **Nothing from the game goes into git:** no executables, shaders, textures, keys, dumps or screenshots. `cargo xtask policy` checks this. Run it before every commit, or once run `cargo xtask install-hooks`.
- **One fix per commit.** A commit taken from another fork names its source at the start of the message, for example `from belobug da580e4e: ...`. Our own Mac changes start with `mac:`. This keeps a bad change easy to find and drop.
- **New behaviour goes behind a `VICTORY_*` switch,** off by default until it has been measured. Mac-only changes stay in their own files or functions, so fixes from the Windows forks still apply cleanly.
- **Build from source only.** Never run prebuilt binaries from any fork.

## Build and test

```bash
./setup.sh                    # first time: Wine, D3DMetal, VIctory
cargo test --workspace        # native tests
WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast   # GPU tests under Wine
cargo xtask policy            # asset check
```

`VICTORY_DIAGNOSTICS=1 tools/macos/victory.sh ...` builds with the debugging switches (logs, frame capture).

## Measuring a change

- **Use fixed views, so runs can be compared:**
  - the prison checkpoint, with a picture at the end;
  - the boat save: the opening cutscene (30 to 90 s after the first world frame) and the driving scene with the water running (180 to 270 s) are very different, so measure and name them apart;
  - the prison yard at 5 min 20 s into the opening.
- **Count frames a minute** in a fixed window. Close other apps. To claim a speed-up under 15%, run with and without in turns (A B A B). A switch that changes how shaders are translated rebuilds hundreds of pipelines each time it flips, so test it in blocks instead: one warm-up run, then three of each. A run-time switch can also be compared within one run: `VICTORY_AB=<SWITCH>:<seconds>` flips it every few seconds (an A/A run, `VICTORY_AB=none:10`, gave 0.0%, and a switch with a known 36% cost showed 36%).
- **Expect a slow state in some runs.** Lately about half the runs at the prison and the boat get slower partway through whatever the switches (GPU submissions wait 3 to 4 times longer), and the first boat run after a new build is often slow. Take three runs a side and report the spread. A slow run shows in the GPU driver's memory counters: `ioreg -r -c AGXAccelerator` "Alloc system memory" minus "In use system memory" stays under 1 GB in fast runs and rises above 1.5 GB (3 to 11 GB) in slow ones. Sample it every 30 s during the run and leave slow runs out of a comparison, or count them apart.
- **Don't build while the game boots.** Background builds slowed the game's start several times over here, so scripted key presses missed the start screens and runs never reached the world.
- **Measure GPU changes with frame replay,** without starting the game: one recorded frame replays in seconds. See [docs/replay.md](docs/replay.md) and `tools/macos/replay.sh`. Recordings contain game shaders, so keep them local and never commit them.
- **Look at the picture first, speed second.**
