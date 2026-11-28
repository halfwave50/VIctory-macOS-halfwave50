# Notes for AI coding agents

- **What this is:** VIctory (a runtime that runs the Xbox Series build of GTA VI) built as a Windows program, run on Apple Silicon under CrossOver Wine plus Apple's D3DMetal. Rust workspace. The working branch is `main-mac`.
- **Start with:** `README.md` (status, how to run), `DEVELOPING.md` (rules, measuring), `WHATS-NEW.md` (latest changes), `docs/macos.md` (every Mac problem and fix), `docs/architecture.md` (VIctory's design).
- **Code map:**
  - `crates/victory-loader`: maps and patches the game image.
  - `crates/victory-hle`: the console runtime.
  - `crates/victory-gpu`: D3D12.X to D3D12, frame replay in `src/replay/`.
  - `crates/victory-shader`: the shader recompiler.
  - `tools/macos/`: the Wine build, the launchers and the test runner.
- **Commands:**
  - `cargo test --workspace`
  - `tools/macos/test-wine.sh --workspace` (GPU tests)
  - `cargo xtask policy` (must pass)
  - `tools/macos/play-main-mac.sh <exe>` (run with the measured settings)
- **Hard rule:** nothing from the game in git (executables, shaders, textures, keys, dumps, screenshots). The game files are the user's own, outside the repository.
- **Conventions:**
  - One fix per commit, with its source at the start of the message (`from <fork> <hash>:` or `mac:`).
  - New behaviour behind a `VICTORY_*` switch, off until measured.
  - Mac-only code kept apart, so fixes from the Windows forks still apply.
