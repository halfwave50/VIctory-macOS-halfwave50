# Halfwave: VIctory for Apple Silicon Macs

<img width="1600" alt="GTA VI on an M4 Pro: player control in the prison cell at sunset, VIctory's fps line in the window title" src="https://github.com/user-attachments/assets/3ef3c9ee-b506-40e6-a2e9-7ec69acc906a" />

**Halfwave** runs the Xbox Series build of GTA VI on an Apple Silicon Mac (Wine plus Apple's D3DMetal), from a copy of the game that you own. It combines our own Mac fixes with fixes from the Windows forks, and every change is measured on a MacBook Pro M4 Pro (48 GB).

**Source code only, nothing from the game.** You need your own copy of the game.

## Status (28 November 2026)

**It works:**
- The game loads the world and gives player control. Saved games load, free roam included.
- Sunlight and shadows are right, rooms have fill light, and characters and clothes draw.
- Screen reflections reach the picture: water reflects the sky, and the steel objects in the prison cell get light.
- The water simulation runs, so the boat's wake and spray show.

**Speed:**

| Place | Series X profile | With 1920x1080 output |
|---|---|---|
| Prison checkpoint | 3.1–3.4 fps | 3.0 fps |
| Boat mission opening (cutscene on the boat) | 4.7–5.0 fps | 4.9 fps |
| Driving the boat (play, "Follow the plane") | | 2.0 fps (median of a 9-minute play run) |

Prison numbers are for the still view once the camera has settled; the first minute runs at 2.3–2.8 fps. The boat-opening numbers are for the cutscene that starts the mission, measured 30–90 s after the first world frame. Earlier figures of 4.7 fps (6.0 at 1920x1080) were the same cutscene. It had dropped to 4.1 fps with a wider flicker setting in the launcher; back on the narrower one it runs at 4.7–5.0 (5.0 in three runs on the 28 November build). Driving is the speed you get in play: about 2 fps. There Apple's Metal HUD shows the GPU busy for only 75–100 ms of a 500 ms frame, so most of the time goes to waiting, most likely between GPU submissions. The next speed work targets it. The prison 1920x1080 number comes from an earlier run.

**Not yet:**
- Playable speed. The next goal is 5 fps, then 10 fps at 1080p.
- Some runs fall into a slow state partway through: about 1 boat run in 3, and lately some prison runs (2 of the last 3 on the default settings, at about 2.2 fps in the first minute). Each GPU submission then waits 3 to 4 times longer while the GPU itself is mostly idle; samples put the extra time inside macOS's GPU submit call. The cause is not found yet; memory pressure is the main guess (the game uses 34–35 GB of 48 here).
- In free roam some objects flicker black for one frame. The launcher fixes the main cause: a texture slot the game rewrites keeps its old texture until the new one is ready. It is not yet checked by eye in free roam. The launcher also keeps short button taps and lets the world run closer to real time; neither is tried in play yet.
- In one play run the controller stopped responding soon after driving began. The launcher now restarts a pad read that stops (`VICTORY_PAD_REARM=1`) and doesn't send the game's rumble to the pad, the likeliest cause; not tried with a pad yet.
- One ray-traced reflection pass is still left out by default, because it drew rectangular blocks on water. Its fix is in, behind `VICTORY_HALVES_WHOLE_WAVE=1` (off by default): the programs that sort and count rays work across a 64-lane wave, and here each 32-lane half answered alone or raced the other. With the switch on and the pass back in (`VICTORY_SKIP_PIPELINES=` empty), the boat pictures show no blocks and the cutscene runs at 4.9–5.0 fps (two runs), the same as without. A narrower form keeps the fix's new barriers to the ray programs only (`VICTORY_HALVES_WHOLE_WAVE_LDS=<program list>`); in four boat runs the water stayed clean at the defaults' speed. It is not a launcher default yet. The game's ray-tracing structures build on the Mac (`VICTORY_BVH_BUILDER=1`, off). The builder alone kept 5.0 fps in one boat run, but once the rays reach all their memory (`VICTORY_POINTER_PAGES=3`) the cutscene drops to about 1.1–1.3 fps, so ray tracing waits for more speed.
- Distant trees and buildings appear 5 to 35 s late after a camera cut, so wide shots miss them. The game streams them in too slowly here.
- Night streets and rooms look flatter than on the console, because our fill light's minimum level outweighs the lamps. The launcher now uses a lower minimum (0.1 instead of 0.3), which brings back part of the contrast. One black patch on water.
- At 1920x1080 output the game's interface is drawn twice too large.

Recent changes: [WHATS-NEW.md](WHATS-NEW.md).

## How it works

| Part | What it does | Status |
|---|---|---|
| Loader | VIctory's PE loader maps the game's Xbox Series executable into a Windows x64 process under Wine, which runs on Apple Silicon through Rosetta 2 | ✅ |
| Console runtime | VIctory's HLE layer stands in for the console's system libraries (`xgameruntime`, `d3d12_xs`) | ✅ |
| GPU layer | Turns the console's D3D12 calls, descriptors and command lists into PC D3D12, which Apple's D3DMetal runs on Metal | ✅ world, sunlight, shadows, reflections · 🟡 one ray-traced pass left out |
| Shader recompiler | Recompiles the console GPU's RDNA 2 shaders to HLSL and DXIL; D3DMetal turns DXIL into Metal | ✅ all 69 shader checks pass on Metal |
| Textures | Detiles the console's tiled textures for the Mac GPU | ✅ |
| Speed | | 🟡 2 fps driving, 3–3.4 in the prison cell, up to 5.0 in cutscenes (M4 Pro); the goal is 5 fps, then 10 at 1080p |

Tests: `cargo test --workspace` runs the loader, HLE and shader tests; [docs/macos.md](docs/macos.md) has the results under Wine.

## Tested on

| Mac | Memory | Speed | Note | When (UTC) | Build |
|---|---|---|---|---|---|
| MacBook Pro M4 Pro | 48 GB | 3.1–3.4 fps prison, 2.0 driving the boat at 1080p; 4.8–4.9 boat opening cutscene (4.9 at 1080p) | Our test Mac | 27 Nov 17:35 | [2204c12](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/2204c1234360) |
| MacBook Pro M5 Pro | 24 GB | 3.9 fps prison cell (259 ms; 4K output, scale 0.5, render targets sized by the scale, `VICTORY_TIME_STEP_MAX=0.25`); 3.7 fps (273 ms) without | "the cell holds 259 ms (3.9 fps) at 12.5 GB wired" ([1](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7206500809), [2](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7208611413), [3](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1#issuecomment-7231869693)) | 28 Nov 01:43 | main-mac f230de6 + local patches |
| MacBook Air M5 | 16 GB | 1.9 fps at the start (Series S), slowing to 0.25 fps over a 20-minute run | "The best result I've seen on my 16 GB M5 MacBook Air" ([report](https://github.com/pragyaangaur/VIctory-macOS/issues/455)) | 27 Nov 01:24 | 8616bd6 + memory marks (#456) |
| MacBook Pro M2 Pro | 32 GB | 1.3 fps prison cell (2.7 fps at 1080p); 1.4 fps prologue yard (0.7 upstream) | "The picture is very close to your screenshot"; no world stall in 10 runs ([1](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1#issuecomment-7207381523), [2](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1#issuecomment-7208668352)) | 27 Nov 03:03 | main-mac + upstream #445 |
| Mac mini M5 Pro (Mac17,16) | 24 GB | 6.1 fps prison cell (163 ms median; 1080p output, scale 0.5); 5.1 fps (195 ms) before their picks; 4.1 fps (242 ms) on 27 Nov | Upstream picks, guard work and flights on; "21 minutes in the cell on the final build, with no panic and no crash" ([PR #8](https://github.com/halfwave50/VIctory-macOS-halfwave50/pull/8), [PR #2](https://github.com/halfwave50/VIctory-macOS-halfwave50/pull/2)) | 28 Nov 03:46 | public main-mac cbf96d2 + 54 commits (PR #8) |
| MacBook Pro M5 Pro | 64 GB | 2.85 fps prison cell (~350 ms, release build); 2.2 fps while recording | Opening cell to the guard with player control; "a backlit player goes near-black" ([clip](https://github.com/pragyaangaur/VIctory-macOS/issues/513#issuecomment-7212488761)) | 27 Nov 07:27 | their integration branch (main d2b4889 + patches) |
| M4 (model not given) | 16 GB | 1.1–1.4 fps prison cell (750–880 ms, GPU-bound; Series S, 720p output) | Inmates whole; with level 8 "0 dropped mesh pairs from the intro to the cell", haze gone with `estimate:0.7` + `VICTORY_AMBIENT_AIR=1.5` ([1](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7214957684), [2](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7220336191)) | 27 Nov 14:06 | current main + our mesh row merge levels 3–8 |
| MacBook Pro M5 Pro | 48 GB | 1.9 fps scripted opening (529 ms median, 342 ms GPU; Series X, warm run) | Port of our mesh row merge: all four ground pairs link, "no dropped mesh draws" ([comment](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7221435422)) | 27 Nov 14:57 | main a31fcca1 + #578 |
| MacBook Pro M4 Pro (Mac16,8) | 48 GB | 3.0 fps prison cell, standing (334 ms average, 331 ms median; Series X, 720p output); 2.2 fps on a31fcca1 | "Almost all of the gain is GPU time" (372 to 207 ms); picture the same ([comment](https://github.com/pragyaangaur/VIctory-macOS/issues/131#issuecomment-7222088458)) | 27 Nov 15:27 | main d1609d9 |
| M5 (model not given) | 16 GB | 2.7 fps (369 ms median, 113 ms GPU; Series S, native 720p) with `VICTORY_LOOP_ENTRY_EXIT=1`; 1.8 fps (559 ms) without | Strand hair pass `CS_VisHair` no longer spins to the loop cap, "with the same picture" ([comment](https://github.com/pragyaangaur/VIctory-macOS/issues/131#issuecomment-7225530964)) | 27 Nov 18:28 | upstream PR #610 |
| MacBook Pro M1 Max | 32 GB | 1.0 fps prologue cell (1,044 ms average; Series X, 1080p output) | World draws with upstream #514 (`VICTORY_BARRIER_AFTER_CAP=1`); white flashes, grey skin, an occasional crash ([clip](https://github.com/pragyaangaur/VIctory-macOS/issues/513#issuecomment-7225964806)) | 27 Nov 18:52 | main 2bb3ae9 + upstream #499, #500, #514 |
| MacBook Pro M1 Pro | 16 GB | 0.6–0.9 fps prologue cell (1.1–1.6 s a frame; Series S, 720p output) with upstream #445; 3.3–4.5 s without | #445 ends the GPU aborts (~25 a minute to 0), "hair drawn in both" ([issue #7](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/7)) | 27 Nov 23:23 | main-mac 38a6dff + upstream #445 |
| M5 Pro (model not given) | 48 GB | 3.1 fps prison room, standing (318 ms, 163 ms GPU; Series X, 720p output) with `VICTORY_LOOP_ENTRY_EXIT=1`; 2.4 fps (409 ms) without | GPU time per pipeline: hair pass `CS_VisHair` 74 ms a frame, "nothing stands out" after the next one ([comment](https://github.com/pragyaangaur/VIctory-macOS/issues/131#issuecomment-7230563868)) | 27 Nov 23:59 | main d1609d9 + upstream #646 (diagnostics) |

Tried it on your Mac? Post your Mac, memory and fps in [issue #1](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1) and we'll add a row.

<img width="1608" alt="Four scenes on an M4 Pro: beach at sunset, boat in daylight, night street, apartment" src="https://github.com/user-attachments/assets/c89e61c0-9552-4217-b547-7fa180598d32" />

More pictures, questions and results from other Macs: [issue #1](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1).

## Mac efforts

Public GTA VI efforts for Apple Silicon that we know of. Each status is that project's own claim, taken from its README and issues; checked daily, last on 27 Nov 2026.

| Project | Approach | Status, as stated | Speed, as reported |
|---|---|---|---|
| [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) | VIctory + Wine + D3DMetal | The world draws on testers' Macs, up to the prologue's first gameplay prompt; not playable yet | About 0.9–1.1 s a frame in the opening on an M5 Pro; 165–180 ms at 1280x720 on an M4 with 16 GB ([#131](https://github.com/pragyaangaur/VIctory-macOS/issues/131)) |
| [64bitMango/VIctory-macOS](https://github.com/64bitMango/VIctory-macOS) | VIctory + Wine + D3DMetal | A fork of VIctory for macOS with more speed work; not playable yet | About 201–209 ms a frame (4.8–5 fps) on an M5 Pro, Series S profile with the PC upscaler |
| [Native-Victory-macOS](https://github.com/LakshanSivakumar/Native-Victory-macOS) | Native: no Wine, straight to Metal | Menus at 30 fps natively; the prologue runs, the world is still dark | About 0.5 fps in the world on an M5 Pro |
| [gta-vi-macos-first-scene](https://github.com/AEMONET-Dev/gta-vi-macos-first-scene) | VIctory + Wine + D3DMetal: renderer branches, experiment notes and frame-time data | The first scene shows recognisable people and streets (Series S profile) | Median 2.7 s between frames on an M5 Max, diagnostics on (snapshot of 25 Nov) |
| [MetVL](https://github.com/cloverfield11/MetVL) | Native: VIctory's loader under Rosetta 2, own Metal window | The game's code runs and presents frames; Direct3D 12 to Metal translation not started yet | None reported for the world |
| Halfwave (this project) | VIctory + Wine + D3DMetal | Player control, lit world, saves load | About 2 fps driving and 3.1–3.4 fps in the prison cell on an M4 Pro (more in "Tested on") |

Know another one? Tell us in [issue #1](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1).

## Try it

You need:
- An Apple Silicon Mac. Confirmed working with 16 GB (MacBook Air M5) and up; see "Tested on".
- macOS 15 or newer.
- [Homebrew](https://brew.sh).
- About 15 GB of free space.

1. Get the code:
   ```bash
   git clone https://github.com/halfwave50/VIctory-macOS-halfwave50.git
   cd VIctory-macOS-halfwave50
   ```
2. Download Apple's **Game Porting Toolkit**, version 4.0 beta 3 or newer, from [Apple's developer downloads](https://developer.apple.com/download/all/?q=game%20porting%20toolkit). A free Apple account works. Leave the `.dmg` in Downloads, unopened.
3. Build everything:
   ```bash
   ./setup.sh
   ```
   The first run takes about an hour. If it stops, fix what it says and run it again.
4. Point `game_dir` in `victory.toml` at your game folder. Every package's files go directly in that folder.
5. Start the game with our settings:
   ```bash
   tools/macos/play-main-mac.sh <the game's executable>
   ```

[docs/macos.md](docs/macos.md) explains every Mac fix and problem in detail.

## Main fixes

| Problem | Fix | Commit |
|---|---|---|
| No direct sunlight | Packed 16-bit textures that D3DMetal can't store to are kept as R8G8B8A8 (from Belobug's fork) | [010f0f9](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/010f0f9b9d6a) |
| Characters and clothes missing | Mesh pipelines that were just over Metal's limit now fit | [df9aab4](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/df9aab472803), [f1f590a](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/f1f590a93234) |
| Black bands in sunlight | The tile-list programs run exactly | [e566e1d](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/e566e1d14b7e) |
| Dark rooms | Estimated fill light with a minimum level (`VICTORY_AMBIENT=estimate`) | [9158460](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/915846063859) |
| Blocky shadow edges | One shadow program gets its image size (`VICTORY_UAV_SIZE_LEVEL0`) | in `play-main-mac.sh` |
| Slow buffer copies | Many of the game's buffers are shared with the GPU instead of copied every frame (`VICTORY_ALIAS_BUFFERS=1`) | [f517ed2](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/f517ed2f43e1) |
| Threads waiting on one lock | The list of what each shader resource slot reads is updated in batches, once per barrier (`VICTORY_CENSUS_FAST=1`); about 20% faster in the prison cell | [bee1f89](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/bee1f89ba2e2) |
| No screen reflections | The game writes its reflection image through one texture and reads it through a second view of the same memory; that view now uses the written texture (`VICTORY_PLACED_TWIN=1`), with a minimum reflection level of 0.15 | [5b2830a](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/5b2830a7bb1f) |
| Objects black for one frame | When the game moves a texture and rewrites its shader slot, the slot no longer stays empty until a shader has read it: it keeps its old texture until the new one is ready (`VICTORY_PLACED_AT_ONCE=keep`) | [f298867](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/f298867beb74), [c4a2248](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/c4a2248b3cfb) |
| No wake or spray on water | A water-simulation program that reads raw descriptor data as numbers now translates, so the simulation gets its time step (`VICTORY_DESCRIPTOR_AS_DATA=1`, from ScarlettMac) | [e6b84a0](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/e6b84a002e3e) |

Merged into [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS): the sunlight fix ([#478](https://github.com/pragyaangaur/VIctory-macOS/pull/478)) and the shadow-edge fix ([#488](https://github.com/pragyaangaur/VIctory-macOS/pull/488)).

## Develop or fork it

[DEVELOPING.md](DEVELOPING.md) covers how the work is done: one fix per commit with its source named, new code behind switches, measuring, and the asset rule. AI coding tools can start from [AGENTS.md](AGENTS.md).

## Credits and licence

This builds on:
- [VIctory](https://github.com/Marvisak/VIctory) by Marvisak
- [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) by pragyaangaur and its contributors. Its own README has more on how it works, testing and debugging switches.
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm, trustno1-coder and paveledits (ScarlettMac)

The licence is GPL-3.0 ([LICENSE](LICENSE)).
