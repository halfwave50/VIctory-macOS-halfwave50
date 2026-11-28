# Halfwave: VIctory for Apple Silicon Macs

<img width="1600" alt="GTA VI on an M4 Pro: player control in the prison cell at sunset, VIctory's fps line in the window title" src="https://github.com/user-attachments/assets/3ef3c9ee-b506-40e6-a2e9-7ec69acc906a" />

**Halfwave** runs the Xbox Series build of GTA VI on an Apple Silicon Mac (Wine plus Apple's D3DMetal), from a copy of the game that you own. It combines our own Mac fixes with fixes from the Windows forks, and every change is measured on a MacBook Pro M4 Pro (48 GB).

**Source code only, nothing from the game.** You need your own copy of the game.

## Status (27 November 2026)

**It works:**
- The game loads the world and gives player control. Saved games load, free roam included.
- Sunlight and shadows are right, rooms have fill light, and characters and clothes draw.
- Screen reflections reach the picture: water reflects the sky, and the steel objects in the prison cell get light.

**Speed:**

| Place | Series X profile | With 1920x1080 output |
|---|---|---|
| Prison checkpoint | 2.9–3.1 fps | 3.0 fps |
| Boat | 4.7 fps | 6.0 fps |

The 1920x1080 numbers were measured just before the latest speed-up.

**Not yet:**
- Playable speed. The next goal is 5 fps, then 10 fps at 1080p.
- In free roam some objects flicker black for one frame. The launcher now keeps short button taps and lets the world run closer to real time; neither is tried in play yet.
- One ray-traced reflection pass is left out, because it drew rectangular blocks on water.
- Dark car interiors at night, and one black patch on water.
- At 1920x1080 output the game's interface is drawn twice too large.

Recent changes: [WHATS-NEW.md](WHATS-NEW.md).

## Tested on

| Mac | Memory | Speed | Note | When (UTC) | Build |
|---|---|---|---|---|---|
| MacBook Pro M4 Pro | 48 GB | 3.1 fps prison, 4.7 boat (3.0 / 6.0 at 1080p) | Our test Mac | 27 Nov 06:48 | [cc3908c](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/cc3908cee8fd) |
| MacBook Pro M5 Pro | 24 GB | 3.8 fps prison cell with `VICTORY_CENSUS_FAST=1`, 3.0 without (render scale 0.5) | "By a wide margin the best picture and the best speed I have had on this Mac" ([1](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7206500809), [2](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7208611413)) | 27 Nov 02:59 | ad0f9fc + 2 memory fixes |
| MacBook Air M5 | 16 GB | 1.9 fps at the start (Series S), slowing to 0.25 fps over a 20-minute run | "The best result I've seen on my 16 GB M5 MacBook Air" ([report](https://github.com/pragyaangaur/VIctory-macOS/issues/455)) | 27 Nov 01:24 | 8616bd6 + memory marks (#456) |
| MacBook Pro M2 Pro | 32 GB | 1.3 fps prison cell (2.7 fps at 1080p); 1.4 fps prologue yard (0.7 upstream) | "The picture is very close to your screenshot"; no world stall in 10 runs ([1](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1#issuecomment-7207381523), [2](https://github.com/halfwave50/VIctory-macOS-halfwave50/issues/1#issuecomment-7208668352)) | 27 Nov 03:03 | main-mac + upstream #445 |
| Mac mini (Mac17,16) | 24 GB | 4.1 fps prison checkpoint at 1080p output (242 ms median); 0.2 fps at 4K | "Run at the checkpoint without visible problems … for over 25 minutes" ([PR #2](https://github.com/halfwave50/VIctory-macOS-halfwave50/pull/2)) | 27 Nov 06:17 | 6660053 + upstream #381, #456, #415, #416, #436, #452 |
| MacBook Pro M5 Pro | 64 GB | 2.85 fps prison cell (~350 ms, release build); 2.2 fps while recording | Opening cell to the guard with player control; "a backlit player goes near-black" ([clip](https://github.com/pragyaangaur/VIctory-macOS/issues/513#issuecomment-7212488761)) | 27 Nov 07:27 | their integration branch (main d2b4889 + patches) |

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
| Halfwave (this project) | VIctory + Wine + D3DMetal | Player control, lit world, saves load | 2.9–4.7 fps on an M4 Pro, 6.0 at 1080p output (more in "Tested on") |

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
   On a Mac with 24 GB, use `tools/macos/play-24gb.sh <game folder>` instead (run it once with `warmup` after it first, and after every translator change): the settings this branch was measured with, 163-164 ms a frame in the prison cell on a Mac mini M5 Pro (halfwave50/VIctory-macOS-halfwave50#8).

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

Merged into [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS): the sunlight fix ([#478](https://github.com/pragyaangaur/VIctory-macOS/pull/478)) and the shadow-edge fix ([#488](https://github.com/pragyaangaur/VIctory-macOS/pull/488)).

## Develop or fork it

[DEVELOPING.md](DEVELOPING.md) covers how the work is done: one fix per commit with its source named, new code behind switches, measuring, and the asset rule. AI coding tools can start from [AGENTS.md](AGENTS.md).

## Credits and licence

This builds on:
- [VIctory](https://github.com/Marvisak/VIctory) by Marvisak
- [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) by pragyaangaur and its contributors. Its own README has more on how it works, testing and debugging switches.
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder

The licence is GPL-3.0 ([LICENSE](LICENSE)).
