#!/bin/sh
# Starts the game on a 24 GB Mac with the settings cherrypicker-ottco is
# measured with (prison cell, Mac mini M5 Pro 24 GB, Series X 1.0.2.1:
# 163-164 ms a frame, ~6.1 fps; halfwave50/VIctory-macOS-halfwave50#8,
# pragyaangaur/VIctory-macOS#664). Everything play-main-mac.sh sets, plus the
# memory and output settings below. Any switch you set yourself wins.
#
#   tools/macos/play-24gb.sh <game folder>           start the game
#   tools/macos/play-24gb.sh <game folder> warmup    shader cache + pipeline
#                                                    library first (once, and
#                                                    after a translator change)
#
# <game folder> is the macOS folder that holds game_scarlett_master.exe.
[ -n "$1" ] || { echo "usage: $0 <game folder> [warmup]" >&2; exit 2; }
game=$(cd "$1" 2>/dev/null && pwd) || { echo "no such folder: $1" >&2; exit 2; }
root=$(cd "$(dirname "$0")/../.." && pwd)
# Wine's Z: drive is the Mac's /.
wine_game="Z:$(printf '%s' "$game" | tr '/' '\\')"
d() { eval "[ -n \"\${$1+x}\" ]" || export "$1=$2"; }
# Memory on 24 GB: aliasing off (it took free memory to 15 % with ~4 GB of
# swap here); low-memory marks 0.12/0.16 (Wine reads free memory lower than
# macOS does: at 0.20/0.26 copies were given up and remade all the time).
d VICTORY_ALIAS_BUFFERS 0;          d VICTORY_SMALL_LOOSE_WINDOWS 1
d VICTORY_RAM_LOW 0.12;             d VICTORY_RAM_FREE 0.16
d VICTORY_EVICT_HEAP_UAV 1;         d VICTORY_SPILL_SHADERS 1
d VICTORY_STEP_OVER pool;           d VICTORY_NO_HANG_WATCHDOG 1
# Host textures peak at 3.3-3.9 GB here; 3000 MB made them churn.
d VICTORY_GPU_BUDGET_MB 4500
# 1080p output at half render scale: 3840x2160 ran 24 GB out of memory.
d VICTORY_OUTPUT 1920x1080;         d VICTORY_RENDER_SCALE 0.5
d VICTORY_CENSUS_BATCH 1
if [ "$2" = warmup ]; then
    # The same translation switches as the run, so the cache keys match:
    # play-main-mac.sh's defaults, without starting the game.
    env_file=$(mktemp "${TMPDIR:-/tmp}/victory-play-env.XXXXXX")
    sed '$d' "$root/tools/macos/play-main-mac.sh" | sed 's/^exe="\$1"; shift$/exe=x/; s/^\[ -n "\$1" \].*$//' > "$env_file"
    . "$env_file"; rm -f "$env_file"
    exec "$root/tools/macos/victory.sh" translate --cache --pipelines "$wine_game"
fi
exec "$root/tools/macos/play-main-mac.sh" "$wine_game\\game_scarlett_master.exe"
