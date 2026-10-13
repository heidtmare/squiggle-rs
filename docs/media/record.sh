#!/bin/sh
# Records docs/media/showcase.mp4 and the README's showcase.gif.
# Needs ffmpeg. Run from the repository root.
set -eu

out=target/showcase
app=target/release/squiggle-app
mkdir -p "$out"
cargo build --release -p squiggle-app -p squiggle-train -p squiggle-io
export BEVY_ASSET_ROOT=squiggle-app

# Datasets, then the packages the tour imports.
if [ ! -f "$out/forest/spec.toml" ] || [ ! -f "$out/clusters/spec.toml" ]; then
    "$app" --tour docs/media/prepare.tour
    target/release/squiggle-train "$out/catalogue" --out "$out/forest" --model forest --name forest
    target/release/squiggle-train cluster "$out/families" --out "$out/clusters"
fi

# A sync server for the shared-project scene.
rm -rf "$out/server" "$out/families.squiggle"
target/release/squiggle-server --port 18100 --dir "$out/server" &
server=$!
trap 'kill $server' EXIT

# Headless: renders offscreen, so the screen may be locked or asleep.
"$app" --tour docs/media/showcase.tour --record docs/media/showcase.mp4 \
    --squiggles 100 --samples 10000 --analyst ana

# README preview: a clip of each scene (times from the tour's caption log),
# 10 fps, 720 px wide.
clips="[0:v]trim=4.5:11,setpts=PTS-STARTPTS[v0];[0:v]trim=20:24,setpts=PTS-STARTPTS[v1];[0:v]trim=40:44,setpts=PTS-STARTPTS[v2];[0:v]trim=58.5:62,setpts=PTS-STARTPTS[v3];[0:v]trim=97:101.5,setpts=PTS-STARTPTS[v4];[0:v]trim=113:117.5,setpts=PTS-STARTPTS[v5];[0:v]trim=162:166,setpts=PTS-STARTPTS[v6];[0:v]trim=234:242,setpts=PTS-STARTPTS[v7];[0:v]trim=264:271,setpts=PTS-STARTPTS[v8]"
ffmpeg -y -loglevel error -i docs/media/showcase.mp4 -filter_complex \
    "$clips;[v0][v1][v2][v3][v4][v5][v6][v7][v8]concat=n=9:v=1:a=0,fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
    docs/media/showcase.gif
