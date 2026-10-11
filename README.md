# Squiggle

An interactive waveform plotter built on [Bevy](https://bevy.org). Pan, zoom
and pick your way around hundreds of high-density polylines at once, then run
them through Fourier and wavelet transforms without leaving the plot.

## Features

- **Fast at scale.** Samples are uploaded to the GPU once and projected in a
  vertex shader, so panning and zooming cost nothing per point. A hundred
  squiggles of ten thousand samples each stay smooth.
- **Stacked multi-sensor tracks.** Traces from different sensors share one
  time axis, from 1 px/day to 10 ps/px, so a 40 Hz seismometer and a
  1 GS/s capture line up. A min/max level-of-detail pyramid keeps tens of
  millions of samples per trace interactive.
- **Seismic, RF and digitizer formats.** miniSEED (Steim1/2), SAC, SigMF
  (complex I/Q), WAV, CSV and raw binary with a described layout.
- **Independent axis zoom.** Zoom x and y together or separately, anchored on
  the point under the cursor — waveform axes rarely share a unit.
- **Even strokes at any zoom.** Lines are widened in screen space after
  projection, so they keep the same thickness however the axes are stretched.
- **Linear and logarithmic axes**, with ticks and labels that follow the view.
- **Exact picking.** Click a line to select it and see its individual samples.
- **Signal transforms.** Fourier, DFT, FFT, STFT, inverse Fourier and wavelet
  decompositions, with the results plotted alongside the source.
- **Self-checking demos** that run each transform on a signal with a known
  answer and show the numerical checks next to the picture.

## Running

```sh
cargo run -p squiggle-app --release
```

The plot starts empty; **Add Squiggle** puts one on it. To start with a full
plot instead:

```sh
cargo run -p squiggle-app --release -- --squiggles 100 --samples 10000
```

`--release` matters: a debug build is usable but not smooth. The workspace
already compiles dependencies with optimizations in dev profiles.

| Flag | Effect |
| --- | --- |
| `--project FILE` | Open a `.squiggle` project |
| `WAVEFORM...` | Import these files (any format below) |
| `--squiggles N` | Start with `N` demo squiggles |
| `--samples N` | Samples per demo squiggle |
| `--demo fourier\|wavelet` | Load a transform demo at startup |
| `--screenshot FILE` | Save the window once the plot settles, then quit |

## Using it

The toolbar switches between two views. **Tracks** shows imported traces,
one row each, on a shared time axis; it opens when traces are loaded.
**Overlay** shows free squiggles: demos and transform results.

| Tracks | Control |
| --- | --- |
| Zoom time about the cursor | Scroll |
| Pan time / scroll rows | Drag, or horizontal scroll / scroll over the labels |
| Scale amplitude (all rows) | Pinch, or `Control` + scroll |
| Select a track | Click its row or label |
| Y scale (normalized, auto, fixed), I/Q component | Right-click a row |
| Frame every trace | **Fit**, or double-click |
| Order rows | **Rows by** in the toolbar |

The gold cursor spans every row; its time is shown on the axis, and its UTC
time and the pixel size at the top right.

| Overlay | Control |
| --- | --- |
| Pan | Drag with the left mouse button |
| Zoom both axes | Scroll |
| Zoom y only | `Alt` + scroll |
| Zoom x only | `Control` + scroll |
| Select a squiggle | Click it — its samples appear as dots |
| Add a demo squiggle | **Add Squiggle** in the toolbar |
| Frame everything | **Fit** in the toolbar |
| List and delete squiggles | The **Squiggles** sidebar handle on the left |
| Delete the selection | **Edit ▸ Delete selected** |
| Transform a squiggle | Select it, then pick from the **Transform** menu |

Zoom is multiplicative, so each scroll notch changes the scale by the same
ratio at every magnification. When a squiggle is selected, sample dots closer
than six pixels apart are thinned so dense data stays readable.

## Projects

A project (`.squiggle`, SQLite, format `heidtmare.squiggle.project/1`) holds
sensors, their phenomenology assignments, and traces. It stores an
append-only op log, not state: opening a project replays the log. Samples
stay in their source files, which are hashed on import and refused if they
change.

| Action | Control |
| --- | --- |
| New / open / save as | **File** menu |
| Import waveforms | **File ▸ Import Waveforms…**, or drop files on the window |
| Review sensors and traces | **Project ▸ Sensors…** |
| Add or edit phenomenologies | **Project ▸ Phenomenologies…** |

Once saved, every edit is written immediately.

**Import.** Each channel is listed for confirmation: sensor code, start time
(UTC, ISO 8601), sample rate, phenomenology. Values not found in the file are
flagged. A sensor already in the project keeps its assignment; otherwise a
suggestion (from a SEED channel code, or RF for SigMF) is pre-selected and
marked. Traces plot in seconds from the project reference (earliest start),
coloured by phenomenology.

| Format | Start | Rate | Sensor code |
| --- | --- | --- | --- |
| miniSEED 2 | record headers (+ B1001 µs, time correction) | header | `NET.STA.LOC.CHA` |
| SAC | `nz*` + `b` | `1/delta` | `NET.STA.LOC.CMP` |
| SigMF | first capture's `core:datetime` | `core:sample_rate` | file stem, `.chN` per channel |
| WAV | supplied (defaults to file mtime) | header | file stem, `.chN` per channel |
| CSV | ISO `time` column, else supplied | `time`/`t`/`timestamp`/`seconds` column, else supplied | column header |
| Raw binary | supplied | supplied | file stem, `.chN` per channel |

- **miniSEED**: encodings int16, int32, float32, float64, Steim1, Steim2;
  records are merged per channel, gaps become missing samples (drawn as
  breaks), overlaps are dropped. miniSEED 3 is not supported yet.
- **SAC**: evenly spaced time series, either byte order.
- **SigMF**: real or complex, any `core:datatype`; select either file of the
  pair. Integer types are scaled to ±1. Both files are hashed.
- **Raw binary**: any file no reader recognises. The import dialog asks for
  its layout (number format, byte order, real or I/Q, interleaved channels,
  header bytes, scale); the layout is stored with each trace, and the last
  one used is offered for the next file.

Complex (I/Q) traces plot magnitude by default; right-click the row for
phase, I or Q.

**Phenomenology.** Built-ins: seismic, acoustic, infrasound, RF, EMP. Each
has quantity, units, sample kind, propagation (constant speed, phases, or
unknown) and colour. Edits create a new version; assignments record the
version and the analyst. Sensors without one are plotted grey and refused by
**Transform**.

## Transforms

The **Transform** menu turns each selected squiggle into new ones, added to
the plot in the source's colour. **Transform ▸ Settings…** holds the
parameters.

| Transform | Result |
| --- | --- |
| Fourier | The magnitude of the continuous Fourier integral, approximated at each sample's real x, so uneven spacing is honoured |
| DFT | The discrete Fourier transform, summed directly (O(N²)) |
| FFT | The same spectrum by FFT — radix-2, or Bluestein for lengths that are not powers of two |
| STFT | One spectrum per windowed frame (Rectangular, Hann, Hamming or Blackman), stacked as a waterfall |
| Inverse Fourier | Back to the signal. Exact for anything Fourier, DFT, FFT or STFT produced; any other squiggle is read as an amplitude spectrum and synthesised from cosines |
| Wavelet | Haar or Daubechies 2–4 decomposition, as signal-aligned bands that sum to the original, or as raw coefficients per level |

Spectra plot amplitude, power, decibels or phase, one- or two-sided. DFT, FFT
and STFT amplitudes are in the signal's own units: a sine of amplitude 3 peaks
at 3. Everything but the continuous Fourier transform assumes evenly spaced
samples, taking the interval from the first and last x.

A transformed squiggle keeps its complex spectrum alongside the plotted
points, which is what lets the inverse reproduce the original exactly rather
than reconstruct it from magnitudes alone.

### Demos

The **Demo** menu, or `--demo` at startup, loads a test signal whose transform
is known in advance. The result is plotted beside the expected answer, and a
side panel lists numerical checks with pass or fail marks: peak heights,
phase, Parseval, DFT against FFT, exact inverse reconstruction, wavelet bands
summing to the signal, energy kept by the coefficients, and each feature
showing up in its own band at its own time. The same checks run as unit tests.

```sh
cargo run -p squiggle-app --release -- --demo wavelet
cargo run -p squiggle-app --release -- --demo fourier --screenshot fourier.png
```

## How it works

### Layout

```
squiggle-core/   model, project op log, time, samples, LOD, transforms; no dependencies (optional serde)
squiggle-io/     readers (miniSEED, SAC, SigMF, WAV, CSV, raw), sources, SQLite project store
squiggle-app/    the Bevy application
  src/workspace.rs   open project, async file I/O
  src/tracks.rs      stacked tracks: LOD meshes, time axis, labels
  src/project_ui.rs  import, sensor and phenomenology windows
  assets/shaders/squiggle_line.wgsl   screen-space polyline expansion
  src/plot/      camera, input, meshing, overlay
  src/ui.rs      menu bar, toolbar, sidebar (egui)
  src/showcase.rs  transform demo scenes and their checks
```

`squiggle-core` has no dependencies at all — the FFT is implemented there —
which keeps the parts worth testing (projection, tick generation, bounds,
hit-testing, transforms) testable with nothing on screen.

### Rendering

Each squiggle is an entity carrying its model and a mesh. The mesh holds two
vertices per sample — one either side of the line — along with each sample's
neighbours. `squiggle_line.wgsl` projects all three, works out the joint angle
*in pixels*, and offsets the vertex along the miter. Widening after projection
is what keeps a stroke the same visual thickness at any zoom, and it is the
only way to get an even stroke when x and y are scaled by different amounts.

Pan and zoom are a single `Transform` on the parent of every squiggle, so
moving the view touches one matrix rather than every point.

Tracks are separate entities, not children of that parent. Each holds a
min/max pyramid of its samples (blocks of 16·4ᵏ, built off-thread, extendable
in place for streaming) and a mesh of about one min/max pair per pixel over
three screen widths, alternating min-first and max-first so the line never
doubles back. The mesh is rebuilt only when the view leaves that window or the
zoom changes by more than ~2.8×; otherwise only the transform moves. Mesh x
is seconds from the window start and the transform is computed in `f64`, so
nanosecond spacing survives far from the reference.

Readers emit through `WaveformSource` (`Header`, `Chunk`, `Gap`, `End`);
loading consumes sources, so a live feed can replace a file without changing
display or ingest.

Axes, crosshairs and the dots on a selected squiggle are drawn with gizmos:
there are never more than a few hundred of them and they change every frame,
which is the opposite trade from the squiggles themselves.

### Picking

A click measures the distance to each line segment, after rejecting squiggles
by their screen-space bounding box. There is no second render pass and no GPU
readback, and the result is exact.

## Testing

```sh
cargo test          # the whole workspace
```

`squiggle-core`'s tests cover the projection round trip, culling, zoom
anchoring, tick placement under pan and zoom, label formatting, bounds
maintenance, hit-testing, and the transforms: FFT against DFT, exact
forward-inverse round trips, amplitude scaling, STFT frame placement, and
wavelet orthonormality and reconstruction, plus LOD ranges and decimation
against brute force (one chunk or many) and time-axis ticks.
`squiggle-io` tests each reader on synthesised files (Steim1/2 round trips,
both SAC byte orders, SigMF pairs, raw layouts), record merging with gaps,
source chunking, and change detection. `squiggle-app` tests the mesh
builder's topology, import validation, LOD bucketing and argument parsing.
