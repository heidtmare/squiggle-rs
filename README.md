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
- **Preprocessing.** Detrend, taper and Butterworth filters (low-, high-,
  band-pass, band-stop; zero-phase or causal) per trace, saved in the project.
- **Alignment.** Cross-correlation or GCC-PHAT delays between traces at any
  rates, reconciled across all pairs by least squares, reviewed before
  applying. Manual shifts by drag (snapping to correlation peaks) or by
  sample. Every edit undoes.
- **Picking and location.** STA/LTA onsets refined by AIC, manual picks by
  key, and a grid-search locator over position, depth and origin time that
  mixes phenomenologies (EMP anchors the origin, slower phases resolve
  position). Moveout removal and pick alignment line up arrivals across
  phenomenologies.
- **Spectrogram strips** per row, two-sided for I/Q.
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
| `--synthetic` | Start with a synthetic event of known delays (see [Alignment](#alignment)) |
| `--located` | Start with a synthetic EMP, seismic and infrasound event of known source (see [Picking and location](#picking-and-location)) |
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
| Waveform / spectrogram / both, FFT size, y scale, I/Q component | Right-click a row |
| Alignment reference, lock / unlock / reset offset | Right-click a row |
| Shift a track in time | `Shift` + drag its row (`Alt` disables snapping) |
| Pick first phase / S at the cursor | `P` / `S` over a row |
| Remove the nearest pick | `Delete` or `Backspace` over a row; or right-click the row |
| Nudge selected tracks | `←` / `→` one sample, with `Shift` ten |
| Frame every trace | **Fit**, or double-click |
| Order rows | **Rows by** in the toolbar |
| Plot without processing | **Raw** in the toolbar |
| Undo / redo | `⌘Z` / `⇧⌘Z` (`Ctrl` elsewhere), or **Edit** |

The gold cursor spans every row; its time is shown on the axis, and its UTC
time and the pixel size at the top right. A row's second line shows its
distance from the saved origin; its third line shows its offset (`Δ`, 🔒 when
locked) and processing; the reference is marked ★. Picks are solid lines
(white manual, blue automatic); arrivals predicted from the saved origin are
dashed green.

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
| Review sensors, traces and locations | **Project ▸ Sensors…** |
| Add or edit phenomenologies | **Project ▸ Phenomenologies…** |

Once saved, every edit is written immediately. Undo and redo append the
inverse or repeated change to the log; the log is never rewritten.
Processing, offsets, picks, sensor locations and the origin undo; imports and
phenomenology assignments do not yet.

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
- **SAC**: evenly spaced time series, either byte order. `stla`/`stlo`/`stel`
  set the sensor location.
- **SigMF**: real or complex, any `core:datatype`; select either file of the
  pair. Integer types are scaled to ±1. Both files are hashed.
  `core:geolocation` sets the sensor location.

A location from the file is set on a sensor that has none. Edit locations
(latitude, longitude in degrees, elevation in metres) in **Project ▸
Sensors…**.
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

## Processing and alignment

**Process ▸ Filter and detrend…** sets the selected tracks' processing:
remove mean or linear trend, Hann taper, then a Butterworth filter of order
1–10 (band filters have twice the poles), zero-phase (forward-backward with
steady-state edges) or causal. Each gap-free run is processed separately.
Processing needs an assigned phenomenology and a corner below Nyquist; it is
stored per trace and applied on load. Complex traces are filtered per
component.

### Alignment

**Process ▸ Align…** measures each selected trace's delay behind a reference
(★; right-click a row to choose) over a time window (default: the visible
range) within a search range:

1. Each pair is resampled onto one grid at the lower rate, capped at 2²⁰
   samples (Kaiser-windowed sinc, anti-aliased).
2. Cross-correlation (normalised) or GCC-PHAT gives the lag, refined to a
   sub-sample by a parabola. Confidence is `1 − second peak / main peak`.
3. With **Every pair**, all pairs are measured and the delays solved by
   weighted least squares (weight = coefficient × confidence); residuals show
   disagreement.

Results list delay, coefficient, confidence, worst residual and the new
offset. Rows start checked only when the reference pair is strong and the
pairs agree within two samples. **Apply** shifts each checked trace by minus
its delay, as one undoable command with automatic provenance. Locked offsets
(any manual shift locks) are reported and never changed; traces of another
phenomenology or unassigned ones are skipped, since waveforms of different
physics do not correlate.

Offsets are per trace and global until events exist; rows are drawn at
clock-corrected start plus offset.

`--synthetic` (or **Demo ▸ Synthetic event**) writes seven SAC files to the
temp directory: five seismic stations (40–200 Hz) and two infrasound sensors
recording one event with known delays under a strong microseism. Align then
shows a *vs truth* column. Unfiltered, the microseism dominates and rows stay
unchecked; after a 1–15 Hz band-pass, delays land within 1.5 ms of truth.

## Picking and location

A pick marks an arrival on a trace: a phase from its phenomenology's
propagation (`P`, `S`, … or an unnamed onset for a constant speed), a time on
the trace's own clock (so it moves with clock corrections and offsets), an
uncertainty and provenance. One per trace and phase. Manual picks lock: an
automatic re-pick never replaces them.

**Process ▸ Pick…** picks the selected tracks' plotted series (complex traces
by magnitude), with STA/LTA windows and thresholds per phenomenology:

1. Classic (short window over the long window before it) or recursive
   STA/LTA on energy; a trigger runs from the *on* ratio until it falls below
   *off*.
2. The first (or strongest) trigger is refined by the Akaike information
   criterion over a window from four STAs before it to one after; the AIC
   minimum is the onset, and the half-width of its basin within 2 of the
   minimum is the uncertainty (at least half a sample).

Defaults: seismic 0.3/5 s, infrasound 0.2/3 s, acoustic 0.01/0.5 s, EMP and
RF 10 µs/1 ms; other phenomenologies 20/400 samples. Results are reviewed,
then applied as automatic picks of the first phase. Hover a row and press `P`
or `S` for a manual pick at the cursor (uncertainty: a pixel, at least a
sample).

**Process ▸ Locate…** uses every included pick whose sensor has a location
and whose phase has a speed:

- Travel time is a straight ray at the phase speed: great-circle surface
  distance (spherical Earth) combined with source depth plus sensor elevation.
  Valid at regional distances.
- A 41×41 grid over latitude and longitude (±2× the sensor spread, at least
  10 km) and 11 depths (0 to the depth limit; 0 fixes the source at the
  surface) is refined eight times around the best node.
- At each node the origin time is solved in closed form, weighting each pick
  by `1/σ²`; χ² is the misfit. Precise EMP and RF picks therefore anchor the
  origin time, and slower phases resolve position.

The result shows position, depth, origin, RMS, χ²/dof, per-pick residuals and
a map of the first grid's misfit with sensors (white outline: used),
solution and, for synthetic data, truth. **Save origin** stores it (one per
project until events exist); rows then show predicted arrivals, can be
ordered by distance, and Shift+drag snaps to them.

**Align across phenomenologies** (in the Locate window):

- **Remove moveout** shifts every placed trace by minus its first-phase travel
  time from the saved origin, so arrivals of every phenomenology line up at
  the origin time.
- **Align picks to ★ reference** shifts selected tracks so their pick sits on
  the reference's pick of the same phase.

Both are undoable, automatic and leave locked offsets alone.

`--located` (or **Demo ▸ Synthetic located event**) writes twelve SAC files
with station coordinates: three EMP sensors (1 MHz, 20 ms captures), six
seismic stations (50–200 Hz, P and S) and three infrasound sensors (50 Hz)
recording a 0.5 km deep explosion. Arrivals follow the built-in speeds. With
default settings every pick lands within a sample of truth, and the location
is within 3 m, depth 17 m and origin time 1 µs.

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
squiggle-core/   model, project op log, time, samples, LOD, transforms, DSP; no dependencies (optional serde)
squiggle-io/     readers (miniSEED, SAC, SigMF, WAV, CSV, raw), sources, SQLite project store
squiggle-app/    the Bevy application
  src/workspace.rs   open project, async file I/O
  src/tracks.rs      stacked tracks: LOD meshes, time axis, labels, shifting
  src/strips.rs      spectrogram strips
  src/process_ui.rs  filter and align windows
  src/locate_ui.rs   pick and locate windows, alignment across phenomenologies
  src/synthetic.rs   synthetic alignment and located events
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

A row showing a spectrogram has a sprite beneath its line: Welch-averaged
columns of about one per pixel over the same three-width window, rebuilt
off-thread when the view leaves it or the zoom drifts 2.5×.

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
against brute force (one chunk or many) and time-axis ticks. DSP tests check
Butterworth responses against −3 dB corners, chunked against whole filtering
and resampling, zero-phase pulses, alias rejection, spectrogram bins, delay
recovery (known shifts, sub-sample, gaps, noise confidence), the least-squares
solve, and known delays at mixed rates recovered within one sample. Picker
tests check STA/LTA forms (streamed equals whole), triggers, AIC and onsets
within two samples; locator tests recover sources from exact arrivals, with
mixed phenomenologies anchored on the fastest. Project tests cover locks,
reverts, processing validation, picks (per-phase replacement, manual locks,
clock correction) and sensor locations.
`squiggle-io` tests each reader on synthesised files (Steim1/2 round trips,
both SAC byte orders, SigMF pairs, raw layouts), record merging with gaps,
source chunking, and change detection. `squiggle-app` tests the mesh
builder's topology, import validation, LOD bucketing, undo/redo through the
log, the synthetic events, manual pick placement, moveout and pick alignment,
argument parsing, and the M4 self-check: pick and locate the synthetic
located event within tolerance.
