# Squiggle

An interactive waveform plot: pan, zoom and pick your way around many
high-density polylines at once.

This is a Rust port of the JavaFX application at [`heidtmare/squiggle`](https://github.com/heidtmare/squiggle),
rebuilt on [Bevy](https://bevy.org). The original drew everything onto a
JavaFX `Canvas` from a 30 fps animation timer, projecting every coordinate on
the CPU each frame. Here the points are uploaded to the GPU once and the
projection happens in a vertex shader, so panning and zooming no longer cost
anything per point.

## Running

```sh
cargo run -p squiggle-app --release
```

The plot starts empty; **Add Squiggle** puts one on it. To start with a full
plot instead — which is what the Java `SquigglePlotExample` did:

```sh
cargo run -p squiggle-app --release -- --squiggles 100 --samples 10000
```

`--release` matters: a debug build of the plot is playable but not smooth. The
workspace already compiles dependencies with optimizations in dev profiles.

## Using it

| Action | Control |
| --- | --- |
| Pan | Drag with the left mouse button |
| Zoom both axes | Scroll |
| Zoom y only | `Alt` + scroll |
| Zoom x only | `Control` + scroll |
| Select a squiggle | Click it — its samples appear as dots |
| Add a demo squiggle | **Add Squiggle** in the toolbar |
| Frame everything | **Fit** in the toolbar |
| List and delete squiggles | The **Squiggles** sidebar handle on the left |

Independent x and y zoom is deliberate: waveform axes rarely share a unit.

## Layout

```
squiggle-core/   model and coordinate math, no rendering dependencies
squiggle-app/    the Bevy application
  assets/shaders/squiggle_line.wgsl   screen-space polyline expansion
  src/plot/      camera, input, meshing, overlay
  src/ui.rs      menu bar, toolbar, sidebar (egui)
```

`squiggle-core` has no dependencies at all, which keeps the parts worth testing
— projection, tick generation, bounds, hit-testing — testable with
`cargo test -p squiggle-core` and nothing on screen.

## How the pieces map to the Java original

| Java | Rust |
| --- | --- |
| `Squiggle` / `SimpleSquiggle` | `squiggle_core::Squiggle` |
| `SquiggleCoordinate` | `squiggle_core::Point` |
| `RenderingAttributes` | `squiggle_core::Style` |
| `RenderingContext` (transform half) | `squiggle_core::PlotView` |
| `RenderingContext` (drawing half) | `plot::line`, `plot::overlay` |
| `LinearXAxis`, `LinearYAxis`, `Logrithmic*Axis` | `squiggle_core::Axis` |
| `SquigglePlot` control | `plot::PlotPlugin` and the `Plot` resource |
| `Sidebar`, `SidebarItem`, FXML view | `ui.rs` |
| `SquigglePlotExample` | the `--squiggles` flag, in `demo.rs` |
| `LazyPropertyWrapper` | dropped; Bevy change detection covers it |

### Rendering

Each squiggle is an entity carrying its model and a mesh. The mesh holds two
vertices per sample — one either side of the line — along with each sample's
neighbours. `squiggle_line.wgsl` projects all three, works out the joint angle
*in pixels*, and offsets the vertex along the miter. Widening after projection
is what keeps a stroke the same visual thickness at any zoom, and it is the
only way to get an even stroke when x and y are scaled by different amounts.

Pan and zoom are a single `Transform` on the parent of every squiggle, so
moving the view touches one matrix rather than every point.

Axes, crosshairs and the dots on a selected squiggle are drawn with gizmos:
there are never more than a few hundred of them and they change every frame,
which is the opposite trade from the squiggles themselves.

### Picking

The Java version picked by rendering the scene again with every object in a
unique colour and reading back the pixel under the cursor — an extra full draw
and a GPU stall per click. This port measures the distance from the click to
each line segment instead, after rejecting squiggles by their screen-space
bounding box. It is cheaper, exact, and needs no second canvas.

## Deliberate differences

These are places where the port does not match the original's behaviour,
because the original's behaviour was a bug or a stub.

- **Zoom is multiplicative.** The Java code added a flat `0.05` to the scale per
  scroll notch, which crawls when zoomed in and jumps when zoomed out. A constant
  ratio behaves the same at every magnification. Zoom also holds the point under
  the cursor still, which the original did not.
- **Logarithmic axes work.** `LogrithmicXAxis` left its minor ticks as a `TODO`
  and only labelled powers in one direction; `LogrithmicYAxis` was a copy of the
  linear axis with no log behaviour at all. `AxisKind::Logarithmic` now places
  major ticks on decade boundaries in both directions and subdivides each decade.
- **Tick generation is bounded.** The Java loops would spin indefinitely on a
  zero or negative tick step. Steps that produce no ticks yield none, and an
  axis will not emit more ticks than it could possibly show.
- **Absent z and m are `None`.** The original signalled them with `NaN`, which
  quietly propagated into bounds; bounds now ignore non-finite samples.
- **The sidebar does something.** Its content pane was an empty placeholder in
  the original. It now lists the squiggles with their colour and sample count,
  and selecting or deleting there matches what clicking on the plot does.
- **Menu items are wired up.** File ▸ Close, Edit ▸ Delete and Help ▸ About did
  nothing in the original.
- **Selected samples are thinned.** A selected squiggle drew a dot per sample,
  including thousands that landed on the same pixel. Dots closer than six pixels
  on screen are skipped, in the same spirit as the original's habit of averaging
  samples that shared a screen column. They also take the squiggle's own colour
  rather than a fixed cyan, which in the original matched the selected line and
  so was invisible against it.

## Testing

```sh
cargo test          # the whole workspace
```

`squiggle-core`'s tests cover the projection round trip, culling, zoom
anchoring, tick placement under pan and zoom, label formatting, bounds
maintenance and hit-testing. `squiggle-app` tests the mesh builder's topology.
