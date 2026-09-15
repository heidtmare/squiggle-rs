//! The plot itself: the squiggles, the axes, and the camera that frames them.
//!
//! This is the port of the Java `SquigglePlot` control. There, one class held
//! the canvas, the animation timer, the mouse handlers, the render loop and the
//! model. Here the model lives in `squiggle-core`, each squiggle is an entity,
//! and Bevy supplies the loop.

pub mod input;
pub mod line;
pub mod overlay;

use bevy::prelude::*;
use bevy::sprite_render::Material2dPlugin;
use squiggle_core::{Axis, Bounds, Coord, Insets, Orientation, PlotView, Rgba, Squiggle};

use line::SquiggleLineMaterial;

/// Gutters reserved for the axes, matching the Java plot's insets.
pub const AXIS_GUTTER: f64 = 20.5;

/// Squiggles sit furthest back; later ones are nudged forward so they stack in
/// the order they were added.
const Z_SQUIGGLES: f32 = -10.0;
const Z_SQUIGGLE_STEP: f32 = 0.001;
/// The gutters mask any squiggle that runs past the plot area.
const Z_GUTTER: f32 = -1.0;

/// Hairline width for the axis and crosshair gizmos.
pub const GIZMO_LINE_WIDTH: f32 = 1.0;

/// How a selected squiggle is highlighted, as in the Java renderer.
const SELECTED_COLOR: Rgba = Rgba::CYAN;
const SELECTED_WIDTH_BONUS: f32 = 1.0;

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Material2dPlugin::<SquiggleLineMaterial>::default())
            .init_resource::<Plot>()
            .init_resource::<input::DragState>()
            .add_message::<AddSquiggle>()
            .add_message::<RemoveSquiggle>()
            .add_message::<FitToData>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    add_squiggles,
                    remove_squiggles,
                    rebuild_meshes,
                    sync_materials,
                    input::track_cursor,
                    input::pan,
                    input::zoom,
                    input::pick,
                    fit_to_data,
                    sync_plot_transform,
                    sync_gutters,
                    overlay::draw,
                )
                    .chain(),
            );
    }
}

/// Marks the camera that renders the plot, as opposed to the one egui draws to.
#[derive(Component)]
pub struct PlotCamera;

/// The entity whose transform carries the plot's pan and zoom. Squiggles are
/// its children, so they are stored in cartesian space and never rebuilt when
/// the view moves.
#[derive(Component)]
pub struct PlotRoot;

/// One of the opaque strips behind an axis.
#[derive(Component)]
pub struct AxisGutter(pub Orientation);

/// A squiggle's model data. The mesh beside it is derived from this.
#[derive(Component, Deref, DerefMut)]
pub struct SquiggleModel(pub Squiggle);

/// Set when a squiggle's points change and its mesh no longer matches.
#[derive(Component)]
pub struct MeshOutOfDate;

/// Ask for a squiggle to be added to the plot.
#[derive(Message)]
pub struct AddSquiggle(pub Squiggle);

/// Ask for a squiggle to be removed.
#[derive(Message)]
pub struct RemoveSquiggle(pub Entity);

/// Ask the view to frame everything currently plotted.
#[derive(Message, Default)]
pub struct FitToData;

/// The plot's view state and display options.
///
/// The Java control exposed these as JavaFX properties so the FXML could bind
/// to them; a resource plays the same role here.
#[derive(Resource)]
pub struct Plot {
    pub name: String,
    /// Pan, zoom and the size of the area the plot draws into.
    pub view: PlotView,
    pub axes: Vec<Axis>,
    pub axes_visible: bool,
    pub crosshairs_visible: bool,
    /// When off, the plot ignores drag, scroll and click, as in the original.
    pub controls_enabled: bool,
    /// Cursor position in plot-local logical pixels, if it is over the plot.
    pub cursor: Option<Coord>,
    /// Where the plot area starts within the window, in logical pixels.
    pub origin: Vec2,
    /// The window's device pixel ratio, needed to size strokes in physical pixels.
    pub scale_factor: f32,
}

impl Default for Plot {
    fn default() -> Self {
        Self {
            name: "Squiggle Plot".to_string(),
            view: PlotView {
                insets: Insets::new(0.0, 0.0, AXIS_GUTTER, AXIS_GUTTER),
                ..PlotView::default()
            },
            axes: vec![
                Axis::linear("My X Axis", Orientation::X),
                Axis::linear("My Y Axis", Orientation::Y),
            ],
            axes_visible: true,
            crosshairs_visible: true,
            controls_enabled: true,
            cursor: None,
            origin: Vec2::ZERO,
            scale_factor: 1.0,
        }
    }
}

impl Plot {
    /// Converts plot-local screen pixels into the plot camera's world space,
    /// whose origin is the centre of the viewport with y pointing up.
    pub fn to_world(&self, screen: Coord) -> Vec2 {
        Vec2::new(
            screen.x as f32 - self.view.width as f32 / 2.0,
            self.view.height as f32 / 2.0 - screen.y as f32,
        )
    }

    /// Whether a plot-local point falls inside the plot area, excluding gutters.
    pub fn plot_area_contains(&self, screen: Coord) -> bool {
        let origin = self.view.plot_origin();
        screen.x >= origin.x
            && screen.x <= origin.x + self.view.plot_width()
            && screen.y >= origin.y
            && screen.y <= origin.y + self.view.plot_height()
    }

    pub fn is_usable(&self) -> bool {
        self.view.plot_width() > 0.0 && self.view.plot_height() > 0.0
    }
}

fn setup(mut commands: Commands, mut gizmo_config: ResMut<GizmoConfigStore>) {
    // The Java axes and crosshairs were hairlines; gizmos default to two pixels.
    let (config, _) = gizmo_config.config_mut::<DefaultGizmoConfigGroup>();
    config.line.width = GIZMO_LINE_WIDTH;

    // The plot camera draws the world; `ui` narrows its viewport to whatever
    // the egui panels leave free.
    commands.spawn((
        PlotCamera,
        Camera2d,
        Camera { order: 0, ..default() },
        // JavaFX painted the plot onto a black background.
        Msaa::Sample4,
    ));

    commands.spawn((PlotRoot, Transform::default(), Visibility::default()));

    for orientation in [Orientation::X, Orientation::Y] {
        commands.spawn((
            AxisGutter(orientation),
            Sprite { color: Color::BLACK, custom_size: Some(Vec2::ZERO), ..default() },
            Transform::from_xyz(0.0, 0.0, Z_GUTTER),
        ));
    }
}

fn add_squiggles(
    mut commands: Commands,
    mut requests: MessageReader<AddSquiggle>,
    root: Single<Entity, With<PlotRoot>>,
    existing: Query<(), With<SquiggleModel>>,
    plot: Res<Plot>,
    mut materials: ResMut<Assets<SquiggleLineMaterial>>,
) {
    let mut depth = existing.iter().count() as f32;

    for AddSquiggle(squiggle) in requests.read() {
        let material =
            materials.add(SquiggleLineMaterial::new(squiggle.style.stroke, squiggle.style.line_width, plot.scale_factor));

        commands.spawn((
            SquiggleModel(squiggle.clone()),
            MeshOutOfDate,
            MeshMaterial2d(material),
            Transform::from_xyz(0.0, 0.0, Z_SQUIGGLES + depth * Z_SQUIGGLE_STEP),
            ChildOf(*root),
        ));
        depth += 1.0;
    }
}

fn remove_squiggles(mut commands: Commands, mut requests: MessageReader<RemoveSquiggle>) {
    for RemoveSquiggle(entity) in requests.read() {
        commands.entity(*entity).despawn();
    }
}

/// Uploads geometry for squiggles whose points have changed.
fn rebuild_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    stale: Query<(Entity, &SquiggleModel), With<MeshOutOfDate>>,
) {
    for (entity, model) in &stale {
        let mut entity = commands.entity(entity);
        entity.remove::<MeshOutOfDate>();

        // A squiggle with fewer than two points has no geometry at all, so it
        // carries no mesh rather than an empty one.
        match line::build_mesh(model.points()) {
            Some(mesh) => {
                entity.insert(Mesh2d(meshes.add(mesh)));
            }
            None => {
                entity.remove::<Mesh2d>();
            }
        }
    }
}

/// Keeps each squiggle's material in step with its style and selection.
fn sync_materials(
    plot: Res<Plot>,
    changed: Query<(&SquiggleModel, &MeshMaterial2d<SquiggleLineMaterial>)>,
    mut materials: ResMut<Assets<SquiggleLineMaterial>>,
) {
    for (model, material) in &changed {
        let (color, width) = if model.selected {
            (SELECTED_COLOR, model.style.line_width + SELECTED_WIDTH_BONUS)
        } else {
            (model.style.stroke, model.style.line_width)
        };

        let wanted = SquiggleLineMaterial::new(color, width, plot.scale_factor);
        let Some(mut existing) = materials.get_mut(&material.0) else {
            continue;
        };
        // Only write when something actually differs, so the material is not
        // re-uploaded every frame.
        if existing.line.color != wanted.line.color
            || existing.line.half_width != wanted.line.half_width
        {
            *existing = wanted;
        }
    }
}

/// Folds the view's pan and zoom into the root transform, so the GPU applies
/// them to every squiggle at once.
fn sync_plot_transform(plot: Res<Plot>, mut root: Single<&mut Transform, With<PlotRoot>>) {
    let view = &plot.view;
    let translation = Vec3::new(
        (view.insets.left + view.translation.x - view.width / 2.0) as f32,
        (view.insets.bottom + view.translation.y - view.height / 2.0) as f32,
        0.0,
    );
    let scale = Vec3::new(view.scale.x as f32, view.scale.y as f32, 1.0);

    if root.translation != translation || root.scale != scale {
        root.translation = translation;
        root.scale = scale;
    }
}

/// Sizes the opaque strips that stop squiggles from bleeding into the axes.
fn sync_gutters(plot: Res<Plot>, mut gutters: Query<(&AxisGutter, &mut Sprite, &mut Transform)>) {
    let view = &plot.view;
    for (gutter, mut sprite, mut transform) in &mut gutters {
        // Each strip runs the full width or height so the corner is covered once.
        let (size, centre) = match gutter.0 {
            Orientation::X => (
                Vec2::new(view.width as f32, view.insets.bottom as f32),
                Coord::new(view.width / 2.0, view.height - view.insets.bottom / 2.0),
            ),
            Orientation::Y => (
                Vec2::new(view.insets.left as f32, view.height as f32),
                Coord::new(view.insets.left / 2.0, view.height / 2.0),
            ),
        };

        sprite.custom_size = Some(size);
        let world = plot.to_world(centre);
        transform.translation = Vec3::new(world.x, world.y, Z_GUTTER);
    }
}

fn fit_to_data(
    mut requests: MessageReader<FitToData>,
    mut plot: ResMut<Plot>,
    squiggles: Query<&SquiggleModel>,
) {
    if requests.read().count() == 0 || !plot.is_usable() {
        return;
    }

    let bounds = squiggles
        .iter()
        .filter_map(|model| model.bounds())
        .reduce(Bounds::union);

    if let Some(bounds) = bounds {
        plot.view.fit(bounds, 0.05);
    }
}
