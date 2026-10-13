//! Scripted tours: `--tour SCRIPT [--record OUT.mp4]`.
//!
//! A script drives the app with synthetic input (cursor, buttons, wheel, keys)
//! written as Bevy messages, so it never touches the system cursor. Widgets are
//! found by their label through egui's accessibility tree. Captions, title
//! cards and a pointer are painted on top. The app runs headless (no winit)
//! and the cameras render to an offscreen image the size of the window, so a
//! hidden, locked or sleeping screen still records.
//! With `--record`, every frame is captured and piped to `ffmpeg`; the script
//! runs on a fixed 30 fps clock.
//!
//! One command per line, `#` comments:
//!
//! ```text
//! title "Heading" "Subtitle"   # full-window card; `title` alone clears it
//! caption "Text"               # bottom banner; `caption` alone clears it
//! wait 1.5
//! waitfor "Label" [timeout]    # until a widget with this label exists
//! move T [secs]                # glide the pointer
//! click T | rclick T | dclick T [mods]
//! drag T T [T…] [secs] [mods]     # through each target in turn
//! scroll T notches [secs] [mods]
//! key K [mods]                 # K: a letter, digit, or Escape/Delete/Left/Right/[/]
//! type "text"                  # letters, digits and spaces into the focused field
//! pick "path"                  # answer the next file dialog with this path
//! dump                         # print every labelled widget to stderr
//! end
//! ```
//!
//! A target `T` is `"Label"` (`"Prefix*"` matches by prefix), `"Label"@n` (n-th by position), either with an
//! optional `+dx,dy` offset in points, or `x,y` as fractions of the window.
//! Modifiers: `shift`, `ctrl`, `alt`, `cmd`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command as Process, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::{CursorMoved, PrimaryWindow, WindowEvent};
use bevy_egui::{EguiContextSettings, EguiContexts, EguiFullOutput, EguiPostUpdateSet, EguiPrimaryContextPass, PrimaryEguiContext, egui};

use crate::demo::DemoOptions;
use crate::plot::PlotCamera;

/// The answer to the next file dialog, set by a `pick` step.
static PICK: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Takes the path a tour has given for the next file dialog.
pub fn picked() -> Option<PathBuf> {
    PICK.lock().ok()?.take()
}

pub const FPS: f64 = 30.0;
const DT: f64 = 1.0 / FPS;
/// Default pointer glide.
const GLIDE: f64 = 0.6;
/// How long a target label may take to appear before the step is skipped.
const FIND_TIMEOUT: f64 = 4.0;

pub struct TourPlugin;

impl Plugin for TourPlugin {
    fn build(&self, app: &mut App) {
        let Some(path) = app.world().resource::<DemoOptions>().tour.clone() else {
            return;
        };
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read tour {}: {e}", path.display()));
        let steps = parse(&source).unwrap_or_else(|e| panic!("tour {}: {e}", path.display()));
        let record = app.world().resource::<DemoOptions>().record.clone();

        app.insert_resource(Tour::new(steps, record))
            .add_systems(Update, offscreen)
            .add_systems(First, drive)
            .add_systems(PreUpdate, place_cursor.after(bevy::input::InputSystems))
            .add_systems(EguiPrimaryContextPass, overlay)
            .add_systems(
                PostUpdate,
                (
                    read_widgets.after(EguiPostUpdateSet::EndPass).before(EguiPostUpdateSet::ProcessOutput),
                    restore_cursor.after(EguiPostUpdateSet::PostProcessOutput),
                ),
            )
            .add_systems(Last, (capture, pace).chain());
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Mods {
    shift: bool,
    ctrl: bool,
    alt: bool,
    cmd: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum Target {
    Label { text: String, nth: usize, offset: Vec2 },
    Fraction(Vec2),
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    Title(Option<(String, String)>),
    Caption(Option<String>),
    Wait(f64),
    WaitFor(String, f64),
    Move(Target, f64),
    Click(Target, MouseButton, u32, Mods),
    Drag(Vec<Target>, f64, Mods),
    Scroll(Target, f32, f64, Mods),
    Key(KeyCode, Key, Mods),
    Type(String),
    Pick(PathBuf),
    Dump,
    End,
}

/// A labelled widget from the last egui pass, in logical points.
struct Widget {
    label: String,
    rect: Rect,
}

#[derive(Resource)]
pub struct Tour {
    steps: Vec<Step>,
    index: usize,
    /// Tour clock, seconds; advances one frame period per frame.
    clock: f64,
    /// When the current step started, on the tour clock.
    step_start: f64,
    /// Per-step scratch: where a glide began, resolved targets, wheel already sent.
    from: Vec2,
    resolved: Option<(Vec2, Vec2)>,
    path: Vec<Vec2>,
    sent: f32,
    phase: u32,
    cursor: Vec2,
    pressed: bool,
    title: Option<(String, String)>,
    caption: Option<String>,
    widgets: Vec<Widget>,
    dump: bool,
    real_cursor: Option<Option<Vec2>>,
    recorder: Option<Recorder>,
    finished: Option<u32>,
    /// The offscreen image the cameras render to, once set up.
    target: Option<Handle<Image>>,
    started: Option<Instant>,
    frame: u64,
}

impl Tour {
    fn new(steps: Vec<Step>, record: Option<PathBuf>) -> Self {
        Self {
            steps,
            index: 0,
            clock: 0.0,
            step_start: 0.0,
            from: Vec2::ZERO,
            resolved: None,
            path: Vec::new(),
            sent: 0.0,
            phase: 0,
            cursor: Vec2::new(640.0, 400.0),
            pressed: false,
            title: None,
            caption: None,
            widgets: Vec::new(),
            dump: false,
            real_cursor: None,
            recorder: record.map(Recorder::new),
            finished: None,
            target: None,
            started: None,
            frame: 0,
        }
    }

    fn next(&mut self) {
        self.index += 1;
        self.step_start = self.clock;
        self.from = self.cursor;
        self.resolved = None;
        self.path.clear();
        self.sent = 0.0;
        self.phase = 0;
    }

    fn find(&self, target: &Target, window: Vec2) -> Option<Vec2> {
        match target {
            Target::Fraction(f) => Some(*f * window),
            Target::Label { text, nth, offset } => {
                let mut matches: Vec<&Widget> = self.widgets.iter().filter(|w| matches(&w.label, text)).collect();
                matches.sort_by(|a, b| {
                    (a.rect.min.y, a.rect.min.x).partial_cmp(&(b.rect.min.y, b.rect.min.x)).unwrap()
                });
                matches.get(nth.saturating_sub(1)).map(|w| w.rect.center() + *offset)
            }
        }
    }
}

/// A label pattern matches exactly, or by prefix when it ends in `*`.
fn matches(label: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => label.starts_with(prefix),
        None => label == pattern,
    }
}

/// Writes the input messages for this frame and advances the script.
#[allow(clippy::too_many_arguments)]
fn drive(
    mut tour: ResMut<Tour>,
    window: Single<(Entity, &Window), With<PrimaryWindow>>,
    mut writers: Writers,
    mut exit: MessageWriter<AppExit>,
) {
    let (entity, window) = *window;
    let size = window.size();
    let tour = &mut *tour;

    if let Some(frames) = &mut tour.finished {
        *frames += 1;
        let pending = tour.recorder.as_ref().is_some_and(|r| r.pending());
        if *frames > 10 && (!pending || *frames > 300) {
            if let Some(recorder) = tour.recorder.take() {
                recorder.finish();
            }
            exit.write(AppExit::Success);
        }
        return;
    }

    // Egui's context renders unscaled and zooms; it reads physical positions.
    let scale = if tour.target.is_some() { window.scale_factor() } else { 1.0 };
    let mut input = Input { entity, scale, w: &mut writers };

    // Several instantaneous steps may run in one frame; anything that takes
    // time returns and resumes next frame.
    loop {
        let Some(step) = tour.steps.get(tour.index).cloned() else {
            tour.finished = Some(0);
            return;
        };
        let elapsed = tour.clock - tour.step_start;
        match step {
            Step::Title(t) => tour.title = t,
            Step::Caption(c) => {
                // Logged so cuts can be made at scene changes.
                info!("tour: {:.2} s caption {:?}", tour.clock, c.as_deref().unwrap_or(""));
                tour.caption = c;
            }
            Step::Dump => tour.dump = true,
            Step::Pick(path) => *PICK.lock().unwrap() = Some(path),
            Step::End => {
                tour.index = tour.steps.len();
                continue;
            }
            Step::Wait(secs) => {
                if elapsed < secs {
                    break;
                }
            }
            Step::WaitFor(label, timeout) => {
                let found = tour.widgets.iter().any(|w| matches(&w.label, &label));
                if !found && elapsed < timeout {
                    break;
                }
                if !found {
                    warn!("tour: {label:?} did not appear");
                }
            }
            Step::Move(target, secs) => {
                let Some(to) = resolve(tour, &target, size, elapsed) else { break };
                if !glide(tour, &mut input, to, elapsed, secs) {
                    break;
                }
            }
            Step::Click(target, button, count, mods) => {
                let Some(to) = resolve(tour, &target, size, elapsed) else { break };
                if tour.phase == 0 {
                    if !glide(tour, &mut input, to, elapsed, GLIDE) {
                        break;
                    }
                    tour.phase = 1;
                    tour.step_start = tour.clock;
                    input.modifiers(mods, ButtonState::Pressed);
                    break;
                }
                // Press and release on alternate frames so each registers.
                let frame = tour.phase - 1;
                if frame < count * 4 {
                    match frame % 4 {
                        0 => {
                            input.button(button, ButtonState::Pressed);
                            tour.pressed = true;
                        }
                        2 => {
                            input.button(button, ButtonState::Released);
                            tour.pressed = false;
                        }
                        _ => {}
                    }
                    tour.phase += 1;
                    break;
                }
                input.modifiers(mods, ButtonState::Released);
                if elapsed < 0.25 {
                    break;
                }
            }
            Step::Drag(targets, secs, mods) => {
                if tour.path.is_empty() {
                    let Some(path) = targets.iter().map(|t| tour.find(t, size)).collect::<Option<Vec<_>>>() else {
                        if elapsed > FIND_TIMEOUT {
                            warn!("tour: drag target not found");
                            tour.next();
                            continue;
                        }
                        break;
                    };
                    tour.path = path;
                }
                let a = tour.path[0];
                let legs = tour.path.len() as u32 - 1;
                match tour.phase {
                    0 => {
                        if !glide(tour, &mut input, a, elapsed, GLIDE) {
                            break;
                        }
                        input.modifiers(mods, ButtonState::Pressed);
                        tour.phase = 1;
                        tour.step_start = tour.clock;
                        break;
                    }
                    1 => {
                        input.button(MouseButton::Left, ButtonState::Pressed);
                        tour.pressed = true;
                        tour.phase = 2;
                        tour.step_start = tour.clock;
                        tour.from = a;
                        break;
                    }
                    leg if leg < 2 + legs => {
                        let b = tour.path[(leg - 1) as usize];
                        if !glide(tour, &mut input, b, elapsed, secs / f64::from(legs)) {
                            break;
                        }
                        tour.phase += 1;
                        tour.step_start = tour.clock;
                        tour.from = b;
                        break;
                    }
                    _ => {
                        input.button(MouseButton::Left, ButtonState::Released);
                        input.modifiers(mods, ButtonState::Released);
                        tour.pressed = false;
                    }
                }
            }
            Step::Scroll(target, notches, secs, mods) => {
                let Some(at) = resolve(tour, &target, size, elapsed) else { break };
                if tour.phase == 0 {
                    if !glide(tour, &mut input, at, elapsed, GLIDE) {
                        break;
                    }
                    input.modifiers(mods, ButtonState::Pressed);
                    tour.phase = 1;
                    tour.step_start = tour.clock;
                    break;
                }
                let done = (elapsed / secs.max(DT)).min(1.0) as f32;
                // Smooth pixel scrolling, as a trackpad sends it.
                let total = notches * 50.0;
                let delta = total * ease(done) - tour.sent;
                if delta != 0.0 {
                    input.wheel(delta);
                    tour.sent += delta;
                }
                if done < 1.0 {
                    break;
                }
                input.modifiers(mods, ButtonState::Released);
            }
            Step::Type(text) => {
                // One character every three frames: press, release, rest.
                let n = tour.phase as usize;
                if n < text.chars().count() * 3 {
                    let c = text.chars().nth(n / 3).unwrap();
                    let (code, key) = if c == ' ' {
                        (KeyCode::Space, Key::Space)
                    } else {
                        parse_key(&c.to_string()).map(|(code, _)| (code, Key::Character(c.to_string().into()))).unwrap_or((KeyCode::Space, Key::Space))
                    };
                    match n % 3 {
                        0 => input.key(code, key, ButtonState::Pressed),
                        1 => input.key(code, key, ButtonState::Released),
                        _ => {}
                    }
                    tour.phase += 1;
                    break;
                }
            }
            Step::Key(code, key, mods) => {
                match tour.phase {
                    0 => {
                        input.modifiers(mods, ButtonState::Pressed);
                        tour.phase = 1;
                        break;
                    }
                    1 => {
                        input.key(code, key.clone(), ButtonState::Pressed);
                        tour.phase = 2;
                        break;
                    }
                    2 => {
                        input.key(code, key, ButtonState::Released);
                        tour.phase = 3;
                        break;
                    }
                    _ => input.modifiers(mods, ButtonState::Released),
                }
            }
        }
        tour.next();
    }
}

/// Finds a step's target, following it while it moves (windows settle over a
/// few frames), giving up on it (and skipping the step) after a while.
fn resolve(tour: &mut Tour, target: &Target, size: Vec2, elapsed: f64) -> Option<Vec2> {
    match (tour.find(target, size), tour.resolved) {
        (Some(at), Some(_)) => {
            tour.resolved = Some((at, at));
            Some(at)
        }
        (None, Some((at, _))) => Some(at),
        (Some(at), None) => {
            tour.resolved = Some((at, at));
            tour.step_start = tour.clock;
            tour.from = tour.cursor;
            Some(at)
        }
        (None, None) => {
            if elapsed > FIND_TIMEOUT {
                warn!("tour: target {target:?} not found");
                tour.next();
            }
            None
        }
    }
}

/// Moves the pointer towards `to`; true once there.
fn glide(tour: &mut Tour, input: &mut Input, to: Vec2, elapsed: f64, secs: f64) -> bool {
    let done = (elapsed / secs.max(DT)).min(1.0) as f32;
    let at = tour.from.lerp(to, ease(done));
    if at != tour.cursor {
        tour.cursor = at;
        input.cursor(at);
    }
    done >= 1.0
}

fn ease(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[derive(bevy::ecs::system::SystemParam)]
struct Writers<'w> {
    moved: MessageWriter<'w, CursorMoved>,
    buttons: MessageWriter<'w, MouseButtonInput>,
    wheel: MessageWriter<'w, MouseWheel>,
    keys: MessageWriter<'w, KeyboardInput>,
    events: MessageWriter<'w, WindowEvent>,
}

struct Input<'a, 'w> {
    entity: Entity,
    scale: f32,
    w: &'a mut Writers<'w>,
}

impl Input<'_, '_> {
    fn cursor(&mut self, position: Vec2) {
        let message = CursorMoved { window: self.entity, position, delta: None };
        self.w.moved.write(message.clone());
        self.w.events.write(WindowEvent::CursorMoved(CursorMoved { position: position * self.scale, ..message }));
    }

    fn button(&mut self, button: MouseButton, state: ButtonState) {
        let message = MouseButtonInput { button, state, window: self.entity };
        self.w.buttons.write(message);
        self.w.events.write(WindowEvent::MouseButtonInput(message));
    }

    fn wheel(&mut self, y: f32) {
        let message = MouseWheel { unit: MouseScrollUnit::Pixel, x: 0.0, y, window: self.entity, phase: TouchPhase::Moved };
        self.w.wheel.write(message);
        self.w.events.write(WindowEvent::MouseWheel(message));
    }

    fn key(&mut self, key_code: KeyCode, logical_key: Key, state: ButtonState) {
        let text = match &logical_key {
            Key::Character(c) if state == ButtonState::Pressed => Some(c.clone()),
            _ => None,
        };
        let message = KeyboardInput { key_code, logical_key, state, text, repeat: false, window: self.entity };
        self.w.keys.write(message.clone());
        self.w.events.write(WindowEvent::KeyboardInput(message));
    }

    fn modifiers(&mut self, mods: Mods, state: ButtonState) {
        for (on, code, key) in [
            (mods.shift, KeyCode::ShiftLeft, Key::Shift),
            (mods.ctrl, KeyCode::ControlLeft, Key::Control),
            (mods.alt, KeyCode::AltLeft, Key::Alt),
            (mods.cmd, KeyCode::SuperLeft, Key::Super),
        ] {
            if on {
                self.key(code, key, state);
            }
        }
    }
}

/// The overlay plot reads the window's cursor; point it at the tour's for the
/// frame and put the real one back before winit would warp the system cursor.
fn place_cursor(tour: Option<ResMut<Tour>>, mut window: Single<&mut Window, With<PrimaryWindow>>) {
    let Some(mut tour) = tour else { return };
    tour.real_cursor = Some(window.cursor_position());
    let at = tour.cursor;
    window.set_cursor_position(Some(at));
}

fn restore_cursor(tour: Option<ResMut<Tour>>, mut window: Single<&mut Window, With<PrimaryWindow>>) {
    let Some(mut tour) = tour else { return };
    if let Some(real) = tour.real_cursor.take() {
        window.set_cursor_position(real);
    }
}

/// Collects labelled widgets from egui's accessibility tree.
fn read_widgets(mut tour: ResMut<Tour>, outputs: Query<&EguiFullOutput, With<PrimaryEguiContext>>) {
    let Ok(output) = outputs.single() else { return };
    let Some(update) = output.0.as_ref().and_then(|o| o.platform_output.accesskit_update.as_ref()) else {
        return;
    };
    tour.widgets = update
        .nodes
        .iter()
        .filter_map(|(_, node)| {
            let label = node.label().or_else(|| node.value())?.to_string();
            let b = node.bounds()?;
            Some(Widget {
                label,
                rect: Rect::new(b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32),
            })
        })
        .collect();
    if std::mem::take(&mut tour.dump) {
        let mut lines: Vec<_> = tour.widgets.iter().map(|w| (w.rect.min.y as i32, w.rect.min.x as i32, w.label.clone())).collect();
        lines.sort();
        eprintln!("tour dump at {:.1} s", tour.clock);
        for (y, x, label) in lines {
            eprintln!("tour widget {x:>5},{y:>5}  {label:?}");
        }
    }
}

/// Paints the title card, caption and pointer over everything.
fn overlay(mut contexts: EguiContexts, tour: Res<Tour>, window: Single<&Window, With<PrimaryWindow>>) -> Result {
    let ctx = contexts.ctx_mut()?;
    ctx.enable_accesskit();
    if tour.target.is_some() && ctx.zoom_factor() != window.scale_factor() {
        ctx.set_zoom_factor(window.scale_factor());
    }
    let screen = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Debug, egui::Id::new("tour")));
    let accent = egui::Color32::from_rgb(255, 196, 64);

    if let Some((heading, sub)) = &tour.title {
        painter.rect_filled(screen, 0.0, egui::Color32::from_black_alpha(235));
        let center = screen.center();
        painter.text(center - egui::vec2(0.0, 24.0), egui::Align2::CENTER_CENTER, heading, egui::FontId::proportional(46.0), egui::Color32::WHITE);
        painter.text(center + egui::vec2(0.0, 30.0), egui::Align2::CENTER_CENTER, sub, egui::FontId::proportional(20.0), accent);
    }

    if let Some(caption) = &tour.caption {
        let galley = painter.layout(caption.clone(), egui::FontId::proportional(22.0), egui::Color32::WHITE, screen.width() - 120.0);
        let size = galley.size();
        let pos = egui::pos2(screen.center().x - size.x / 2.0, screen.max.y - 46.0 - size.y);
        let back = egui::Rect::from_min_size(pos, size).expand2(egui::vec2(18.0, 10.0));
        painter.rect_filled(back, 8.0, egui::Color32::from_black_alpha(215));
        painter.rect_stroke(back, 8.0, egui::Stroke::new(1.5, accent), egui::StrokeKind::Outside);
        painter.galley(pos, galley, egui::Color32::WHITE);
    }

    if tour.title.is_none() {
        let p = egui::pos2(tour.cursor.x, tour.cursor.y);
        if tour.pressed {
            painter.circle_filled(p, 14.0, accent.gamma_multiply(0.45));
        }
        let arrow = [(0.0, 0.0), (0.0, 17.0), (4.5, 13.0), (7.5, 19.5), (10.0, 18.5), (7.0, 12.0), (12.5, 12.0)];
        let points: Vec<_> = arrow.iter().map(|(x, y)| p + egui::vec2(*x, *y)).collect();
        painter.add(egui::Shape::convex_polygon(points.clone(), egui::Color32::WHITE, egui::Stroke::NONE));
        painter.add(egui::Shape::closed_line(points, egui::Stroke::new(1.2, egui::Color32::BLACK)));
    }
    Ok(())
}

/// Points every window camera at an image of the window's size (again when
/// that changes, as it does once a scale override lands). Egui keeps routing
/// the window's input to its context, which was mapped when created.
///
/// An image target with a scale factor other than 1 renders nothing here, so
/// the image is unscaled and the window's scale is applied instead by egui's
/// zoom (see `overlay`) and the plot camera's projection.
fn offscreen(
    mut tour: ResMut<Tour>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(&mut RenderTarget, &mut Projection, Has<PlotCamera>), With<Camera>>,
    mut images: ResMut<Assets<Image>>,
    mut egui: Query<&mut EguiContextSettings, With<PrimaryEguiContext>>,
) {
    // There is no OS window for egui to set a cursor or IME position on.
    for mut settings in &mut egui {
        if settings.enable_ime || settings.enable_cursor_icon_updates {
            settings.enable_ime = false;
            settings.enable_cursor_icon_updates = false;
        }
    }
    let size = window.physical_size();
    let current = tour.target.as_ref().and_then(|h| images.get(h)).map(|i| i.size());
    if size.x == 0 || current == Some(size) {
        return;
    }
    let image = Image::new_target_texture(
        size.x,
        size.y,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        None,
    );
    let handle = images.add(image);
    for (mut target, mut projection, plot) in &mut cameras {
        if plot && let Projection::Orthographic(ortho) = &mut *projection {
            ortho.scale = 1.0 / window.scale_factor();
        }
        let ours = matches!(&*target, RenderTarget::Image(t) if Some(&t.handle) == tour.target.as_ref());
        if ours || matches!(*target, RenderTarget::Window(_)) {
            *target = RenderTarget::Image(ImageRenderTarget { handle: handle.clone(), scale_factor: 1.0 });
        }
    }
    tour.target = Some(handle);
    info!("tour: rendering offscreen at {}x{}", size.x, size.y);
}

/// Captures this frame, when recording.
fn capture(mut commands: Commands, mut tour: ResMut<Tour>) {
    let index = tour.frame;
    tour.frame += 1;
    if tour.finished.is_some() {
        return;
    }
    let Some(target) = tour.target.clone() else { return };
    let Some(recorder) = &mut tour.recorder else { return };
    recorder.sent += 1;
    let frames = recorder.frames.clone();
    commands.spawn(Screenshot::image(target)).observe(move |captured: On<ScreenshotCaptured>| {
        let image = &captured.image;
        let format = format!("{:?}", image.texture_descriptor.format);
        let size = image.size();
        if let Some(data) = image.data.clone() {
            let _ = frames.send(Frame { index, width: size.x, height: size.y, bgra: format.starts_with("Bgra"), data });
        }
    });
}

/// Holds the frame rate to the tour clock, and advances it.
fn pace(mut tour: ResMut<Tour>) {
    let started = *tour.started.get_or_insert_with(Instant::now);
    tour.clock += DT;
    let due = started + Duration::from_secs_f64(tour.clock);
    if let Some(wait) = due.checked_duration_since(Instant::now()) {
        std::thread::sleep(wait);
    }
}

struct Frame {
    index: u64,
    width: u32,
    height: u32,
    bgra: bool,
    data: Vec<u8>,
}

/// Feeds captured frames, in order, to `ffmpeg` on its own thread.
struct Recorder {
    frames: Sender<Frame>,
    sent: u64,
    written: std::sync::Arc<std::sync::atomic::AtomicU64>,
    thread: JoinHandle<()>,
}

impl Recorder {
    fn new(out: PathBuf) -> Self {
        let (frames, receive) = channel();
        let written = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let count = written.clone();
        let thread = std::thread::spawn(move || encode(receive, out, count));
        Self { frames, sent: 0, written, thread }
    }

    fn pending(&self) -> bool {
        self.written.load(std::sync::atomic::Ordering::Relaxed) < self.sent
    }

    fn finish(self) {
        drop(self.frames);
        let _ = self.thread.join();
    }
}

fn encode(frames: Receiver<Frame>, out: PathBuf, written: std::sync::Arc<std::sync::atomic::AtomicU64>) {
    let mut ffmpeg: Option<Child> = None;
    let mut waiting = BTreeMap::new();
    let mut next = None;
    for frame in frames {
        // Screenshots start a frame or two late; begin with the first to arrive.
        let first = *next.get_or_insert(frame.index);
        if frame.index < first {
            continue;
        }
        let child = ffmpeg.get_or_insert_with(|| {
            Process::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-f", "rawvideo"])
                .args(["-pix_fmt", if frame.bgra { "bgra" } else { "rgba" }])
                .args(["-s", &format!("{}x{}", frame.width, frame.height)])
                .args(["-r", &FPS.to_string(), "-i", "-"])
                .args(["-vf", "scale=1920:-2:flags=lanczos,format=yuv420p"])
                // Flat UI, mostly static: long GOPs and animation tuning
                // halve the size with no visible loss.
                .args(["-c:v", "libx264", "-preset", "veryslow", "-tune", "animation", "-crf", "28"])
                .args(["-x264-params", "keyint=900:min-keyint=30", "-movflags", "+faststart"])
                .arg(&out)
                .stdin(Stdio::piped())
                .spawn()
                .expect("ffmpeg must be on PATH to record")
        });
        waiting.insert(frame.index, frame.data);
        let stdin = child.stdin.as_mut().unwrap();
        while let Some(data) = next.and_then(|n| waiting.remove(&n)) {
            if stdin.write_all(&data).is_err() {
                return;
            }
            next = next.map(|n| n + 1);
            written.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if let Some(mut child) = ffmpeg {
        drop(child.stdin.take());
        let _ = child.wait();
    }
}

fn parse(source: &str) -> Result<Vec<Step>, String> {
    source
        .lines()
        .enumerate()
        .filter_map(|(n, line)| {
            let words = match split(line) {
                Ok(words) => words,
                Err(e) => return Some(Err(format!("line {}: {e}", n + 1))),
            };
            (!words.is_empty()).then(|| step(&words).map_err(|e| format!("line {}: {e}", n + 1)))
        })
        .collect()
}

/// Splits a line into words, keeping quoted strings whole (quotes retained).
fn split(line: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut chars = line.trim().chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '#' {
            break;
        } else {
            let mut word = String::new();
            let mut quoted = false;
            while let Some(&c) = chars.peek() {
                if c == '"' {
                    quoted = !quoted;
                } else if c.is_whitespace() && !quoted {
                    break;
                }
                word.push(c);
                chars.next();
            }
            if quoted {
                return Err("unterminated quote".into());
            }
            words.push(word);
        }
    }
    Ok(words)
}

fn step(words: &[String]) -> Result<Step, String> {
    let rest = &words[1..];
    let number = |i: usize, default: f64| -> Result<f64, String> {
        rest.get(i).filter(|w| w.parse::<f64>().is_ok()).map_or(Ok(default), |w| w.parse().map_err(|_| format!("bad number {w}")))
    };
    let mods = || -> Result<Mods, String> {
        let mut m = Mods::default();
        for w in rest.iter().filter(|w| w.chars().all(|c| c.is_ascii_lowercase())) {
            match w.as_str() {
                "shift" => m.shift = true,
                "ctrl" => m.ctrl = true,
                "alt" => m.alt = true,
                "cmd" => m.cmd = true,
                other => return Err(format!("unknown modifier {other}")),
            }
        }
        Ok(m)
    };
    let target = |i: usize| rest.get(i).ok_or("missing target".to_string()).and_then(|w| parse_target(w));
    Ok(match words[0].as_str() {
        "title" => Step::Title(match rest {
            [] => None,
            [a] => Some((unquote(a), String::new())),
            [a, b, ..] => Some((unquote(a), unquote(b))),
        }),
        "caption" => Step::Caption(rest.first().map(|c| unquote(c))),
        "wait" => Step::Wait(number(0, 1.0)?),
        "waitfor" => Step::WaitFor(unquote(rest.first().ok_or("missing label")?), number(1, 10.0)?),
        "move" => Step::Move(target(0)?, number(1, GLIDE)?),
        "click" => Step::Click(target(0)?, MouseButton::Left, 1, mods()?),
        "dclick" => Step::Click(target(0)?, MouseButton::Left, 2, mods()?),
        "rclick" => Step::Click(target(0)?, MouseButton::Right, 1, mods()?),
        "drag" => {
            let targets = rest.iter().filter(|w| w.starts_with('"') || w.contains(',')).map(|w| parse_target(w)).collect::<Result<Vec<_>, _>>()?;
            if targets.len() < 2 {
                return Err("drag needs two or more targets".into());
            }
            let secs = rest.iter().find(|w| w.parse::<f64>().is_ok()).map_or(1.0, |w| w.parse().unwrap());
            Step::Drag(targets, secs, mods()?)
        }
        "scroll" => Step::Scroll(target(0)?, number(1, 1.0)? as f32, number(2, 0.6)?, mods()?),
        "key" => {
            let (code, key) = parse_key(rest.first().ok_or("missing key")?)?;
            Step::Key(code, key, mods()?)
        }
        "pick" => Step::Pick(unquote(rest.first().ok_or("missing path")?).into()),
        "type" => Step::Type(unquote(rest.first().ok_or("missing text")?)),
        "dump" => Step::Dump,
        "end" => Step::End,
        other => return Err(format!("unknown command {other}")),
    })
}

fn unquote(word: &str) -> String {
    word.trim_matches('"').replace("\\n", "\n")
}

fn parse_target(word: &str) -> Result<Target, String> {
    if let Some(body) = word.strip_prefix('"') {
        let end = body.rfind('"').ok_or("unterminated label")?;
        let text = body[..end].to_string();
        let mut tail = &body[end + 1..];
        let mut nth = 1;
        let mut offset = Vec2::ZERO;
        if let Some(t) = tail.strip_prefix('@') {
            let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
            nth = digits.parse().map_err(|_| format!("bad index in {word}"))?;
            tail = &t[digits.len()..];
        }
        if let Some(t) = tail.strip_prefix('+') {
            let (x, y) = t.split_once(',').ok_or(format!("bad offset in {word}"))?;
            offset = Vec2::new(x.parse().map_err(|_| format!("bad offset in {word}"))?, y.parse().map_err(|_| format!("bad offset in {word}"))?);
        }
        return Ok(Target::Label { text, nth, offset });
    }
    let (x, y) = word.split_once(',').ok_or(format!("bad target {word}"))?;
    Ok(Target::Fraction(Vec2::new(
        x.parse().map_err(|_| format!("bad target {word}"))?,
        y.parse().map_err(|_| format!("bad target {word}"))?,
    )))
}

fn parse_key(word: &str) -> Result<(KeyCode, Key), String> {
    let named = |code, key| Ok((code, key));
    match word {
        "Escape" => named(KeyCode::Escape, Key::Escape),
        "Delete" => named(KeyCode::Delete, Key::Delete),
        "Backspace" => named(KeyCode::Backspace, Key::Backspace),
        "Enter" => named(KeyCode::Enter, Key::Enter),
        "Left" => named(KeyCode::ArrowLeft, Key::ArrowLeft),
        "Right" => named(KeyCode::ArrowRight, Key::ArrowRight),
        "[" => Ok((KeyCode::BracketLeft, Key::Character("[".into()))),
        "]" => Ok((KeyCode::BracketRight, Key::Character("]".into()))),
        "?" => Ok((KeyCode::Slash, Key::Character("?".into()))),
        w if w.len() == 1 => {
            let c = w.chars().next().unwrap();
            let code = match c.to_ascii_uppercase() {
                'A' => KeyCode::KeyA, 'B' => KeyCode::KeyB, 'C' => KeyCode::KeyC, 'D' => KeyCode::KeyD,
                'E' => KeyCode::KeyE, 'F' => KeyCode::KeyF, 'G' => KeyCode::KeyG, 'H' => KeyCode::KeyH,
                'I' => KeyCode::KeyI, 'J' => KeyCode::KeyJ, 'K' => KeyCode::KeyK, 'L' => KeyCode::KeyL,
                'M' => KeyCode::KeyM, 'N' => KeyCode::KeyN, 'O' => KeyCode::KeyO, 'P' => KeyCode::KeyP,
                'Q' => KeyCode::KeyQ, 'R' => KeyCode::KeyR, 'S' => KeyCode::KeyS, 'T' => KeyCode::KeyT,
                'U' => KeyCode::KeyU, 'V' => KeyCode::KeyV, 'W' => KeyCode::KeyW, 'X' => KeyCode::KeyX,
                'Y' => KeyCode::KeyY, 'Z' => KeyCode::KeyZ,
                '0' => KeyCode::Digit0, '1' => KeyCode::Digit1, '2' => KeyCode::Digit2, '3' => KeyCode::Digit3,
                '4' => KeyCode::Digit4, '5' => KeyCode::Digit5, '6' => KeyCode::Digit6, '7' => KeyCode::Digit7,
                '8' => KeyCode::Digit8, '9' => KeyCode::Digit9,
                _ => return Err(format!("unknown key {w}")),
            };
            Ok((code, Key::Character(c.to_ascii_lowercase().to_string().into())))
        }
        other => Err(format!("unknown key {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_parses() {
        let steps = parse(
            "# intro\ntitle \"Squiggle\" \"waveforms\"\nwait 2\nclick \"Demo\"\nclick \"Fit\"@2+4,-3 shift\n\
             drag 0.5,0.5 0.6,0.5 1.5 cmd\nscroll 0.5,0.5 -3\nkey P\nkey Z cmd\ncaption\n",
        )
        .expect("parses");
        assert_eq!(steps.len(), 9);
        assert_eq!(steps[0], Step::Title(Some(("Squiggle".into(), "waveforms".into()))));
        assert_eq!(
            steps[3],
            Step::Click(
                Target::Label { text: "Fit".into(), nth: 2, offset: Vec2::new(4.0, -3.0) },
                MouseButton::Left,
                1,
                Mods { shift: true, ..Mods::default() }
            )
        );
        assert_eq!(steps[8], Step::Caption(None));
    }

    #[test]
    fn bad_lines_are_reported() {
        assert!(parse("wobble").is_err());
        assert!(parse("click \"Demo").is_err());
    }
}
