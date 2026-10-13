//! The demo waveform, and the command-line options that seed it.
//!
//! The Java project generated this shape in two places: the toolbar's *Add
//! Squiggle* button, and `SquigglePlotExample`, which filled a plot with a
//! hundred of them to see how the renderer coped. Both live here, the latter
//! as a startup flag.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use rand::Rng;
use squiggle_core::{Point, Rgba, Squiggle, Style};

use crate::plot::AddSquiggle;
use crate::showcase::{self, LoadScene, Scene, Showcase};
use crate::workspace::{SyntheticEvent, WorkspaceRequest};

/// Samples per squiggle, as the Java toolbar used.
pub const DEFAULT_SAMPLE_COUNT: usize = 1000;

/// Horizontal spacing between samples, and the amplitude of the wave.
const SPACER: f64 = 10.0;

/// Startup options gathered from the command line.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct DemoOptions {
    /// How many squiggles to put on the plot at startup.
    pub seed_count: usize,
    /// How many samples each generated squiggle carries.
    pub sample_count: usize,
    /// A transform demo scene to open at startup.
    pub scene: Option<Scene>,
    /// Save a screenshot here once the plot has settled, then quit.
    pub screenshot: Option<std::path::PathBuf>,
    /// Project file to open at startup.
    pub project: Option<std::path::PathBuf>,
    /// Waveform files to import at startup.
    pub import: Vec<std::path::PathBuf>,
    /// Start with a synthetic event.
    pub synthetic: Option<SyntheticEvent>,
    /// Connect to this live feed (`host:port`).
    pub live: Option<String>,
    /// Start with a live replay of the event families.
    pub live_demo: bool,
    /// Who edits; else `$SQUIGGLE_ANALYST`, else the OS user.
    pub analyst: Option<String>,
    /// Sync server (`host:port`) the Shared Project window starts with.
    pub server: Option<String>,
    /// Tour script to drive the app with (see `tour`).
    pub tour: Option<std::path::PathBuf>,
    /// Record the tour to this video file.
    pub record: Option<std::path::PathBuf>,
}

impl Default for DemoOptions {
    fn default() -> Self {
        Self {
            seed_count: 0,
            sample_count: DEFAULT_SAMPLE_COUNT,
            scene: None,
            screenshot: None,
            project: None,
            import: Vec::new(),
            synthetic: None,
            live: None,
            live_demo: false,
            analyst: None,
            server: None,
            tour: None,
            record: None,
        }
    }
}

pub const USAGE: &str = "\
Usage: squiggle-app [--project FILE | --synthetic | --located | --swarm | --mixed | --catalogue | --families]
                    [--live HOST:PORT | --live-demo] [--analyst NAME] [--server HOST:PORT]
                    [--squiggles N] [--samples N]
                    [--demo SCENE] [--screenshot FILE] [--tour SCRIPT [--record OUT.mp4]]
                    [WAVEFORM...]

  WAVEFORM...        import these files at startup (miniSEED, SAC, SigMF,
                     WAV, CSV; anything else as raw binary)
  --project FILE     open a .squiggle project
  --synthetic        start with a synthetic event of known delays (alignment demo)
  --located          start with a synthetic EMP, seismic and infrasound event of
                     known source (picking and location demo)
  --swarm            start with two minutes of synthetic seismic data holding
                     several events and glitches (detection demo)
  --mixed            start with seven minutes of synthetic earthquakes and
                     explosions on seismic and infrasound sensors
                     (classification demo)
  --catalogue        start with 27 minutes of 40 varied synthetic earthquakes and
                     explosions on the same sensors (training data for models)
  --families         start with 16 minutes of repeats from four source families,
                     one weak repeat each (clustering and template demo)
  --live HOST:PORT   connect to a live feed (heidtmare.squiggle.stream/1)
  --live-demo        start with the event families replayed live at ×10
  --analyst NAME     record edits as NAME (default $SQUIGGLE_ANALYST, else the OS user)
  --server HOST:PORT sync server for sharing projects (heidtmare.squiggle.sync/1)
  --squiggles N      put N squiggles on the plot at startup (default 0)
  --samples N        samples per generated squiggle (default 1000)
  --demo SCENE       open a transform demo: fourier or wavelet
  --screenshot FILE  save a PNG of the window once the plot settles, then quit
  --tour SCRIPT      drive the app with a scripted tour, then quit
  --record OUT.mp4   record the tour to a video (needs ffmpeg)
";

impl DemoOptions {
    /// Parses the arguments after the program name.
    pub fn from_args<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut options = Self::default();
        let mut args = args.into_iter();

        while let Some(argument) = args.next() {
            let value = |args: &mut I::IntoIter| {
                args.next()
                    .ok_or_else(|| format!("{argument} needs a number"))?
                    .parse::<usize>()
                    .map_err(|_| format!("{argument} needs a number"))
            };

            match argument.as_str() {
                "--squiggles" => options.seed_count = value(&mut args)?,
                "--samples" => options.sample_count = value(&mut args)?,
                "--demo" => {
                    let name = args.next().ok_or_else(|| format!("{argument} needs a scene"))?;
                    options.scene = Some(
                        Scene::parse(&name)
                            .ok_or_else(|| format!("unknown demo scene: {name}\n\n{USAGE}"))?,
                    );
                }
                "--screenshot" => {
                    let path = args.next().ok_or_else(|| format!("{argument} needs a file"))?;
                    options.screenshot = Some(path.into());
                }
                "--tour" => {
                    let path = args.next().ok_or_else(|| format!("{argument} needs a file"))?;
                    options.tour = Some(path.into());
                }
                "--record" => {
                    let path = args.next().ok_or_else(|| format!("{argument} needs a file"))?;
                    options.record = Some(path.into());
                }
                "--project" => {
                    let path = args.next().ok_or_else(|| format!("{argument} needs a file"))?;
                    options.project = Some(path.into());
                }
                "--synthetic" => options.synthetic = Some(SyntheticEvent::Alignment),
                "--located" => options.synthetic = Some(SyntheticEvent::Located),
                "--swarm" => options.synthetic = Some(SyntheticEvent::Swarm),
                "--mixed" => options.synthetic = Some(SyntheticEvent::Mixed),
                "--catalogue" => options.synthetic = Some(SyntheticEvent::Catalogue),
                "--families" => options.synthetic = Some(SyntheticEvent::Families),
                "--live" => options.live = Some(args.next().ok_or_else(|| format!("{argument} needs host:port"))?),
                "--live-demo" => options.live_demo = true,
                "--analyst" => options.analyst = Some(args.next().filter(|a| !a.trim().is_empty()).ok_or_else(|| format!("{argument} needs a name"))?),
                "--server" => options.server = Some(args.next().ok_or_else(|| format!("{argument} needs host:port"))?),
                "--help" | "-h" => return Err(USAGE.to_string()),
                other if other.starts_with('-') => {
                    return Err(format!("unrecognised argument: {other}\n\n{USAGE}"));
                }
                path => options.import.push(path.into()),
            }
        }

        Ok(options)
    }
}

/// Fills the plot at startup from `--project`, waveform paths, `--squiggles` or `--demo`.
pub fn seed(
    options: Res<DemoOptions>,
    mut add: MessageWriter<AddSquiggle>,
    mut load: MessageWriter<LoadScene>,
    mut workspace: MessageWriter<WorkspaceRequest>,
) {
    if let Some(path) = &options.project {
        workspace.write(WorkspaceRequest::Open(path.clone()));
    }
    if let Some(event) = options.synthetic {
        workspace.write(WorkspaceRequest::Synthetic(event));
    }
    if !options.import.is_empty() {
        workspace.write(WorkspaceRequest::Import(options.import.clone()));
    }
    if let Some(scene) = options.scene {
        load.write(LoadScene(scene));
    }
    for index in 0..options.seed_count {
        add.write(AddSquiggle(generate(index, options.sample_count)));
    }
}

/// Builds one demo waveform: a sine riding on a baseline that steps up with
/// each successive squiggle, in a random colour.
pub fn generate(index: usize, sample_count: usize) -> Squiggle {
    let points = (0..sample_count).map(|sample| {
        let x = sample as f64 * SPACER;
        let y = index as f64 * SPACER + x.sin() * SPACER;
        Point::new(x, y)
    });

    Squiggle::with_points(format!("Squiggle {index}"), points)
        .with_style(Style::new(random_color(), Style::SQUIGGLE.line_width))
}

/// Frames to let the plot render after it settles before capturing it, and to
/// let the capture reach the disk before quitting.
const SCREENSHOT_SETTLE_FRAMES: u32 = 20;
const SCREENSHOT_SAVE_FRAMES: u32 = 30;

/// Takes the `--screenshot` capture once the plot has settled, then quits.
pub fn screenshot(
    mut commands: Commands,
    options: Res<DemoOptions>,
    showcase: Res<Showcase>,
    mut frames: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(path) = &options.screenshot else {
        return;
    };
    // Without a scene there is nothing to wait for but the first frames.
    if options.scene.is_some() && !showcase::settled(&showcase) {
        return;
    }

    *frames += 1;
    if *frames == SCREENSHOT_SETTLE_FRAMES {
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path.clone()));
    } else if *frames == SCREENSHOT_SETTLE_FRAMES + SCREENSHOT_SAVE_FRAMES {
        exit.write(AppExit::Success);
    }
}

fn random_color() -> Rgba {
    let mut rng = rand::rng();
    Rgba::new(rng.random(), rng.random(), rng.random())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<DemoOptions, String> {
        DemoOptions::from_args(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn no_arguments_seeds_an_empty_plot() {
        assert_eq!(parse(&[]), Ok(DemoOptions::default()));
    }

    #[test]
    fn both_counts_can_be_set() {
        assert_eq!(
            parse(&["--squiggles", "100", "--samples", "10000"]),
            Ok(DemoOptions { seed_count: 100, sample_count: 10_000, ..DemoOptions::default() })
        );
    }

    #[test]
    fn a_demo_scene_and_screenshot_can_be_asked_for() {
        let options = parse(&["--demo", "Wavelet", "--screenshot", "out.png"]).expect("parses");
        assert_eq!(options.scene, Some(Scene::Wavelet));
        assert_eq!(options.screenshot, Some("out.png".into()));
    }

    #[test]
    fn an_unknown_demo_scene_is_rejected() {
        assert!(parse(&["--demo", "laplace"]).is_err());
    }

    #[test]
    fn a_missing_value_is_an_error_not_a_silent_default() {
        assert!(parse(&["--squiggles"]).is_err());
    }

    #[test]
    fn a_non_numeric_value_is_rejected() {
        assert!(parse(&["--samples", "lots"]).is_err());
    }

    #[test]
    fn a_project_and_waveforms_can_be_opened() {
        let options = parse(&["--project", "p.squiggle", "a.wav", "b.csv"]).expect("parses");
        assert_eq!(options.project, Some("p.squiggle".into()));
        assert_eq!(options.import, vec![std::path::PathBuf::from("a.wav"), "b.csv".into()]);
    }

    #[test]
    fn the_synthetic_event_can_be_asked_for() {
        assert_eq!(parse(&["--synthetic"]).unwrap().synthetic, Some(SyntheticEvent::Alignment));
        assert_eq!(parse(&["--located"]).unwrap().synthetic, Some(SyntheticEvent::Located));
        assert_eq!(parse(&["--swarm"]).unwrap().synthetic, Some(SyntheticEvent::Swarm));
        assert_eq!(parse(&["--mixed"]).unwrap().synthetic, Some(SyntheticEvent::Mixed));
        assert_eq!(parse(&["--catalogue"]).unwrap().synthetic, Some(SyntheticEvent::Catalogue));
        assert_eq!(parse(&["--families"]).unwrap().synthetic, Some(SyntheticEvent::Families));
        assert_eq!(parse(&["--live", "h:1"]).unwrap().live.as_deref(), Some("h:1"));
        assert!(parse(&["--live-demo"]).unwrap().live_demo);
        assert!(parse(&["--live"]).is_err());
        assert_eq!(parse(&["--analyst", "bo", "--server", "h:2"]).unwrap().analyst.as_deref(), Some("bo"));
        assert_eq!(parse(&["--server", "h:2"]).unwrap().server.as_deref(), Some("h:2"));
        assert!(parse(&["--analyst"]).is_err());
    }

    #[test]
    fn unknown_arguments_are_reported_with_the_usage() {
        let error = parse(&["--colour", "red"]).expect_err("should not parse");
        assert!(error.contains("--colour"), "{error}");
        assert!(error.contains("Usage"), "{error}");
    }

    #[test]
    fn a_generated_squiggle_has_the_samples_it_was_asked_for() {
        let squiggle = generate(3, 64);
        assert_eq!(squiggle.len(), 64);
        assert_eq!(squiggle.name, "Squiggle 3");
    }

    #[test]
    fn successive_squiggles_sit_above_one_another() {
        // The baseline steps up by SPACER per index, so the later squiggle's
        // band of values sits higher.
        let lower = generate(0, 256).bounds().expect("has points");
        let upper = generate(5, 256).bounds().expect("has points");
        assert!(upper.min_y > lower.min_y, "{upper:?} vs {lower:?}");
    }
}
