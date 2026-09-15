//! The demo waveform, and the command-line options that seed it.
//!
//! The Java project generated this shape in two places: the toolbar's *Add
//! Squiggle* button, and `SquigglePlotExample`, which filled a plot with a
//! hundred of them to see how the renderer coped. Both live here, the latter
//! as a startup flag.

use bevy::prelude::*;
use rand::Rng;
use squiggle_core::{Point, Rgba, Squiggle, Style};

use crate::plot::AddSquiggle;

/// Samples per squiggle, as the Java toolbar used.
pub const DEFAULT_SAMPLE_COUNT: usize = 1000;

/// Horizontal spacing between samples, and the amplitude of the wave.
const SPACER: f64 = 10.0;

/// Startup options gathered from the command line.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemoOptions {
    /// How many squiggles to put on the plot at startup.
    pub seed_count: usize,
    /// How many samples each generated squiggle carries.
    pub sample_count: usize,
}

impl Default for DemoOptions {
    fn default() -> Self {
        Self { seed_count: 0, sample_count: DEFAULT_SAMPLE_COUNT }
    }
}

pub const USAGE: &str = "\
Usage: squiggle-app [--squiggles N] [--samples N]

  --squiggles N  put N squiggles on the plot at startup (default 0)
  --samples N    samples per generated squiggle (default 1000)
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
                "--help" | "-h" => return Err(USAGE.to_string()),
                other => return Err(format!("unrecognised argument: {other}\n\n{USAGE}")),
            }
        }

        Ok(options)
    }
}

/// Fills the plot at startup when `--squiggles` asked for it.
pub fn seed(options: Res<DemoOptions>, mut add: MessageWriter<AddSquiggle>) {
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
            Ok(DemoOptions { seed_count: 100, sample_count: 10_000 })
        );
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
