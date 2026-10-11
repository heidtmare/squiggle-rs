//! Spectrogram strips: a per-row spectrogram drawn as a sprite beneath the
//! row's waveform.
//!
//! Like the waveform mesh, a strip covers three screen widths at about one
//! column per pixel and is rebuilt off-thread when the view leaves it or the
//! zoom drifts more than 2.5× from its column width. Complex traces show the
//! two-sided spectrum of their processed I/Q, centred on DC.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use squiggle_core::dsp::spectrogram;
use squiggle_core::transform::Complex;
use squiggle_core::{SampleKind, Window};

use crate::plot::{Plot, PlotMode};
use crate::tracks::{RowDisplay, Series, Track, TrackView};

const Z_STRIPS: f32 = -20.0;
/// Widest texture built; wider caches use longer columns.
const MAX_COLUMNS: usize = 8192;
/// Colour range: these quantiles of the strip's dB values.
const DB_QUANTILES: (f64, f64) = (0.05, 0.995);

/// What a strip was computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StripKey {
    lo: usize,
    hi: usize,
    hop: usize,
    nfft: usize,
    /// Identity of the series (changes with processing or component).
    series: usize,
}

struct StripImage {
    key: StripKey,
    columns: usize,
    bins: usize,
    rgba: Vec<u8>,
}

#[derive(Component)]
pub struct StripTask(Task<StripImage>);

/// On a track: its current strip.
#[derive(Component)]
pub struct Strip {
    key: StripKey,
    columns: usize,
    bins: usize,
    sprite: Entity,
}

/// On a strip sprite: the track it belongs to.
#[derive(Component)]
pub struct StripOf(Entity);

fn series_id(series: &Arc<Series>) -> usize {
    Arc::as_ptr(series) as usize
}

/// Starts a strip build for rows that show a spectrogram and lack a suitable one.
pub fn request_strips(
    mut commands: Commands,
    plot: Res<Plot>,
    mode: Res<PlotMode>,
    view: Res<TrackView>,
    tracks: Query<(Entity, &Track, Option<&Strip>), Without<StripTask>>,
) {
    if *mode != PlotMode::Tracks || !plot.is_usable() {
        return;
    }
    let width = plot.view.plot_width();
    for (entity, track, strip) in &tracks {
        let (Some(series), true, Some(_)) = (&track.series, track.display.has_spectrogram(), track.row) else {
            continue;
        };
        let (lo, hi) = track.indices(view.time.start, view.time.end(width));
        if hi <= lo {
            continue;
        }
        let ideal = ((hi - lo) as f64 / width).max(1.0);
        let id = series_id(series);
        let suits = strip.is_some_and(|s| {
            let k = s.key;
            let ratio = k.hop as f64 / ideal;
            k.series == id && k.nfft == track.nfft && k.lo <= lo && k.hi >= hi && (ratio >= 0.4 || k.hop == 1) && ratio <= 2.5
        });
        if suits {
            continue;
        }

        let span = hi - lo;
        let cache_hi = (hi + span).min(track.len);
        let mut hop = ideal.round().max(1.0) as usize;
        let cache_lo = lo.saturating_sub(span) / hop * hop;
        if (cache_hi - cache_lo).div_ceil(hop) > MAX_COLUMNS {
            hop = (cache_hi - cache_lo).div_ceil(MAX_COLUMNS);
        }
        let key = StripKey { lo: cache_lo, hi: cache_hi, hop, nfft: track.nfft, series: id };
        let (series, complex) = (series.clone(), track.kind == SampleKind::Complex);
        let task = AsyncComputeTaskPool::get().spawn(async move { build(&series, key, complex) });
        commands.entity(entity).insert(StripTask(task));
    }
}

fn build(series: &Series, key: StripKey, complex: bool) -> StripImage {
    let input: Vec<Complex> = if complex {
        series.samples.complex_range(key.lo, key.hi)
    } else {
        series.values.range(key.lo, key.hi).map(|v| Complex::from(f64::from(v))).collect()
    };
    let s = spectrogram(&input, key.nfft, key.hop, Window::Hann, complex);
    let (low, high) = s.db_range(DB_QUANTILES.0, DB_QUANTILES.1);
    // Row 0 is the highest bin.
    let mut rgba = Vec::with_capacity(s.columns * s.bins * 4);
    for bin in (0..s.bins).rev() {
        for column in 0..s.columns {
            let db = s.at(column, bin);
            let pixel = if db.is_finite() { colormap((db - low) / (high - low)) } else { [0; 4] };
            rgba.extend_from_slice(&pixel);
        }
    }
    StripImage { key, columns: s.columns, bins: s.bins, rgba }
}

/// Inferno-like ramp, `t` in `0..=1`.
pub(crate) fn colormap(t: f32) -> [u8; 4] {
    const STOPS: [[f32; 3]; 6] = [
        [0.0, 0.0, 4.0],
        [40.0, 11.0, 84.0],
        [101.0, 21.0, 110.0],
        [188.0, 55.0, 84.0],
        [249.0, 142.0, 9.0],
        [252.0, 255.0, 164.0],
    ];
    let x = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (x as usize).min(STOPS.len() - 2);
    let f = x - i as f32;
    let channel = |c: usize| (STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f).round() as u8;
    [channel(0), channel(1), channel(2), 255]
}

pub fn poll_strips(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut tracks: Query<(Entity, &mut StripTask, Option<&mut Strip>)>,
    mut sprites: Query<&mut Sprite, With<StripOf>>,
) {
    for (entity, mut task, strip) in &mut tracks {
        let Some(built) = check_ready(&mut task.0) else {
            continue;
        };
        commands.entity(entity).remove::<StripTask>();
        if built.columns == 0 || built.bins == 0 {
            continue;
        }
        let mut image = Image::new(
            Extent3d { width: built.columns as u32, height: built.bins as u32, depth_or_array_layers: 1 },
            TextureDimension::D2,
            built.rgba,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        // Rows usually have fewer pixels than bins; linear filtering avoids moiré.
        image.sampler = ImageSampler::linear();
        let handle = images.add(image);
        match strip {
            Some(mut strip) => {
                if let Ok(mut sprite) = sprites.get_mut(strip.sprite) {
                    sprite.image = handle;
                }
                strip.key = built.key;
                strip.columns = built.columns;
                strip.bins = built.bins;
            }
            None => {
                let sprite = commands.spawn((Sprite::from_image(handle), Visibility::Hidden, StripOf(entity))).id();
                commands.entity(entity).insert(Strip { key: built.key, columns: built.columns, bins: built.bins, sprite });
            }
        }
    }
}

/// Places each strip's visible part over its row.
pub fn draw_strips(
    plot: Res<Plot>,
    mode: Res<PlotMode>,
    view: Res<TrackView>,
    tracks: Query<(&Track, &Strip)>,
    mut sprites: Query<(&mut Sprite, &mut Transform, &mut Visibility), With<StripOf>>,
) {
    let area = &plot.view;
    let width = area.plot_width();
    let active = *mode == PlotMode::Tracks && plot.is_usable();
    for (track, strip) in &tracks {
        let Ok((mut sprite, mut transform, mut visibility)) = sprites.get_mut(strip.sprite) else {
            continue;
        };
        let k = strip.key;
        let time_of = |x: f64| track.drawn_start() + (k.lo as f64 + x * k.hop as f64) / track.rate;
        let from = view.time.start.max(time_of(0.0));
        let to = view.time.end(width).min(time_of(strip.columns as f64));
        let (Some((top, height)), true, true) = (track.row, active && to > from, track.display.has_spectrogram()) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        if top > area.insets.top + area.plot_height() || top + height < area.insets.top {
            *visibility = Visibility::Hidden;
            continue;
        }
        let texel = |t: f64| ((t - time_of(0.0)) * track.rate / k.hop as f64) as f32;
        sprite.rect = Some(Rect::new(texel(from), 0.0, texel(to), strip.bins as f32));
        sprite.custom_size = Some(Vec2::new(((to - from) * view.time.px_per_s) as f32, (height - 2.0).max(1.0) as f32));
        let centre_x = area.insets.left + view.time.x_of((from + to) / 2.0);
        *transform = Transform::from_xyz(
            (centre_x - area.width / 2.0) as f32,
            (area.height / 2.0 - (top + height / 2.0)) as f32,
            Z_STRIPS,
        );
        *visibility = Visibility::Visible;
    }
}

/// Drops strips of removed tracks or rows that stopped showing a spectrogram.
pub fn prune_strips(
    mut commands: Commands,
    sprites: Query<(Entity, &StripOf)>,
    tracks: Query<&Track>,
) {
    for (sprite, owner) in &sprites {
        let keep = tracks.get(owner.0).is_ok_and(|t| t.display != RowDisplay::Waveform);
        if !keep {
            commands.entity(sprite).despawn();
            if let Ok(mut owner) = commands.get_entity(owner.0) {
                owner.remove::<Strip>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_colormap_runs_dark_to_light() {
        let luma = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
        let samples: Vec<u32> = (0..=10).map(|i| luma(colormap(i as f32 / 10.0))).collect();
        assert!(samples.windows(2).all(|w| w[0] <= w[1]), "{samples:?}");
        assert_eq!(colormap(-1.0), colormap(0.0));
        assert_eq!(colormap(f32::NAN)[3], 255);
    }
}
