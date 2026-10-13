//! The look shared by every window: style, fonts, the palette, button styles
//! and where windows first open.

use std::sync::Arc;

use bevy_egui::egui;
use squiggle_core::Rgba;

/// Fills the button that carries out a window's main action, and selections.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 112, 160);
/// The accent where it outlines or underlines rather than fills.
pub const ACCENT_BRIGHT: egui::Color32 = egui::Color32::from_rgb(70, 170, 220);
/// Fills a destructive button while it is hovered or pressed.
pub const DANGER: egui::Color32 = egui::Color32::from_rgb(190, 55, 55);

/// Passed, accepted, confident.
pub const GOOD: egui::Color32 = egui::Color32::from_rgb(90, 210, 110);
/// Needs a look: weak, stale, unsaved, offline.
pub const WARN: egui::Color32 = egui::Color32::from_rgb(240, 190, 60);
/// Failed or rejected.
pub const BAD: egui::Color32 = egui::Color32::from_rgb(235, 80, 80);
/// Secondary text drawn by hand, where `ui.weak` cannot reach.
pub const MUTED: egui::Color32 = egui::Color32::from_gray(160);

/// The two labels the classifier knows by sight.
pub const EXPLOSION: egui::Color32 = egui::Color32::from_rgb(240, 140, 70);
pub const EARTHQUAKE: egui::Color32 = egui::Color32::from_rgb(110, 170, 240);

/// Distinct colours for categories without one of their own; the first two
/// match the classifier's labels.
pub const CATEGORIES: [egui::Color32; 10] = [
    EARTHQUAKE,
    EXPLOSION,
    egui::Color32::from_rgb(120, 200, 120),
    egui::Color32::from_rgb(210, 110, 200),
    egui::Color32::from_rgb(240, 210, 90),
    egui::Color32::from_rgb(90, 210, 210),
    egui::Color32::from_rgb(230, 100, 110),
    egui::Color32::from_rgb(160, 140, 240),
    egui::Color32::from_rgb(190, 160, 110),
    egui::Color32::from_rgb(150, 220, 170),
];

/// A class label's colour.
pub fn label_color(label: &str) -> egui::Color32 {
    match label {
        squiggle_core::taxonomy::EXPLOSION => EXPLOSION,
        squiggle_core::taxonomy::EARTHQUAKE => EARTHQUAKE,
        _ => egui::Color32::from_gray(190),
    }
}

/// A project colour (a phenomenology's, a trace's) in egui's terms.
pub fn color32(color: Rgba) -> egui::Color32 {
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgb(byte(color.red), byte(color.green), byte(color.blue))
}

/// Gap between a docked window and the edges of the plot.
const MARGIN: f32 = 10.0;

/// Sets the style and fonts; once, at startup.
pub fn install(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, style);
    install_fonts(ctx);
}

/// egui's dark style with the accent for selections, a little more room
/// between widgets, and rounder, shadowed windows.
fn style(style: &mut egui::Style) {
    let spacing = &mut style.spacing;
    spacing.item_spacing = egui::vec2(8.0, 5.0);
    spacing.button_padding = egui::vec2(6.0, 3.0);
    spacing.window_margin = egui::Margin::same(10);
    spacing.menu_margin = egui::Margin::same(6);

    let visuals = &mut style.visuals;
    visuals.selection.bg_fill = ACCENT;
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    visuals.hyperlink_color = ACCENT_BRIGHT;
    visuals.warn_fg_color = WARN;
    visuals.error_fg_color = BAD;
    visuals.window_corner_radius = egui::CornerRadius::same(8);
    visuals.menu_corner_radius = egui::CornerRadius::same(6);
    visuals.window_shadow = egui::Shadow { offset: [0, 6], blur: 18, spread: 0, color: egui::Color32::from_black_alpha(110) };
    visuals.popup_shadow = egui::Shadow { offset: [0, 3], blur: 10, spread: 0, color: egui::Color32::from_black_alpha(90) };
    visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(70));
    for widget in [&mut visuals.widgets.inactive, &mut visuals.widgets.hovered, &mut visuals.widgets.active, &mut visuals.widgets.open] {
        widget.corner_radius = egui::CornerRadius::same(4);
    }
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT_BRIGHT);
}

/// egui's default fonts are subset and lack arrows, check marks, warning
/// signs and bullets; DejaVu Sans covers them as the last fallback.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "dejavu".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!("../assets/fonts/DejaVuSans.ttf"))),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("dejavu".to_owned());
    }
    ctx.set_fonts(fonts);
}

/// A button for the action a window exists to perform.
pub fn primary(text: impl Into<String>) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text).color(egui::Color32::WHITE)).fill(ACCENT)
}

/// Adds a button that turns red under the pointer, for removing and rejecting.
pub fn danger(ui: &mut egui::Ui, enabled: bool, button: egui::Button) -> egui::Response {
    ui.scope(|ui| {
        let widgets = &mut ui.visuals_mut().widgets;
        for state in [&mut widgets.hovered, &mut widgets.active] {
            state.weak_bg_fill = DANGER;
            state.bg_fill = DANGER;
            state.fg_stroke.color = egui::Color32::WHITE;
        }
        ui.add_enabled(enabled, button)
    })
    .inner
}

/// A table's header row; the columns named in `numeric` hold numbers and
/// are right-aligned to match them.
pub fn headings(ui: &mut egui::Ui, headings: &[&str], numeric: &[&str]) {
    for heading in headings {
        if numeric.contains(heading) {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.strong(*heading));
        } else {
            ui.strong(*heading);
        }
    }
}

/// A number in a table: monospace and right-aligned, so digits line up.
pub fn number(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    number_colored(ui, text, ui.visuals().text_color())
}

/// A number in a table, coloured by what it says.
pub fn number_colored(ui: &mut egui::Ui, text: impl Into<String>, color: egui::Color32) -> egui::Response {
    let text = egui::RichText::new(text).monospace().color(color);
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(text)).inner
}

/// Remembers the plot's rectangle, so windows can open over its edge.
pub fn set_plot_rect(ctx: &egui::Context, rect: egui::Rect) {
    ctx.data_mut(|d| d.insert_temp(plot_rect_id(), rect));
}

/// A tool window, first opened against the plot's top-right corner: clear of
/// the track labels and the sidebar. Windows already docked there push it
/// down and left a step, so each one's title bar stays in view.
pub fn tool_window(ctx: &egui::Context, title: impl Into<egui::WidgetText>) -> egui::Window<'static> {
    const STEP: f32 = 28.0;
    let plot = ctx.data(|d| d.get_temp(plot_rect_id())).unwrap_or_else(|| ctx.content_rect());
    let corners: Vec<egui::Pos2> = ctx.memory(|m| {
        m.areas()
            .visible_layer_ids()
            .into_iter()
            .filter(|layer| layer.order == egui::Order::Middle)
            .filter_map(|layer| m.area_rect(layer.id))
            .map(|rect| rect.right_top())
            .collect()
    });
    let mut corner = plot.right_top() + egui::vec2(-MARGIN, MARGIN);
    while corners.iter().any(|c| c.distance(corner) < STEP / 2.0) {
        corner += egui::vec2(-STEP, STEP);
    }
    egui::Window::new(title).pivot(egui::Align2::RIGHT_TOP).default_pos(corner)
}

/// A dialog, first opened in the middle of the window.
pub fn dialog(ctx: &egui::Context, title: impl Into<egui::WidgetText>) -> egui::Window<'static> {
    egui::Window::new(title)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
        .collapsible(false)
        .resizable(false)
}

fn plot_rect_id() -> egui::Id {
    egui::Id::new("squiggle_plot_rect")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_in_the_ui_has_a_font() {
        let ctx = egui::Context::default();
        install_fonts(&ctx);
        let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();

        let sources = [
            include_str!("ui.rs"),
            include_str!("tracks.rs"),
            include_str!("process_ui.rs"),
            include_str!("locate_ui.rs"),
            include_str!("events_ui.rs"),
            include_str!("classify_ui.rs"),
            include_str!("models_ui.rs"),
            include_str!("templates_ui.rs"),
            include_str!("scatter_ui.rs"),
            include_str!("project_ui.rs"),
            include_str!("sync_ui.rs"),
            include_str!("live_ui.rs"),
            include_str!("showcase.rs"),
            include_str!("plot/overlay.rs"),
        ];
        // `has_glyph` misses glyphs from the face that also holds `�`, so
        // compare what is drawn with what `�` draws instead.
        let drawn = |c: char| {
            ctx.fonts_mut(|f| {
                let galley = f.layout_no_wrap(c.to_string(), egui::FontId::proportional(14.0), egui::Color32::WHITE);
                galley.rows[0].row.glyphs[0].uv_rect
            })
        };
        let replacement = drawn(char::REPLACEMENT_CHARACTER);
        let mut missing: Vec<char> = sources
            .iter()
            .flat_map(|source| source.chars())
            .filter(|c| !c.is_ascii() && drawn(*c) == replacement)
            .collect();
        missing.sort_unstable();
        missing.dedup();
        assert!(missing.is_empty(), "no font has {missing:?}");
    }
}
