//! The look shared by every window: fonts, the accent and danger colours,
//! button styles and where windows first open.

use std::sync::Arc;

use bevy_egui::egui;

/// Fills the button that carries out a window's main action.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 112, 160);
/// Fills a destructive button while it is hovered or pressed.
pub const DANGER: egui::Color32 = egui::Color32::from_rgb(190, 55, 55);

/// Gap between a docked window and the edges of the plot.
const MARGIN: f32 = 10.0;

/// egui's default fonts are subset and lack arrows, check marks, warning
/// signs and bullets; DejaVu Sans covers them as the last fallback.
pub fn install_fonts(ctx: &egui::Context) {
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
