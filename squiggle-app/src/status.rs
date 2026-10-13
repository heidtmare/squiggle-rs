//! The status bar along the bottom, the notices stacked above it, the empty
//! state and the drop target.

use bevy::prelude::*;
use bevy::window::FileDragAndDrop;
use bevy_egui::egui;

use crate::theme;
use crate::workspace::Workspace;

/// How long a notice that is not an error stays up, seconds.
const NOTICE_SECONDS: f64 = 5.0;
/// How long the last edit stays in the status bar, seconds.
const EDIT_SECONDS: f64 = 8.0;
/// Background work that runs at least this long says when it is done, seconds.
const NOTIFY_AFTER: f64 = 2.0;
/// Notices shown at once; older ones wait.
const MAX_NOTICES: usize = 4;

#[derive(Resource, Default)]
pub struct Status {
    notices: Vec<Notice>,
    /// The undo and redo depths last seen, to notice edits, undos and redos.
    depth: (usize, usize),
    /// The newest edit, undo or redo: what happened, how to reverse it, and when.
    last_edit: Option<(String, String, f64)>,
    /// Background work in progress and when it started.
    busy: Vec<(&'static str, f64)>,
    /// Files are being dragged over the window.
    pub dragging: bool,
}

struct Notice {
    title: String,
    text: String,
    error: bool,
    /// When it was posted, app seconds; set when first drawn.
    since: Option<f64>,
}

impl Status {
    /// A notice that stays until dismissed.
    pub fn error(&mut self, title: impl Into<String>, text: impl Into<String>) {
        self.notices.push(Notice { title: title.into(), text: text.into(), error: true, since: None });
    }

    /// Notes the work running now; work that ran a while says when it is done.
    pub fn observe_busy(&mut self, busy: &[&'static str], now: f64) {
        let finished: Vec<&'static str> = self
            .busy
            .iter()
            .filter(|(job, since)| !busy.contains(job) && now - since >= NOTIFY_AFTER)
            .map(|(job, _)| *job)
            .collect();
        for job in finished {
            self.info("Done", job);
        }
        self.busy.retain(|(job, _)| busy.contains(job));
        for job in busy {
            if !self.busy.iter().any(|(j, _)| j == job) {
                self.busy.push((job, now));
            }
        }
    }

    /// A notice that goes away by itself.
    pub fn info(&mut self, title: impl Into<String>, text: impl Into<String>) {
        self.notices.push(Notice { title: title.into(), text: text.into(), error: false, since: None });
    }

    /// Notices an edit, undo or redo since the last frame.
    pub fn observe_history(&mut self, workspace: &Workspace, shortcuts: (&str, &str), now: f64) {
        let (undo, redo) = workspace.history_depth();
        let (was_undo, was_redo) = std::mem::replace(&mut self.depth, (undo, redo));
        let (undo_key, redo_key) = shortcuts;
        let label = |l: Option<&str>| l.unwrap_or_default().to_string();
        let edit = if undo > was_undo && redo == 0 {
            Some((label(workspace.undo_label()), format!("{undo_key} to undo")))
        } else if undo < was_undo && redo > was_redo {
            Some((format!("Undid {}", label(workspace.redo_label())), format!("{redo_key} to redo")))
        } else if undo > was_undo && redo < was_redo {
            Some((format!("Redid {}", label(workspace.undo_label())), format!("{undo_key} to undo")))
        } else {
            None
        };
        if edit.is_some() || (undo, redo) == (0, 0) {
            self.last_edit = edit.map(|(what, hint)| (what, hint, now));
        }
    }
}

/// Follows files dragged over the window.
pub fn track_drags(mut drags: MessageReader<FileDragAndDrop>, mut status: ResMut<Status>) {
    for drag in drags.read() {
        status.dragging = matches!(drag, FileDragAndDrop::HoveredFile { .. });
    }
}

/// What the status bar shows, gathered by the chrome.
pub struct Readout {
    /// Work running in the background.
    pub busy: Vec<&'static str>,
    /// Where the pointer is, in the plot's terms.
    pub cursor: Option<String>,
    pub selection: Option<String>,
    /// The live feed: what it says and whether all is well.
    pub live: Option<(String, bool)>,
    pub sync: Option<(String, bool)>,
}

/// The bar's contents, left to right: work in progress or the last edit,
/// then the cursor, the selection, the live feed and the shared project.
pub fn bar(ui: &mut egui::Ui, status: &Status, readout: Readout) {
    ui.horizontal(|ui| {
        let now = ui.input(|i| i.time);
        if !readout.busy.is_empty() {
            ui.spinner();
            ui.label(format!("{}…", readout.busy.join(", ")));
        } else if let Some((what, hint, at)) = status.last_edit.as_ref().filter(|(_, _, at)| now - at < EDIT_SECONDS) {
            ui.label(egui::RichText::new("✔").color(theme::GOOD));
            ui.label(what);
            ui.weak(format!("— {hint}"));
            ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(EDIT_SECONDS - (now - at)));
        } else {
            ui.weak("Press ? for keyboard shortcuts");
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut first = true;
            let mut gap = |ui: &mut egui::Ui| {
                if !std::mem::take(&mut first) {
                    ui.separator();
                }
            };
            if let Some((text, ok)) = readout.sync {
                gap(ui);
                ui.label(egui::RichText::new(format!("⇅ {text}")).color(if ok { theme::MUTED } else { theme::WARN }));
            }
            if let Some((text, ok)) = readout.live {
                gap(ui);
                ui.label(egui::RichText::new("●").color(if ok { theme::GOOD } else { theme::WARN }));
                ui.label(text);
            }
            if let Some(text) = readout.selection {
                gap(ui);
                ui.label(text);
            }
            if let Some(text) = readout.cursor {
                gap(ui);
                ui.label(egui::RichText::new(text).monospace());
            }
        });
    });
}

/// Notices stacked up from the bottom-right corner of the plot, newest at
/// the bottom; errors stay until dismissed.
pub fn notices(ctx: &egui::Context, status: &mut Status, plot: egui::Rect) {
    let now = ctx.input(|i| i.time);
    status.notices.retain(|n| n.error || n.since.is_none_or(|since| now - since < NOTICE_SECONDS));
    if status.notices.is_empty() {
        return;
    }

    let mut dismissed = None;
    egui::Area::new("squiggle_notices".into())
        .order(egui::Order::Foreground)
        .pivot(egui::Align2::RIGHT_BOTTOM)
        .fixed_pos(plot.right_bottom() - egui::vec2(12.0, 12.0))
        .show(ctx, |ui| {
            ui.set_max_width(360.0);
            let shown = status.notices.len().saturating_sub(MAX_NOTICES);
            for (i, notice) in status.notices.iter_mut().enumerate().skip(shown) {
                let since = *notice.since.get_or_insert(now);
                if !notice.error {
                    ctx.request_repaint_after(std::time::Duration::from_secs_f64(NOTICE_SECONDS - (now - since)));
                }
                let edge = if notice.error { theme::BAD } else { theme::ACCENT_BRIGHT };
                egui::Frame::popup(ui.style()).stroke(egui::Stroke::new(1.0, edge)).show(ui, |ui| {
                    ui.horizontal_top(|ui| {
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new(&notice.title).strong().color(edge));
                            ui.label(&notice.text);
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            if ui.small_button("✕").on_hover_text("Dismiss").clicked() {
                                dismissed = Some(i);
                            }
                        });
                    });
                });
                ui.add_space(6.0);
            }
        });
    if let Some(i) = dismissed {
        status.notices.remove(i);
    }
}

/// Outlines the plot while files are dragged over the window.
pub fn drop_target(ctx: &egui::Context, plot: egui::Rect) {
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, "squiggle_drop".into()));
    painter.rect_filled(plot, 0.0, theme::ACCENT.gamma_multiply(0.15));
    painter.rect_stroke(plot.shrink(4.0), 8.0, egui::Stroke::new(2.0, theme::ACCENT_BRIGHT), egui::StrokeKind::Inside);
    painter.text(plot.center(), egui::Align2::CENTER_CENTER, "Drop to import", egui::FontId::proportional(22.0), egui::Color32::WHITE);
}
